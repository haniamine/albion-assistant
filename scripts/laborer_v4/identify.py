"""Detect-then-classify laborer identification.

v3 ran a full-screen masked search per laborer template - 17 candidates x 2
monitors x ~98 ms = ~3.7 s, which is why it needed a prefetch thread. v4
locates the icon *once* and then classifies that single crop against every
candidate: 0.28 ms each, so the whole set costs single-digit milliseconds.

Classification is two-stage, because the two mistakes are different in kind
and only one of them is visible to a whole-icon score:

  profession  Different professions have different portraits and separate
              cleanly by score - the worst measured gap is 0.067.

  tier        Sibling tiers are *pixel-identical over 95% of the icon* once
              aligned; they differ only in a ~25x23 badge box. A whole-icon
              score puts them within 0.004 of each other, so it cannot decide
              between them at all. Measured on this repo's assets, an icon
              wearing an unknown (t6) badge still scored 0.986 against a t4
              template - above v3's 0.97 gate. Tier is therefore decided by a
              comparison weighted towards exactly the pixels that disagree
              between siblings, which puts the runner-up at least 1.8x worse
              even under heavy capture noise.

Both stages demand a *margin* over the runner-up, not just an absolute
threshold: a threshold cannot tell "clearly t5" from "t5 and t4 are tied", and
a tie is precisely when v3 guessed wrong.

Tier is decided against *every* loaded sibling, including tiers we hold no
journal for. Narrowing the comparison set to the inventory - as v3 did - means
a ready t3-bs is scored only against t5-bs, matches it at ~0.98, and is handed
the wrong journal. Here it is identified correctly as t3-bs and then refused
for want of a journal, which is a different and much safer outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assets import laborer_profession
from .config import Config
from .vision import Frame, Rect, Template, match_template, weighted_patch_distance

log = logging.getLogger("laborer.identify")


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    # top_left corrected for this asset's crop offset, giving one shared
    # coordinate space for differently-cropped templates.
    aligned_top_left: Tuple[int, int]


@dataclass(frozen=True)
class LaborerVerdict:
    name: str
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    icon_origin: Tuple[int, int]
    profession_margin: Optional[float]
    tier_distance: float
    tier_ratio: Optional[float]
    runner_up: Optional[str]

    @property
    def center(self) -> Tuple[int, int]:
        return self.top_left[0] + self.size[0] // 2, self.top_left[1] + self.size[1] // 2

    def describe(self) -> str:
        parts = [f"{self.name} score={self.score:.3f} tier-dist={self.tier_distance:.1f}"]
        if self.tier_ratio is not None and self.runner_up:
            parts.append(f"next={self.runner_up} x{self.tier_ratio:.2f}")
        if self.profession_margin is not None:
            parts.append(f"prof-margin={self.profession_margin:.3f}")
        return " ".join(parts)


@dataclass(frozen=True)
class Rejection:
    reason: str
    detail: str
    # Set when the icon was identified but is unusable, e.g. an unknown tier.
    likely_name: Optional[str] = None

    def describe(self) -> str:
        return f"{self.reason}: {self.detail}"


def score_candidates(
    frame: Frame,
    rect: Optional[Rect],
    templates: Dict[str, Template],
) -> List[Candidate]:
    """Score every template against one crop, best first."""
    if rect is None:
        source, region = frame.gray, frame.rect
    else:
        sub = frame.sub_gray(rect)
        if sub is None:
            return []
        source, region = sub

    candidates: List[Candidate] = []
    for name, template in templates.items():
        scored = match_template(source, template)
        if scored is None:
            continue
        score, loc = scored
        top_left = (region.left + loc[0], region.top + loc[1])
        offset = template.align_offset
        candidates.append(
            Candidate(
                name=name,
                score=score,
                top_left=top_left,
                size=template.size,
                aligned_top_left=(top_left[0] - offset[0], top_left[1] - offset[1]),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def classify(
    frame: Frame,
    rect: Optional[Rect],
    templates: Dict[str, Template],
    tier_weights: Dict[str, np.ndarray],
    config: Config,
) -> object:
    """Identify the laborer in ``rect``, or explain why we refuse to.

    Returns a ``LaborerVerdict`` or a ``Rejection``. It never guesses.
    """
    candidates = score_candidates(frame, rect, templates)
    if not candidates:
        return Rejection("no_candidates", "region too small for any laborer template")

    best = candidates[0]
    threshold = config.match.laborer_score
    if best.score < threshold:
        return Rejection("no_laborer", f"best {best.name} {best.score:.3f} < {threshold:.3f}")

    # --- stage 1: profession, by whole-icon score ----------------------
    profession = laborer_profession(best.name)
    rival = next(
        (candidate for candidate in candidates if laborer_profession(candidate.name) != profession),
        None,
    )
    profession_margin: Optional[float] = None
    if rival is not None:
        profession_margin = best.score - rival.score
        if profession_margin < config.match.laborer_profession_margin:
            return Rejection(
                "ambiguous_profession",
                f"{best.name} {best.score:.3f} vs {rival.name} {rival.score:.3f} "
                f"(margin {profession_margin:.3f} < {config.match.laborer_profession_margin:.3f})",
            )

    if frame.bgr is None:
        # Tier discrimination needs colour. Refusing beats guessing: grayscale
        # cannot separate sibling tiers at all.
        return Rejection("no_color_capture", "tier cannot be decided from a grayscale capture")

    # --- stage 2: tier, by badge-weighted comparison -------------------
    # One canonical icon origin from the best match, so every sibling is
    # measured against the same pixels. v3 sampled each sibling at its own
    # match location, which can compare two different places on screen.
    origin = best.aligned_top_left
    siblings = [name for name in templates if laborer_profession(name) == profession]

    distances: List[Tuple[float, str]] = []
    for name in siblings:
        template = templates[name]
        weight = tier_weights.get(name)
        if weight is None or weight.shape[:2] != (template.height, template.width):
            continue
        patch = frame.bgr_patch(
            (origin[0] + template.align_offset[0], origin[1] + template.align_offset[1]),
            template.width,
            template.height,
        )
        if patch is None:
            continue
        distance = weighted_patch_distance(patch, template, weight)
        if distance is not None:
            distances.append((distance, name))

    if not distances:
        return Rejection("no_tier_sample", f"could not sample the tier badge near {origin}")

    distances.sort()
    winner_distance, winner_name = distances[0]

    ratio: Optional[float] = None
    runner_up: Optional[str] = None
    if len(distances) > 1:
        runner_distance, runner_up = distances[1]
        ratio = runner_distance / max(winner_distance, 1e-6)

    if winner_distance > config.match.tier_max_distance:
        # Nothing we hold a template for looks like this badge. Almost always a
        # tier above what has been captured; acting on the closest match would
        # hand a journal to the wrong laborer.
        return Rejection(
            "unknown_tier",
            f"closest is {winner_name} at {winner_distance:.1f} > "
            f"{config.match.tier_max_distance:.1f}; the icon is probably a tier with no template",
            likely_name=winner_name,
        )

    if ratio is not None and ratio < config.match.tier_ratio_min:
        return Rejection(
            "ambiguous_tier",
            f"{winner_name} ({winner_distance:.1f}) vs {runner_up} "
            f"({winner_distance * ratio:.1f}) - only x{ratio:.2f} apart, "
            f"need x{config.match.tier_ratio_min:.2f}",
        )

    winning = next((c for c in candidates if c.name == winner_name), best)
    return LaborerVerdict(
        name=winner_name,
        score=winning.score,
        top_left=winning.top_left,
        size=winning.size,
        icon_origin=origin,
        profession_margin=profession_margin,
        tier_distance=winner_distance,
        tier_ratio=ratio,
        runner_up=runner_up,
    )


def format_scores(candidates: Sequence[Candidate], limit: int = 6) -> str:
    return ", ".join(f"{c.name}={c.score:.3f}" for c in candidates[:limit])
