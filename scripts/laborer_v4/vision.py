"""Screen capture, template loading and matching primitives.

Two things here matter more than anything else for v4's speed:

1.  A single long-lived ``mss`` instance per thread. Constructing one per
    lookup (as v3 did, six call sites deep) costs ~12 ms each in DC setup.
2.  Templates whose alpha channel is fully opaque carry *no* mask. Every asset
    in this repo is alpha=255 everywhere, and an all-255 mask is numerically
    identical to no mask (verified: max abs difference 5e-7) while costing
    2.4x the time - 98 ms vs 41 ms for a 90x90 template over 1080p.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def inflate(self, padding: int) -> "Rect":
        return Rect(
            self.left - padding,
            self.top - padding,
            self.width + 2 * padding,
            self.height + 2 * padding,
        )

    def union(self, other: "Rect") -> "Rect":
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        return Rect(
            left,
            top,
            max(self.right, other.right) - left,
            max(self.bottom, other.bottom) - top,
        )

    def clamp_to(self, bounds: "Rect") -> Optional["Rect"]:
        left = max(self.left, bounds.left)
        top = max(self.top, bounds.top)
        right = min(self.right, bounds.right)
        bottom = min(self.bottom, bounds.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def contains_point(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_mss(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    @staticmethod
    def from_mss(monitor: Dict[str, int]) -> "Rect":
        return Rect(
            int(monitor["left"]),
            int(monitor["top"]),
            int(monitor["width"]),
            int(monitor["height"]),
        )

    @staticmethod
    def around(center: Tuple[int, int], width: int, height: int) -> "Rect":
        return Rect(center[0] - width // 2, center[1] - height // 2, width, height)


@dataclass
class Frame:
    """One captured region. ``gray`` is always present, ``bgr`` only on demand."""

    rect: Rect
    gray: np.ndarray
    bgr: Optional[np.ndarray] = None

    def sub_gray(self, rect: Rect) -> Optional[Tuple[np.ndarray, Rect]]:
        """Numpy view of a sub-rect - free, no re-capture."""
        clamped = rect.clamp_to(self.rect)
        if clamped is None:
            return None
        x = clamped.left - self.rect.left
        y = clamped.top - self.rect.top
        return self.gray[y:y + clamped.height, x:x + clamped.width], clamped

    def bgr_patch(self, top_left: Tuple[int, int], width: int, height: int) -> Optional[np.ndarray]:
        if self.bgr is None:
            return None
        x = top_left[0] - self.rect.left
        y = top_left[1] - self.rect.top
        if x < 0 or y < 0:
            return None
        patch = self.bgr[y:y + height, x:x + width]
        if patch.shape[0] != height or patch.shape[1] != width:
            return None
        return patch


class Grabber:
    """Thread-local, long-lived mss instances."""

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def sct(self) -> mss.base.MSSBase:
        instance = getattr(self._local, "sct", None)
        if instance is None:
            instance = mss.mss()
            self._local.sct = instance
        return instance

    def monitors(self) -> List[Rect]:
        """Per-monitor rects, index 0 being the virtual desktop (as mss does)."""
        return [Rect.from_mss(monitor) for monitor in self.sct.monitors]

    def virtual_desktop(self) -> Rect:
        return Rect.from_mss(self.sct.monitors[0])

    def monitor_at(self, point: Tuple[int, int]) -> Optional[int]:
        for index, monitor in enumerate(self.sct.monitors):
            if index == 0:
                continue
            if Rect.from_mss(monitor).contains_point(point):
                return index
        return None

    def grab(self, rect: Rect, color: bool = False) -> Frame:
        raw = np.asarray(self.sct.grab(rect.to_mss()))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR) if color else None
        return Frame(rect=rect, gray=gray, bgr=bgr)

    def close(self) -> None:
        instance = getattr(self._local, "sct", None)
        if instance is not None:
            instance.close()
            self._local.sct = None


@dataclass(frozen=True)
class Template:
    name: str
    path: Path
    gray: np.ndarray
    mask: Optional[np.ndarray]
    gray_half: np.ndarray
    mask_half: Optional[np.ndarray]
    bgr: Optional[np.ndarray]
    has_alpha: bool
    # Offset of this template's content within the shared icon frame, measured
    # at load time. Lets differently-cropped assets be compared in one space.
    align_offset: Tuple[int, int] = (0, 0)

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def with_alignment(self, offset: Tuple[int, int]) -> "Template":
        return Template(
            name=self.name,
            path=self.path,
            gray=self.gray,
            mask=self.mask,
            gray_half=self.gray_half,
            mask_half=self.mask_half,
            bgr=self.bgr,
            has_alpha=self.has_alpha,
            align_offset=offset,
        )


def load_template(path: Path, name: Optional[str] = None, half_scale: float = 0.5) -> Template:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"unable to load image: {path}")

    has_alpha = image.ndim == 3 and image.shape[2] == 4
    mask: Optional[np.ndarray] = None
    if has_alpha:
        alpha = image[:, :, 3]
        # A fully opaque alpha channel carries no information; keeping it as a
        # mask would cost 2.4x for an identical score.
        if int(alpha.min()) < 255:
            mask = alpha.copy()
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3:
        bgr = image
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_half = cv2.resize(gray, None, fx=half_scale, fy=half_scale, interpolation=cv2.INTER_AREA)
    mask_half = (
        cv2.resize(mask, None, fx=half_scale, fy=half_scale, interpolation=cv2.INTER_AREA)
        if mask is not None
        else None
    )
    return Template(
        name=name or path.stem,
        path=path,
        gray=gray,
        mask=mask,
        gray_half=gray_half,
        mask_half=mask_half,
        bgr=bgr,
        has_alpha=has_alpha,
    )


@dataclass(frozen=True)
class Match:
    name: str
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]

    @property
    def center(self) -> Tuple[int, int]:
        return self.top_left[0] + self.size[0] // 2, self.top_left[1] + self.size[1] // 2

    @property
    def rect(self) -> Rect:
        return Rect(self.top_left[0], self.top_left[1], self.size[0], self.size[1])


def score_at(
    source: np.ndarray,
    template_gray: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    """Best (score, top-left) of ``template_gray`` inside ``source``.

    TM_SQDIFF_NORMED rather than TM_CCORR_NORMED: CCORR is not mean-subtracted,
    so bright uniform areas score ~1.0 and beat the real sprite. Returned as
    ``1 - min_val`` so higher is better, matching v3's scale exactly.
    """
    template_height, template_width = template_gray.shape[:2]
    source_height, source_width = source.shape[:2]
    if source_height < template_height or source_width < template_width:
        return None

    result = cv2.matchTemplate(source, template_gray, cv2.TM_SQDIFF_NORMED, mask=template_mask)
    result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
    min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
    return 1.0 - float(min_val), (int(min_loc[0]), int(min_loc[1]))


def match_template(source: np.ndarray, template: Template) -> Optional[Tuple[float, Tuple[int, int]]]:
    return score_at(source, template.gray, template.mask)


def locate_in(
    frame: Frame,
    template: Template,
    rect: Optional[Rect],
    threshold: float,
) -> Optional[Match]:
    """Locate ``template`` inside ``rect`` of an already-captured frame."""
    if rect is None:
        source, region = frame.gray, frame.rect
    else:
        sub = frame.sub_gray(rect)
        if sub is None:
            return None
        source, region = sub

    scored = match_template(source, template)
    if scored is None:
        return None
    score, loc = scored
    if score < threshold:
        return None
    return Match(
        name=template.name,
        score=score,
        top_left=(region.left + loc[0], region.top + loc[1]),
        size=template.size,
    )


def locate_peaks(
    source: np.ndarray,
    template: np.ndarray,
    threshold: float,
    suppression: Tuple[int, int],
    limit: int = 64,
) -> List[Tuple[float, Tuple[int, int]]]:
    """Every place ``template`` fits, not just the single best.

    ``score_at`` returns one global argmax, which is the right answer only when
    the thing being looked for is unique on screen. An inventory holds a dozen
    icons that differ by a few percent, so the global best for one template is
    routinely a *different* item's slot - the caller needs all the candidate
    positions and must decide what occupies each one separately.

    Peaks are taken greedily, blanking a ``suppression``-sized box around each
    so one icon cannot yield a cluster of near-identical hits.
    """
    template_height, template_width = template.shape[:2]
    source_height, source_width = source.shape[:2]
    if source_height < template_height or source_width < template_width:
        return []

    result = cv2.matchTemplate(source, template, cv2.TM_SQDIFF_NORMED)
    result = 1.0 - np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)

    half_x = max(1, suppression[0] // 2)
    half_y = max(1, suppression[1] // 2)
    peaks: List[Tuple[float, Tuple[int, int]]] = []

    for _ in range(limit):
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break
        peaks.append((float(max_val), (int(max_loc[0]), int(max_loc[1]))))
        left = max(0, max_loc[0] - half_x)
        top = max(0, max_loc[1] - half_y)
        result[top:max_loc[1] + half_y + 1, left:max_loc[0] + half_x + 1] = -1.0

    return peaks


def coarse_locate(
    frame: Frame,
    template: Template,
    scale: float,
    scaled_source: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    """Half-scale search returning an approximate full-scale top-left.

    Cuts a 1080p search from ~41 ms to ~9 ms per template; the caller refines
    the winner at full resolution inside a small box.
    """
    source = scaled_source
    if source is None:
        source = cv2.resize(frame.gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    scored = score_at(source, template.gray_half, template.mask_half)
    if scored is None:
        return None
    score, loc = scored
    return score, (
        frame.rect.left + int(round(loc[0] / scale)),
        frame.rect.top + int(round(loc[1] / scale)),
    )


def scaled_source(frame: Frame, scale: float) -> np.ndarray:
    """Downscale a frame once so every template can reuse it."""
    return cv2.resize(frame.gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def pyramid_locate(
    frame: Frame,
    template: Template,
    threshold: float,
    scale: float,
    slack: float,
    refine_padding: int,
    small: Optional[np.ndarray] = None,
) -> Optional[Match]:
    """Coarse-to-fine full-region search."""
    coarse = coarse_locate(frame, template, scale, small)
    if coarse is None:
        return None
    coarse_score, coarse_top_left = coarse
    if coarse_score < threshold - slack:
        return None

    refine_rect = Rect(
        coarse_top_left[0] - refine_padding,
        coarse_top_left[1] - refine_padding,
        template.width + 2 * refine_padding,
        template.height + 2 * refine_padding,
    )
    refined = locate_in(frame, template, refine_rect, threshold)
    if refined is not None:
        return refined

    # Coarse stage was confident but the refine box missed: the estimate can be
    # off by more than the padding on a downscale. Fall back to the full region
    # rather than declaring the element absent.
    return locate_in(frame, template, None, threshold)


def project_onto(target: Template, other: Template) -> np.ndarray:
    """``other``'s pixels expressed in ``target``'s coordinate frame.

    Uses the measured crop alignment, so two assets cut with different bounds
    still line up. Non-overlapping pixels come back as NaN.
    """
    height, width = target.height, target.width
    canvas = np.full((height, width, 3), np.nan, np.float32)
    dx = other.align_offset[0] - target.align_offset[0]
    dy = other.align_offset[1] - target.align_offset[1]
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(width, other.width + dx), min(height, other.height + dy)
    if x1 > x0 and y1 > y0 and other.bgr is not None:
        canvas[y0:y1, x0:x1] = other.bgr[y0 - dy:y1 - dy, x0 - dx:x1 - dx].astype(np.float32)
    return canvas


def build_tier_weights(
    siblings: Sequence[Template],
    min_overlap: int = 2,
) -> Dict[str, np.ndarray]:
    """Per-pixel weights for telling sibling tiers apart.

    Measured on this repo's assets: once aligned, two tiers of the same
    profession are *pixel-identical* over 95% of the icon and differ only in a
    ~25x23 badge box. A whole-icon score therefore puts siblings within 0.004
    of each other - far too close to decide on. Weighting each pixel by how
    much the siblings disagree there concentrates the comparison on the 5% that
    carries the tier, which pushes the runner-up to 1.8x worse even under heavy
    capture noise.

    The weights are derived from the assets themselves, so re-capturing or
    adding a tier keeps working with no hand-tuned badge rectangle.
    """
    weights: Dict[str, np.ndarray] = {}
    for target in siblings:
        stack = np.stack([project_onto(target, other) for other in siblings])
        counts = (~np.isnan(stack).any(-1)).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = np.nanstd(stack, axis=0).mean(-1)
        weights[target.name] = np.where(
            counts >= min_overlap, np.nan_to_num(deviation), 0.0
        ).astype(np.float32)
    return weights


def weighted_patch_distance(
    patch: np.ndarray,
    template: Template,
    weight: np.ndarray,
) -> Optional[float]:
    """Weighted RMS difference between a captured patch and a template."""
    if template.bgr is None or patch.shape[:2] != (template.height, template.width):
        return None
    total = float(weight.sum())
    if total <= 0.0:
        return None
    difference = patch.astype(np.float32) - template.bgr.astype(np.float32)
    energy = float(np.sum(weight[..., None] * difference * difference))
    return (energy / (total * 3.0)) ** 0.5


def align_templates(templates: Sequence[Template], search_padding: int = 24) -> Dict[str, Tuple[int, int]]:
    """Measure how differently each laborer asset was cropped.

    The laborer icons in this repo range from 83x83 to 95x102 - they were cut
    with different bounds, so the same on-screen icon matches each of them at a
    slightly different top-left. Cross-correlating against the largest template
    recovers each one's offset, which lets sibling match locations be compared
    in a single coordinate space instead of naively.
    """
    if not templates:
        return {}

    reference = max(templates, key=lambda item: item.gray.size)
    padded = cv2.copyMakeBorder(
        reference.gray,
        search_padding, search_padding, search_padding, search_padding,
        cv2.BORDER_REPLICATE,
    )

    offsets: Dict[str, Tuple[int, int]] = {}
    for template in templates:
        if template.name == reference.name:
            offsets[template.name] = (0, 0)
            continue
        if template.height > padded.shape[0] or template.width > padded.shape[1]:
            offsets[template.name] = (0, 0)
            continue

        result = cv2.matchTemplate(padded, template.gray, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        dx = max_loc[0] - search_padding
        dy = max_loc[1] - search_padding
        if max_val < 0.35 or abs(dx) > search_padding or abs(dy) > search_padding:
            # Alignment is only an optimisation for the sibling-agreement check;
            # falling back to (0, 0) degrades gracefully to v3 behaviour.
            offsets[template.name] = (0, 0)
        else:
            offsets[template.name] = (int(dx), int(dy))
    return offsets


def annotate(frame: Frame, boxes: Iterable[Tuple[Rect, str, Tuple[int, int, int]]]) -> np.ndarray:
    canvas = frame.bgr.copy() if frame.bgr is not None else cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
    for rect, label, color in boxes:
        x = rect.left - frame.rect.left
        y = rect.top - frame.rect.top
        cv2.rectangle(canvas, (x, y), (x + rect.width, y + rect.height), color, 2)
        cv2.putText(canvas, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas
