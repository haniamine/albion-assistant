"""Inventory journal scanning and slot verification.

The scan is the source of truth for two things:

  * where to shift-click each journal, and
  * which laborer tiers are candidates at all (see AssetLibrary.active_laborers).

Detect the slots, then decide what is in them - never the other way round
-------------------------------------------------------------------------

Journal icons are the same book drawn in slightly different colours. Measured
on this repo's assets, in *grayscale*, every journal template matches every
other journal's slot at 0.87-0.97, against a self-match of 1.000: t3-common
scores 0.968 on the t2-common slot. There is no threshold that separates them,
because the wrong answer outscores what a correct answer degrades to on a live
capture.

The original scan asked each template "where are you?", took its single global
argmax, and then dropped whichever of two templates lost a collision. With
scores that close the argmax is regularly a *different* journal's slot, so a
journal that was plainly in the bag got deleted from the cache - and worse, the
survivor could be recorded at a slot holding something else.

This module asks the inverse question, per slot: "what is in you?". Detection
runs in grayscale, where the templates being near-identical is a feature - any
of them finds every journal-like icon. Identity is then decided at each fixed
slot, in colour, by ranking every template against that one patch. At a fixed
position the correct template wins by 0.023-0.088 over the runner-up, and that
margin holds under heavy noise because every candidate degrades together.

Two collisions therefore become impossible by construction: a slot has exactly
one occupant, and a journal is recorded at exactly one slot.

The other rule kept from v4: every journal lookup - the initial scan *and* any
later re-resolve - is bounded to the inventory panel. v3's stale-cache fallback
searched the whole monitor, so it could lock onto the required-journal icon
drawn inside the laborer dialog itself and cache that as an inventory slot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import Config
from .state import JournalSlot
from .vision import Frame, Grabber, Rect, Template, locate_peaks, score_at

log = logging.getLogger("laborer.inventory")


@dataclass
class ScanReport:
    slots: Dict[str, JournalSlot]
    missing: List[str]
    collisions: List[str]
    duration: float
    # Slots that hold something journal-shaped but could not be named.
    unidentified: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unidentified is None:
            self.unidentified = []

    def summary(self) -> List[str]:
        lines = [f"journal scan: {len(self.slots)} found in {self.duration * 1000:.0f} ms"]
        for name, slot in sorted(self.slots.items()):
            lines.append(f"  {name:<12} @ {slot.position}  ({slot.score:.3f})")
        if self.missing:
            lines.append("  not in inventory: " + ", ".join(sorted(self.missing)))
        for note in self.unidentified:
            lines.append(f"  ? {note}")
        for collision in self.collisions:
            lines.append(f"  ! {collision}")
        return lines


@dataclass(frozen=True)
class SlotVerdict:
    name: str
    score: float
    margin: float
    runner_up: Optional[str]


def inventory_rect(monitor: Rect, left_ratio: float) -> Rect:
    """The inventory panel: the right-hand slice of the game area."""
    offset = int(monitor.width * left_ratio)
    return Rect(monitor.left + offset, monitor.top, monitor.width - offset, monitor.height)


def detect_slots(
    frame: Frame,
    journals: Dict[str, Template],
    config: Config,
) -> List[Tuple[int, int]]:
    """Every position in the panel holding a journal-shaped icon.

    Grayscale, and every template contributes its peaks rather than just its
    best. Because the templates are near-identical this over-reports wildly -
    the same slot is found by all ten - so the pooled hits are then collapsed by
    position. Over-reporting is the safe direction: an undetected slot is a
    journal silently missing from the cache, which is the bug being fixed.
    """
    if not journals:
        return []

    suppression = (
        max(template.width for template in journals.values()),
        max(template.height for template in journals.values()),
    )

    pooled: List[Tuple[float, Tuple[int, int]]] = []
    for template in journals.values():
        for score, loc in locate_peaks(
            frame.gray, template.gray, config.match.journal_slot_score, suppression
        ):
            center = (
                frame.rect.left + loc[0] + template.width // 2,
                frame.rect.top + loc[1] + template.height // 2,
            )
            pooled.append((score, center))

    # Strongest first, so a cluster collapses onto its best-scoring member.
    pooled.sort(key=lambda item: item[0], reverse=True)
    separation = max(config.match.journal_min_separation, min(suppression) // 2)

    centers: List[Tuple[int, int]] = []
    for _score, center in pooled:
        if any(
            abs(center[0] - kept[0]) < separation and abs(center[1] - kept[1]) < separation
            for kept in centers
        ):
            continue
        centers.append(center)
    return centers


def identify_slot(
    frame: Frame,
    center: Tuple[int, int],
    journals: Dict[str, Template],
    config: Config,
) -> Optional[SlotVerdict]:
    """Rank every journal against one slot, in colour.

    Scoring each template inside a window around the slot rather than at an
    exact top-left absorbs the differing crop bounds of the assets (39x59 to
    50x64) without needing them re-cut to one size.
    """
    if frame.bgr is None:
        return None

    padding = config.roi.journal_verify_padding
    width = max(template.width for template in journals.values())
    height = max(template.height for template in journals.values())
    window = Rect(center[0] - width // 2, center[1] - height // 2, width, height).inflate(padding)

    clamped = window.clamp_to(frame.rect)
    if clamped is None:
        return None
    x = clamped.left - frame.rect.left
    y = clamped.top - frame.rect.top
    patch = frame.bgr[y:y + clamped.height, x:x + clamped.width]

    ranked: List[Tuple[float, str]] = []
    for name, template in journals.items():
        if template.bgr is None:
            continue
        scored = score_at(patch, template.bgr, None)
        if scored is not None:
            ranked.append((scored[0], name))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    score, name = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else None
    margin = score - ranked[1][0] if len(ranked) > 1 else 1.0
    return SlotVerdict(name=name, score=score, margin=margin, runner_up=runner_up)


def scan_journals(
    grabber: Grabber,
    region: Rect,
    journals: Dict[str, Template],
    config: Config,
    monitor_index: int,
) -> ScanReport:
    """One colour capture, then detect-then-identify over the whole panel."""
    import time

    started = time.perf_counter()
    frame = grabber.grab(region, color=True)

    found: Dict[str, Tuple[JournalSlot, Tuple[int, int]]] = {}
    unidentified: List[str] = []

    for center in detect_slots(frame, journals, config):
        verdict = identify_slot(frame, center, journals, config)
        if verdict is None:
            continue
        if verdict.score < config.match.journal_score:
            unidentified.append(
                f"slot at {center}: best is {verdict.name} at {verdict.score:.3f} "
                f"< {config.match.journal_score:.3f} - not one of ours"
            )
            continue
        if verdict.margin < config.match.journal_margin:
            # Two journals fit this slot equally well. Recording either one
            # risks handing the wrong journal to a laborer, so record neither.
            unidentified.append(
                f"slot at {center}: {verdict.name} ({verdict.score:.3f}) vs "
                f"{verdict.runner_up} ({verdict.score - verdict.margin:.3f}) - "
                f"only {verdict.margin:.4f} apart, need {config.match.journal_margin:.4f}"
            )
            continue

        slot = JournalSlot(position=center, score=verdict.score, monitor_index=monitor_index)
        previous = found.get(verdict.name)
        if previous is not None and previous[0].score >= verdict.score:
            # The same journal in two slots is a split stack, not an error;
            # keep the cleaner-looking one.
            continue
        found[verdict.name] = (slot, center)

    slots, collisions = _resolve_collisions(found, config.match.journal_min_separation)
    missing = [name for name in journals if name not in slots]

    return ScanReport(
        slots=slots,
        missing=sorted(set(missing)),
        collisions=collisions,
        duration=time.perf_counter() - started,
        unidentified=unidentified,
    )


def _resolve_collisions(
    found: Dict[str, Tuple[JournalSlot, Tuple[int, int]]],
    min_separation: int,
) -> Tuple[Dict[str, JournalSlot], List[str]]:
    """Keep the higher-scoring journal when two land on the same slot."""
    ordered = sorted(found.items(), key=lambda item: item[1][0].score, reverse=True)
    kept: Dict[str, JournalSlot] = {}
    collisions: List[str] = []

    for name, (slot, _top_left) in ordered:
        clash = next(
            (
                other
                for other, other_slot in kept.items()
                if abs(other_slot.position[0] - slot.position[0]) < min_separation
                and abs(other_slot.position[1] - slot.position[1]) < min_separation
            ),
            None,
        )
        if clash is not None:
            collisions.append(
                f"{name} ({slot.score:.3f}) collides with {clash} "
                f"({kept[clash].score:.3f}) at {slot.position}; keeping {clash}"
            )
            continue
        kept[name] = slot

    return kept, collisions


def check_journal_at(
    grabber: Grabber,
    name: str,
    journals: Dict[str, Template],
    position: Tuple[int, int],
    config: Config,
    bounds: Optional[Rect] = None,
) -> Tuple[bool, str]:
    """Is the journal named ``name`` - and not some lookalike - still at ``position``?

    Checked immediately before every shift-click, so a reflowed inventory can
    never cause a click on the wrong item. That guarantee needs the same ranking
    the scan uses: an earlier version scored one template in grayscale against a
    0.90 floor, and a *different* journal in the slot scores up to 0.97 that
    way, so the check passed on exactly the reflow it existed to catch.

    Returns ``(ok, reason)``. The reason separates the two failures, which look
    identical to a boolean but are not:

      *obscured*  nothing there resembles any journal. The panel is mid-redraw
                  or something is drawn over it. Transient - worth waiting out.
      *different* another journal is clearly there. A real reflow; the cache is
                  wrong and waiting will not fix it.
    """
    template = journals.get(name)
    if template is None:
        return False, f"'{name}' is not a journal we know"

    padding = config.roi.journal_verify_padding
    width = max(item.width for item in journals.values())
    height = max(item.height for item in journals.values())
    rect = Rect(
        position[0] - width // 2,
        position[1] - height // 2,
        width,
        height,
    ).inflate(padding)
    if bounds is not None:
        clamped = rect.clamp_to(bounds)
        if clamped is None or clamped.width < template.width or clamped.height < template.height:
            return False, "the slot lies outside the search region"
        rect = clamped

    frame = grabber.grab(rect, color=True)
    verdict = identify_slot(frame, position, journals, config)
    if verdict is None:
        return False, "obscured: could not sample the slot"

    if verdict.score < config.match.journal_score:
        # Nothing here looks like any journal at all. Reporting this as "now
        # holds <closest name>" reads as a reflow and is simply wrong - at 0.57
        # the closest name means nothing.
        return False, (
            f"obscured: nothing journal-like at {position} "
            f"(closest {verdict.name} only {verdict.score:.3f} < {config.match.journal_score:.2f})"
        )
    if verdict.name != name:
        return False, (
            f"different: {verdict.name} ({verdict.score:.3f}) is in the slot, not {name}"
        )
    if verdict.margin < config.match.journal_margin:
        return False, (
            f"obscured: {verdict.name} vs {verdict.runner_up} only {verdict.margin:.4f} apart"
        )
    return True, "ok"


def journal_at(
    grabber: Grabber,
    name: str,
    journals: Dict[str, Template],
    position: Tuple[int, int],
    config: Config,
    bounds: Optional[Rect] = None,
) -> bool:
    return check_journal_at(grabber, name, journals, position, config, bounds)[0]
