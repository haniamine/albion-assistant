"""Persisted state: learned UI anchors and the journal slot cache.

Both live in one JSON file so a screen-geometry change can invalidate them
together. v3 persisted only journal positions, and re-derived every element
location from scratch on every lookup.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ANCHOR_REFERENCE
from .vision import Rect

log = logging.getLogger("laborer.state")

STATE_VERSION = 1


@dataclass
class AnchorModel:
    """Learned pixel offsets between the elements of the laborer dialog.

    Every element we look for - take-all, advance-tier, ready, accept, the
    laborer portrait - sits inside the same dialog, so their relative offsets
    are constant. Learn them once and every subsequent lookup becomes a ~1 ms
    match inside a small box instead of a ~41 ms full-screen sweep. If a
    predicted box misses, the caller falls back to a full search and the model
    re-learns itself, so a moved or resized UI heals automatically.
    """

    reference: str = ANCHOR_REFERENCE
    # element -> (element_top_left - reference_top_left)
    offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Last known screen top-left of the reference element.
    origin: Optional[Tuple[int, int]] = None
    monitor_index: Optional[int] = None

    def clear(self) -> None:
        self.offsets.clear()
        self.origin = None
        self.monitor_index = None

    def knows(self, element: str) -> bool:
        if self.origin is None:
            return False
        return element == self.reference or element in self.offsets

    def predict_top_left(self, element: str) -> Optional[Tuple[int, int]]:
        if self.origin is None:
            return None
        if element == self.reference:
            return self.origin
        offset = self.offsets.get(element)
        if offset is None:
            return None
        return self.origin[0] + offset[0], self.origin[1] + offset[1]

    def element_rect(self, element: str, size: Tuple[int, int], padding: int) -> Optional[Rect]:
        top_left = self.predict_top_left(element)
        if top_left is None:
            return None
        return Rect(top_left[0], top_left[1], size[0], size[1]).inflate(padding)

    def dialog_rect(self, sizes: Dict[str, Tuple[int, int]], padding: int) -> Optional[Rect]:
        """Union of every known element box - one grab serves the whole cycle."""
        if self.origin is None:
            return None
        combined: Optional[Rect] = None
        for element, size in sizes.items():
            top_left = self.predict_top_left(element)
            if top_left is None:
                continue
            rect = Rect(top_left[0], top_left[1], size[0], size[1])
            combined = rect if combined is None else combined.union(rect)
        if combined is None:
            return None
        return combined.inflate(padding)

    def observe(self, positions: Dict[str, Tuple[int, int]], drift_tolerance: int = 6) -> None:
        """Fold one cycle's observed top-lefts into the model."""
        if not positions:
            return

        origin = positions.get(self.reference)
        if origin is None:
            # The reference was not seen, but any element with a known offset
            # lets us back out where it would have been.
            for element, position in positions.items():
                offset = self.offsets.get(element)
                if offset is not None:
                    origin = (position[0] - offset[0], position[1] - offset[1])
                    break
        if origin is None:
            return

        self.origin = origin
        for element, position in positions.items():
            if element == self.reference:
                continue
            offset = (position[0] - origin[0], position[1] - origin[1])
            previous = self.offsets.get(element)
            if previous is not None:
                drift = max(abs(previous[0] - offset[0]), abs(previous[1] - offset[1]))
                if drift > drift_tolerance:
                    log.debug("anchor %s drifted %d px (%s -> %s)", element, drift, previous, offset)
            self.offsets[element] = offset

    # NOTE: take-all, advance-tier and accept all learn *the same* offset, and
    # that is correct - they are one button slot in the dialog showing different
    # labels at different moments. An earlier version treated coincident anchors
    # as evidence of a false match and deleted them, which threw away good data;
    # which button is currently in that slot is decided by ranking the templates
    # against the pixels (see CONFUSABLE_ELEMENTS), never by position.

    def to_dict(self) -> Dict[str, object]:
        return {
            "reference": self.reference,
            "origin": list(self.origin) if self.origin else None,
            "monitor_index": self.monitor_index,
            "offsets": {name: list(offset) for name, offset in sorted(self.offsets.items())},
        }

    @staticmethod
    def from_dict(data: object) -> "AnchorModel":
        model = AnchorModel()
        if not isinstance(data, dict):
            return model
        reference = data.get("reference")
        if isinstance(reference, str):
            model.reference = reference
        origin = _parse_point(data.get("origin"))
        model.origin = origin
        monitor = data.get("monitor_index")
        model.monitor_index = monitor if isinstance(monitor, int) else None
        offsets = data.get("offsets")
        if isinstance(offsets, dict):
            for name, raw in offsets.items():
                point = _parse_point(raw)
                if isinstance(name, str) and point is not None:
                    model.offsets[name] = point
        return model


@dataclass
class JournalSlot:
    position: Tuple[int, int]
    score: float
    monitor_index: int


@dataclass
class JournalCache:
    """Where each journal type sits in the inventory, as of the last scan.

    This is the list that drives candidate reduction: a laborer tier whose
    journal is not in here cannot be acted on, so its template is never even
    matched against the screen.
    """

    slots: Dict[str, JournalSlot] = field(default_factory=dict)
    scanned_at: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.slots)

    def names(self) -> List[str]:
        return sorted(self.slots)

    def position(self, journal_name: str) -> Optional[Tuple[int, int]]:
        slot = self.slots.get(journal_name)
        return slot.position if slot else None

    def replace(self, slots: Dict[str, JournalSlot]) -> None:
        self.slots = dict(slots)
        self.scanned_at = time.time()

    def drop(self, journal_name: str) -> None:
        self.slots.pop(journal_name, None)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scanned_at": self.scanned_at,
            "slots": {
                name: {
                    "position": list(slot.position),
                    "score": round(slot.score, 4),
                    "monitor_index": slot.monitor_index,
                }
                for name, slot in sorted(self.slots.items())
            },
        }

    @staticmethod
    def from_dict(data: object) -> "JournalCache":
        cache = JournalCache()
        if not isinstance(data, dict):
            return cache
        scanned_at = data.get("scanned_at")
        cache.scanned_at = float(scanned_at) if isinstance(scanned_at, (int, float)) else 0.0
        slots = data.get("slots")
        if not isinstance(slots, dict):
            return cache
        for name, raw in slots.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                continue
            position = _parse_point(raw.get("position"))
            if position is None:
                log.warning("dropping invalid cached journal slot for %s", name)
                continue
            score = raw.get("score")
            monitor = raw.get("monitor_index")
            cache.slots[name] = JournalSlot(
                position=position,
                score=float(score) if isinstance(score, (int, float)) else 0.0,
                monitor_index=int(monitor) if isinstance(monitor, int) else 0,
            )
        return cache


@dataclass
class AppState:
    anchors: AnchorModel = field(default_factory=AnchorModel)
    journals: JournalCache = field(default_factory=JournalCache)
    # Geometry signature; a monitor change invalidates every cached pixel.
    screen_signature: str = ""
    _dirty: bool = field(default=False, repr=False)

    def mark_dirty(self) -> None:
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def apply_signature(self, signature: str) -> bool:
        """Return True if the cached state was invalidated by a geometry change."""
        if self.screen_signature and self.screen_signature != signature:
            log.warning(
                "screen geometry changed (%s -> %s); clearing anchors and journal cache",
                self.screen_signature, signature,
            )
            self.anchors.clear()
            self.journals.replace({})
            self.screen_signature = signature
            self._dirty = True
            return True
        self.screen_signature = signature
        return False

    def save(self, path: Path) -> None:
        payload = {
            "version": STATE_VERSION,
            "screen_signature": self.screen_signature,
            "anchors": self.anchors.to_dict(),
            "journals": self.journals.to_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        self._dirty = False

    def flush(self, path: Path) -> None:
        """Persist only if something changed - never on the click path."""
        if self._dirty:
            self.save(path)

    @staticmethod
    def load(path: Path) -> "AppState":
        state = AppState()
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            log.warning("could not read %s (%s); starting from empty state", path, error)
            return state
        if not isinstance(data, dict):
            return state
        if data.get("version") != STATE_VERSION:
            log.warning("state file version mismatch; starting from empty state")
            return state
        signature = data.get("screen_signature")
        state.screen_signature = signature if isinstance(signature, str) else ""
        state.anchors = AnchorModel.from_dict(data.get("anchors"))
        state.journals = JournalCache.from_dict(data.get("journals"))
        return state


def import_v3_positions(path: Path, monitor_index: int) -> Optional[JournalCache]:
    """One-time migration of v3's journal-positions.json."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    cache = JournalCache()
    for name, raw in data.items():
        point = _parse_point(raw)
        if isinstance(name, str) and point is not None:
            cache.slots[name] = JournalSlot(position=point, score=0.0, monitor_index=monitor_index)
    if not cache.slots:
        return None
    cache.scanned_at = 0.0  # unknown age: treat as needing verification
    return cache


def _parse_point(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return int(x), int(y)
