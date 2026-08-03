"""Cycle outcomes, audible feedback and logging setup.

You cannot read a console while playing, so an outcome that needs a human gets
a distinct beep pattern. Only outcomes that need one have a pattern: a routine
success or a not-ready laborer is silent, because a beep on every keypress is
noise you learn to ignore, which defeats the point. Beeps run on a daemon
thread: winsound.Beep is blocking and must never sit on the hotkey loop.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import winsound
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class Outcome(Enum):
    SUCCESS = "success"
    NOT_READY = "not-ready"
    STOP_LABORER = "stop-laborer"
    PICKED_UP = "picked-up"
    PICKUP_FAILED = "pickup-failed"
    NO_JOURNAL = "no-journal"
    UNKNOWN_LABORER = "unknown-laborer"
    JOURNAL_FAILED = "journal-failed"
    ACCEPT_FAILED = "accept-failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    ERROR = "error"

    @property
    def is_failure(self) -> bool:
        return self in {
            Outcome.UNKNOWN_LABORER,
            Outcome.JOURNAL_FAILED,
            Outcome.ACCEPT_FAILED,
            Outcome.PICKUP_FAILED,
            Outcome.ERROR,
        }


# (frequency Hz, duration ms) pairs. Distinct shapes, not just pitches, so they
# are told apart through game audio.
#
# Outcomes absent from this table are deliberately silent: SUCCESS, NOT_READY,
# PICKED_UP, ABORTED and BLOCKED are all either the normal path or something
# the operator just did on purpose, and beeping on them buries the ones that
# actually need attention.
BEEP_PATTERNS: dict[Outcome, Sequence[Tuple[int, int]]] = {
    # The t6 alert - two long high tones, played the moment the tier is
    # recognised rather than at the end of the cycle.
    Outcome.STOP_LABORER: ((1200, 160), (1200, 160)),
    Outcome.NO_JOURNAL: ((520, 110), (520, 110), (520, 110)),
    Outcome.UNKNOWN_LABORER: ((620, 90), (440, 160)),
    Outcome.JOURNAL_FAILED: ((440, 140), (330, 220)),
    Outcome.ACCEPT_FAILED: ((440, 140), (330, 220)),
    Outcome.PICKUP_FAILED: ((440, 140), (330, 220)),
    Outcome.ERROR: ((300, 420),),
}


class Feedback:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: "queue.Queue[Sequence[Tuple[int, int]]]" = queue.Queue(maxsize=8)
        self._worker = threading.Thread(target=self._run, name="laborer-beeps", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            pattern = self._queue.get()
            for frequency, duration in pattern:
                try:
                    winsound.Beep(frequency, duration)
                except RuntimeError:
                    return
                time.sleep(0.03)

    def play(self, outcome: Outcome) -> None:
        if not self.enabled:
            return
        pattern = BEEP_PATTERNS.get(outcome)
        if not pattern:
            return
        try:
            self._queue.put_nowait(pattern)
        except queue.Full:
            pass


class _Formatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[38;5;244m",
        logging.INFO: "",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[1;31m",
    }
    RESET = "\x1b[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{text}{self.RESET}" if color else text


def setup_logging(verbose: bool, log_path: Optional[str] = None) -> None:
    root = logging.getLogger("laborer")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(_Formatter(use_color=sys.stdout.isatty()))
    root.addHandler(console)

    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
        root.addHandler(file_handler)


class Timer:
    """Per-step timings, printed as one line at the end of each cycle."""

    def __init__(self) -> None:
        self.steps: List[Tuple[str, float]] = []
        self._started = time.perf_counter()
        self._mark = self._started

    def step(self, name: str) -> None:
        now = time.perf_counter()
        self.steps.append((name, now - self._mark))
        self._mark = now

    @property
    def total(self) -> float:
        return time.perf_counter() - self._started

    def summary(self) -> str:
        parts = " ".join(f"{name}={duration * 1000:.0f}ms" for name, duration in self.steps)
        return f"total={self.total * 1000:.0f}ms {parts}".strip()
