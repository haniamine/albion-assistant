"""The left mouse button as an action starter.

The hard part is not reading the button - ``GetAsyncKeyState`` does that - it is
that the engine's whole job is to inject left clicks. Every shift-click on a
journal and every press of take-all, accept and yes is a left button down/up
pair indistinguishable, to any polling read, from the operator's own. A naive
"is the button down?" trigger would therefore re-arm itself off the very clicks
it just caused and never stop.

Three things keep that from happening:

*   **Edges, not levels.** A cycle starts on a transition, so a button that is
    simply held does nothing after the first one.
*   **A suppression gate.** After every cycle the trigger refuses to fire until
    it has *seen the button up*. The engine's clicks all land while
    ``run_cycle`` is running and the loop is not polling, so what the gate
    really covers is the operator still holding the button that started the
    cycle - and any injected click still settling as the cycle returns.
*   **Location.** The click has to happen in the game: its window in the
    foreground, and the pointer inside its rect.

Firing on the release rather than the press is the same argument from the other
side: the engine injects a mouse-down within milliseconds of the trigger, and
doing that while the physical button is still held hands the game a down with
no up in between.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Tuple

import win32gui

from laborer_v4.winput import GameWindow, find_game_window, foreground_hwnd, is_pressed

log = logging.getLogger("laborer.trigger")

VK_LBUTTON = 0x01

# Rising pair on, falling pair off - the shape says which way it went without
# having to look at a console that is behind a full-screen game.
BEEP_ON: Sequence[Tuple[int, int]] = ((760, 70), (1140, 90))
BEEP_OFF: Sequence[Tuple[int, int]] = ((1140, 70), (760, 90))


class WindowTracker:
    """The game window, cached, with the rect kept current cheaply.

    ``find_game_window`` walks every top-level window and opens a process
    handle per candidate, which is far too much to run on a 10 ms poll. Once a
    handle is known the rect can be re-read from it directly, so the full
    search only runs when there is no valid handle - i.e. before the game
    starts, and again if it is restarted.
    """

    def __init__(self, titles: Sequence[str], processes: Sequence[str], refresh_seconds: float = 2.0) -> None:
        self.titles = list(titles)
        self.processes = list(processes)
        self.refresh_seconds = refresh_seconds
        self._window: Optional[GameWindow] = None
        self._next_search = 0.0

    def current(self) -> Optional[GameWindow]:
        window = self._window
        if window is not None:
            try:
                if win32gui.IsWindow(window.hwnd):
                    window.rect = win32gui.GetWindowRect(window.hwnd)
                    return window
            except win32gui.error:
                pass
            self._window = None

        now = time.monotonic()
        if now < self._next_search:
            return None
        self._next_search = now + self.refresh_seconds
        self._window = find_game_window(self.titles, self.processes)
        return self._window


class ClickTrigger:
    """Edge-detecting, self-click-proof left button trigger."""

    def __init__(self, config, window: WindowTracker, enabled: bool = False) -> None:
        self.cfg = config
        self.window = window
        self.enabled = enabled
        self._down = False
        # Nothing fires until a release has been observed, so enabling the
        # trigger with the button already down cannot start a cycle either.
        self._suppressed = True

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        self.suppress()
        return self.enabled

    def suppress(self) -> None:
        """Refuse to fire again until the button has been seen released."""
        self._suppressed = True

    def poll(self) -> bool:
        """True exactly once per qualifying click."""
        down = is_pressed(VK_LBUTTON)
        was_down, self._down = self._down, down

        if not self.enabled:
            # Keep tracking the level so re-enabling mid-hold is not an edge,
            # but require a release before the first cycle either way.
            self._suppressed = True
            return False

        if self._suppressed:
            if not down:
                self._suppressed = False
            return False

        fired = (down and not was_down) if self.cfg.fire_on_press else (was_down and not down)
        if not fired:
            return False
        if self.cfg.fire_on_press:
            # The button is still down and the engine is about to inject its
            # own; nothing more may fire until this one is let go.
            self._suppressed = True
        return self._in_game()

    def _in_game(self) -> bool:
        if not (self.cfg.require_foreground or self.cfg.require_inside_window):
            return True

        window = self.window.current()
        if window is None:
            # Refusing is the safe answer: the alternative is starting a click
            # sequence off a click we cannot place, which is how a stray click
            # in another application ends up driving the mouse.
            log.debug("click ignored: the game window has not been found")
            return False
        if self.cfg.require_foreground and foreground_hwnd() != window.hwnd:
            log.debug("click ignored: '%s' is not the foreground window", window.title)
            return False
        if self.cfg.require_inside_window:
            x, y = win32gui.GetCursorPos()
            left, top, right, bottom = window.rect
            if not (left <= x < right and top <= y < bottom):
                log.debug("click ignored: (%d,%d) is outside the game window", x, y)
                return False
        return True

    def describe(self) -> str:
        return (
            f"{'ON' if self.enabled else 'off'} "
            f"(fires on {'press' if self.cfg.fire_on_press else 'release'})"
        )
