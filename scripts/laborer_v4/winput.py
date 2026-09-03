"""Windows input injection, DPI awareness and the foreground-window gate.

The SendInput / shift-handling *timings* are carried over from v3 unchanged.
What is new: process DPI awareness (without it, capture coordinates and cursor
coordinates diverge on any display scaling other than 100%), a foreground
window gate, a panic key that can abort a cycle mid-flight - and the routing
guarantees below, which are what makes the shift actually reach the game.

Why a shift-click can arrive as a plain click
---------------------------------------------

Windows routes the two halves of a shift-click through different paths:

    mouse    -> the window under the *cursor*
    keyboard -> the window with *focus*

So whenever the game is not the focused window, the click still lands on it
while the shift key-down goes somewhere else entirely. The game sees an
unmodified click: the journal comes onto the cursor and nothing is handed over.

Focus is therefore the condition that decides whether the modifier arrives, and
it is checked directly - before the shift goes down and again before every
click, because focus can be stolen mid-sequence.

Why the modifier must not be *verified* mid-sequence
----------------------------------------------------

The tempting check is to read shift out of the game's own input queue with
``AttachThreadInput`` + ``GetKeyState``. That check cannot be used while shift
is held, because of this line in the ``AttachThreadInput`` documentation:

    Note that key state, which can be ascertained by calls to the GetKeyState
    or GetKeyboardState function, is reset after a call to AttachThreadInput.

Attaching resets the keyboard state of the threads it touches - so asking the
game "do you have shift down?" is itself capable of dropping the shift, and the
answer it returns may already be the post-reset one. Running that check after
the press and again between clicks made shift-clicks fail intermittently and
then reported the failure it had just caused.

The rule this module now keeps: **never attach to the target's input queue
while an injected shift is held.** ``attached_input`` enforces it rather than
trusting callers. The queue read survives as a pre-flight probe (it is the only
way to detect an elevated game whose queue we cannot touch at all) and as a
diagnostic, both of which run with no modifier down.

``SetCursorPos`` has a related problem: it moves the OS cursor without putting a
movement event in the queue, so a game that tracks its own hover target from
input events can click whatever it last thought was under the cursor. Moves are
therefore injected through ``SendInput`` *first* - an injected move to where the
cursor already is carries no movement at all, so the injection has to happen
before the pointer is placed, not after it.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import win32api
import win32con
import win32gui
import win32process

log = logging.getLogger("laborer.input")

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Both return SHORT. Left at the default c_int the sign bit lands in the wrong
# place on the way back out, so declare them.
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.GetKeyState.restype = ctypes.c_short

# QueryFullProcessImageNameW rather than GetModuleFileNameEx: it needs only
# PROCESS_QUERY_LIMITED_INFORMATION and works across the 32/64-bit boundary,
# which matters because the client is a 32-bit process.
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
_kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
)
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

_MAX_PATH = 1024

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_SCANCODE = 0x0008

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

VK_CONTROL = 0x11
VK_MENU = 0x12
VK_SHIFT = 0x10
VK_LEFT_SHIFT = 0xA0
LEFT_SHIFT_SCANCODE = 0x2A

# True between an injected shift-down and its matching shift-up. Module scope
# rather than driver state so the attach guard below cannot be bypassed by a
# code path that does not hold the driver.
_shift_held = False

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HardwareInput(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class InputUnion(ctypes.Union):
    _fields_ = (("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput))


class Input(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", InputUnion))


class InputError(RuntimeError):
    """SendInput was rejected by the OS (usually UIPI against an elevated window)."""


class PanicAbort(Exception):
    """The panic key was pressed; unwind the current cycle immediately."""


def enable_dpi_awareness() -> str:
    """Make capture coordinates and cursor coordinates agree.

    Without this, a DPI-unaware process gets virtualised cursor coordinates
    while mss reports physical pixels, so every click lands off-target on any
    scaling other than 100%. v3 worked only because the display was at 100%.
    """
    try:
        # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        _user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):
        return "none"


def is_pressed(vk_code: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk_code) & 0x8000)


def combo_pressed(combo: Sequence[int]) -> bool:
    return all(is_pressed(code) for code in combo)


def any_combo_pressed(combos: Sequence[Sequence[int]]) -> bool:
    return any(combo_pressed(combo) for combo in combos)


def ignore_console_ctrl_c() -> None:
    """Stop Ctrl+C from killing the process when it is also an action hotkey."""
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)


def send_input(*inputs: Input) -> None:
    array = (Input * len(inputs))(*inputs)
    sent = _user32.SendInput(len(array), ctypes.byref(array), ctypes.sizeof(Input))
    if sent != len(inputs):
        # v3 called ctypes.get_last_error() on a DLL loaded without
        # use_last_error=True, so the reported code was always meaningless.
        raise InputError(
            f"SendInput sent {sent}/{len(inputs)} events "
            f"(WinError {ctypes.get_last_error()}); is the game running elevated?"
        )


def _keyboard_scancode_input(scan_code: int, flags: int) -> Input:
    return Input(
        type=INPUT_KEYBOARD,
        union=InputUnion(
            ki=KeyboardInput(wVk=0, wScan=scan_code, dwFlags=KEYEVENTF_SCANCODE | flags, time=0, dwExtraInfo=0)
        ),
    )


def _keyboard_vk_input(vk_code: int, flags: int) -> Input:
    scan_code = _user32.MapVirtualKeyW(vk_code, 0)
    return Input(
        type=INPUT_KEYBOARD,
        union=InputUnion(
            ki=KeyboardInput(wVk=vk_code, wScan=scan_code, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def _mouse_input(flags: int) -> Input:
    return Input(
        type=INPUT_MOUSE,
        union=InputUnion(mi=MouseInput(dx=0, dy=0, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)),
    )


def _absolute_move_input(x: int, y: int) -> Input:
    """A real movement event at (x, y), in virtual-desktop coordinates.

    SendInput takes absolute coordinates normalised to 0..65535 across the
    whole virtual desktop, not pixels, and the mapping is over ``size - 1``.
    """
    left = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = max(1, _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN) - 1)
    height = max(1, _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN) - 1)
    normalised_x = int(round((x - left) * 65535.0 / width))
    normalised_y = int(round((y - top) * 65535.0 / height))
    return Input(
        type=INPUT_MOUSE,
        union=InputUnion(
            mi=MouseInput(
                dx=max(0, min(65535, normalised_x)),
                dy=max(0, min(65535, normalised_y)),
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


# --- reading key state as the *target window* sees it ---------------------


def window_thread_id(hwnd: int) -> int:
    if not hwnd:
        return 0
    return int(_user32.GetWindowThreadProcessId(hwnd, None))


class attached_input:
    """Briefly share ``hwnd``'s thread input queue.

    Needed because ``GetKeyState`` reports the *calling thread's* queue, so the
    only way to ask "does the game have shift down?" is to be attached to the
    game's queue while asking.

    Kept as short-lived as possible on purpose: while two threads are attached
    they share input state, and this process never pumps a message loop, so
    holding the attachment across a sleep could stall the game's own input.
    Attach, read, detach - never attach across an injected event.

    Refuses to attach at all while an injected shift is held: the attach resets
    keyboard state (see the module docstring), so doing it there destroys the
    modifier mid-click. That is a property of the API, not of any one caller,
    so it is enforced here rather than documented and hoped for.
    """

    def __init__(self, hwnd: Optional[int], force: bool = False) -> None:
        self.target = window_thread_id(hwnd or 0)
        self.self_thread = int(_kernel32.GetCurrentThreadId())
        self.attached = False
        # ``force`` is for the diagnostic path only, where the measurement is
        # the point and no click follows the perturbation it causes.
        self.force = force

    def __enter__(self) -> "attached_input":
        if _shift_held and not self.force:
            log.debug("not attaching to thread %s: an injected shift is held", self.target)
            return self
        if self.target and self.target != self.self_thread:
            self.attached = bool(_user32.AttachThreadInput(self.self_thread, self.target, True))
        return self

    def __exit__(self, *_exc: object) -> bool:
        if self.attached:
            _user32.AttachThreadInput(self.self_thread, self.target, False)
            self.attached = False
        return False


def key_state_in_window(hwnd: Optional[int], vk_code: int, force: bool = False) -> Optional[bool]:
    """Is ``vk_code`` down according to ``hwnd``'s input queue?

    ``None`` means the question could not be asked - no window, the attach was
    refused (a higher-integrity target, typically a game run as admin), or an
    injected shift is currently held. A ``None`` from a refused attach is itself
    a finding: input injection into that window is unlikely to work at all.

    Reading this costs an ``AttachThreadInput``, which resets keyboard state.
    Only call it with no modifier down - as a pre-flight probe or a diagnostic,
    never inside a click sequence. ``force`` overrides the held-shift guard for
    the one caller that measures shift on purpose and sends no click after it.
    """
    if not hwnd:
        return None
    try:
        with attached_input(hwnd, force=force) as attachment:
            if not attachment.attached:
                return None
            return bool(_user32.GetKeyState(vk_code) & 0x8000)
    except OSError:
        return None


def focus_window(hwnd: int) -> bool:
    """Make ``hwnd`` the foreground window, so keyboard input routes to it.

    SetForegroundWindow is refused outright for a process that does not own the
    current foreground window, so attach to that window's input queue first -
    the documented way to be allowed to make the call.
    """
    if not hwnd:
        return False
    if win32gui.GetForegroundWindow() == hwnd:
        return True
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except win32gui.error:
        pass

    with attached_input(win32gui.GetForegroundWindow()):
        try:
            win32gui.SetForegroundWindow(hwnd)
        except win32gui.error as error:
            log.debug("SetForegroundWindow(%s) refused: %s", hwnd, error)
    return win32gui.GetForegroundWindow() == hwnd


@dataclass
class GameWindow:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    # Basename of the owning executable, lowercased; "" when it could not be
    # read (an elevated process refuses the handle).
    process: str = ""

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        left, top, right, bottom = self.rect
        return left, top, right - left, bottom - top

    @property
    def area(self) -> int:
        return (self.rect[2] - self.rect[0]) * (self.rect[3] - self.rect[1])


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def window_process_name(hwnd: int) -> str:
    """Basename of the executable owning ``hwnd``, lowercased.

    Returns "" when the process cannot be opened - an elevated game refuses a
    handle to a medium-integrity caller. Callers must treat "" as *unknown*,
    never as *not the game*, or running the client as administrator would make
    it invisible to the picker.
    """
    try:
        _thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
    except Exception:  # pragma: no cover - pywin32 raises bare errors here
        return ""
    if not pid:
        return ""

    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(_MAX_PATH)
        buffer = ctypes.create_unicode_buffer(_MAX_PATH)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value.rsplit("\\", 1)[-1].lower()
    finally:
        _kernel32.CloseHandle(handle)


def _title_rank(title: str, needles: Sequence[str]) -> int:
    """3 exact, 2 prefix, 1 substring, 0 no match."""
    lowered = title.lower()
    best = 0
    for needle in needles:
        if lowered == needle:
            return 3
        if lowered.startswith(needle):
            best = max(best, 2)
        elif needle in lowered:
            best = max(best, 1)
    return best


def find_game_window(
    title_contains: Sequence[str],
    process_contains: Sequence[str] = (),
) -> Optional[GameWindow]:
    """The game's own window - not merely the biggest one mentioning it.

    Picking the largest title match is wrong on any machine that has other
    windows talking *about* Albion, and those are exactly the windows a player
    keeps open. Observed on this setup:

        AlbionOnline - StatisticsAnalysisTool   2115796 px   monitor 1
        Albion Online Client                    2073600 px   monitor 2   <- the game
        Albion gestion - Google Sheets - Chrome 2044416 px   monitor 1
        laborer-v4.py - albion-assistant - VS   2044416 px   monitor 1

    The market tool's window is 42k px larger than the client, so "largest
    wins" selected it, and with it monitor 1 - the whole search region then sat
    on the wrong screen. Every lookup was doomed from there, which surfaced as
    an inventory scan that found no journals at all.

    Identity therefore comes from the *executable*, which no window title can
    imitate. The title stays as the tie-break for two windows of the same
    process (launcher vs client) and as the whole answer when the process name
    is unreadable.
    """
    needles = [needle.lower() for needle in title_contains if needle]
    processes = [name.lower() for name in process_contains if name]
    if not needles and not processes:
        return None

    found: List[GameWindow] = []

    def _callback(hwnd: int, _extra: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        try:
            rect = win32gui.GetWindowRect(hwnd)
        except win32gui.error:
            return True
        if rect[2] - rect[0] < 200 or rect[3] - rect[1] < 200:
            return True
        process = window_process_name(hwnd) if processes else ""
        if not _title_rank(title, needles) and not _process_matches(process, processes):
            return True
        found.append(GameWindow(hwnd=hwnd, title=title, rect=rect, process=process))
        return True

    win32gui.EnumWindows(_callback, None)
    return select_game_window(found, needles, processes)


def select_game_window(
    found: Sequence[GameWindow],
    needles: Sequence[str],
    processes: Sequence[str],
) -> Optional[GameWindow]:
    """Rank already-enumerated candidates. Split out so it can be tested."""
    if not found:
        return None

    by_process = [window for window in found if _process_matches(window.process, processes)]
    if by_process:
        candidates = by_process
    else:
        # No process matched. Anything whose process we *could* read is now
        # known not to be the game, so only the unreadable ones stay plausible;
        # falling back to every title match would re-select the market tool.
        # If that leaves nothing, the game is not running.
        candidates = [
            window for window in found
            if (not processes or not window.process) and _title_rank(window.title, needles)
        ]
        if not candidates:
            log.warning(
                "no window belongs to %s; ignoring %d title-only match(es): %s",
                "/".join(processes) or "(no process filter)",
                len(found),
                ", ".join(f"{window.title!r} ({window.process or 'unknown'})" for window in found),
            )
            return None

    best = max(candidates, key=lambda window: (_title_rank(window.title, needles), window.area))
    rejected = [window for window in found if window.hwnd != best.hwnd]
    if rejected:
        log.debug(
            "game window %r (%s); ignored %s",
            best.title, best.process or "unknown process",
            ", ".join(f"{window.title!r}" for window in rejected),
        )
    return best


def _process_matches(process: str, needles: Sequence[str]) -> bool:
    return bool(process) and any(needle in process for needle in needles)


def foreground_hwnd() -> int:
    return int(win32gui.GetForegroundWindow() or 0)


def foreground_title() -> str:
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return ""
    try:
        return win32gui.GetWindowText(hwnd) or ""
    except win32gui.error:
        return ""


def foreground_matches(title_contains: Sequence[str]) -> bool:
    title = foreground_title().lower()
    return any(needle.lower() in title for needle in title_contains if needle)


class InputDriver:
    """All cursor and click injection, with dry-run and panic support."""

    def __init__(
        self,
        config,
        dry_run: bool = False,
        panic: Optional[Callable[[], bool]] = None,
        target_titles: Sequence[str] = (),
        target_processes: Sequence[str] = (),
    ) -> None:
        self.cfg = config
        self.dry_run = dry_run
        self._panic = panic or (lambda: False)
        self.target_titles = list(target_titles)
        self.target_processes = list(target_processes)
        # Set by the engine each time geometry is refreshed. Everything that
        # needs to know where input is being routed reads it from here.
        self.target_hwnd: Optional[int] = None
        self._preflight_done = False
        # Set when a release did not take, so the next plain click can undo it.
        self._shift_stuck = False

    def check_panic(self) -> None:
        if self._panic():
            raise PanicAbort("panic key pressed")

    def cursor_position(self) -> Tuple[int, int]:
        return win32api.GetCursorPos()

    def resolve_target(self) -> Optional[int]:
        """The game window, re-found if the cached handle has gone stale."""
        hwnd = self.target_hwnd
        if hwnd and win32gui.IsWindow(hwnd):
            return hwnd
        # The same process filter the engine uses. Re-finding the target by
        # title alone would focus whichever window is biggest - and focusing
        # the market tool instead of the game sends the shift there, so the
        # click that follows arrives at the game unmodified.
        if self.target_titles or self.target_processes:
            window = find_game_window(self.target_titles, self.target_processes)
        else:
            window = None
        self.target_hwnd = window.hwnd if window else None
        return self.target_hwnd

    def ensure_target_focus(self) -> bool:
        """Give the game keyboard focus before any modified input goes out.

        Without this a shift-click degrades into a plain click: the button
        press is routed by cursor position and reaches the game, while the
        shift is routed by focus and does not.
        """
        hwnd = self.resolve_target()
        if hwnd is None:
            # Nothing to focus and nothing to verify against. Callers that care
            # about the modifier will fail the verification step instead.
            return not self.cfg.focus_target_before_input
        if foreground_hwnd() == hwnd:
            return True
        if not self.cfg.focus_target_before_input:
            log.warning(
                "the game is not the foreground window and focus_target_before_input is off; "
                "a shift-click will arrive unmodified"
            )
            return False
        if self.dry_run:
            log.info("[dry-run] focus game window %s", hwnd)
            return True
        if focus_window(hwnd):
            log.info("brought the game window to the foreground before injecting input")
            time.sleep(self.cfg.focus_settle)
            return True
        log.error("could not give the game window focus; refusing to inject a modified click")
        return False

    def _place_cursor(self, x: int, y: int) -> None:
        """Put the cursor at (x, y) and leave a real movement event behind it.

        The order is the whole point. ``SetCursorPos`` moves the pointer without
        queueing anything, and an injected move to a position the cursor already
        occupies carries no movement - so placing first and injecting second can
        silently reduce to no movement event at all, which leaves a game that
        tracks its own hover target still pointing at the previous slot. Inject
        first, then pin the exact pixel: the absolute mapping is over a 0..65535
        grid and rounds, so ``SetCursorPos`` still has a job to do afterwards.
        """
        if self.cfg.inject_mouse_move:
            try:
                if win32api.GetCursorPos() == (x, y):
                    # Already parked here, so there is nothing to move *into*
                    # the slot. Step off first; the game needs to see the
                    # pointer enter, not merely be present.
                    nudge = max(1, int(self.cfg.cursor_nudge_pixels))
                    send_input(_absolute_move_input(x + nudge, y + nudge))
                    time.sleep(self.cfg.cursor_settle)
                send_input(_absolute_move_input(x, y))
            except InputError as error:
                log.debug("injected move rejected: %s", error)
        win32api.SetCursorPos((x, y))

    def move_cursor(self, position: Tuple[int, int]) -> bool:
        x, y = int(position[0]), int(position[1])
        if self.dry_run:
            log.info("[dry-run] move cursor to (%d, %d)", x, y)
            return True
        for _ in range(max(1, self.cfg.cursor_move_attempts)):
            self._place_cursor(x, y)
            time.sleep(self.cfg.cursor_settle)
            if win32api.GetCursorPos() == (x, y):
                return True
        log.warning("cursor did not settle at (%d, %d)", x, y)
        return False

    def send_left_click(self, hold_seconds: float) -> None:
        if self.dry_run:
            log.info("[dry-run] left click (hold %.3fs)", hold_seconds)
            return
        send_input(_mouse_input(MOUSEEVENTF_LEFTDOWN))
        time.sleep(hold_seconds)
        send_input(_mouse_input(MOUSEEVENTF_LEFTUP))

    def click(self, position: Tuple[int, int]) -> bool:
        """A plain left click at ``position``; False if nothing was sent.

        Refuses to click when the cursor could not be placed - the click would
        otherwise go out wherever the pointer actually is, which on this UI is
        another inventory slot or another button.
        """
        self.check_panic()
        self.release_stuck_shift()
        if not self.move_cursor(position):
            log.error("cursor did not reach %s; not clicking a position we are not on", position)
            return False
        self.send_left_click(self.cfg.click_hold)
        return True

    def restore_cursor(self, position: Tuple[int, int]) -> None:
        if self.dry_run:
            log.info("[dry-run] restore cursor to %s", position)
            return
        try:
            win32api.SetCursorPos((int(position[0]), int(position[1])))
        except win32api.error as error:
            log.warning("could not restore cursor to %s: %s", position, error)

    # --- shift handling ------------------------------------------------

    @staticmethod
    def shift_registered() -> bool:
        """Did the OS accept our injected shift? Says nothing about routing."""
        return is_pressed(VK_SHIFT) or is_pressed(VK_LEFT_SHIFT)

    def routing_ok(self) -> Optional[bool]:
        """Would a keystroke injected right now reach the game?

        Keyboard input goes to the focused window, so this is a foreground
        comparison and nothing more - which is exactly why it is safe to call
        between clicks: no attach, no key-state reset, sub-millisecond.
        ``None`` means there is nothing to compare against.
        """
        if not self.target_titles and not self.target_processes:
            return None
        hwnd = self.resolve_target()
        if hwnd:
            return foreground_hwnd() == hwnd
        return foreground_matches(self.target_titles)

    def shift_reaches_target(self) -> bool:
        """The gate every click has to pass, cheap enough to re-check per click.

        Two independent conditions, both necessary: the OS has to have taken
        the key at all, and the game has to be the window keyboard input is
        being routed to. Losing focus mid-sequence is the failure that turns a
        shift-click into a plain click, and it is caught here rather than
        assumed away by a check made once before the first click.
        """
        if not self.shift_registered():
            return False
        routed = self.routing_ok()
        if routed is None:
            # No window to verify against (window.title_contains is empty).
            # The async state is all there is; say so once, in pre-flight.
            return True
        return routed

    def shift_in_target(self) -> Optional[bool]:
        """Does the *game's* input queue have shift down? ``None`` if unknowable.

        Pre-flight and diagnostics only - this attaches to the game's input
        queue, which resets keyboard state. ``attached_input`` refuses while an
        injected shift is held, so calling it there answers ``None`` rather than
        breaking the click.
        """
        if not self.cfg.verify_shift_in_target:
            return None
        hwnd = self.resolve_target()
        state = key_state_in_window(hwnd, VK_SHIFT)
        if state is None:
            state = key_state_in_window(hwnd, VK_LEFT_SHIFT)
        return state

    def preflight_shift(self) -> None:
        """Warn about conditions that make a shift-click impossible, once.

        Runs before the modifier goes down, because the only check that can
        detect an unreadable (elevated) input queue is the one that must not run
        afterwards. Advisory: it never blocks a click on its own, since an
        unreadable queue is a strong hint and not a verdict.
        """
        if self._preflight_done:
            return

        if self.routing_ok() is None:
            self._preflight_done = True
            log.warning(
                "window.title_contains is empty, so there is no way to tell which window "
                "keyboard input reaches; shift-clicks will be sent on the OS key state alone"
            )
            return
        if not self.cfg.verify_shift_in_target:
            self._preflight_done = True
            return
        if not self.resolve_target():
            # The window is not up yet; leave the probe pending rather than
            # burning the one warning on a question that had no answer.
            return
        self._preflight_done = True
        if self.shift_in_target() is None:
            log.warning(
                "the game window's input queue cannot be read (AttachThreadInput refused), "
                "which usually means the game is elevated and this script is not. Injected "
                "input may be dropped entirely - run --test-shift for the full diagnosis."
            )

    @staticmethod
    def _try_send(*inputs: Input) -> bool:
        """SendInput, reporting rejection instead of raising.

        Used for the modifier only. A rejected key press has a fallback worth
        trying and a diagnosis worth printing; raising here would abort the
        cycle before the release ever ran.
        """
        try:
            send_input(*inputs)
            return True
        except InputError as error:
            log.debug("key injection rejected: %s", error)
            return False

    def shift_down(self) -> bool:
        global _shift_held
        if self.dry_run:
            return True
        # Flagged before the first injection and cleared only in shift_up, so
        # nothing can attach to the game's input queue (and reset the modifier)
        # for as long as it is held - including on the failure paths below.
        _shift_held = True
        for _ in range(max(1, self.cfg.shift_press_attempts)):
            # One key-down, sent as a scancode: that is what a game reading raw
            # input or DirectInput looks at, and Windows still synthesises
            # VK_LSHIFT into the window message for anything reading that.
            # Sending the VK form *and* the scancode form, as v3 did, is two
            # presses of one key - noise that nothing needed.
            self._try_send(_keyboard_scancode_input(LEFT_SHIFT_SCANCODE, 0))
            time.sleep(self.cfg.shift_state_settle)
            if self.shift_reaches_target():
                return True
            # Fallbacks in increasing order of desperation, for the case where
            # SendInput itself is being filtered.
            self._try_send(_keyboard_vk_input(VK_LEFT_SHIFT, 0))
            time.sleep(self.cfg.shift_state_settle)
            if self.shift_reaches_target():
                return True
            win32api.keybd_event(VK_LEFT_SHIFT, LEFT_SHIFT_SCANCODE, 0, 0)
            time.sleep(self.cfg.shift_state_settle)
            if self.shift_reaches_target():
                return True
        log.warning("shift key did not register as pressed")
        return False

    def shift_up(self) -> bool:
        """Release shift, and say so loudly if it did not take.

        A shift that stays down is worse than one that never went down: every
        following plain click in the cycle becomes a shift-click, so the accept
        button or the settings gear gets clicked with a modifier the caller
        does not know about.
        """
        global _shift_held
        if self.dry_run:
            return True
        released = False
        try:
            for _ in range(max(1, self.cfg.shift_press_attempts)):
                # Both forms on the way out, unlike the press: whichever one the
                # game latched onto has to be told the key is up.
                self._try_send(
                    _keyboard_scancode_input(LEFT_SHIFT_SCANCODE, win32con.KEYEVENTF_KEYUP),
                    _keyboard_vk_input(VK_LEFT_SHIFT, win32con.KEYEVENTF_KEYUP),
                )
                time.sleep(self.cfg.shift_state_settle)
                if not self.shift_registered():
                    released = True
                    break
                win32api.keybd_event(VK_LEFT_SHIFT, LEFT_SHIFT_SCANCODE, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(self.cfg.shift_state_settle)
                if not self.shift_registered():
                    released = True
                    break
        finally:
            # Cleared either way: the release events have been sent, and leaving
            # the flag set would disable the queue probe for the rest of the run.
            _shift_held = False

        self._shift_stuck = not released
        if not released:
            log.error(
                "shift is still down after %d release attempts - any plain click that follows "
                "would carry it. If you are holding shift yourself, let go.",
                max(1, self.cfg.shift_press_attempts),
            )
        return released

    def release_stuck_shift(self) -> None:
        """Undo a shift left down by a failed release, before a plain click.

        Gated on our own release having failed, not merely on shift being down:
        an operator resting on the shift key is not something to fight over on
        every click, and injecting a release under their hand achieves nothing.
        """
        if self.dry_run or not self._shift_stuck:
            return
        if not self.shift_registered():
            self._shift_stuck = False
            return
        log.warning("shift is still down from the last hand-over; releasing it before clicking")
        if self.shift_up():
            self._shift_stuck = False

    def wait_for_modifier_release(self) -> None:
        """Wait out any modifier the operator is still physically holding.

        Ctrl+C is one of the trigger combos, and Alt is easy to be leaning on.
        Either one turns the intended Shift+click into Ctrl+Shift+click or
        Alt+Shift+click, which the game treats as a different action.
        """
        modifiers = (("ctrl", VK_CONTROL), ("alt", VK_MENU))
        deadline = time.monotonic() + self.cfg.modifier_release_timeout
        while time.monotonic() < deadline:
            if not any(is_pressed(code) for _name, code in modifiers):
                return
            time.sleep(0.005)
        held = [name for name, code in modifiers if is_pressed(code)]
        if held:
            log.warning(
                "%s still held after %.2fs; the click will carry it",
                "+".join(held), self.cfg.modifier_release_timeout,
            )

    def shift_click(
        self,
        position: Tuple[int, int],
        clicks: Optional[int] = None,
        delay_scale: float = 1.0,
    ) -> bool:
        """Shift-click ``position``, or do nothing and say so.

        Returns True only if every click went out with the OS holding shift and
        the game holding focus. A False return means no *unmodified* click was
        fired either - firing one would pick the journal up onto the cursor, and
        the next retry would then drop it into whatever slot it landed over.

        ``clicks`` and ``delay_scale`` are the retry knobs: the caller sends one
        click at normal speed first and, if that produced nothing, comes back
        with more clicks spaced further apart. ``delay_scale`` stretches the
        pauses around the clicks only - the button *hold* is what the game reads
        as a click rather than a drag, so it is deliberately left alone.
        """
        self.check_panic()
        self.wait_for_modifier_release()

        if not self.ensure_target_focus():
            return False
        # Anything that needs the game's input queue happens here, with no
        # modifier down - reading it later would reset the shift we are about
        # to hold.
        self.preflight_shift()

        count = max(1, int(self.cfg.shift_click_count if clicks is None else clicks))
        scale = max(1.0, float(delay_scale))

        target = (int(position[0]), int(position[1]))
        if not self.move_cursor(target):
            log.error("cursor did not reach %s; not shift-clicking a position we are not on", target)
            return False
        time.sleep(self.cfg.shift_move_delay * scale)

        if self.dry_run:
            log.info("[dry-run] shift-click x%d at %s", count, position)
            return True

        clicks_sent = 0
        all_registered = True
        try:
            # Inside the try: once any key-down has gone out, the release has to
            # run no matter what happens next.
            self.shift_down()
            time.sleep(self.cfg.shift_before_click_delay * scale)
            for _ in range(count):
                # Shift can drop between clicks - a focus steal is enough, and
                # focus is what decides whether the modifier arrives at all.
                # Re-press before any click that would otherwise land unmodified
                # and pick the journal up onto the cursor.
                if not self.shift_reaches_target():
                    log.debug("shift is no longer reaching the game, re-pressing")
                    self.shift_down()
                    if not self.shift_reaches_target():
                        all_registered = False
                        # Firing unmodified would silently move one item, which
                        # is worse for the caller's retry logic than doing nothing.
                        log.warning(
                            "shift is still not reaching the game%s; skipping this click "
                            "rather than sending it unmodified",
                            "" if self.shift_registered() else " (the OS did not take it either)",
                        )
                        continue
                # The operator's hand is on the mouse and a retry burst leaves
                # hundreds of milliseconds between its clicks; a nudge of the
                # physical mouse in that gap would drop the next click on a
                # neighbouring slot.
                if win32api.GetCursorPos() != target and not self.move_cursor(target):
                    all_registered = False
                    log.warning("cursor moved away from %s mid-sequence; stopping here", target)
                    break
                self.send_left_click(self.cfg.shift_click_hold)
                clicks_sent += 1
                time.sleep(self.cfg.shift_click_gap * scale)
            time.sleep(self.cfg.shift_after_click_hold * scale)
        finally:
            self.shift_up()
            time.sleep(self.cfg.shift_release_delay * scale)

        if not all_registered:
            self._explain_lost_shift()
        return all_registered and clicks_sent > 0

    def _explain_lost_shift(self) -> None:
        """Turn 'shift did not register' into something actionable, once.

        Called after the release, so the queue probe below is allowed to attach
        again - during the click it would have been refused.
        """
        hwnd = self.resolve_target()
        if hwnd is None:
            log.error(
                "no window matching %s was found, so shift cannot be routed to the game. "
                "Check window.title_contains in the config.",
                self.target_titles or "(no titles configured)",
            )
            return
        foreground = foreground_hwnd()
        if foreground != hwnd:
            log.error(
                "'%s' is the foreground window, not the game - keyboard input goes there "
                "while the click goes to the game underneath the cursor.",
                foreground_title() or "(none)",
            )
            return
        if key_state_in_window(hwnd, VK_SHIFT) is None:
            log.error(
                "the game window's input queue could not be read (AttachThreadInput refused). "
                "That normally means the game is running elevated and this script is not - "
                "run the script as administrator too, or the game unelevated."
            )
            return
        log.error(
            "the game had focus, but the OS never reported the injected shift as down. "
            "SendInput is being filtered or another process is clearing the modifier; "
            "try running the script as administrator."
        )


def diagnose_shift(
    driver: "InputDriver",
    title_contains: Sequence[str],
    process_contains: Sequence[str] = (),
) -> int:
    """Answer 'why does the game not see my shift?' without touching the game.

    Injects shift with the cursor left where it is and no click at all, then
    reports what each layer believes. Every line below is a different place the
    key can be lost, in the order it can be lost.
    """
    log.info("--- shift diagnosis ---")

    window = find_game_window(title_contains, process_contains)
    if window is None:
        log.error(
            "no visible window belongs to %s with a title containing %s - nothing to inject into",
            list(process_contains), list(title_contains),
        )
        log.info("fix: start the game, or correct window.process_contains / title_contains in the config")
        return 1
    log.info("game window: '%s' (%s, hwnd %s)", window.title, window.process or "unknown process", window.hwnd)
    driver.target_hwnd = window.hwnd

    focused = foreground_hwnd() == window.hwnd
    log.info(
        "foreground window: '%s'%s",
        foreground_title() or "(none)",
        "  <- the game, keyboard input routes here" if focused
        else "  <- NOT the game; keyboard goes here, clicks go to the game",
    )

    if not focused:
        if focus_window(window.hwnd):
            log.info("focus restored to the game")
            time.sleep(driver.cfg.focus_settle)
        else:
            log.error("could not focus the game window - shift cannot reach it")
            return 1

    readable = key_state_in_window(window.hwnd, VK_SHIFT) is not None
    log.info(
        "game input queue readable: %s%s",
        readable,
        "" if readable else "  <- AttachThreadInput refused; the game is almost certainly elevated",
    )

    log.info("injecting shift down (no click will be sent)")
    try:
        driver.shift_down()
        time.sleep(0.05)
        os_state = driver.shift_registered()
        routed = driver.routing_ok()
        # Deliberately destructive: reading the game's queue resets keyboard
        # state, which is why the click path never does this. Here it is the
        # measurement being asked for and nothing is clicked afterwards.
        game_state = key_state_in_window(window.hwnd, VK_SHIFT, force=True)
        log.info("  OS async key state: %s", "down" if os_state else "UP - SendInput was dropped")
        log.info("  keyboard routing:   %s", "to the game" if routed else "NOT to the game")
        log.info(
            "  game input queue:   %s  (indicative only - reading it resets key state)",
            "unreadable" if game_state is None else ("down" if game_state else "reports UP"),
        )
    finally:
        driver.shift_up()

    # The two conditions that actually decide whether the modifier arrives, and
    # the only two that can be measured without disturbing what is measured.
    if not os_state:
        log.error("the OS did not even accept the injected key. Run the script as administrator.")
        return 1
    if routed is False:
        log.error("the game lost focus during the test, so the modifier went to another window.")
        return 1

    if game_state is None:
        log.warning(
            "the game's queue could not be read, so this test can only confirm that the OS took "
            "the key and that the game has focus - which is exactly what the click path gates on."
        )
    elif not game_state:
        # Do not report this as a fault. It is the expected reading: the probe
        # resets the state it is asking about, so a focused game holding an
        # injected shift still answers "up" here. Treating this answer as a
        # verdict is what used to make shift-clicks fail intermittently.
        log.warning(
            "the game's queue reports shift UP while the OS reports it down. That is the probe "
            "disturbing what it measures - AttachThreadInput resets keyboard state - not "
            "evidence that the game refused the key. The click path never makes this call, and "
            "gates on the two answers above instead; both of those are good."
        )

    log.info("shift reaches the game correctly; a shift-click here should work")
    return 0
