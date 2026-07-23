from __future__ import annotations

import ctypes
import json
import time
import winsound
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from ctypes import wintypes

import cv2
import mss
import numpy as np
import win32api
import win32con


ROOT_DIR = Path(__file__).resolve().parents[1]
MAPPING_FILE = ROOT_DIR / "mapping.json"
JOURNAL_POSITIONS_FILE = ROOT_DIR / "journal-positions.json"
TAKE_ALL_IMAGE = ROOT_DIR / "assets" / "menu" / "take-all.png"
ADVANCE_TIER_IMAGE = ROOT_DIR / "assets" / "menu" / "advance-tier.png"
READY_IMAGE = ROOT_DIR / "assets" / "menu" / "ready.png"
ACCEPT_IMAGE = ROOT_DIR / "assets" / "menu" / "accept.png"
LABORER_ASSETS_DIR = ROOT_DIR / "assets" / "laborers"

MATCH_THRESHOLD = 0.80
LABORER_MATCH_THRESHOLD = 0.97
JOURNAL_MATCH_THRESHOLD = 0.90
CAPTURE_MONITOR_INDEX: Optional[int] = None
INVENTORY_RIGHT_START_RATIO = 0.50
POLL_SECONDS = 0.01
DEBOUNCE_SECONDS = 0.20
LAST_MATCH_SEARCH_PADDING = 90
ADVANCE_TIER_TIMEOUT_SECONDS = 0.50
ADVANCE_TIER_RESTORE_DELAY_SECONDS = 0.30
READY_TIMEOUT_SECONDS = 0.50
ACCEPT_TIMEOUT_SECONDS = 1.00
ACCEPT_RETRY_ATTEMPTS = 2
PREFETCH_LABORER_WAIT_SECONDS = 1.50
LABORER_HINT_PADDING = 120
TAKE_ALL_NOT_FOUND_DELAY_SECONDS = 0.10
SHIFT_MOVE_DELAY_SECONDS = 0.12
SHIFT_BEFORE_CLICK_DELAY_SECONDS = 0.05
SHIFT_CLICK_COUNT = 2
SHIFT_CLICK_HOLD_SECONDS = 0.06
SHIFT_CLICK_GAP_SECONDS = 0.05
SHIFT_AFTER_CLICK_HOLD_SECONDS = 0.10
SHIFT_RELEASE_DELAY_SECONDS = 0.04
SHIFT_CLICK_RETRY_ATTEMPTS = 3
SHIFT_CLICK_VERIFY_DELAY_SECONDS = 0.15
RESTORE_MOUSE_DELAY_SECONDS = 0.30
CURSOR_MOVE_ATTEMPTS = 3
CURSOR_SETTLE_SECONDS = 0.02
CLICK_HOLD_SECONDS = 0.04
MODIFIER_RELEASE_TIMEOUT_SECONDS = 0.50
JOURNAL_CACHE_VERIFY_PADDING = 12
SHIFT_PRESS_ATTEMPTS = 3
SHIFT_STATE_SETTLE_SECONDS = 0.02
LOG_LABORER_SCORES = False
STOP_LABORER_PREFIX = "t6-"
STOP_LABORER_BEEP_FREQUENCY = 1200
STOP_LABORER_BEEP_DURATION_MS = 350
LABORER_BADGE_REGION = (0, 0, 34, 34)
LABORER_BADGE_THRESHOLD = 0.85
LABORER_BADGE_COLOR_MAX_DISTANCE = 25.0

VK_1 = 0x31
VK_2 = 0x32
VK_C = 0x43
VK_CONTROL = 0x11
VK_NUMPAD1 = 0x61
VK_NUMPAD2 = 0x62
VK_SUPERSCRIPT_TWO = 0xC0
VK_SHIFT = 0x10
VK_LEFT_SHIFT = 0xA0
LEFT_SHIFT_SCANCODE = 0x2A
KEYEVENTF_SCANCODE = 0x0008
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

ImageMapping = Dict[str, Dict[str, Path]]
JournalPositionCache = Dict[str, Tuple[int, int]]
LaborerMatch = Tuple[str, float, int, Tuple[int, int]]

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
    _fields_ = (
        ("mi", MouseInput),
        ("ki", KeyboardInput),
        ("hi", HardwareInput),
    )


class Input(ctypes.Structure):
    _fields_ = (
        ("type", wintypes.DWORD),
        ("union", InputUnion),
    )

@dataclass(frozen=True)
class MaskedTemplate:
    image: np.ndarray
    mask: Optional[np.ndarray]


@dataclass
class TemplateSearchHint:
    position: Optional[Tuple[int, int]] = None
    monitor_index: Optional[int] = None


@dataclass
class LaborerSearchHint:
    position: Optional[Tuple[int, int]] = None
    monitor_index: Optional[int] = None
    laborer_name: Optional[str] = None


TemplateMapping = Dict[str, MaskedTemplate]


def load_mapping(path: Path) -> ImageMapping:
    with path.open("r", encoding="utf-8") as mapping_file:
        raw_mapping = json.load(mapping_file)

    mapping: ImageMapping = {}
    for laborer_name, journal_name in raw_mapping.items():
        laborer_path = ROOT_DIR / "assets" / "laborers" / f"{laborer_name}.png"
        journal_path = ROOT_DIR / "assets" / "journals" / f"{journal_name}.png"
        if not laborer_path.exists():
            raise FileNotFoundError(f"Laborer image not found for '{laborer_name}': {laborer_path}")
        if not journal_path.exists():
            raise FileNotFoundError(f"Journal image not found for '{journal_name}': {journal_path}")
        mapping[laborer_name] = {
            "laborer": laborer_path,
            "journal": journal_path,
        }

    return mapping


def parse_position(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, list) or len(value) != 2:
        return None

    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None

    return int(x), int(y)


def load_journal_positions(path: Path) -> JournalPositionCache:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as positions_file:
        raw_positions = json.load(positions_file)

    if not isinstance(raw_positions, dict):
        print(f"journal positions file has invalid format: {path}")
        return {}

    positions: JournalPositionCache = {}
    for journal_name, raw_position in raw_positions.items():
        if not isinstance(journal_name, str):
            continue

        position = parse_position(raw_position)
        if position is None:
            print(f"invalid cached journal position for {journal_name}: {raw_position}")
            continue

        positions[journal_name] = position

    return positions


def save_journal_positions(path: Path, positions: JournalPositionCache) -> None:
    serializable_positions = {
        journal_name: [position[0], position[1]]
        for journal_name, position in sorted(positions.items())
    }
    with path.open("w", encoding="utf-8") as positions_file:
        json.dump(serializable_positions, positions_file, indent=2)
        positions_file.write("\n")


def load_template(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def load_masked_template(path: Path) -> MaskedTemplate:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[:, :, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return MaskedTemplate(image=image, mask=mask)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return MaskedTemplate(image=image, mask=None)


def ordered_monitor_indexes(
    screen_capture: mss.mss,
    preferred_monitor_index: Optional[int] = None,
) -> list[int]:
    if CAPTURE_MONITOR_INDEX is None:
        monitor_indexes = list(range(1, len(screen_capture.monitors)))
    else:
        monitor_indexes = [CAPTURE_MONITOR_INDEX]

    if preferred_monitor_index in monitor_indexes:
        return [preferred_monitor_index] + [
            monitor_index
            for monitor_index in monitor_indexes
            if monitor_index != preferred_monitor_index
        ]

    return monitor_indexes


def capture_source(
    screen_capture: mss.mss,
    region: dict[str, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    screenshot = np.array(screen_capture.grab(region))
    source = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    return source, (region["left"], region["top"])


def capture_monitor_source(
    screen_capture: mss.mss,
    monitor_index: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    return capture_source(screen_capture, screen_capture.monitors[monitor_index])


def iter_captured_screens(preferred_monitor_index: Optional[int] = None):
    with mss.mss() as screen_capture:
        for monitor_index in ordered_monitor_indexes(screen_capture, preferred_monitor_index):
            source, origin = capture_monitor_source(screen_capture, monitor_index)
            yield monitor_index, source, origin


def capture_source_with_color(
    screen_capture: mss.mss,
    region: dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    screenshot = np.array(screen_capture.grab(region))
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    color = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    return gray, color, (region["left"], region["top"])


def capture_monitor_source_with_color(
    screen_capture: mss.mss,
    monitor_index: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    return capture_source_with_color(screen_capture, screen_capture.monitors[monitor_index])


def capture_screen_by_index_with_color(
    monitor_index: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    with mss.mss() as screen_capture:
        return capture_monitor_source_with_color(screen_capture, monitor_index)


def iter_captured_screens_with_color(preferred_monitor_index: Optional[int] = None):
    with mss.mss() as screen_capture:
        for monitor_index in ordered_monitor_indexes(screen_capture, preferred_monitor_index):
            gray, color, origin = capture_monitor_source_with_color(screen_capture, monitor_index)
            yield monitor_index, gray, color, origin


def template_match_score(
    source: np.ndarray,
    template: np.ndarray,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    template_height, template_width = template.shape[:2]
    source_height, source_width = source.shape[:2]
    if source_height < template_height or source_width < template_width:
        return None

    result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc


def template_center_position(
    origin: Tuple[int, int],
    max_loc: Tuple[int, int],
    template: np.ndarray,
) -> Tuple[int, int]:
    height, width = template.shape[:2]
    return origin[0] + max_loc[0] + width // 2, origin[1] + max_loc[1] + height // 2


def locate_template_in_source(
    source: np.ndarray,
    origin: Tuple[int, int],
    template: np.ndarray,
    match_threshold: float,
) -> Optional[Tuple[Tuple[int, int], float]]:
    score_match = template_match_score(source, template)
    if score_match is None:
        return None

    score, max_loc = score_match
    if score < match_threshold:
        return None

    return template_center_position(origin, max_loc, template), score


def template_hint_region(
    monitor: dict[str, int],
    position: Tuple[int, int],
    template: np.ndarray,
    padding: int = LAST_MATCH_SEARCH_PADDING,
) -> Optional[dict[str, int]]:
    center_x, center_y = position
    monitor_left = monitor["left"]
    monitor_top = monitor["top"]
    monitor_right = monitor_left + monitor["width"]
    monitor_bottom = monitor_top + monitor["height"]
    if not (monitor_left <= center_x < monitor_right and monitor_top <= center_y < monitor_bottom):
        return None

    template_height, template_width = template.shape[:2]
    left = max(monitor_left, center_x - template_width // 2 - padding)
    top = max(monitor_top, center_y - template_height // 2 - padding)
    right = min(monitor_right, center_x + (template_width + 1) // 2 + padding)
    bottom = min(monitor_bottom, center_y + (template_height + 1) // 2 + padding)

    width = right - left
    height = bottom - top
    if width < template_width or height < template_height:
        return None

    return {
        "left": int(left),
        "top": int(top),
        "width": int(width),
        "height": int(height),
    }


def remember_template_match(
    hint: Optional[TemplateSearchHint],
    position: Tuple[int, int],
    monitor_index: int,
) -> None:
    if hint is None:
        return

    hint.position = position
    hint.monitor_index = monitor_index


def find_template_from_hint(
    screen_capture: mss.mss,
    template: np.ndarray,
    match_threshold: float,
    hint: Optional[TemplateSearchHint],
) -> Optional[Tuple[Tuple[int, int], int]]:
    if hint is None or hint.position is None or hint.monitor_index is None:
        return None
    if hint.monitor_index <= 0 or hint.monitor_index >= len(screen_capture.monitors):
        return None
    if CAPTURE_MONITOR_INDEX is not None and hint.monitor_index != CAPTURE_MONITOR_INDEX:
        return None

    monitor = screen_capture.monitors[hint.monitor_index]
    region = template_hint_region(monitor, hint.position, template)
    if region is None:
        return None

    source, origin = capture_source(screen_capture, region)
    located = locate_template_in_source(source, origin, template, match_threshold)
    if located is None:
        return None

    position, score = located
    remember_template_match(hint, position, hint.monitor_index)
    print(f"matched near last position on monitor {hint.monitor_index} ({score:.3f}).")
    return position, hint.monitor_index


def capture_inventory_regions() -> list[Tuple[int, np.ndarray, Tuple[int, int]]]:
    with mss.mss() as screen_capture:
        captures = []
        for monitor_index in ordered_monitor_indexes(screen_capture):
            monitor = screen_capture.monitors[monitor_index]
            left_offset = int(monitor["width"] * INVENTORY_RIGHT_START_RATIO)
            region = {
                "left": monitor["left"] + left_offset,
                "top": monitor["top"],
                "width": monitor["width"] - left_offset,
                "height": monitor["height"],
            }
            source, origin = capture_source(screen_capture, region)
            captures.append((monitor_index, source, origin))
    return captures


def capture_screen_by_index(monitor_index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    with mss.mss() as screen_capture:
        return capture_monitor_source(screen_capture, monitor_index)


def find_template_on_screen(
    template: np.ndarray,
    match_threshold: float = MATCH_THRESHOLD,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> Optional[Tuple[int, int]]:
    match = find_template_on_screen_with_monitor(
        template,
        match_threshold,
        preferred_monitor_index,
        hint,
    )
    if match is None:
        return None
    position, _monitor_index = match
    return position


def find_template_on_screen_with_monitor(
    template: np.ndarray,
    match_threshold: float = MATCH_THRESHOLD,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> Optional[Tuple[Tuple[int, int], int]]:
    best_match: Optional[Tuple[int, Tuple[int, int], float]] = None
    with mss.mss() as screen_capture:
        hint_match = find_template_from_hint(screen_capture, template, match_threshold, hint)
        if hint_match is not None:
            return hint_match

        monitor_indexes = ordered_monitor_indexes(screen_capture, preferred_monitor_index)
        for monitor_index in monitor_indexes:
            source, origin = capture_monitor_source(screen_capture, monitor_index)
            score_match = template_match_score(source, template)
            if score_match is None:
                continue

            score, max_loc = score_match
            position = template_center_position(origin, max_loc, template)
            if best_match is None or score > best_match[2]:
                best_match = (monitor_index, position, score)

            if score >= match_threshold and (
                monitor_index == preferred_monitor_index or len(monitor_indexes) == 1
            ):
                remember_template_match(hint, position, monitor_index)
                print(f"matched on preferred monitor {monitor_index} ({score:.3f}).")
                return position, monitor_index

    if best_match is None or best_match[2] < match_threshold:
        return None

    monitor_index, position, score = best_match
    remember_template_match(hint, position, monitor_index)
    print(f"matched on monitor {monitor_index} ({score:.3f}).")
    return position, monitor_index


def move_cursor(position: Tuple[int, int]) -> bool:
    x, y = map(int, position)
    for _ in range(CURSOR_MOVE_ATTEMPTS):
        win32api.SetCursorPos((x, y))
        time.sleep(CURSOR_SETTLE_SECONDS)
        if win32api.GetCursorPos() == (x, y):
            return True
    print(f"cursor did not settle at {(x, y)}.")
    return False


def click(position: Tuple[int, int]) -> None:
    move_cursor(position)
    send_left_click(hold_seconds=CLICK_HOLD_SECONDS)


def send_input(*inputs: Input) -> None:
    input_array = (Input * len(inputs))(*inputs)
    sent = ctypes.windll.user32.SendInput(
        len(input_array),
        ctypes.byref(input_array),
        ctypes.sizeof(Input),
    )
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())


def keyboard_input(scan_code: int, flags: int) -> Input:
    return Input(
        type=INPUT_KEYBOARD,
        union=InputUnion(
            ki=KeyboardInput(
                wVk=0,
                wScan=scan_code,
                dwFlags=KEYEVENTF_SCANCODE | flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def mouse_input(flags: int) -> Input:
    return Input(
        type=INPUT_MOUSE,
        union=InputUnion(
            mi=MouseInput(
                dx=0,
                dy=0,
                mouseData=0,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def send_left_click(hold_seconds: float = 0.05) -> None:
    send_input(mouse_input(MOUSEEVENTF_LEFTDOWN))
    time.sleep(hold_seconds)
    send_input(mouse_input(MOUSEEVENTF_LEFTUP))


def keyboard_vk_input(vk_code: int, flags: int) -> Input:
    scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)
    return Input(
        type=INPUT_KEYBOARD,
        union=InputUnion(
            ki=KeyboardInput(
                wVk=vk_code,
                wScan=scan_code,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def is_shift_registered() -> bool:
    return is_pressed(VK_SHIFT) or is_pressed(VK_LEFT_SHIFT)


def shift_down() -> None:
    for _ in range(SHIFT_PRESS_ATTEMPTS):
        send_input(
            keyboard_vk_input(VK_LEFT_SHIFT, 0),
            keyboard_input(LEFT_SHIFT_SCANCODE, 0),
        )
        time.sleep(SHIFT_STATE_SETTLE_SECONDS)
        if is_shift_registered():
            return

        win32api.keybd_event(VK_LEFT_SHIFT, LEFT_SHIFT_SCANCODE, 0, 0)
        time.sleep(SHIFT_STATE_SETTLE_SECONDS)
        if is_shift_registered():
            return

    print("shift key did not register as pressed.")


def shift_up() -> None:
    for _ in range(SHIFT_PRESS_ATTEMPTS):
        send_input(
            keyboard_input(LEFT_SHIFT_SCANCODE, win32con.KEYEVENTF_KEYUP),
            keyboard_vk_input(VK_LEFT_SHIFT, win32con.KEYEVENTF_KEYUP),
        )
        time.sleep(SHIFT_STATE_SETTLE_SECONDS)
        if not is_shift_registered():
            return

        win32api.keybd_event(VK_LEFT_SHIFT, LEFT_SHIFT_SCANCODE, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(SHIFT_STATE_SETTLE_SECONDS)
        if not is_shift_registered():
            return

    print("shift key did not register as released.")


def wait_for_ctrl_release(timeout_seconds: float = MODIFIER_RELEASE_TIMEOUT_SECONDS) -> None:
    # Ctrl+C is a trigger key: if Ctrl is still physically held, the game
    # sees Ctrl+Shift+click instead of Shift+click.
    deadline = time.monotonic() + timeout_seconds
    while is_pressed(VK_CONTROL) and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)


def shift_click(position: Tuple[int, int]) -> bool:
    wait_for_ctrl_release()
    move_cursor(position)
    time.sleep(SHIFT_MOVE_DELAY_SECONDS)
    shift_down()
    all_registered = True
    try:
        time.sleep(SHIFT_BEFORE_CLICK_DELAY_SECONDS)
        for _ in range(SHIFT_CLICK_COUNT):
            # Shift can drop between clicks (game steals focus, OS coalesces
            # key state, etc.). Re-press it before any click that would
            # otherwise land as an unmodified click and move a single item.
            if not is_shift_registered():
                print("shift dropped mid-click, re-pressing before click.")
                shift_down()
                if not is_shift_registered():
                    all_registered = False
                    # GetAsyncKeyState only confirms the OS accepted our own
                    # injected key state, not that the game saw it - but if
                    # it can't even confirm that much, firing the click
                    # unmodified would silently move a single item instead
                    # of nothing, which is worse for the caller's retry logic.
                    print("shift still unregistered, skipping this click.")
                    continue
            send_left_click(hold_seconds=SHIFT_CLICK_HOLD_SECONDS)
            time.sleep(SHIFT_CLICK_GAP_SECONDS)
        time.sleep(SHIFT_AFTER_CLICK_HOLD_SECONDS)
    finally:
        shift_up()
        time.sleep(SHIFT_RELEASE_DELAY_SECONDS)

    if not all_registered:
        print("shift-click finished without confirmed shift on every click.")
    return all_registered


def shift_click_journal(
    position: Tuple[int, int],
    journal_template: MaskedTemplate,
    attempts: int = SHIFT_CLICK_RETRY_ATTEMPTS,
) -> bool:
    # shift_click()'s own return value only confirms the OS accepted our
    # injected shift key state - it can't confirm the game actually acted on
    # it, so it's true almost every time and can't drive a retry. The only
    # reliable signal is the game's own state: is the journal actually gone
    # from this slot after we clicked it. Retry against that instead.
    for attempt in range(attempts):
        shift_click(position)
        time.sleep(SHIFT_CLICK_VERIFY_DELAY_SECONDS)
        if not journal_at_position(journal_template, position):
            return True
        if attempt < attempts - 1:
            print(f"journal still present after shift-click, retrying ({attempt + 2}/{attempts}).")

    print("journal still present after all shift-click retries; giving up.")
    return False


def find_template_until(
    template: np.ndarray,
    timeout_seconds: float,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> Optional[Tuple[int, int]]:
    match = find_template_until_with_monitor(
        template,
        timeout_seconds,
        preferred_monitor_index,
        hint,
    )
    if match is None:
        return None
    position, _monitor_index = match
    return position


def find_template_until_with_monitor(
    template: np.ndarray,
    timeout_seconds: float,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> Optional[Tuple[Tuple[int, int], int]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        match = find_template_on_screen_with_monitor(
            template,
            preferred_monitor_index=preferred_monitor_index,
            hint=hint,
        )
        if match is not None:
            return match
        time.sleep(POLL_SECONDS)
    return None


def is_pressed(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def is_laborer_action_pressed() -> bool:
    return (
        is_pressed(VK_1)
        or is_pressed(VK_NUMPAD1)
        or (is_pressed(VK_CONTROL) and is_pressed(VK_C))
    )


def ignore_console_ctrl_c() -> None:
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)


def check_ready(
    ready_template: np.ndarray,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> Optional[int]:
    ready_match = find_template_until_with_monitor(
        ready_template,
        READY_TIMEOUT_SECONDS,
        preferred_monitor_index,
        hint,
    )
    if ready_match is None:
        print("laborer is not ready yet.")
        return None
    _ready_position, monitor_index = ready_match
    print(f"laborer is ready on monitor {monitor_index}.")
    return monitor_index


def is_stop_laborer(laborer_name: str) -> bool:
    return laborer_name.startswith(STOP_LABORER_PREFIX)


def play_stop_laborer_beep(laborer_name: str) -> None:
    print(f"stop laborer found: {laborer_name}. Beeping and skipping action.")
    winsound.Beep(STOP_LABORER_BEEP_FREQUENCY, STOP_LABORER_BEEP_DURATION_MS)


def load_laborer_templates(mapping: ImageMapping) -> TemplateMapping:
    templates: TemplateMapping = {}
    for laborer_name, paths in mapping.items():
        templates[laborer_name] = load_masked_template(paths["laborer"])

    for laborer_path in sorted(LABORER_ASSETS_DIR.glob(f"{STOP_LABORER_PREFIX}*.png")):
        laborer_name = laborer_path.stem
        if laborer_name not in templates:
            templates[laborer_name] = load_masked_template(laborer_path)

    return templates


def filter_laborer_templates_by_inventory(
    laborer_templates: TemplateMapping,
    mapping: ImageMapping,
    journal_positions: JournalPositionCache,
) -> TemplateMapping:
    # Only journals actually present in the inventory (as of the last '2' scan)
    # are real candidates, so narrowing the search to those tiers rules out
    # look-alike tiers (e.g. t2 vs t5 of the same profession) you don't even
    # have a journal for. Stop laborers are exempt: they don't need a journal.
    filtered: TemplateMapping = {}
    for laborer_name, template in laborer_templates.items():
        if is_stop_laborer(laborer_name):
            filtered[laborer_name] = template
            continue
        paths = mapping.get(laborer_name)
        if paths is not None and paths["journal"].stem in journal_positions:
            filtered[laborer_name] = template
    return filtered


def load_journal_templates(mapping: ImageMapping) -> TemplateMapping:
    templates: TemplateMapping = {}
    for laborer_name, paths in mapping.items():
        templates[laborer_name] = load_masked_template(paths["journal"])
    return templates


def load_journal_templates_by_name(mapping: ImageMapping) -> TemplateMapping:
    templates: TemplateMapping = {}
    for paths in mapping.values():
        journal_path = paths["journal"]
        journal_name = journal_path.stem
        if journal_name not in templates:
            templates[journal_name] = load_masked_template(journal_path)
    return templates


def masked_template_match(
    source: np.ndarray,
    template: MaskedTemplate,
) -> Tuple[float, Tuple[int, int]]:
    template_height, template_width = template.image.shape[:2]
    source_height, source_width = source.shape[:2]
    if source_height < template_height or source_width < template_width:
        return 0.0, (0, 0)

    # TM_SQDIFF_NORMED instead of TM_CCORR_NORMED: CCORR is not mean-subtracted,
    # so bright uniform areas (panel edges, screen corners) score ~1.0 and win
    # over the real sprite. SQDIFF measures actual pixel difference.
    result = cv2.matchTemplate(
        source,
        template.image,
        cv2.TM_SQDIFF_NORMED,
        mask=template.mask,
    )
    result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
    min_val, _, min_loc, _ = cv2.minMaxLoc(result)
    return 1.0 - float(min_val), min_loc


def masked_fixed_region_score(
    source: np.ndarray,
    template: MaskedTemplate,
    max_loc: Tuple[int, int],
    region: Tuple[int, int, int, int],
) -> float:
    region_x, region_y, region_width, region_height = region
    source_x = max_loc[0] + region_x
    source_y = max_loc[1] + region_y
    source_crop = source[
        source_y:source_y + region_height,
        source_x:source_x + region_width,
    ]
    template_crop = template.image[
        region_y:region_y + region_height,
        region_x:region_x + region_width,
    ]
    if source_crop.shape[:2] != template_crop.shape[:2]:
        return 0.0

    mask_crop = None
    if template.mask is not None:
        mask_crop = template.mask[
            region_y:region_y + region_height,
            region_x:region_x + region_width,
        ]

    result = cv2.matchTemplate(
        source_crop,
        template_crop,
        cv2.TM_CCOEFF_NORMED,
        mask=mask_crop,
    )
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def is_verified_laborer_match(
    source: np.ndarray,
    laborer_name: str,
    template: MaskedTemplate,
    max_loc: Tuple[int, int],
) -> bool:
    # Different tiers of the same laborer share almost identical art aside from
    # the small tier badge in the corner, so an overall score above threshold
    # can still be the wrong tier (e.g. t2 matching a t5 icon). Verify the
    # fixed badge region separately against this candidate's own template.
    badge_score = masked_fixed_region_score(
        source,
        template,
        max_loc,
        LABORER_BADGE_REGION,
    )
    if badge_score >= LABORER_BADGE_THRESHOLD:
        return True

    print(f"rejected {laborer_name} candidate: tier badge score {badge_score:.3f}.")
    return False


BadgeColor = Tuple[float, float, float]
LaborerBadgeColors = Dict[str, BadgeColor]


def compute_badge_reference_color(
    path: Path,
    region: Tuple[int, int, int, int] = LABORER_BADGE_REGION,
) -> Optional[BadgeColor]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    region_x, region_y, region_width, region_height = region
    crop = image[region_y:region_y + region_height, region_x:region_x + region_width]
    if crop.shape[0] != region_height or crop.shape[1] != region_width:
        return None
    return (
        float(np.median(crop[:, :, 0])),
        float(np.median(crop[:, :, 1])),
        float(np.median(crop[:, :, 2])),
    )


def load_laborer_badge_colors(mapping: ImageMapping) -> LaborerBadgeColors:
    # The tier badge's color (e.g. t3 green vs t4 blue) is far more
    # discriminative than its grayscale shape - same-profession tiers share
    # almost identical art and their badge shapes correlate deceptively well
    # in grayscale (see is_verified_laborer_match), but their colors are
    # reliably tens of BGR units apart.
    colors: LaborerBadgeColors = {}
    for laborer_name, paths in mapping.items():
        color = compute_badge_reference_color(paths["laborer"])
        if color is not None:
            colors[laborer_name] = color

    for laborer_path in sorted(LABORER_ASSETS_DIR.glob(f"{STOP_LABORER_PREFIX}*.png")):
        laborer_name = laborer_path.stem
        if laborer_name not in colors:
            color = compute_badge_reference_color(laborer_path)
            if color is not None:
                colors[laborer_name] = color

    return colors


def masked_region_median_color(
    color_source: np.ndarray,
    top_left: Tuple[int, int],
    region: Tuple[int, int, int, int],
) -> Optional[BadgeColor]:
    region_x, region_y, region_width, region_height = region
    x = top_left[0] + region_x
    y = top_left[1] + region_y
    crop = color_source[y:y + region_height, x:x + region_width]
    if crop.shape[0] != region_height or crop.shape[1] != region_width:
        return None
    return (
        float(np.median(crop[:, :, 0])),
        float(np.median(crop[:, :, 1])),
        float(np.median(crop[:, :, 2])),
    )


def badge_color_distance(a: BadgeColor, b: BadgeColor) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def scan_inventory_journal_positions(journal_templates: TemplateMapping) -> JournalPositionCache:
    captures = capture_inventory_regions()
    positions: JournalPositionCache = {}

    for journal_name, journal_template in journal_templates.items():
        best_match: Optional[Tuple[int, Tuple[int, int], Tuple[int, int], float]] = None

        for monitor_index, source, origin in captures:
            score, max_loc = masked_template_match(source, journal_template)
            if best_match is None or score > best_match[3]:
                best_match = (monitor_index, origin, max_loc, score)

        if best_match is None or best_match[3] < JOURNAL_MATCH_THRESHOLD:
            print(f"journal position not found for {journal_name}.")
            continue

        monitor_index, origin, max_loc, score = best_match
        height, width = journal_template.image.shape[:2]
        position = (
            origin[0] + max_loc[0] + width // 2,
            origin[1] + max_loc[1] + height // 2,
        )
        positions[journal_name] = position
        print(f"cached {journal_name} at {position} on monitor {monitor_index} ({score:.3f}).")

    return positions


def ordered_template_names(
    templates: TemplateMapping,
    preferred_name: Optional[str] = None,
):
    yielded: set[str] = set()

    for template_name in templates:
        if is_stop_laborer(template_name):
            yielded.add(template_name)
            yield template_name

    if preferred_name in templates and preferred_name not in yielded:
        yielded.add(preferred_name)
        yield preferred_name

    for template_name in templates:
        if template_name not in yielded:
            yield template_name


def laborer_profession_suffix(laborer_name: str) -> str:
    _tier, _sep, suffix = laborer_name.partition("-")
    return suffix


def find_laborer_in_source(
    laborer_templates: TemplateMapping,
    source: np.ndarray,
    origin: Tuple[int, int],
    monitor_index: int,
    preferred_laborer_name: Optional[str] = None,
    color_source: Optional[np.ndarray] = None,
    badge_colors: Optional[LaborerBadgeColors] = None,
) -> Optional[LaborerMatch]:
    # Different tiers of the same profession (e.g. t2-bs vs t5-bs) share almost
    # identical art - frame, background, weapon - aside from the small tier
    # badge and the portrait. TM_SQDIFF_NORMED over the whole image can score
    # the wrong tier above threshold because the shared frame dominates the
    # pixel count, and even the badge's grayscale *shape* can correlate
    # deceptively well between two different tier numerals (e.g. III vs IV
    # can score ~0.86, above the 0.85 gate). The badge's *color* is what
    # actually reliably differs between tiers, so when a color capture is
    # available it settles same-profession ties instead of the shape score.
    candidates: Dict[str, Tuple[float, Tuple[int, int], Tuple[int, int]]] = {}
    best_stop_match: Optional[LaborerMatch] = None
    best_overall_score = 0.0

    for laborer_name in ordered_template_names(laborer_templates, preferred_laborer_name):
        template = laborer_templates[laborer_name]
        score, max_loc = masked_template_match(source, template)
        if LOG_LABORER_SCORES:
            print(f"laborer score {laborer_name} on monitor {monitor_index}: {score:.3f}")
        if score < LABORER_MATCH_THRESHOLD:
            continue

        height, width = template.image.shape[:2]
        position = (
            origin[0] + max_loc[0] + width // 2,
            origin[1] + max_loc[1] + height // 2,
        )

        if is_stop_laborer(laborer_name):
            if not is_verified_laborer_match(source, laborer_name, template, max_loc):
                continue
            if best_stop_match is None or score > best_stop_match[1]:
                best_stop_match = (laborer_name, score, monitor_index, position)
            continue

        candidates[laborer_name] = (score, max_loc, position)
        best_overall_score = max(best_overall_score, score)

    if best_stop_match is not None and (not candidates or best_stop_match[1] >= best_overall_score):
        return best_stop_match

    if not candidates:
        return None

    top_name = max(candidates, key=lambda name: candidates[name][0])

    if color_source is None or badge_colors is None:
        # No color capture available (e.g. called without the color-aware
        # capture path): fall back to the less reliable grayscale badge shape
        # check as a best-effort verification of the top overall match.
        score, max_loc, position = candidates[top_name]
        template = laborer_templates[top_name]
        if not is_verified_laborer_match(source, top_name, template, max_loc):
            return None
        return top_name, score, monitor_index, position

    top_suffix = laborer_profession_suffix(top_name)
    sibling_names = [
        name for name in candidates if laborer_profession_suffix(name) == top_suffix
    ]

    best_name: Optional[str] = None
    best_distance: Optional[float] = None
    for name in sibling_names:
        reference_color = badge_colors.get(name)
        if reference_color is None:
            continue
        _score, max_loc, _position = candidates[name]
        observed_color = masked_region_median_color(color_source, max_loc, LABORER_BADGE_REGION)
        if observed_color is None:
            continue
        distance = badge_color_distance(reference_color, observed_color)
        if LOG_LABORER_SCORES:
            print(f"laborer badge color distance {name} on monitor {monitor_index}: {distance:.1f}")
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = name

    if best_name is None or best_distance is None or best_distance > LABORER_BADGE_COLOR_MAX_DISTANCE:
        print(f"rejected {top_name} candidate: tier badge color distance {best_distance}.")
        return None

    best_score, _max_loc, best_position = candidates[best_name]
    return best_name, best_score, monitor_index, best_position


def remember_laborer_match(
    hint: Optional[LaborerSearchHint],
    match: Optional[LaborerMatch],
) -> None:
    if hint is None or match is None:
        return

    laborer_name, _score, monitor_index, position = match
    hint.laborer_name = laborer_name
    hint.monitor_index = monitor_index
    hint.position = position


def find_laborer_from_hint(
    laborer_templates: TemplateMapping,
    hint: Optional[LaborerSearchHint],
    badge_colors: Optional[LaborerBadgeColors] = None,
) -> Optional[LaborerMatch]:
    if hint is None or hint.position is None or hint.monitor_index is None:
        return None
    if hint.laborer_name not in laborer_templates:
        return None

    hint_template = laborer_templates[hint.laborer_name].image
    with mss.mss() as screen_capture:
        if hint.monitor_index <= 0 or hint.monitor_index >= len(screen_capture.monitors):
            return None
        region = template_hint_region(
            screen_capture.monitors[hint.monitor_index],
            hint.position,
            hint_template,
            padding=LABORER_HINT_PADDING,
        )
        if region is None:
            return None
        source, color_source, origin = capture_source_with_color(screen_capture, region)

    match = find_laborer_in_source(
        laborer_templates,
        source,
        origin,
        hint.monitor_index,
        hint.laborer_name,
        color_source,
        badge_colors,
    )
    if match is not None:
        remember_laborer_match(hint, match)
        print(f"laborer matched near last position ({match[1]:.3f}).")
    return match


def find_laborer(
    laborer_templates: TemplateMapping,
    monitor_index: int,
    preferred_laborer_name: Optional[str] = None,
    hint: Optional[LaborerSearchHint] = None,
    badge_colors: Optional[LaborerBadgeColors] = None,
) -> Optional[LaborerMatch]:
    if hint is not None and hint.monitor_index == monitor_index:
        hint_match = find_laborer_from_hint(laborer_templates, hint, badge_colors)
        if hint_match is not None:
            return hint_match

    source, color_source, origin = capture_screen_by_index_with_color(monitor_index)
    match = find_laborer_in_source(
        laborer_templates,
        source,
        origin,
        monitor_index,
        preferred_laborer_name,
        color_source,
        badge_colors,
    )
    remember_laborer_match(hint, match)
    return match


def find_laborer_on_screens(
    laborer_templates: TemplateMapping,
    preferred_laborer_name: Optional[str] = None,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[LaborerSearchHint] = None,
    badge_colors: Optional[LaborerBadgeColors] = None,
) -> Optional[LaborerMatch]:
    hint_match = find_laborer_from_hint(laborer_templates, hint, badge_colors)
    if hint_match is not None:
        return hint_match

    best_match: Optional[LaborerMatch] = None

    for monitor_index, source, color_source, origin in iter_captured_screens_with_color(preferred_monitor_index):
        laborer_match = find_laborer_in_source(
            laborer_templates,
            source,
            origin,
            monitor_index,
            preferred_laborer_name,
            color_source,
            badge_colors,
        )
        if laborer_match is None:
            continue

        if (
            laborer_match[0] == preferred_laborer_name
            and (preferred_monitor_index is None or monitor_index == preferred_monitor_index)
        ):
            remember_laborer_match(hint, laborer_match)
            return laborer_match

        if best_match is None or laborer_match[1] > best_match[1]:
            best_match = laborer_match

    remember_laborer_match(hint, best_match)
    return best_match


def find_journal(
    journal_template: MaskedTemplate,
    monitor_index: int,
    match_threshold: float = JOURNAL_MATCH_THRESHOLD,
) -> Optional[Tuple[Tuple[int, int], float]]:
    source, origin = capture_screen_by_index(monitor_index)
    score, max_loc = masked_template_match(source, journal_template)
    if score < match_threshold:
        return None

    height, width = journal_template.image.shape[:2]
    position = (
        origin[0] + max_loc[0] + width // 2,
        origin[1] + max_loc[1] + height // 2,
    )
    print(f"journal matched on monitor {monitor_index} ({score:.3f}).")
    return position, score


def check_laborer(
    laborer_templates: TemplateMapping,
    monitor_index: int,
    preferred_laborer_name: Optional[str] = None,
    hint: Optional[LaborerSearchHint] = None,
    badge_colors: Optional[LaborerBadgeColors] = None,
) -> Optional[Tuple[str, int]]:
    laborer_match = find_laborer(laborer_templates, monitor_index, preferred_laborer_name, hint, badge_colors)
    if laborer_match is None:
        print("laborer image not found.")
        return None

    laborer_name, score, monitor_index, _position = laborer_match
    print(f"laborer image found: {laborer_name} on monitor {monitor_index} ({score:.3f})")
    return laborer_name, monitor_index


def get_prefetched_laborer(
    laborer_future: Future[Optional[LaborerMatch]],
    monitor_index: int,
) -> Optional[Tuple[str, int]]:
    try:
        laborer_match = laborer_future.result(timeout=PREFETCH_LABORER_WAIT_SECONDS)
    except FutureTimeoutError:
        print("prefetched laborer result not ready yet.")
        return None
    except Exception as error:
        print(f"prefetched laborer check failed: {error}")
        return None

    if laborer_match is None:
        print("prefetched laborer image not found.")
        return None

    laborer_name, score, matched_monitor_index, _position = laborer_match
    if matched_monitor_index != monitor_index:
        print(
            "prefetched laborer monitor mismatch "
            f"(expected {monitor_index}, got {matched_monitor_index})."
        )
        return None

    print(f"laborer image found from prefetch: {laborer_name} on monitor {monitor_index} ({score:.3f})")
    return laborer_name, monitor_index


def journal_at_position(
    journal_template: MaskedTemplate,
    position: Tuple[int, int],
) -> bool:
    with mss.mss() as screen_capture:
        for monitor_index in ordered_monitor_indexes(screen_capture):
            region = template_hint_region(
                screen_capture.monitors[monitor_index],
                position,
                journal_template.image,
                padding=JOURNAL_CACHE_VERIFY_PADDING,
            )
            if region is None:
                continue
            source, _origin = capture_source(screen_capture, region)
            score, _loc = masked_template_match(source, journal_template)
            if score >= JOURNAL_MATCH_THRESHOLD:
                return True
    return False


def select_journal(
    laborer_name: str,
    monitor_index: int,
    mapping: ImageMapping,
    journal_templates: TemplateMapping,
    journal_positions: JournalPositionCache,
    journal_positions_path: Optional[Path] = None,
) -> Optional[Tuple[int, int]]:
    journal_name = mapping[laborer_name]["journal"].stem
    journal_template = journal_templates[laborer_name]
    cached_position = journal_positions.get(journal_name)
    if cached_position is not None:
        if journal_at_position(journal_template, cached_position):
            print(f"using cached journal position for {journal_name}: {cached_position}.")
            shift_click_journal(cached_position, journal_template)
            print(f"shift-clicked cached journal for {laborer_name}.")
            return cached_position

        print(f"cached journal position for {journal_name} is stale, rescanning.")
        del journal_positions[journal_name]
        if journal_positions_path is not None:
            save_journal_positions(journal_positions_path, journal_positions)

    journal_match = find_journal(journal_template, monitor_index)
    if journal_match is None:
        print(f"journal image not found for {laborer_name} ({journal_name}).")
        return None

    journal_position, score = journal_match
    print(f"journal position for {laborer_name} ({journal_name}): {journal_position}.")
    journal_positions[journal_name] = journal_position
    if journal_positions_path is not None:
        save_journal_positions(journal_positions_path, journal_positions)
        print(f"cached journal position for {journal_name}.")
    shift_click_journal(journal_position, journal_template)
    print(f"shift-clicked journal twice for {laborer_name} ({score:.3f}).")
    return journal_position


def click_accept(
    accept_template: np.ndarray,
    preferred_monitor_index: Optional[int] = None,
    hint: Optional[TemplateSearchHint] = None,
) -> bool:
    accept_target = find_template_until(
        accept_template,
        ACCEPT_TIMEOUT_SECONDS,
        preferred_monitor_index,
        hint,
    )
    if accept_target is None:
        print("accept image not found.")
        return False

    click(accept_target)
    print(f"clicked accept at {accept_target}.")
    return True


def restore_mouse(position: Tuple[int, int]) -> None:
    win32api.SetCursorPos(position)
    print(f"restored mouse position to {position}.")


def main() -> None:
    ignore_console_ctrl_c()
    mapping = load_mapping(MAPPING_FILE)
    take_all_template = load_template(TAKE_ALL_IMAGE)
    advance_tier_template = load_template(ADVANCE_TIER_IMAGE)
    ready_template = load_template(READY_IMAGE)
    accept_template = load_template(ACCEPT_IMAGE)
    laborer_templates = load_laborer_templates(mapping)
    laborer_badge_colors = load_laborer_badge_colors(mapping)
    journal_templates = load_journal_templates(mapping)
    journal_templates_by_name = load_journal_templates_by_name(mapping)
    journal_positions = load_journal_positions(JOURNAL_POSITIONS_FILE)
    print("Listening for '&' / '1' / Ctrl+C. Press '2' to scan journal positions. Press superscript-2 to stop.")
    print("'1' / Ctrl+C is blocked until at least one journal position is cached via '2'.")
    print(f"Loaded {len(mapping)} laborer mapping(s).")
    print(f"Loaded {len(journal_positions)} cached journal position(s).")

    laborer_executor = ThreadPoolExecutor(max_workers=1)
    take_all_hint = TemplateSearchHint()
    advance_tier_hint = TemplateSearchHint()
    ready_hint = TemplateSearchHint()
    accept_hint = TemplateSearchHint()
    laborer_hint = LaborerSearchHint()
    last_ready_monitor_index: Optional[int] = None
    last_laborer_name: Optional[str] = None
    last_action = 0.0
    try:
        while True:
            if is_pressed(VK_SUPERSCRIPT_TWO):
                print("Stopped.")
                return

            if is_pressed(VK_2) or is_pressed(VK_NUMPAD2):
                now = time.monotonic()
                if now - last_action >= DEBOUNCE_SECONDS:
                    print()
                    print("=" * 48)
                    print("journal position scan started.")
                    journal_positions = scan_inventory_journal_positions(journal_templates_by_name)
                    save_journal_positions(JOURNAL_POSITIONS_FILE, journal_positions)
                    print(f"saved {len(journal_positions)} journal position(s) to {JOURNAL_POSITIONS_FILE}.")
                    last_action = now

            elif is_laborer_action_pressed():
                now = time.monotonic()
                if now - last_action >= DEBOUNCE_SECONDS:
                    if not journal_positions:
                        print("no journal positions cached yet; press '2' to scan the inventory first.")
                        last_action = now
                        time.sleep(POLL_SECONDS)
                        continue

                    original_mouse_position = win32api.GetCursorPos()
                    laborer_future: Optional[Future[Optional[LaborerMatch]]] = None
                    print()
                    print("=" * 48)
                    print("laborer action started.")
                    try:
                        # Restrict candidates to laborers whose journal was actually
                        # found in the inventory, so look-alike tiers we don't hold a
                        # journal for can't be mistaken for the one that's ready.
                        active_laborer_templates = filter_laborer_templates_by_inventory(
                            laborer_templates,
                            mapping,
                            journal_positions,
                        )
                        preferred_monitor_index = last_ready_monitor_index
                        laborer_future = laborer_executor.submit(
                            find_laborer_on_screens,
                            active_laborer_templates,
                            last_laborer_name,
                            preferred_monitor_index,
                            laborer_hint,
                            laborer_badge_colors,
                        )
                        print("started laborer prefetch.")

                        take_all_match = find_template_on_screen_with_monitor(
                            take_all_template,
                            preferred_monitor_index=preferred_monitor_index,
                            hint=take_all_hint,
                        )
                        action_monitor_index = preferred_monitor_index
                        tier_advanced = False
                        if take_all_match is None:
                            print("take-all image not found.")
                            time.sleep(TAKE_ALL_NOT_FOUND_DELAY_SECONDS)
                        else:
                            take_all_target, action_monitor_index = take_all_match
                            click(take_all_target)
                            print(f"clicked take-all at {take_all_target}.")

                            advance_tier_match = find_template_until_with_monitor(
                                advance_tier_template,
                                ADVANCE_TIER_TIMEOUT_SECONDS,
                                action_monitor_index,
                                advance_tier_hint,
                            )
                            if advance_tier_match is None:
                                print("advance-tier image not found.")
                            else:
                                advance_tier_target, action_monitor_index = advance_tier_match
                                click(advance_tier_target)
                                print(f"clicked advance-tier at {advance_tier_target}.")
                                tier_advanced = True
                                time.sleep(ADVANCE_TIER_RESTORE_DELAY_SECONDS)
                                restore_mouse(original_mouse_position)
                                click(original_mouse_position)
                                print("returned to the original cursor position and clicked there.")

                        ready_monitor_index = check_ready(
                            ready_template,
                            action_monitor_index,
                            ready_hint,
                        )
                        if ready_monitor_index is not None:
                            last_ready_monitor_index = ready_monitor_index
                            laborer_match = None
                            if tier_advanced:
                                # advance-tier changes the laborer's sprite/tier, so the
                                # prefetch (started before the click) reflects the old tier
                                # and must be discarded in favor of a fresh, post-click scan.
                                print("tier advanced, discarding prefetch and rechecking laborer image.")
                                if laborer_future is not None and not laborer_future.done():
                                    laborer_future.cancel()
                            else:
                                laborer_match = get_prefetched_laborer(
                                    laborer_future,
                                    ready_monitor_index,
                                )
                            if laborer_match is None:
                                laborer_match = check_laborer(
                                    active_laborer_templates,
                                    ready_monitor_index,
                                    last_laborer_name,
                                    laborer_hint,
                                    laborer_badge_colors,
                                )

                            if laborer_match is not None:
                                laborer_name, monitor_index = laborer_match
                                last_laborer_name = laborer_name
                                if is_stop_laborer(laborer_name):
                                    play_stop_laborer_beep(laborer_name)
                                    continue

                                journal_position = select_journal(
                                    laborer_name,
                                    monitor_index,
                                    mapping,
                                    journal_templates,
                                    journal_positions,
                                    JOURNAL_POSITIONS_FILE,
                                )
                                if journal_position is not None:
                                    for attempt in range(ACCEPT_RETRY_ATTEMPTS + 1):
                                        if click_accept(accept_template, monitor_index, accept_hint):
                                            break
                                        if attempt < ACCEPT_RETRY_ATTEMPTS:
                                            # If the journal is gone from this position, the
                                            # first shift-click likely already took it and the
                                            # accept prompt was just slow/missed. Re-clicking
                                            # blindly here would hit whatever slot the
                                            # inventory reflowed into, i.e. the wrong item.
                                            if not journal_at_position(
                                                journal_templates[laborer_name],
                                                journal_position,
                                            ):
                                                print(
                                                    "journal no longer at cached position; "
                                                    "skipping shift-click retry to avoid the wrong slot."
                                                )
                                                break
                                            print("accept timed out, retrying shift click.")
                                            shift_click(journal_position)
                    finally:
                        if laborer_future is not None and not laborer_future.done():
                            laborer_future.cancel()
                        time.sleep(RESTORE_MOUSE_DELAY_SECONDS)
                        restore_mouse(original_mouse_position)
                        last_action = now

            time.sleep(POLL_SECONDS)
    finally:
        laborer_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
