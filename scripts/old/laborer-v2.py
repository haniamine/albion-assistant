from __future__ import annotations

import ctypes
import json
import time
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

MATCH_THRESHOLD = 0.80
LABORER_MATCH_THRESHOLD = 0.97
JOURNAL_MATCH_THRESHOLD = 0.90
CAPTURE_MONITOR_INDEX: Optional[int] = None
INVENTORY_RIGHT_START_RATIO = 0.50
POLL_SECONDS = 0.01
DEBOUNCE_SECONDS = 0.20
ADVANCE_TIER_TIMEOUT_SECONDS = 0.50
ADVANCE_TIER_RESTORE_DELAY_SECONDS = 0.30
READY_TIMEOUT_SECONDS = 0.50
ACCEPT_TIMEOUT_SECONDS = 1.00
PREFETCH_LABORER_WAIT_SECONDS = 0.05
TAKE_ALL_NOT_FOUND_DELAY_SECONDS = 0.50
SHIFT_AFTER_CLICK_HOLD_SECONDS = 0.25
RESTORE_MOUSE_DELAY_SECONDS = 0.30
LOG_LABORER_SCORES = False

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
LaborerMatch = Tuple[str, float, int]

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


def capture_monitor(monitor: dict[str, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    with mss.mss() as screen_capture:
        screenshot = np.array(screen_capture.grab(monitor))
    source = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    return source, (monitor["left"], monitor["top"])


def capture_screens() -> list[Tuple[int, np.ndarray, Tuple[int, int]]]:
    with mss.mss() as screen_capture:
        if CAPTURE_MONITOR_INDEX is None:
            monitor_indexes = range(1, len(screen_capture.monitors))
        else:
            monitor_indexes = [CAPTURE_MONITOR_INDEX]

        captures = []
        for monitor_index in monitor_indexes:
            monitor = screen_capture.monitors[monitor_index]
            screenshot = np.array(screen_capture.grab(monitor))
            source = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            captures.append((monitor_index, source, (monitor["left"], monitor["top"])))
    return captures


def capture_screen_by_index(monitor_index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    with mss.mss() as screen_capture:
        monitor = screen_capture.monitors[monitor_index]
        screenshot = np.array(screen_capture.grab(monitor))
    source = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    return source, (monitor["left"], monitor["top"])


def capture_inventory_regions() -> list[Tuple[int, np.ndarray, Tuple[int, int]]]:
    with mss.mss() as screen_capture:
        if CAPTURE_MONITOR_INDEX is None:
            monitor_indexes = range(1, len(screen_capture.monitors))
        else:
            monitor_indexes = [CAPTURE_MONITOR_INDEX]

        captures = []
        for monitor_index in monitor_indexes:
            monitor = screen_capture.monitors[monitor_index]
            left_offset = int(monitor["width"] * INVENTORY_RIGHT_START_RATIO)
            region = {
                "left": monitor["left"] + left_offset,
                "top": monitor["top"],
                "width": monitor["width"] - left_offset,
                "height": monitor["height"],
            }
            screenshot = np.array(screen_capture.grab(region))
            source = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
            captures.append((monitor_index, source, (region["left"], region["top"])))
    return captures


def find_template_on_screen(
    template: np.ndarray,
    match_threshold: float = MATCH_THRESHOLD,
) -> Optional[Tuple[int, int]]:
    match = find_template_on_screen_with_monitor(template, match_threshold)
    if match is None:
        return None
    position, _monitor_index = match
    return position


def find_template_on_screen_with_monitor(
    template: np.ndarray,
    match_threshold: float = MATCH_THRESHOLD,
) -> Optional[Tuple[Tuple[int, int], int]]:
    best_match: Optional[Tuple[int, Tuple[int, int], Tuple[int, int], float]] = None
    for monitor_index, source, origin in capture_screens():
        result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best_match is None or max_val > best_match[3]:
            best_match = (monitor_index, origin, max_loc, max_val)

    if best_match is None or best_match[3] < match_threshold:
        return None

    monitor_index, origin, max_loc, max_val = best_match
    height, width = template.shape[:2]
    position = (origin[0] + max_loc[0] + width // 2, origin[1] + max_loc[1] + height // 2)
    print(f"matched on monitor {monitor_index} ({max_val:.3f}).")
    return position, monitor_index


def click(position: Tuple[int, int]) -> None:
    x, y = map(int, position)
    win32api.SetCursorPos((x, y))
    time.sleep(0.01)

    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


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


def triple_click(position: Tuple[int, int]) -> None:
    x, y = map(int, position)
    win32api.SetCursorPos((x, y))
    time.sleep(0.12)
    for _ in range(3):
        send_left_click()
        time.sleep(0.08)


def shift_down() -> None:
    send_input(
        keyboard_vk_input(VK_SHIFT, 0),
        keyboard_vk_input(VK_LEFT_SHIFT, 0),
        keyboard_input(LEFT_SHIFT_SCANCODE, 0),
    )
    win32api.keybd_event(
        VK_SHIFT,
        LEFT_SHIFT_SCANCODE,
        0,
        0,
    )
    win32api.keybd_event(
        VK_LEFT_SHIFT,
        LEFT_SHIFT_SCANCODE,
        KEYEVENTF_SCANCODE,
        0,
    )


def shift_up() -> None:
    win32api.keybd_event(
        VK_LEFT_SHIFT,
        LEFT_SHIFT_SCANCODE,
        KEYEVENTF_SCANCODE | win32con.KEYEVENTF_KEYUP,
        0,
    )
    win32api.keybd_event(
        VK_SHIFT,
        LEFT_SHIFT_SCANCODE,
        win32con.KEYEVENTF_KEYUP,
        0,
    )
    send_input(
        keyboard_input(LEFT_SHIFT_SCANCODE, win32con.KEYEVENTF_KEYUP),
        keyboard_vk_input(VK_LEFT_SHIFT, win32con.KEYEVENTF_KEYUP),
        keyboard_vk_input(VK_SHIFT, win32con.KEYEVENTF_KEYUP),
    )


def shift_click(position: Tuple[int, int]) -> None:
    x, y = map(int, position)
    win32api.SetCursorPos((x, y))
    time.sleep(0.20)
    shift_down()
    try:
        time.sleep(0.20)
        for _ in range(2):
            send_left_click(hold_seconds=0.15)
            time.sleep(0.15)
        time.sleep(SHIFT_AFTER_CLICK_HOLD_SECONDS)
    finally:
        shift_up()
        time.sleep(0.10)


def find_template_until(template: np.ndarray, timeout_seconds: float) -> Optional[Tuple[int, int]]:
    match = find_template_until_with_monitor(template, timeout_seconds)
    if match is None:
        return None
    position, _monitor_index = match
    return position


def find_template_until_with_monitor(
    template: np.ndarray,
    timeout_seconds: float,
) -> Optional[Tuple[Tuple[int, int], int]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        match = find_template_on_screen_with_monitor(template)
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


def check_ready(ready_template: np.ndarray) -> Optional[int]:
    ready_match = find_template_until_with_monitor(ready_template, READY_TIMEOUT_SECONDS)
    if ready_match is None:
        print("laborer is not ready yet.")
        return None
    _ready_position, monitor_index = ready_match
    print(f"laborer is ready on monitor {monitor_index}.")
    return monitor_index


def load_laborer_templates(mapping: ImageMapping) -> TemplateMapping:
    templates: TemplateMapping = {}
    for laborer_name, paths in mapping.items():
        templates[laborer_name] = load_masked_template(paths["laborer"])
    return templates


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
    result = cv2.matchTemplate(
        source,
        template.image,
        cv2.TM_CCORR_NORMED,
        mask=template.mask,
    )
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc


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


def find_laborer_in_source(
    laborer_templates: TemplateMapping,
    source: np.ndarray,
    monitor_index: int,
) -> Optional[LaborerMatch]:
    best_name: Optional[str] = None
    best_score = 0.0

    for laborer_name, template in laborer_templates.items():
        score, _ = masked_template_match(source, template)
        if LOG_LABORER_SCORES:
            print(f"laborer score {laborer_name} on monitor {monitor_index}: {score:.3f}")
        if score > best_score:
            best_name = laborer_name
            best_score = score

    if best_name is None or best_score < LABORER_MATCH_THRESHOLD:
        return None
    return best_name, best_score, monitor_index


def find_laborer(
    laborer_templates: TemplateMapping,
    monitor_index: int,
) -> Optional[LaborerMatch]:
    source, _origin = capture_screen_by_index(monitor_index)
    return find_laborer_in_source(laborer_templates, source, monitor_index)


def find_laborer_on_screens(laborer_templates: TemplateMapping) -> Optional[LaborerMatch]:
    best_match: Optional[LaborerMatch] = None

    for monitor_index, source, _origin in capture_screens():
        laborer_match = find_laborer_in_source(laborer_templates, source, monitor_index)
        if laborer_match is None:
            continue

        if best_match is None or laborer_match[1] > best_match[1]:
            best_match = laborer_match

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
) -> Optional[Tuple[str, int]]:
    laborer_match = find_laborer(laborer_templates, monitor_index)
    if laborer_match is None:
        print("laborer image not found.")
        return None

    laborer_name, score, monitor_index = laborer_match
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

    laborer_name, score, matched_monitor_index = laborer_match
    if matched_monitor_index != monitor_index:
        print(
            "prefetched laborer monitor mismatch "
            f"(expected {monitor_index}, got {matched_monitor_index})."
        )
        return None

    print(f"laborer image found from prefetch: {laborer_name} on monitor {monitor_index} ({score:.3f})")
    return laborer_name, monitor_index


def select_journal(
    laborer_name: str,
    monitor_index: int,
    mapping: ImageMapping,
    journal_templates: TemplateMapping,
    journal_positions: JournalPositionCache,
) -> bool:
    journal_name = mapping[laborer_name]["journal"].stem
    cached_position = journal_positions.get(journal_name)
    if cached_position is not None:
        print(f"using cached journal position for {journal_name}: {cached_position}.")
        shift_click(cached_position)
        print(f"shift-clicked cached journal for {laborer_name}.")
        return True

    journal_template = journal_templates[laborer_name]
    journal_match = find_journal(journal_template, monitor_index)
    if journal_match is None:
        print(f"journal image not found for {laborer_name} ({journal_name}).")
        return False

    journal_position, score = journal_match
    print(f"journal position for {laborer_name} ({journal_name}): {journal_position}.")
    shift_click(journal_position)
    print(f"shift-clicked journal twice for {laborer_name} ({score:.3f}).")
    return True


def click_accept(accept_template: np.ndarray) -> None:
    accept_target = find_template_until(accept_template, ACCEPT_TIMEOUT_SECONDS)
    if accept_target is None:
        print("accept image not found.")
        return

    click(accept_target)
    print(f"clicked accept at {accept_target}.")


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
    journal_templates = load_journal_templates(mapping)
    journal_templates_by_name = load_journal_templates_by_name(mapping)
    journal_positions = load_journal_positions(JOURNAL_POSITIONS_FILE)
    print("Listening for '&' / '1' / Ctrl+C. Press '2' to scan journal positions. Press superscript-2 to stop.")
    print(f"Loaded {len(mapping)} laborer mapping(s).")
    print(f"Loaded {len(journal_positions)} cached journal position(s).")

    laborer_executor = ThreadPoolExecutor(max_workers=1)
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
                    original_mouse_position = win32api.GetCursorPos()
                    laborer_future: Optional[Future[Optional[LaborerMatch]]] = None
                    print()
                    print("=" * 48)
                    print("laborer action started.")
                    try:
                        laborer_future = laborer_executor.submit(
                            find_laborer_on_screens,
                            laborer_templates,
                        )
                        print("started laborer prefetch.")

                        take_all_target = find_template_on_screen(take_all_template)
                        if take_all_target is None:
                            print("take-all image not found.")
                            time.sleep(TAKE_ALL_NOT_FOUND_DELAY_SECONDS)
                        else:
                            click(take_all_target)
                            print(f"clicked take-all at {take_all_target}.")

                            advance_tier_target = find_template_until(
                                advance_tier_template,
                                ADVANCE_TIER_TIMEOUT_SECONDS,
                            )
                            if advance_tier_target is None:
                                print("advance-tier image not found.")
                            else:
                                click(advance_tier_target)
                                print(f"clicked advance-tier at {advance_tier_target}.")
                                time.sleep(ADVANCE_TIER_RESTORE_DELAY_SECONDS)
                                restore_mouse(original_mouse_position)
                                click(original_mouse_position)
                                print("returned to the original cursor position and clicked there.")

                        ready_monitor_index = check_ready(ready_template)
                        if ready_monitor_index is not None:
                            laborer_match = get_prefetched_laborer(
                                laborer_future,
                                ready_monitor_index,
                            )
                            if laborer_match is None:
                                laborer_match = check_laborer(laborer_templates, ready_monitor_index)

                            if laborer_match is not None:
                                laborer_name, monitor_index = laborer_match
                                if select_journal(
                                    laborer_name,
                                    monitor_index,
                                    mapping,
                                    journal_templates,
                                    journal_positions,
                                ):
                                    click_accept(accept_template)
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
