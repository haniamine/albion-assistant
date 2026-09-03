"""Configuration for laborer-v4.

Everything tunable lives here as a dataclass tree and is mirrored to
``laborer-v4.config.json`` at the repo root. v3 kept ~40 magic constants at
module scope, which meant re-editing the source to tune a threshold; the
config file lets the timings be adjusted between runs without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
LABORER_ASSETS_DIR = ASSETS_DIR / "laborers"
JOURNAL_ASSETS_DIR = ASSETS_DIR / "journals"
MENU_ASSETS_DIR = ASSETS_DIR / "menu"
MAPPING_PATH = ROOT_DIR / "mapping.json"
CONFIG_PATH = ROOT_DIR / "laborer-v4.config.json"
STATE_PATH = ROOT_DIR / "laborer-v4.state.json"
DEBUG_DIR = ROOT_DIR / "debug"

# The dialog elements v4 anchors against. "laborer" is the portrait and is
# handled separately because its template varies per tier.
MENU_ELEMENTS = ("take_all", "advance_tier", "ready", "accept", "settings", "pick_up", "yes")
MENU_ASSET_NAMES = {
    "take_all": "take-all",
    "advance_tier": "advance-tier",
    "ready": "ready",
    "accept": "accept",
    "settings": "settings",
    "pick_up": "pick-up",
    "yes": "yes",
}
LABORER_ELEMENT = "laborer"
ANCHOR_REFERENCE = "ready"

# take-all, advance-tier and accept are the same button art with different text,
# drawn in the same place at different points in the dialog's life. Measured
# cross-scores against each other's clean button:
#
#     take-all     on accept        0.9560      accept    on take-all      0.9546
#     accept       on advance-tier  0.9149      take-all  on advance-tier  0.8884
#     advance-tier on accept        0.7959      advance-tier on take-all   0.7928
#
# No threshold separates those from a real match, so a hit on any of the three
# is re-scored against the other two and dropped unless it is the best fit for
# the pixels actually there. This is what stops "accept" being found on the
# take-all button and clicked - which repeats take-all instead of confirming.
CONFUSABLE_ELEMENTS = {
    "advance_tier": ("take_all", "accept"),
    "take_all": ("accept", "advance_tier"),
    "accept": ("take_all", "advance_tier"),
}

VK_CODES: Dict[str, int] = {
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45, "f": 0x46,
    "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A, "k": 0x4B, "l": 0x4C,
    "m": 0x4D, "n": 0x4E, "o": 0x4F, "p": 0x50, "q": 0x51, "r": 0x52,
    "s": 0x53, "t": 0x54, "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58,
    "y": 0x59, "z": 0x5A,
    "numpad0": 0x60, "numpad1": 0x61, "numpad2": 0x62, "numpad3": 0x63,
    "numpad4": 0x64, "numpad5": 0x65, "numpad6": 0x66, "numpad7": 0x67,
    "numpad8": 0x68, "numpad9": 0x69,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "ctrl": 0x11, "shift": 0x10, "alt": 0x12,
    "esc": 0x1B, "space": 0x20, "tab": 0x09, "enter": 0x0D,
    # 0xC0 is VK_OEM_3: backtick on QWERTY, the superscript-2 key on AZERTY.
    "oem3": 0xC0, "backtick": 0xC0, "superscript2": 0xC0,
}


@dataclass
class WindowConfig:
    """Which window may receive injected input, and where to look for the UI."""

    # v3 polled GetAsyncKeyState globally, so typing '1' in a browser or hitting
    # Ctrl+C to copy text started a full click sequence. Gating on the game
    # window being foreground removes that entire class of accident.
    require_foreground: bool = True
    # The executable that *is* the game. This is the identity check: a title is
    # something any window can claim, and a player's other windows all claim it
    # - the market tool, a Sheets tab, this repo open in an editor. Matched
    # against the lowercased basename, substring.
    process_contains: List[str] = field(default_factory=lambda: ["albion-online.exe"])
    # Tie-break between windows of the same process, and the whole answer when
    # the process name cannot be read (an elevated client). "Albion" alone was
    # matching StatisticsAnalysisTool, whose window is *larger* than the
    # client's - so the picker chose it, and with it the wrong monitor.
    title_contains: List[str] = field(default_factory=lambda: ["Albion Online Client"])
    # None -> derive from the game window; otherwise an mss monitor index (1-based).
    monitor_index: Optional[int] = None
    # Bound every full-screen search to the game window rect instead of the
    # whole monitor. Cheaper, and nothing outside the game can ever match.
    restrict_to_window: bool = True


@dataclass
class MatchConfig:
    # A menu element is fixed UI art on a static dialog and matches at 0.95-1.00
    # when it is really there. 0.80 was low enough to accept scenery: an accept
    # prompt was "found" at 0.804 in a corner of the screen nowhere near the
    # dialog, and clicked.
    menu_score: float = 0.90
    # advance-tier gets its own, much stricter floor. Its template matches the
    # take-all button at ~0.79 clean / ~0.85 live, so at the shared 0.80 floor
    # v4 clicked "advance tier" on the take-all button it had just pressed.
    advance_tier_score: float = 0.93
    # Padding around a candidate match when re-scoring it against the element it
    # is confusable with; big enough to fit the rival template (124x52 vs
    # 141x50) inside the box.
    confusion_padding: int = 20
    # Presence floor only - it decides "is a laborer icon here", never which
    # tier it is. Measured worst case for a correct icon under heavy capture
    # degradation is 0.9746, so 0.95 leaves headroom.
    laborer_score: float = 0.95
    # Profession is decided by whole-icon score, which separates it cleanly:
    # the worst measured gap to another profession is 0.067.
    laborer_profession_margin: float = 0.02
    # Tier is decided by a weighted comparison over the pixels that actually
    # differ between sibling tiers (~4.9% of the icon - the tier badge).
    # Measured: correct tier 20-32 under heavy degradation, unknown tier 83-93.
    tier_max_distance: float = 55.0
    # Runner-up must be at least this many times worse than the winner.
    # Measured worst case for a correct call under heavy degradation: 1.80.
    tier_ratio_min: float = 1.35
    # Absolute floor the winning journal must clear at a slot, in colour.
    # Measured for a correct icon under brutal capture noise: 0.960-0.971.
    journal_score: float = 0.90
    # Permissive floor for *detecting* that a slot holds some journal, in
    # grayscale. Deliberately loose: every journal matches every other journal's
    # slot at 0.87-0.97 in grayscale, which makes any template a fine detector
    # and a useless identifier. Identity is decided afterwards, in colour.
    journal_slot_score: float = 0.86
    # How far the winning journal must beat the runner-up *at the same slot*.
    # Measured worst case for a correct call under brutal noise: 0.0231
    # (t3-common over t2-common), and it barely moves as noise rises because
    # both candidates degrade together. Half of that is the working margin.
    journal_margin: float = 0.012
    # Two journals resolving to slots closer than this are a scan collision.
    journal_min_separation: int = 24
    coarse_scale: float = 0.5
    # Half-scale matching loses a little score; allow for it at the coarse stage.
    coarse_score_slack: float = 0.15
    refine_padding: int = 14


@dataclass
class RoiConfig:
    anchor_padding: int = 34
    laborer_padding: int = 26
    journal_verify_padding: int = 12
    dialog_padding: int = 48
    inventory_left_ratio: float = 0.50


@dataclass
class TimingConfig:
    poll_seconds: float = 0.01
    debounce_seconds: float = 0.25
    # Fixed pause after clicking take-all. Nothing on the dialog can be trusted
    # while it is still redrawing the frame we just clicked in.
    take_all_settle: float = 0.35
    # After that pause, how long to keep waiting for the take-all button to
    # actually go away - i.e. for the new interface to appear. Not fatal if it
    # never does; the advance-tier lookup is gated on its own score either way.
    interface_change_timeout: float = 1.00
    advance_tier_timeout: float = 0.60
    advance_tier_settle: float = 0.30
    ready_timeout: float = 0.60
    accept_timeout: float = 1.00
    # A retry burst is deliberately slow (see input.retry_delay_scale), so the
    # prompt it produces arrives later than the first click's would. Waiting
    # only accept_timeout after it would declare failure while the hand-over
    # the retry just triggered is still animating in.
    retry_accept_timeout: float = 2.50
    # Pause between the accept prompt first matching and the click on it. The
    # prompt animates in, so the frame where its template crosses the threshold
    # is not the frame where it is stable or interactive - clicking on that
    # first frame hits a button still sliding into place.
    accept_settle: float = 0.30
    accept_attempts: int = 3
    # t6 pick-up: find the gear, click it, let the context menu open, click
    # "pick up".
    settings_timeout: float = 1.00
    pickup_settle: float = 0.30
    pickup_timeout: float = 1.50
    # "Pick up" raises a confirmation; same shape again - let it draw, then
    # find and click Yes.
    confirm_settle: float = 0.30
    confirm_timeout: float = 1.50
    verify_delay: float = 0.15
    # How long a cached journal slot may look wrong before it is believed to be
    # wrong. Advancing a tier redraws over the panel, and a check landing in
    # that window sees nothing journal-like at all (measured: best score 0.573)
    # - which is a frame to wait out, not a reflow to rescan for.
    journal_verify_timeout: float = 0.60
    restore_mouse_delay: float = 0.25
    # Hard ceiling on one cycle; anything longer means the UI is not in the
    # state we think it is, so bail out to recovery rather than keep clicking.
    # It has to cover the worst legitimate case, which is the full retry ladder:
    # a fast click plus two deliberately slow bursts, each followed by
    # retry_accept_timeout, is ~11 s of hand-over on its own. At 15 s the
    # ceiling fired *during* the last retry and reported a broken UI when the
    # only thing that had happened was the retry being slow on purpose.
    cycle_timeout: float = 25.0


@dataclass
class InputConfig:
    # These are v3's tuned values, kept verbatim: the shift-click timing is the
    # part of v3 that demonstrably works and there is no reason to disturb it.
    cursor_move_attempts: int = 3
    cursor_settle: float = 0.02
    # How far to step off the target before moving back onto it when the cursor
    # is already parked there. Without a step off there is no movement to
    # inject, and a game that tracks its own hover target never sees the pointer
    # enter the slot. Small enough to stay inside the same inventory cell.
    cursor_nudge_pixels: int = 3
    click_hold: float = 0.04
    shift_move_delay: float = 0.12
    shift_before_click_delay: float = 0.05
    # One click hands the stack over. v3 sent two because it had no way to tell
    # a delivered journal from a missed one, so a duplicate looked free; v4 has
    # the accept prompt as the success signal, which makes the second click pure
    # downside - it lands after the game has already taken the first and moves
    # another journal, or it re-opens a prompt that was about to be clicked.
    shift_click_count: int = 1
    shift_click_hold: float = 0.06
    shift_click_gap: float = 0.05
    shift_after_click_hold: float = 0.10
    shift_release_delay: float = 0.04
    shift_click_attempts: int = 3
    # A retry means the single click produced no prompt at all, so the game
    # either never saw it or was still busy redrawing. The retry therefore stops
    # being gentle: several clicks, with every delay around them multiplied, so
    # a UI that was mid-frame has time to catch at least one of them. Only ever
    # reached after the slot has been re-verified as still holding the journal,
    # so the extra clicks cannot land on something else.
    retry_shift_click_count: int = 3
    retry_delay_scale: float = 4.0
    shift_press_attempts: int = 3
    shift_state_settle: float = 0.02
    modifier_release_timeout: float = 0.50
    click_original_after_advance: bool = True

    # --- getting the pointer out of the way ------------------------------
    # Resting on an inventory slot keeps that item's tooltip open, and the
    # tooltip is drawn over the dialog - i.e. over the accept prompt we are
    # about to look for. So the cursor leaves the slot after every hand-over
    # click, before anything is matched.
    park_cursor_after_shift_click: bool = True
    # How far inside the game window the parking spot has to stay. The spot
    # itself is derived from the window, the bag and the learned dialog box, so
    # there is no screen coordinate to keep in sync here.
    cursor_park_margin: int = 40

    # --- making sure the game actually receives the shift ----------------
    # Windows routes a click by cursor position and a key by focus. If the game
    # is not the focused window the click lands on it while the shift does not,
    # and a shift-click silently degrades into a plain click that picks the
    # journal up onto the cursor. Focus the game first.
    focus_target_before_input: bool = True
    focus_settle: float = 0.12
    # Probe the game's input queue (AttachThreadInput + GetKeyState) once before
    # the modifier goes down, to detect a queue we cannot read at all - which
    # means the game is elevated and injection is likely to be dropped outright.
    # Advisory only, and deliberately *not* used during the click: that probe
    # resets keyboard state, so running it while shift is held drops the very
    # modifier it is checking for. Whether the shift arrives is decided by
    # focus, which is checked before every click instead.
    verify_shift_in_target: bool = True
    # SetCursorPos moves the OS cursor without queueing a movement event, so a
    # game that tracks its own hover target can resolve the click against the
    # slot the cursor was over previously. Inject a real move *first* - an
    # injected move to where the cursor already sits carries no movement, so
    # placing the pointer before injecting can reduce the whole thing to a
    # no-op - then pin the exact pixel with SetCursorPos.
    inject_mouse_move: bool = True


@dataclass
class LaborerConfig:
    stop_prefixes: List[str] = field(default_factory=lambda: ["t6-"])
    # Any laborer whose tier parses above this is treated as a stop laborer even
    # if no dedicated template exists. Defence against t6 professions we have no
    # asset for (currently t6-imb / t6-tk) being read as their t5 sibling.
    max_action_tier: int = 5


@dataclass
class SafetyConfig:
    dry_run: bool = False
    # Trip the breaker rather than keep clicking into a UI we clearly misread.
    max_consecutive_failures: int = 3
    # Re-scan the whole inventory when any cached slot turns out to be stale;
    # an inventory reflow invalidates every slot, not just the one we touched.
    rescan_inventory_on_stale: bool = True
    # A confirmed stop laborer (t6) is picked up: settings gear -> "pick up".
    # Only ever runs on a positive template match for a t6 tier, never on an
    # unidentified portrait - picking up the wrong laborer is not undoable.
    pick_up_stop_laborers: bool = True


@dataclass
class KeyConfig:
    action: List[str] = field(default_factory=lambda: ["1", "numpad1", "ctrl+c"])
    scan: List[str] = field(default_factory=lambda: ["2", "numpad2"])
    quit: List[str] = field(default_factory=lambda: ["oem3"])
    panic: List[str] = field(default_factory=lambda: ["esc"])


# Bumped whenever a *default* changes in a way that must reach an existing
# config file. Writing a new default in the dataclass is not enough on its own:
# load_config applies the stored file over the defaults, so a value that is
# already in the file wins and the code change silently does nothing.
CONFIG_VERSION = 4

# version -> {dotted key: (previous default, new default)}. A stored value is
# only rewritten when it still equals the previous default, i.e. when the
# operator never touched it. Anything deliberately tuned is left alone.
DEFAULT_CHANGES: Dict[int, Dict[str, Tuple[Any, Any]]] = {
    2: {
        # 0.80 accepted scenery: take-all and accept were both "found" at ~0.805
        # in a screen corner and clicked. A real menu match scores 0.999-1.000.
        "match.menu_score": (0.80, 0.90),
    },
    3: {
        # "Albion" matched StatisticsAnalysisTool, Chrome and the editor. The
        # market tool's window is larger than the client's, so the largest-wins
        # picker chose it - and it lives on the other monitor, which put every
        # search region on the wrong screen. Identity moved to the executable
        # name; the title is now only a tie-break and must be the real one.
        "window.title_contains": (["Albion"], ["Albion Online Client"]),
    },
    4: {
        # The hand-over is one click now. The second click of the pair had no
        # accept prompt to check itself against and could only ever arrive after
        # the game had taken the first journal. Retries are where the extra
        # clicks live, and they run slowly and only on a re-verified slot.
        "input.shift_click_count": (2, 1),
        # ...and the retries that replaced it are slow by design, so the cycle
        # ceiling has to be able to contain them.
        "timing.cycle_timeout": (15.0, 25.0),
    },
}


@dataclass
class Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    laborer: LaborerConfig = field(default_factory=LaborerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    keys: KeyConfig = field(default_factory=KeyConfig)
    verbose: bool = False
    debug_dumps: bool = False
    version: int = CONFIG_VERSION


def _to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value


def _apply(target: Any, data: Dict[str, Any], path: str, warnings: List[str]) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            warnings.append(f"unknown config key ignored: {path}{key}")
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{path}{key}.", warnings)
        else:
            setattr(target, key, value)


def load_config(
    path: Path = CONFIG_PATH,
    factory: Callable[[], Config] = Config,
) -> Tuple[Config, List[str]]:
    """Load config, creating it with defaults on first run.

    ``factory`` builds the empty tree the file is applied over, so a later
    version can add its own sections by subclassing ``Config`` instead of
    re-implementing the merge, the migration and the write-back.
    """
    config = factory()
    warnings: List[str] = []
    if not path.exists():
        save_config(config, path)
        warnings.append(f"wrote default config to {path}")
        return config, warnings

    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"could not read {path} ({error}); using defaults")
        return config, warnings

    if not isinstance(data, dict):
        warnings.append(f"{path} is not a JSON object; using defaults")
        return config, warnings

    stored_version = data.get("version")
    stored_version = stored_version if isinstance(stored_version, int) else 1

    _apply(config, data, "", warnings)

    migrated = _migrate(config, stored_version, warnings)
    config.version = CONFIG_VERSION

    # A config written by an older build is missing the keys added since. Write
    # them back with their defaults so the file stays the place you tune from,
    # rather than silently diverging from the code.
    added = _added_keys(_to_dict(config), data)
    if added or migrated or stored_version != CONFIG_VERSION:
        save_config(config, path)
        if added:
            warnings.append(f"added {len(added)} new config key(s) to {path.name}: " + ", ".join(added))

    return config, warnings


def _migrate(config: Config, stored_version: int, warnings: List[str]) -> bool:
    """Move untouched settings onto their new defaults.

    Only rewrites a value that still equals the default it was shipped with, so
    anything deliberately tuned survives. Without this a changed default is
    invisible to anyone who has ever run the program - the file already holds
    the old number and the file wins.
    """
    changed = False
    for version in sorted(DEFAULT_CHANGES):
        if version <= stored_version:
            continue
        for dotted, (old_default, new_default) in DEFAULT_CHANGES[version].items():
            target, _sep, attribute = dotted.rpartition(".")
            holder: Any = config
            for part in target.split(".") if target else []:
                holder = getattr(holder, part, None)
                if holder is None:
                    break
            if holder is None or not hasattr(holder, attribute):
                continue
            current = getattr(holder, attribute)
            if current == old_default:
                setattr(holder, attribute, new_default)
                warnings.append(f"config: {dotted} {old_default} -> {new_default} (updated default)")
                changed = True
            elif current != new_default:
                warnings.append(
                    f"config: {dotted} is {current}; the recommended value is now {new_default} "
                    f"(was {old_default}) - left as you set it"
                )
    return changed


def _added_keys(current: Any, stored: Any, path: str = "") -> List[str]:
    if not isinstance(current, dict) or not isinstance(stored, dict):
        return []
    missing: List[str] = []
    for key, value in current.items():
        if key not in stored:
            missing.append(f"{path}{key}")
        else:
            missing.extend(_added_keys(value, stored[key], f"{path}{key}."))
    return missing


def save_config(config: Config, path: Path = CONFIG_PATH) -> None:
    payload = json.dumps(_to_dict(config), indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_key_combo(combo: str) -> Optional[List[int]]:
    """'ctrl+c' -> [VK_CONTROL, VK_C]; unknown names return None."""
    codes: List[int] = []
    for part in combo.lower().split("+"):
        part = part.strip()
        if not part:
            continue
        code = VK_CODES.get(part)
        if code is None:
            return None
        codes.append(code)
    return codes or None


def parse_key_combos(combos: List[str]) -> Tuple[List[List[int]], List[str]]:
    parsed: List[List[int]] = []
    unknown: List[str] = []
    for combo in combos:
        codes = parse_key_combo(combo)
        if codes is None:
            unknown.append(combo)
        else:
            parsed.append(codes)
    return parsed, unknown
