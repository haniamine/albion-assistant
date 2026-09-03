"""Entry point: hotkey loop, CLI and startup diagnostics."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from .assets import build_library
from .config import (
    CONFIG_PATH,
    MAPPING_PATH,
    ROOT_DIR,
    STATE_PATH,
    Config,
    load_config,
    parse_key_combos,
    save_config,
)
from .engine import LaborerEngine
from .feedback import Feedback, Outcome, setup_logging
from .state import AppState, import_v3_positions
from .vision import Grabber
from .winput import (
    InputDriver,
    any_combo_pressed,
    diagnose_shift,
    enable_dpi_awareness,
    find_game_window,
    ignore_console_ctrl_c,
    key_state_in_window,
)

log = logging.getLogger("laborer.app")

V3_POSITIONS_PATH = ROOT_DIR / "journal-positions.json"


class Hotkeys:
    def __init__(self, config: Config) -> None:
        self.action, unknown_action = parse_key_combos(config.keys.action)
        self.scan, unknown_scan = parse_key_combos(config.keys.scan)
        self.quit, unknown_quit = parse_key_combos(config.keys.quit)
        self.panic, unknown_panic = parse_key_combos(config.keys.panic)
        self.unknown = unknown_action + unknown_scan + unknown_quit + unknown_panic
        self.uses_ctrl_c = any(combo == [0x11, 0x43] for combo in self.action)

    def action_pressed(self) -> bool:
        return any_combo_pressed(self.action)

    def scan_pressed(self) -> bool:
        return any_combo_pressed(self.scan)

    def quit_pressed(self) -> bool:
        return any_combo_pressed(self.quit)

    def panic_pressed(self) -> bool:
        return any_combo_pressed(self.panic)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laborer-v4",
        description="Albion laborer assistant (v4): anchored ROIs, detect-then-classify.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="path to the config file")
    parser.add_argument("--state", type=Path, default=STATE_PATH, help="path to the state cache")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug-level logging")
    parser.add_argument("--log-file", type=Path, default=None, help="also write a debug log here")
    parser.add_argument("--dry-run", action="store_true", help="run every step but never inject input")
    parser.add_argument("--no-sound", action="store_true", help="disable the outcome beeps")
    parser.add_argument("--scan", action="store_true", help="scan the inventory once at startup")
    parser.add_argument("--reset", action="store_true", help="discard the cached state before starting")
    parser.add_argument("--selftest", action="store_true", help="validate assets and timings, then exit")
    parser.add_argument(
        "--test-shift",
        action="store_true",
        help="report whether an injected shift actually reaches the game, then exit (no clicks)",
    )
    parser.add_argument("--write-config", action="store_true", help="rewrite the config file with defaults filled in")
    return parser


def report_startup(
    config: Config,
    library,
    engine: LaborerEngine,
    hotkeys: Hotkeys,
    dpi_mode: str,
    banner: str = "Albion laborer assistant v4",
) -> None:
    log.info("=" * 64)
    log.info("%s", banner)
    log.info("DPI awareness: %s", dpi_mode)
    log.info(
        "assets: %d laborers (%d stop), %d journals, %d menu elements",
        len(library.laborers), len(library.stop_laborers),
        len(library.journals_by_name), len(library.menu),
    )
    for warning in library.warnings:
        log.warning("%s", warning)
    if hotkeys.unknown:
        log.warning("unrecognised key names in config: %s", ", ".join(hotkeys.unknown))

    window = find_game_window(config.window.title_contains, config.window.process_contains)
    if window is None:
        log.warning(
            "no %s window titled %s found; %s",
            config.window.process_contains, config.window.title_contains,
            "actions will be blocked until it appears" if config.window.require_foreground
            else "input gating is off (window.require_foreground=false)",
        )
    else:
        log.info(
            "game window: '%s' (%s) %dx%d at (%d,%d)",
            window.title, window.process or "unknown process",
            window.bounds[2], window.bounds[3], window.bounds[0], window.bounds[1],
        )
        # Say this at startup rather than at the first failed hand-over: if the
        # queue cannot be read, shift-clicks are about to be refused and the
        # reason (an elevation mismatch) is not one the log would otherwise show.
        if config.input.verify_shift_in_target and key_state_in_window(window.hwnd, 0x10) is None:
            log.warning(
                "cannot read the game window's input queue, so a shift-click cannot be "
                "verified - the game is probably running elevated and this script is not. "
                "Run --test-shift for the full diagnosis."
            )

    log.info(
        "hotkeys: action=%s  scan=%s  quit=%s  panic=%s",
        "/".join(config.keys.action), "/".join(config.keys.scan),
        "/".join(config.keys.quit), "/".join(config.keys.panic),
    )
    if config.safety.dry_run:
        log.warning("DRY RUN: no clicks or keystrokes will be sent")
    log.info(
        "stop laborers (%s): %s",
        ", ".join(sorted(library.stop_laborers)) or "none",
        "beep, then settings -> pick up" if config.safety.pick_up_stop_laborers
        else "beep only (safety.pick_up_stop_laborers=false)",
    )
    if engine.state.journals:
        log.info("journal cache: %s", ", ".join(engine.state.journals.names()))
    else:
        log.warning("journal cache is empty - press the scan key before acting")
    if engine.state.anchors.origin:
        log.info(
            "anchors: origin %s, %d learned offsets (%s)",
            engine.state.anchors.origin, len(engine.state.anchors.offsets),
            ", ".join(sorted(engine.state.anchors.offsets)),
        )
    else:
        log.info("anchors: none learned yet - the first cycle will do a full search")
    log.info("=" * 64)


def run_selftest(config: Config, library, engine: LaborerEngine) -> int:
    """Validate assets and measure the matching costs without touching the game."""
    import cv2

    log.info("--- self test ---")
    problems = 0

    masked = [name for name, template in library.laborers.items() if template.mask is not None]
    log.info(
        "laborer templates carrying a real alpha mask: %d/%d%s",
        len(masked), len(library.laborers),
        "" if masked else "  (all opaque - masks correctly dropped, 2.4x faster)",
    )

    sizes = {template.size for template in library.laborers.values()}
    if len(sizes) > 1:
        log.warning(
            "laborer assets have %d distinct sizes %s - cropping is inconsistent; "
            "alignment is auto-corrected but re-cropping them to one size is better",
            len(sizes), sorted(sizes),
        )
        problems += 1

    for warning in library.warnings:
        log.warning("%s", warning)
        problems += 1

    geometry = engine.refresh_geometry()
    if geometry is None:
        log.error("could not resolve a monitor")
        return 1

    frame = engine.grabber.grab(engine.search_rect)
    log.info("captured search region %dx%d", frame.gray.shape[1], frame.gray.shape[0])

    started = time.perf_counter()
    engine.grabber.grab(engine.search_rect)
    grab_ms = (time.perf_counter() - started) * 1000

    from .vision import coarse_locate, scaled_source

    started = time.perf_counter()
    small = scaled_source(frame, config.match.coarse_scale)
    for template in engine.active_templates.values():
        coarse_locate(frame, template, config.match.coarse_scale, small)
    coarse_ms = (time.perf_counter() - started) * 1000

    width, height = library.max_laborer_size()
    crop = frame.gray[:height + 52, :width + 52]
    started = time.perf_counter()
    for template in engine.active_templates.values():
        if crop.shape[0] >= template.height and crop.shape[1] >= template.width:
            cv2.matchTemplate(crop, template.gray, cv2.TM_SQDIFF_NORMED, mask=template.mask)
    classify_ms = (time.perf_counter() - started) * 1000

    log.info("timing: grab=%.1f ms", grab_ms)
    log.info("timing: cold locate (%d candidates, half scale) = %.1f ms", len(engine.active_templates), coarse_ms)
    log.info("timing: warm classify (%d candidates in one ROI) = %.1f ms", len(engine.active_templates), classify_ms)
    log.info("for reference, v3's equivalent full-screen masked sweep was ~%.0f ms", len(engine.active_templates) * 98.0 * 2)

    log.info("--- self test %s ---", "passed with warnings" if problems else "passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    config, config_warnings = load_config(args.config)
    if args.verbose:
        config.verbose = True
    if args.dry_run:
        config.safety.dry_run = True
    if args.write_config:
        save_config(config, args.config)

    setup_logging(config.verbose, str(args.log_file) if args.log_file else None)
    for warning in config_warnings:
        log.warning("%s", warning)

    dpi_mode = enable_dpi_awareness()

    if args.test_shift:
        # Deliberately ahead of the assets, the state cache and the engine: this
        # diagnoses input routing, so it has to still work when those are broken.
        return diagnose_shift(
            InputDriver(
                config.input,
                target_titles=config.window.title_contains,
                target_processes=config.window.process_contains,
            ),
            config.window.title_contains,
            config.window.process_contains,
        )

    try:
        library = build_library(config, MAPPING_PATH)
    except (FileNotFoundError, ValueError) as error:
        log.error("could not load assets: %s", error)
        return 2

    state = AppState() if args.reset else AppState.load(args.state)
    if not state.journals and not args.reset:
        migrated = import_v3_positions(V3_POSITIONS_PATH, monitor_index=0)
        if migrated is not None:
            log.info("imported %d journal position(s) from v3's cache; rescan to refresh", len(migrated.slots))
            state.journals = migrated
            state.mark_dirty()

    grabber = Grabber()
    hotkeys = Hotkeys(config)
    driver = InputDriver(
        config.input,
        dry_run=config.safety.dry_run,
        panic=hotkeys.panic_pressed,
        target_titles=config.window.title_contains,
        target_processes=config.window.process_contains,
    )
    feedback = Feedback(enabled=not args.no_sound)
    engine = LaborerEngine(config, library, state, grabber, driver, feedback, state_path=args.state)

    engine.refresh_geometry()
    engine.refresh_candidates(force=True)

    if args.selftest:
        return run_selftest(config, library, engine)

    report_startup(config, library, engine, hotkeys, dpi_mode)

    if hotkeys.uses_ctrl_c:
        # Ctrl+C doubles as an action hotkey, so the console handler must go.
        ignore_console_ctrl_c()
        log.info("Ctrl+C is an action hotkey; use the quit key to stop")

    if args.scan:
        engine.scan_inventory()

    poll = config.timing.poll_seconds
    debounce = config.timing.debounce_seconds
    last_action = 0.0
    breaker_tripped = False

    try:
        while True:
            if hotkeys.quit_pressed():
                log.info("quit key pressed - stopping")
                break

            now = time.monotonic()

            if hotkeys.scan_pressed():
                if now - last_action >= debounce:
                    log.info("-" * 64)
                    try:
                        engine.scan_inventory()
                    except Exception:
                        log.exception("inventory scan failed")
                    if breaker_tripped:
                        log.info("failure breaker reset")
                        breaker_tripped = False
                        engine.consecutive_failures = 0
                    last_action = time.monotonic()

            elif hotkeys.action_pressed():
                if now - last_action >= debounce:
                    if breaker_tripped:
                        log.error("actions are paused after repeated failures; press the scan key to reset")
                        # BLOCKED is silent by design; this one still needs a
                        # noise, because otherwise the key press does nothing
                        # at all and looks like the script died.
                        feedback.play(Outcome.ERROR)
                        last_action = time.monotonic()
                        time.sleep(poll)
                        continue

                    log.info("-" * 64)
                    # run_cycle owns the beep now: the t6 alert has to sound
                    # mid-cycle, before the pick-up clicks, not after them.
                    outcome = engine.run_cycle()

                    limit = config.safety.max_consecutive_failures
                    if limit > 0 and engine.consecutive_failures >= limit:
                        log.error(
                            "%d consecutive failures - pausing actions. "
                            "Fix the UI state, then press the scan key to resume.",
                            engine.consecutive_failures,
                        )
                        feedback.play(Outcome.ERROR)
                        breaker_tripped = True

                    # Debounce from the *end* of the action, not the start;
                    # v3 timed it from the start so a multi-second cycle left
                    # no debounce at all.
                    last_action = time.monotonic()

            time.sleep(poll)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        state.flush(args.state)
        grabber.close()

    return 0
