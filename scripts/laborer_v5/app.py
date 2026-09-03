"""Entry point: hotkey loop with the left-click trigger, CLI and diagnostics.

The loop is v4's with one branch added. What is new is that a cycle now has two
ways in - the action key, and a left click while the trigger is armed - so the
body that starts one is a function rather than an inline branch, and both ways
share the debounce and the failure breaker.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from laborer_v4.app import V3_POSITIONS_PATH, Hotkeys as V4Hotkeys, report_startup, run_selftest
from laborer_v4.assets import build_library
from laborer_v4.config import MAPPING_PATH, parse_key_combos, save_config
from laborer_v4.engine import LaborerEngine
from laborer_v4.feedback import Feedback, Outcome, setup_logging
from laborer_v4.state import AppState, import_v3_positions
from laborer_v4.vision import Grabber
from laborer_v4.winput import (
    InputDriver,
    any_combo_pressed,
    diagnose_shift,
    enable_dpi_awareness,
    ignore_console_ctrl_c,
)

from .config import CONFIG_PATH, STATE_PATH, V5Config, load_config, seed_state
from .trigger import BEEP_OFF, BEEP_ON, ClickTrigger, WindowTracker

log = logging.getLogger("laborer.app")


class Hotkeys(V4Hotkeys):
    def __init__(self, config: V5Config) -> None:
        super().__init__(config)
        self.toggle_click, unknown_toggle = parse_key_combos(config.keys.toggle_click)
        self.unknown = self.unknown + unknown_toggle

    def toggle_click_pressed(self) -> bool:
        return any_combo_pressed(self.toggle_click)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laborer-v5",
        description="Albion laborer assistant (v5): v4's engine, started by a left click.",
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
    parser.add_argument(
        "--click-trigger",
        action="store_true",
        help="start with the left-click trigger already armed (default: off until the toggle key)",
    )
    return parser


def wait_for_interface(seconds: float, poll: float, panic) -> bool:
    """Let the dialog the click just opened finish animating in.

    Slept in poll-sized steps rather than in one go so the panic key still
    works during the gap - a full second of a dead keyboard is long enough to
    notice, and this is exactly the moment an operator realises they clicked
    the wrong thing. False means panic was pressed and no cycle should start.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if panic():
            return False
        time.sleep(min(poll, 0.05))
    return True


def report_trigger(config: V5Config, hotkeys: Hotkeys, trigger: ClickTrigger) -> None:
    log.info("toggle click trigger: %s", "/".join(config.keys.toggle_click))
    log.info("left-click trigger: %s", trigger.describe())
    gates = [
        name for name, on in (
            ("game in foreground", config.click_trigger.require_foreground),
            ("pointer inside the window", config.click_trigger.require_inside_window),
        ) if on
    ]
    log.info("  a click only counts with %s", " and ".join(gates) if gates else "no gating at all")
    log.info(
        "  %s",
        f"then {config.click_trigger.open_delay:.2f}s for the interface to open before the cycle starts"
        if config.click_trigger.open_delay > 0
        else "the cycle starts immediately (click_trigger.open_delay=0)",
    )
    if not gates:
        log.warning(
            "click_trigger gating is fully off - a left click anywhere, in any "
            "application, will start a cycle once the trigger is armed"
        )
    if not config.click_trigger.action_key_still_starts:
        log.info("  the action key is ignored while the trigger is armed")


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
    if args.state == STATE_PATH and not args.reset:
        config_warnings += seed_state(args.state)
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

    trigger = ClickTrigger(
        config.click_trigger,
        WindowTracker(
            config.window.title_contains,
            config.window.process_contains,
            config.click_trigger.window_refresh_seconds,
        ),
        enabled=args.click_trigger or config.click_trigger.enabled_at_start,
    )

    report_startup(config, library, engine, hotkeys, dpi_mode, banner="Albion laborer assistant v5")
    report_trigger(config, hotkeys, trigger)
    log.info("=" * 64)

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

    def start_cycle(source: str) -> bool:
        """Run one cycle. Returns whether the breaker is now tripped."""
        nonlocal breaker_tripped
        if breaker_tripped:
            log.error("actions are paused after repeated failures; press the scan key to reset")
            # BLOCKED is silent by design; this one still needs a noise, because
            # otherwise the trigger does nothing at all and looks like the
            # script died.
            feedback.play(Outcome.ERROR)
            return True

        log.info("-" * 64)
        log.debug("cycle started by %s", source)
        # run_cycle owns the beep: the t6 alert has to sound mid-cycle, before
        # the pick-up clicks, not after them.
        engine.run_cycle()

        limit = config.safety.max_consecutive_failures
        if limit > 0 and engine.consecutive_failures >= limit:
            log.error(
                "%d consecutive failures - pausing actions. "
                "Fix the UI state, then press the scan key to resume.",
                engine.consecutive_failures,
            )
            feedback.play(Outcome.ERROR)
            breaker_tripped = True
        return breaker_tripped

    try:
        while True:
            if hotkeys.quit_pressed():
                log.info("quit key pressed - stopping")
                break

            now = time.monotonic()
            # Polled every pass, whatever else happens: the trigger works on
            # button *edges*, and an iteration that skipped the read would miss
            # the transition entirely rather than merely delay it.
            clicked = trigger.poll()

            if hotkeys.toggle_click_pressed():
                if now - last_action >= debounce:
                    enabled = trigger.toggle()
                    log.info("left-click trigger %s", "ARMED - a click in game starts a cycle" if enabled else "off")
                    feedback.play_pattern(BEEP_ON if enabled else BEEP_OFF)
                    last_action = time.monotonic()

            elif hotkeys.scan_pressed():
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
                    trigger.suppress()
                    last_action = time.monotonic()

            elif clicked or (
                (config.click_trigger.action_key_still_starts or not trigger.enabled)
                and hotkeys.action_pressed()
            ):
                if now - last_action >= debounce:
                    # A click opens the dialog; the action key is pressed with
                    # it already open. So only the click waits.
                    if clicked and not wait_for_interface(
                        config.click_trigger.open_delay, poll, hotkeys.panic_pressed
                    ):
                        log.info("panic key during the interface gap - no cycle started")
                    else:
                        start_cycle("a left click" if clicked else "the action key")
                    # The engine has just injected a burst of left clicks of its
                    # own, and the operator may still be holding the one that
                    # started this. Either way the trigger stays shut until it
                    # sees the button released.
                    trigger.suppress()
                    # Debounce from the *end* of the action, not the start; v3
                    # timed it from the start so a multi-second cycle left no
                    # debounce at all.
                    last_action = time.monotonic()

            time.sleep(poll)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        state.flush(args.state)
        grabber.close()

    return 0
