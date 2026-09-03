"""Offline tests for laborer-v5: no game, no screen capture, no input.

Only the parts v5 adds are covered here - the click trigger's state machine and
the config layering. Everything v5 inherits is v4's and is tested by
``test_laborer_v4.py``. Run with:

    python scripts\\tests\\test_laborer_v5.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laborer_v4.config import Config  # noqa: E402
from laborer_v5.config import ClickTriggerConfig, V5Config, load_config  # noqa: E402

try:
    from laborer_v5 import trigger as trigger_module  # noqa: E402
except ImportError as _error:  # pywin32 absent, or not running on Windows
    trigger_module = None
    print(f"! trigger tests will be skipped: {_error}")

FAILURES: List[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}")
    else:
        print(f"  ok    {message}")


@contextmanager
def patched(module, **attributes):
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


class FakeWindow:
    """A game window that is always found, focused and under the pointer."""

    def __init__(self, hwnd: int = 77, rect: Tuple[int, int, int, int] = (0, 0, 1920, 1080)) -> None:
        self.hwnd = hwnd
        self.title = "Albion Online Client"
        self.rect = rect

    def current(self) -> Optional["FakeWindow"]:
        return self


class NoWindow:
    def current(self):
        return None


class Button:
    """A scripted sequence of physical button levels, one per poll."""

    def __init__(self, levels: str) -> None:
        # "_" up, "#" down - written as a string so a test reads as a timeline.
        self.levels = [character == "#" for character in levels]
        self.index = 0

    def __call__(self, _vk_code: int) -> bool:
        level = self.levels[min(self.index, len(self.levels) - 1)]
        self.index += 1
        return level


def _fires(levels: str, config: ClickTriggerConfig, window=None, enabled: bool = True) -> str:
    """Run the trigger over a timeline; returns one character per poll."""
    instance = trigger_module.ClickTrigger(config, window or FakeWindow(), enabled=enabled)
    result = []
    with patched(
        trigger_module,
        is_pressed=Button(levels),
        foreground_hwnd=lambda: 77,
        win32gui=type("gui", (), {"GetCursorPos": staticmethod(lambda: (100, 100)), "error": RuntimeError}),
    ):
        for _ in levels:
            result.append("!" if instance.poll() else ".")
    return "".join(result)


def test_a_click_fires_once_on_release() -> None:
    print("\n[trigger: one cycle per click]")
    config = ClickTriggerConfig()
    # A press held over several polls is still one click, and it lands on the
    # release - the poll where the button comes back up.
    check(_fires("_###_", config) == "....!", "a held press fires exactly once, on release")
    check(_fires("_#_#_#_", config) == "..!.!.!", "three clicks fire three times")


def test_nothing_fires_before_a_release_is_seen() -> None:
    print("\n[trigger: the arming gate]")
    config = ClickTriggerConfig()
    # Starting with the button already down: the release that follows is the
    # one that arms the trigger, so it must not itself be read as a click.
    check(_fires("###_#_", config) == ".....!", "the first release only arms; it does not fire")
    check(_fires("_#_", config, enabled=False) == "...", "a disabled trigger never fires")


def test_a_button_down_when_the_cycle_returns_starts_nothing() -> None:
    print("\n[trigger: what suppression is for]")
    # The engine's own clicks all land inside run_cycle, which the loop is not
    # polling during - so they are never sampled as edges. What *is* down when
    # polling resumes is either the last injected press still settling or an
    # impatient second click from the operator, and neither may start a cycle.
    config = ClickTriggerConfig()
    instance = trigger_module.ClickTrigger(config, FakeWindow(), enabled=True)
    button = Button("_#_" + "#__" + "_#_")
    with patched(
        trigger_module,
        is_pressed=button,
        foreground_hwnd=lambda: 77,
        win32gui=type("gui", (), {"GetCursorPos": staticmethod(lambda: (100, 100)), "error": RuntimeError}),
    ):
        check([instance.poll() for _ in range(3)] == [False, False, True], "the operator's click starts a cycle")
        instance.suppress()  # what the loop does after every cycle
        check(not any(instance.poll() for _ in range(3)), "a button already down on return starts nothing")
        check([instance.poll() for _ in range(3)] == [False, False, True], "the next real click works again")


def test_a_click_outside_the_game_is_ignored() -> None:
    print("\n[trigger: location gating]")
    config = ClickTriggerConfig()
    check(_fires("_#_", config, window=NoWindow()) == "...", "no game window found: no cycle")

    instance = trigger_module.ClickTrigger(config, FakeWindow(), enabled=True)
    with patched(
        trigger_module,
        is_pressed=Button("_#_#_"),
        foreground_hwnd=lambda: 1234,  # something else has focus
        win32gui=type("gui", (), {"GetCursorPos": staticmethod(lambda: (100, 100)), "error": RuntimeError}),
    ):
        check(not any(instance.poll() for _ in range(5)), "game not in the foreground: no cycle")

    instance = trigger_module.ClickTrigger(config, FakeWindow(rect=(0, 0, 800, 600)), enabled=True)
    with patched(
        trigger_module,
        is_pressed=Button("_#_#_"),
        foreground_hwnd=lambda: 77,
        win32gui=type("gui", (), {"GetCursorPos": staticmethod(lambda: (900, 100)), "error": RuntimeError}),
    ):
        check(not any(instance.poll() for _ in range(5)), "pointer outside the window rect: no cycle")


def test_press_mode_fires_on_the_way_down() -> None:
    print("\n[trigger: fire_on_press]")
    config = ClickTriggerConfig(fire_on_press=True)
    check(_fires("_###_", config) == ".!...", "the press fires, not the release")
    check(_fires("_###_###_", config) == ".!...!...", "one fire per press, however long it is held")


def test_toggling_off_and_on_needs_a_fresh_release() -> None:
    print("\n[trigger: the toggle]")
    instance = trigger_module.ClickTrigger(ClickTriggerConfig(), FakeWindow(), enabled=False)
    check(instance.toggle() is True and instance.enabled, "the toggle arms it")
    with patched(
        trigger_module,
        is_pressed=Button("###_#_"),
        foreground_hwnd=lambda: 77,
        win32gui=type("gui", (), {"GetCursorPos": staticmethod(lambda: (100, 100)), "error": RuntimeError}),
    ):
        # Toggling while the button happens to be down must not fire on the
        # release of that same press.
        fired = [instance.poll() for _ in range(6)]
        check(fired == [False, False, False, False, False, True], "arming mid-press does not fire on that press")
    check(instance.toggle() is False and not instance.enabled, "the toggle disarms it again")


def test_the_interface_gap_is_waited_out_but_stays_interruptible() -> None:
    print("\n[gap: waiting for the dialog to open]")
    from laborer_v5.app import wait_for_interface

    started = time.monotonic()
    check(wait_for_interface(0.20, 0.01, lambda: False) is True, "an uninterrupted gap runs to completion")
    elapsed = time.monotonic() - started
    check(0.18 <= elapsed <= 0.45, f"and takes about as long as asked ({elapsed:.2f}s)")

    # A whole second of dead keyboard is long enough to notice, and this is
    # exactly when an operator realises they clicked the wrong laborer.
    started = time.monotonic()
    check(wait_for_interface(5.0, 0.01, lambda: True) is False, "the panic key cuts the gap short")
    check(time.monotonic() - started < 1.0, "and does not wait out the rest of it")

    check(wait_for_interface(0.0, 0.01, lambda: False) is True, "a zero gap is not a wait at all")


def test_the_v5_config_is_v4s_plus_the_trigger(tmp_path: Path) -> None:
    print("\n[config: v5 layers over v4]")
    source = tmp_path / "v4.json"
    v4 = json.loads(json.dumps({"match": {"menu_score": 0.97}, "keys": {"action": ["7"]}, "version": 4}))
    source.write_text(json.dumps(v4), encoding="utf-8")

    target = tmp_path / "v5.json"
    with patched(sys.modules["laborer_v5.config"], V4_CONFIG_PATH=source):
        config, warnings = load_config(target)

    check(isinstance(config, V5Config) and isinstance(config, Config), "v5's config is still a v4 config")
    check(config.match.menu_score == 0.97, "a tuned v4 threshold is carried over, not reset")
    check(config.keys.action == ["7"], "a rebound v4 action key is carried over")
    check(config.keys.toggle_click == ["3", "numpad3"], 'the toggle defaults to the " (3) key')
    check(any("seeded" in warning for warning in warnings), "the seeding is reported, not silent")

    stored = json.loads(target.read_text(encoding="utf-8"))
    check("click_trigger" in stored, "the new section is written back to the file")
    check(
        stored["click_trigger"]["enabled_at_start"] is False,
        "the trigger is off by default - left click is walk and attack in game",
    )
    check(stored["click_trigger"]["open_delay"] == 1.0, "the interface gap defaults to 1s and is tunable")

    # Loading again must be a no-op: a config that rewrites itself every start
    # is a config whose edits you cannot trust.
    _config, again = load_config(target)
    check(not again, "a second load changes nothing")


def main() -> int:
    import tempfile

    tests: List[Callable[[], None]] = []
    if trigger_module is not None:
        tests.extend([
            test_a_click_fires_once_on_release,
            test_nothing_fires_before_a_release_is_seen,
            test_a_button_down_when_the_cycle_returns_starts_nothing,
            test_a_click_outside_the_game_is_ignored,
            test_press_mode_fires_on_the_way_down,
            test_toggling_off_and_on_needs_a_fresh_release,
            test_the_interface_gap_is_waited_out_but_stays_interruptible,
        ])

    with tempfile.TemporaryDirectory() as directory:
        tests.append(lambda: test_the_v5_config_is_v4s_plus_the_trigger(Path(directory)))
        for test in tests:
            try:
                test()
            except Exception:
                FAILURES.append("exception")
                traceback.print_exc()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
