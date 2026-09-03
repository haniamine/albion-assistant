"""Configuration for laborer-v5.

v4's tree, plus the settings the click trigger needs. Subclassing rather than
copying means every threshold and timing v4 tunes is inherited automatically -
a new match floor in v4 is a new match floor here, with no second place to
edit.

v5 keeps its own config and state files so the two versions can be run side by
side, but seeds both from v4's on first run: the state cache in particular
holds the learned dialog anchors and the journal slot positions, and starting
from an empty one would mean a cold full-screen search and a rescan.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from laborer_v4.config import (
    CONFIG_PATH as V4_CONFIG_PATH,
    ROOT_DIR,
    STATE_PATH as V4_STATE_PATH,
    Config,
    KeyConfig,
)
from laborer_v4.config import load_config as _load_v4_config

CONFIG_PATH = ROOT_DIR / "laborer-v5.config.json"
STATE_PATH = ROOT_DIR / "laborer-v5.state.json"


@dataclass
class ClickTriggerConfig:
    """The left mouse button as an action starter."""

    # Off at startup, always. In game the left button is walk, attack and
    # interact - i.e. the button you press constantly - so a mode where every
    # one of those presses runs a full click sequence is not a mode to be in
    # by accident. The toggle key arms it deliberately.
    enabled_at_start: bool = False
    # Fire when the button comes *up*, not when it goes down. The engine's
    # first act is to inject its own clicks, and injecting a mouse-down while
    # the physical button is still held gives the game a button that never
    # went up between the two. Waiting for the release means the user's click
    # is complete and the button is idle before anything is injected.
    fire_on_press: bool = False
    # A click only counts while the game itself is the foreground window.
    # Without this the trigger fires on a click in the editor, the browser or
    # anywhere else - the same class of accident that gating the hotkeys on
    # the foreground window removed in v4.
    require_foreground: bool = True
    # ...and only while the pointer is inside the game's rect. Matters on a
    # multi-monitor desk, where the game can hold focus while the click lands
    # on another screen.
    require_inside_window: bool = True
    # The click is what opens the laborer dialog, and the dialog animates in.
    # The engine's first act is to look for take-all, so handing over on the
    # frame the click landed searches a panel that is not drawn yet - it finds
    # nothing, and a cycle that finds nothing counts against the failure
    # breaker. This gap only applies to a click; pressing the action key means
    # the interface is already open.
    open_delay: float = 1.0
    # How often to re-resolve the game window while the trigger is armed and
    # no window is known yet - i.e. when the script was started before the
    # game. Finding it walks every top-level window, so this is throttled
    # rather than run on the poll interval.
    window_refresh_seconds: float = 2.0
    # Keep the action key working while the click trigger is on. Turning this
    # off makes the two mutually exclusive.
    action_key_still_starts: bool = True


@dataclass
class V5KeyConfig(KeyConfig):
    # VK 0x33 is the '3' key: the one marked " on AZERTY, where the action key
    # '1' is marked &. Named by virtual key, so the binding is the same
    # physical key on either layout.
    toggle_click: List[str] = field(default_factory=lambda: ["3", "numpad3"])


@dataclass
class V5Config(Config):
    keys: V5KeyConfig = field(default_factory=V5KeyConfig)
    click_trigger: ClickTriggerConfig = field(default_factory=ClickTriggerConfig)


def seed_from_v4(path: Path, source: Path) -> List[str]:
    """Copy v4's file to ``path`` when v5 has none yet."""
    if path.exists() or not source.exists():
        return []
    try:
        shutil.copyfile(source, path)
    except OSError as error:
        return [f"could not seed {path.name} from {source.name}: {error}"]
    return [f"seeded {path.name} from {source.name}"]


def load_config(path: Path = CONFIG_PATH) -> Tuple[V5Config, List[str]]:
    warnings = seed_from_v4(path, V4_CONFIG_PATH)
    config, more = _load_v4_config(path, factory=V5Config)
    return config, warnings + more  # type: ignore[return-value]


def seed_state(path: Path = STATE_PATH) -> List[str]:
    return seed_from_v4(path, V4_STATE_PATH)
