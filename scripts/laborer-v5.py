"""Launcher for the v5 laborer assistant.

v5 is v4's engine with a second way to start a cycle: a left click, armed and
disarmed with the " key (the 3 key; & / 1 stays the action key).

    python scripts\\laborer-v5.py                 # run, trigger off until the toggle key
    python scripts\\laborer-v5.py --click-trigger # run with the trigger already armed
    python scripts\\laborer-v5.py --selftest      # validate assets and timings
    python scripts\\laborer-v5.py --dry-run       # everything except input injection
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from laborer_v5.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
