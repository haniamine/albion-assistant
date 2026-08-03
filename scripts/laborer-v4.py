"""Launcher for the v4 laborer assistant.

    python scripts\\laborer-v4.py            # run
    python scripts\\laborer-v4.py --selftest # validate assets and timings
    python scripts\\laborer-v4.py --dry-run  # everything except input injection
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from laborer_v4.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
