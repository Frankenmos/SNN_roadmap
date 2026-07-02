"""Launcher for the TinySkirmish rollout CLI (envs/tiny_skirmish).

Bootstraps sys.path so it works when invoked as `python scripts/run_tiny_skirmish.py`
from anywhere; `python -m envs.tiny_skirmish.rollout` from the repo root is equivalent.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.tiny_skirmish.rollout import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
