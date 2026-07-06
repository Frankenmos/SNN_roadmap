"""Headless smoke test for tools/analysis/dashboard.py.

Runs the real dashboard script through streamlit.testing.v1.AppTest.
Streamlit executes every tab container on each script run, so a single
AppTest.run() exercises all five tabs (Overview / Policy / PPO / Reward /
Checkpoint) at once.

Two scenarios:
- every real run discovered under models/ (skipped when none exist),
  which covers the schema variants: legacy, pre-SIL Ray runs, SIL runs;
- a synthetic all-empty DB (tables created, zero rows), which must render
  notices instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from Utility.logger_utils import initialize_db

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "tools" / "analysis" / "dashboard.py"
REAL_MODELS_DIR = REPO_ROOT / "models"


def _real_runs() -> list[str]:
    if not REAL_MODELS_DIR.exists():
        return []
    return sorted(
        child.name
        for child in REAL_MODELS_DIR.iterdir()
        if child.is_dir() and (child / "training_logs.db").exists()
    )


def _make_apptest() -> AppTest:
    return AppTest.from_file(str(DASHBOARD_PATH), default_timeout=300)


def _assert_clean(at: AppTest, context: str) -> None:
    assert not at.exception, (
        f"dashboard raised in {context}: "
        f"{[element.value for element in at.exception]}"
    )


@pytest.mark.skipif(
    not _real_runs(),
    reason="no local runs with training_logs.db under models/",
)
def test_dashboard_renders_every_local_run(monkeypatch):
    monkeypatch.delenv("SNN_DASHBOARD_MODELS_DIR", raising=False)
    at = _make_apptest()
    at.run()
    _assert_clean(at, "initial render")

    run_select = at.sidebar.selectbox[0]
    assert sorted(run_select.options) == _real_runs()
    for run_name in run_select.options:
        run_select.set_value(run_name)
        at.run()
        _assert_clean(at, f"run {run_name}")


def test_dashboard_handles_empty_db(monkeypatch, tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir(parents=True)
    conn = initialize_db(str(run_dir / "training_logs.db"))
    conn.close()

    monkeypatch.setenv("SNN_DASHBOARD_MODELS_DIR", str(tmp_path))
    at = _make_apptest()
    at.run()
    _assert_clean(at, "empty DB render")
    assert at.info, "expected notice elements on an all-empty DB"
