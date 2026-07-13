"""Read-only mission-control helpers for the dashboard.

Cross-run overview, effective-config diffs, and launch-command
generation. Deliberately NO process management: training runs are
launched by the user in their own terminal; this module only reads
models/<run>/ artifacts (databases opened read-only) and formats the
exact `python -m distributed.ray_train` invocation (see
distributed/ray_train.py:_parse_args for the accepted flags).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.registry.core import _flatten_config

SNAPSHOT_FILE_RE = re.compile(r"^policy_u(\d+)\.pth$")


@dataclass
class RunOverview:
    name: str
    updates_total: int | None = None
    last_update_index: int | None = None
    right_click_share_last: float | None = None
    last_eval_mean: float | None = None
    last_eval_policy_version: int | None = None
    snapshot_count: int = 0
    has_checkpoint: bool = False
    has_best: bool = False
    db_modified_iso: str | None = None


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    try:
        return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _one(conn: sqlite3.Connection, sql: str):
    try:
        return conn.execute(sql).fetchone()
    except sqlite3.Error:
        return None


def _overview_for(run_dir: Path) -> RunOverview:
    overview = RunOverview(name=run_dir.name)
    overview.has_checkpoint = (run_dir / "checkpoint.pth").exists()
    overview.has_best = (run_dir / "best_checkpoint.pth").exists()
    snapshot_dir = run_dir / "snapshots"
    if snapshot_dir.is_dir():
        overview.snapshot_count = sum(
            1 for p in snapshot_dir.iterdir() if SNAPSHOT_FILE_RE.match(p.name)
        )

    db_path = run_dir / "training_logs.db"
    if db_path.exists():
        overview.db_modified_iso = datetime.fromtimestamp(
            db_path.stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M")
    conn = _connect_ro(db_path)
    if conn is None:
        return overview
    try:
        row = _one(conn, "SELECT COUNT(*) FROM ppo_updates")
        if row:
            overview.updates_total = int(row[0])
        row = _one(
            conn,
            "SELECT global_update_index, rollout_policy_no_op_count, "
            "rollout_policy_left_click_count, rollout_policy_right_click_count "
            "FROM ppo_updates ORDER BY update_id DESC LIMIT 1",
        )
        if row:
            overview.last_update_index = (
                int(row[0]) if row[0] is not None else None
            )
            counts = [value if value is not None else 0 for value in row[1:4]]
            total = sum(counts)
            if total > 0:
                overview.right_click_share_last = counts[2] / total
        row = _one(
            conn,
            "SELECT mean_reward, policy_version FROM eval_runs "
            "ORDER BY eval_id DESC LIMIT 1",
        )
        if row:
            overview.last_eval_mean = (
                float(row[0]) if row[0] is not None else None
            )
            overview.last_eval_policy_version = (
                int(row[1]) if row[1] is not None else None
            )
    finally:
        conn.close()
    return overview


def list_runs_overview(models_dir: str | Path) -> list[RunOverview]:
    """One row per run directory that has a DB or a checkpoint, newest
    DB activity first."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    overviews = []
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (
            (run_dir / "training_logs.db").exists()
            or (run_dir / "checkpoint.pth").exists()
        ):
            continue
        overviews.append(_overview_for(run_dir))
    overviews.sort(key=lambda o: o.db_modified_iso or "", reverse=True)
    return overviews


def runs_with_config(models_dir: str | Path) -> list[str]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(
        run_dir.name
        for run_dir in models_dir.iterdir()
        if run_dir.is_dir() and (run_dir / "effective_config.json").exists()
    )


def diff_run_configs(
    run_a: str,
    run_b: str,
    models_dir: str | Path,
) -> dict:
    """Key-level diff of two runs' effective_config.json."""
    flats = []
    for run_name in (run_a, run_b):
        config_path = Path(models_dir) / run_name / "effective_config.json"
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"error": f"cannot read effective_config.json for {run_name}"}
        flats.append(_flatten_config(raw))
    flat_a, flat_b = flats
    changed = {
        key: (flat_a[key], flat_b[key])
        for key in sorted(set(flat_a) & set(flat_b))
        if flat_a[key] != flat_b[key]
    }
    return {
        "changed": changed,
        "only_a": sorted(set(flat_a) - set(flat_b)),
        "only_b": sorted(set(flat_b) - set(flat_a)),
    }


def build_launch_command(
    run_name: str,
    num_actors: int | None = 10,
    max_updates: int | None = None,
    fragment_steps: int | None = None,
    global_rollout_steps: int | None = None,
    config_path: str | None = None,
    local_mode: bool = False,
) -> str:
    """The exact ray_train invocation for the user's terminal. Flags
    mirror distributed/ray_train.py:_parse_args; omitted flags fall back
    to config.yaml exactly as the trainer itself would."""
    parts = ["python", "-m", "distributed.ray_train"]
    if num_actors is not None:
        parts += ["--num-actors", str(int(num_actors))]
    if run_name:
        parts += ["--run-name", run_name]
    if max_updates is not None:
        parts += ["--max-updates", str(int(max_updates))]
    if fragment_steps is not None:
        parts += ["--fragment-steps", str(int(fragment_steps))]
    if global_rollout_steps is not None:
        parts += ["--global-rollout-steps", str(int(global_rollout_steps))]
    if config_path:
        quoted = f'"{config_path}"' if " " in config_path else config_path
        parts += ["--config", quoted]
    if local_mode:
        parts.append("--local-mode")
    return " ".join(parts)
