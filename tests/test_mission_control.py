"""Unit tests for tools.analysis.mission_control (read-only helpers)."""

from __future__ import annotations

import json

import torch

from Utility.logger_utils import initialize_db
from tools.analysis.mission_control import (
    build_launch_command,
    diff_run_configs,
    list_runs_overview,
    runs_with_config,
)


def _make_run(tmp_path, name, *, with_db=True, config=None):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    torch.save({"agent_state": {}}, run_dir / "checkpoint.pth")
    if config is not None:
        (run_dir / "effective_config.json").write_text(
            json.dumps(config), encoding="utf-8",
        )
    if with_db:
        conn = initialize_db(str(run_dir / "training_logs.db"))
        with conn:
            conn.executemany(
                "INSERT INTO ppo_updates (global_update_index, "
                "rollout_policy_no_op_count, rollout_policy_left_click_count, "
                "rollout_policy_right_click_count) VALUES (?, ?, ?, ?)",
                [(i, 60, 0, 40) for i in range(1, 6)],
            )
            conn.execute(
                "INSERT INTO eval_runs (episode_index, num_episodes, "
                "mean_reward, std_reward, min_reward, max_reward, "
                "deterministic, policy_version) "
                "VALUES (100, 5, 17.5, 1.0, 15.0, 20.0, 1, 4)",
            )
        conn.close()
    return run_dir


def test_list_runs_overview(tmp_path):
    run_dir = _make_run(tmp_path, "run_a")
    (run_dir / "snapshots").mkdir()
    (run_dir / "snapshots" / "policy_u5.pth").write_bytes(b"x")
    (run_dir / "snapshots" / "policy_u10.pth").write_bytes(b"x")
    (run_dir / "snapshots" / "notes.txt").write_bytes(b"x")  # ignored
    _make_run(tmp_path, "run_b", with_db=False)
    (tmp_path / "not_a_run").mkdir()  # no DB, no checkpoint -> skipped

    overviews = {o.name: o for o in list_runs_overview(tmp_path)}
    assert set(overviews) == {"run_a", "run_b"}

    run_a = overviews["run_a"]
    assert run_a.updates_total == 5
    assert run_a.last_update_index == 5
    assert run_a.right_click_share_last == 0.4
    assert run_a.last_eval_mean == 17.5
    assert run_a.last_eval_policy_version == 4
    assert run_a.snapshot_count == 2
    assert run_a.has_checkpoint is True
    assert run_a.db_modified_iso is not None

    run_b = overviews["run_b"]
    assert run_b.updates_total is None
    assert run_b.snapshot_count == 0
    assert run_b.has_checkpoint is True


def test_config_diff_and_listing(tmp_path):
    _make_run(
        tmp_path,
        "run_a",
        with_db=False,
        config={"ppo": {"lr": 5e-5, "sil_enabled": False}},
    )
    _make_run(
        tmp_path,
        "run_b",
        with_db=False,
        config={"ppo": {"lr": 5e-5, "sil_enabled": True, "sil_coef": 0.5}},
    )
    _make_run(tmp_path, "run_c", with_db=False)  # no config

    assert runs_with_config(tmp_path) == ["run_a", "run_b"]

    diff = diff_run_configs("run_a", "run_b", tmp_path)
    assert diff["changed"] == {"ppo.sil_enabled": (False, True)}
    assert diff["only_a"] == []
    assert diff["only_b"] == ["ppo.sil_coef"]

    missing = diff_run_configs("run_a", "run_c", tmp_path)
    assert "error" in missing


def test_build_launch_command():
    assert build_launch_command("my_run") == (
        "python -m distributed.ray_train --num-actors 10 --run-name my_run"
    )
    full = build_launch_command(
        "my_run",
        num_actors=8,
        max_updates=500,
        fragment_steps=64,
        global_rollout_steps=2048,
        config_path="alt config.yaml",
        local_mode=True,
        resume_weights_only=True,
    )
    assert full == (
        "python -m distributed.ray_train --num-actors 8 --run-name my_run "
        "--max-updates 500 --fragment-steps 64 --global-rollout-steps 2048 "
        '--config "alt config.yaml" --local-mode --resume-weights-only'
    )
    # blank run name / no actors -> flags omitted entirely
    assert build_launch_command("", num_actors=None) == (
        "python -m distributed.ray_train"
    )
