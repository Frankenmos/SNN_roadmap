"""Unit tests for the snapshot schedule (Utility.checkpoint_snapshots)
and the .pth registry (tools.registry) against synthetic checkpoints."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from tools.registry.core import (
    _json_sanitize,
    diff_entries,
    export_run_data,
    fork_run_lineage,
    list_run_entries,
    resolve_ref,
    show_entry,
    tag_ref,
    verify_run_registry,
    write_run_data_json,
)
from Utility.checkpoint_snapshots import (
    SnapshotSchedule,
    build_snapshot_payload,
    save_policy_snapshot,
    snapshot_path,
)
from Utility.logger_utils import initialize_db
from Utility.model_git import sha256_file


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------
def test_schedule_disabled_by_default():
    schedule = SnapshotSchedule()
    assert not schedule.enabled
    assert not any(schedule.is_due(version) for version in range(0, 500))


def test_schedule_two_tier_boundaries():
    schedule = SnapshotSchedule(dense_every=5, dense_until=200, sparse_every=25)
    assert schedule.enabled
    assert not schedule.is_due(0)
    assert not schedule.is_due(3)
    assert schedule.is_due(5)
    assert schedule.is_due(200)  # last dense-tier snapshot
    assert not schedule.is_due(205)  # dense cadence stops after dense_until
    assert not schedule.is_due(210)
    assert schedule.is_due(225)  # sparse tier
    assert schedule.is_due(250)
    assert not schedule.is_due(251)


def test_schedule_single_tier_variants():
    dense_only = SnapshotSchedule(dense_every=10, dense_until=100, sparse_every=0)
    assert dense_only.is_due(100)
    assert not dense_only.is_due(110)

    sparse_only = SnapshotSchedule(dense_every=0, dense_until=0, sparse_every=25)
    assert sparse_only.is_due(25)
    assert sparse_only.is_due(50)
    assert not sparse_only.is_due(30)


def test_schedule_from_config_getter():
    values = {
        "snapshot_dense_every_updates": 5,
        "snapshot_dense_until_update": 200,
        "snapshot_sparse_every_updates": 25,
    }
    schedule = SnapshotSchedule.from_distributed_config(
        lambda name, fallback: values.get(name, fallback),
    )
    assert (schedule.dense_every, schedule.dense_until, schedule.sparse_every) == (
        5,
        200,
        25,
    )
    absent = SnapshotSchedule.from_distributed_config(
        lambda name, fallback: fallback,
    )
    assert not absent.enabled


# ----------------------------------------------------------------------
# Synthetic agent / checkpoints
# ----------------------------------------------------------------------
def _make_policy(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    policy = torch.nn.Sequential(
        OrderedDict(
            [
                ("trunk", torch.nn.Linear(4, 8)),
                ("actor_head", torch.nn.Linear(8, 3)),
            ],
        ),
    )
    policy.register_parameter(
        "fast_alpha", torch.nn.Parameter(torch.rand(8)),
    )
    # Raw value outside [0, 1]: the exporter must report the clamped
    # effective value alongside it (snnTorch clamps in the forward).
    policy.register_parameter(
        "slow_beta", torch.nn.Parameter(torch.full((4,), 1.2)),
    )
    return policy


def _extractor_state() -> dict:
    return {
        "entity_normalizer": {
            "count": 42.0,
            "mean": [0.0] * 15,
            "m2": [1.0] * 15,
        },
    }


def _make_agent(seed: int = 0, update_count: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        policy=_make_policy(seed),
        extractor=SimpleNamespace(state_dict=_extractor_state),
        ppo=SimpleNamespace(update_count=update_count),
    )


def _write_run(tmp_path, run_name: str, versions: list[int], seed: int = 0):
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for version in versions:
        agent = _make_agent(seed=seed, update_count=version)
        path = save_policy_snapshot(
            agent, run_dir=run_dir, episode=version * 10, run_name=run_name,
        )
        assert path is not None and path.exists()
    return run_dir


# ----------------------------------------------------------------------
# Snapshot payload / writer
# ----------------------------------------------------------------------
def test_payload_has_no_optimizer_and_mandatory_extractor(tmp_path):
    agent = _make_agent()
    payload = build_snapshot_payload(
        agent, episode=70, run_dir=tmp_path, run_name="test_run",
    )
    assert "optimizer_state" not in payload
    assert "scheduler_state" not in payload
    assert payload["policy_version"] == 7
    assert payload["episode"] == 70
    assert payload["extractor_state"]["entity_normalizer"]["count"] == 42.0
    assert set(payload["agent_state"]) == set(agent.policy.state_dict())

    empty_extractor = SimpleNamespace(state_dict=dict)
    broken = SimpleNamespace(
        policy=agent.policy, extractor=empty_extractor, ppo=agent.ppo,
    )
    with pytest.raises(ValueError, match="extractor"):
        build_snapshot_payload(broken, episode=1, run_dir=tmp_path)


def test_save_snapshot_writes_versioned_file(tmp_path):
    run_dir = _write_run(tmp_path, "run_a", versions=[5])
    path = snapshot_path(run_dir, 5)
    assert path.exists()
    assert path.name == "policy_u5.pth"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ckpt["policy_version"] == 5
    assert "agent_state" in ckpt and "extractor_state" in ckpt


def test_save_snapshot_never_raises(tmp_path, capsys):
    broken = SimpleNamespace(
        policy=_make_policy(),
        extractor=SimpleNamespace(state_dict=dict),  # empty -> ValueError
        ppo=SimpleNamespace(update_count=3),
    )
    assert save_policy_snapshot(broken, run_dir=tmp_path, episode=1) is None
    assert "snapshot failed" in capsys.readouterr().out


def test_snapshot_refuses_to_misattribute_a_missing_phase(tmp_path):
    (tmp_path / "resume_events.jsonl").write_text(
        json.dumps({"phase_id": 7, "source": {"git_commit": "abc"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No resume event for phase_id=8"):
        build_snapshot_payload(
            _make_agent(),
            episode=1,
            run_dir=tmp_path,
            run_name=tmp_path.name,
            phase_id=8,
        )


# ----------------------------------------------------------------------
# Registry: list / resolve / show
# ----------------------------------------------------------------------
def test_list_entries_and_eval_join(tmp_path):
    run_dir = _write_run(tmp_path, "run_a", versions=[5, 10])
    torch.save(
        {"agent_state": _make_policy().state_dict(), "policy_version": 12},
        run_dir / "checkpoint.pth",
    )
    conn = initialize_db(str(run_dir / "training_logs.db"))
    with conn:
        conn.executemany(
            "INSERT INTO eval_runs (episode_index, num_episodes, mean_reward, "
            "std_reward, min_reward, max_reward, deterministic, policy_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (50, 4, 12.0, 1.0, 10.0, 14.0, 1, 5),
                (90, 4, 20.0, 1.0, 18.0, 22.0, 1, 9),
            ],
        )
    conn.close()

    entries = list_run_entries("run_a", models_dir=tmp_path)
    snapshots = {entry.policy_version: entry for entry in entries if entry.kind == "snapshot"}
    by_name = {entry.name: entry for entry in entries}
    assert set(snapshots) == {5, 10}
    assert "checkpoint.pth" in by_name

    snap5 = snapshots[5]
    assert snap5.kind == "snapshot"
    assert snap5.policy_version == 5
    assert snap5.eval_mean == 12.0 and snap5.eval_policy_version == 5

    snap10 = snapshots[10]
    # nearest eval at or before version 10 is the one at version 9
    assert snap10.eval_mean == 20.0 and snap10.eval_policy_version == 9

    assert by_name["checkpoint.pth"].kind == "checkpoint"
    assert by_name["checkpoint.pth"].policy_version == 12


def test_resolve_ref_forms(tmp_path):
    run_dir = _write_run(tmp_path, "run_a", versions=[5, 10])
    torch.save({"agent_state": {}}, run_dir / "checkpoint.pth")

    u5 = resolve_ref("run_a:u5", models_dir=tmp_path)
    latest = resolve_ref("run_a:latest", models_dir=tmp_path)
    assert torch.load(u5, map_location="cpu", weights_only=False)["policy_version"] == 5
    assert torch.load(latest, map_location="cpu", weights_only=False)["policy_version"] == 10
    assert resolve_ref("run_a", models_dir=tmp_path) == latest
    assert (
        resolve_ref("run_a:checkpoint", models_dir=tmp_path).name
        == "checkpoint.pth"
    )
    direct = run_dir / "snapshots" / "policy_u5.pth"
    assert resolve_ref(str(direct), models_dir=tmp_path) == direct

    with pytest.raises(ValueError):
        resolve_ref("run_a:bogus", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_ref("no_such_run", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_ref("run_a:best", models_dir=tmp_path)  # no best file written


def test_model_git_hash_tags_forks_and_integrity(tmp_path):
    parent_dir = _write_run(tmp_path, "parent", versions=[5])
    parent_path = resolve_ref("parent:u5", models_dir=tmp_path)
    parent_sha = parent_path.stem

    tag_ref("parent", "stable", "parent:u5", models_dir=tmp_path)
    assert resolve_ref("parent:tag/stable", models_dir=tmp_path) == parent_path
    torch.save({"agent_state": {}}, parent_dir / "checkpoint.pth")
    with pytest.raises(ValueError, match="immutable Model Git object"):
        fork_run_lineage("bad_child", "parent:checkpoint", models_dir=tmp_path)

    fork_path = fork_run_lineage("child", "parent:u5", models_dir=tmp_path)
    assert fork_path.exists()
    _write_run(tmp_path, "child", versions=[1], seed=1)
    child_entry = next(
        entry for entry in list_run_entries("child", models_dir=tmp_path)
        if entry.kind == "snapshot"
    )
    assert child_entry.parent_sha256 == parent_sha

    verified = verify_run_registry("child", models_dir=tmp_path)
    assert verified == {
        "ok": True,
        "objects": 1,
        "index_events": 1,
        "errors": [],
    }

    # Corruption is detected against the filename/content digest contract.
    child_entry.path.write_bytes(child_entry.path.read_bytes() + b"tamper")
    broken = verify_run_registry("child", models_dir=tmp_path)
    assert not broken["ok"]
    assert any("object hash mismatch" in error for error in broken["errors"])


def test_same_update_divergence_preserves_compat_ref_and_full_identity(tmp_path):
    run_dir = tmp_path / "forked_run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_name": "forked_run"}),
        encoding="utf-8",
    )
    (run_dir / "effective_config.json").write_text(
        json.dumps({"model": {"attention_embed_dim": 8}}),
        encoding="utf-8",
    )
    (run_dir / "resume_events.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "phase_id": phase,
                    "effective_config_sha256": f"effective-{phase}",
                    "source": {"git_commit": f"commit-{phase}", "git_dirty": False},
                },
            )
            for phase in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )

    first = save_policy_snapshot(
        _make_agent(seed=0, update_count=5),
        run_dir=run_dir,
        episode=50,
        run_name="forked_run",
        phase_id=1,
    )
    first_compat_sha = sha256_file(first)
    first_index_bytes = (run_dir / "model_git" / "index.jsonl").read_bytes()
    checkpoint_path = run_dir / "checkpoint.pth"
    torch.save({"resume_only": True}, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    second = save_policy_snapshot(
        _make_agent(seed=1, update_count=5),
        run_dir=run_dir,
        episode=55,
        run_name="forked_run",
        phase_id=2,
    )

    assert second == first
    assert sha256_file(first) == first_compat_sha
    assert sha256_file(checkpoint_path) == checkpoint_sha
    entries = [
        entry for entry in list_run_entries("forked_run", models_dir=tmp_path)
        if entry.kind == "snapshot"
    ]
    assert len(entries) == 2
    assert entries[0].artifact_sha256 != entries[1].artifact_sha256
    assert entries[1].parent_sha256 == entries[0].artifact_sha256
    assert resolve_ref("forked_run:u5", models_dir=tmp_path) == entries[1].path
    assert entries[0].metadata["phase_id"] == 1
    assert entries[1].metadata["phase_id"] == 2

    index_rows = [
        json.loads(line)
        for line in (run_dir / "model_git" / "index.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert (run_dir / "model_git" / "index.jsonl").read_bytes().startswith(
        first_index_bytes,
    )
    assert index_rows[0]["source_identity"]["git_commit"] == "commit-1"
    assert index_rows[1]["source_identity"]["git_commit"] == "commit-2"
    assert index_rows[0]["phase_effective_config_sha256"] == "effective-1"
    assert index_rows[1]["phase_effective_config_sha256"] == "effective-2"
    assert index_rows[0]["config_sha256"]
    assert index_rows[0]["run_manifest_sha256"]
    assert verify_run_registry("forked_run", models_dir=tmp_path)["ok"] is True


def test_show_entry_param_counts_and_time_constants(tmp_path):
    run_dir = _write_run(tmp_path, "run_a", versions=[5])
    info = show_entry(snapshot_path(run_dir, 5))
    counts = info["module_param_counts"]
    assert counts["trunk"] == 4 * 8 + 8
    assert counts["actor_head"] == 8 * 3 + 3
    assert counts["fast_alpha"] == 8
    assert info["metadata"]["policy_version"] == 5
    assert info["metadata"]["has_extractor_state"] is True
    assert info["metadata"]["has_optimizer_state"] is False
    alpha_rows = [row for row in info["time_constants"] if row["kind"] == "alpha"]
    assert len(alpha_rows) == 1 and alpha_rows[0]["numel"] == 8


# ----------------------------------------------------------------------
# Registry: diff
# ----------------------------------------------------------------------
def test_diff_known_delta_and_config_diff(tmp_path):
    run_a = _write_run(tmp_path, "run_a", versions=[5], seed=0)
    run_b = tmp_path / "run_b"
    run_b.mkdir()

    # run_b snapshot = run_a weights with trunk.weight shifted by exactly +1
    # and one extra tensor, so every diff channel has a known expectation.
    base = torch.load(
        str(snapshot_path(run_a, 5)), map_location="cpu", weights_only=False,
    )
    shifted = {
        name: tensor.clone() for name, tensor in base["agent_state"].items()
    }
    shifted["trunk.weight"] = shifted["trunk.weight"] + 1.0
    shifted["extra.weight"] = torch.zeros(2, 2)
    modified = dict(base)
    modified["agent_state"] = shifted
    modified["policy_version"] = 6
    (run_b / "snapshots").mkdir()
    torch.save(modified, run_b / "snapshots" / "policy_u6.pth")

    for run_dir, lr in ((run_a, 5e-5), (run_b, 1e-4)):
        (run_dir / "effective_config.json").write_text(
            json.dumps({"ppo": {"lr": lr}, "reward": {"name": "v4"}}),
            encoding="utf-8",
        )

    diff = diff_entries(
        snapshot_path(run_a, 5), run_b / "snapshots" / "policy_u6.pth",
    )

    rows = {row["name"]: row for row in diff["layers"]}
    trunk = rows["trunk.weight"]
    assert trunk["l2"] == pytest.approx(math.sqrt(trunk["numel"]), rel=1e-5)
    assert diff["layers"][0]["name"] == "trunk.weight"  # sorted by L2 desc
    for name, row in rows.items():
        if name != "trunk.weight":
            assert row["l2"] == 0.0
            assert row["cosine"] == pytest.approx(1.0, abs=1e-6)

    assert diff["only_in_b"] == ["extra.weight"]
    assert diff["only_in_a"] == []
    assert diff["metadata_diff"]["policy_version"] == (5, 6)
    assert diff["config_diff"]["changed"] == {"ppo.lr": (5e-5, 1e-4)}
    assert diff["total_l2"] == pytest.approx(trunk["l2"], rel=1e-6)


# ----------------------------------------------------------------------
# Registry: export (arch-explorer live bundle)
# ----------------------------------------------------------------------
def _insert_ppo_updates(run_dir, count: int) -> None:
    conn = initialize_db(str(run_dir / "training_logs.db"))
    with conn:
        conn.executemany(
            "INSERT INTO ppo_updates (episode_index, global_update_index, "
            "mean_entropy, grad_norm, rollout_policy_no_op_count, "
            "rollout_policy_left_click_count, "
            "rollout_policy_right_click_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (i * 10, i, 1.0 - i * 0.01, 2.0 - i * 0.05, 3000 - i * 100,
                 0, 100 + i * 50)
                for i in range(1, count + 1)
            ],
        )
        conn.executemany(
            "INSERT INTO eval_runs (episode_index, num_episodes, mean_reward, "
            "std_reward, min_reward, max_reward, deterministic, "
            "policy_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (50, 4, 12.0, 1.0, 10.0, 14.0, 1, 5),
                (90, 4, 20.0, 1.0, 18.0, 22.0, 1, 9),
            ],
        )
    conn.close()


def test_export_run_data_bundle(tmp_path):
    run_dir = _write_run(tmp_path, "run_a", versions=[5, 10])
    torch.save(
        {"agent_state": _make_policy().state_dict(), "policy_version": 12},
        run_dir / "checkpoint.pth",
    )
    (run_dir / "effective_config.json").write_text(
        json.dumps(
            {
                "reward": {"name": "defeat_roaches_v4"},
                "ppo": {"lr": 5e-5, "sil_enabled": True},
                "model": {
                    "spatial_head_type": "coarse_to_fine",
                    "fast_token_snn_alpha": 0.55,
                    "fast_token_snn_beta": 0.65,
                    "slow_token_snn_alpha": 0.92,
                    "slow_token_snn_beta": 0.97,
                },
            },
        ),
        encoding="utf-8",
    )
    _insert_ppo_updates(run_dir, count=12)

    data = export_run_data("run_a", models_dir=tmp_path, max_points=5)

    assert data["schema_version"] == 1
    assert data["kind"] == "arch-explorer-run-data"
    assert data["run"] == "run_a"
    assert data["config"]["reward_name"] == "defeat_roaches_v4"
    assert data["config"]["sil_enabled"] is True
    assert data["config"]["snn_init"]["slow_beta"] == 0.97
    assert data["module_param_counts"]["trunk"] == 4 * 8 + 8

    refs = [entry["ref"] for entry in data["entries"]]
    assert refs == ["run_a:u5", "run_a:u10", "run_a:checkpoint"]

    by_ref = {entry["ref"]: entry for entry in data["entries"]}
    # exact update-row join at each artifact's policy version
    assert by_ref["run_a:u5"]["update_row"]["global_update_index"] == 5
    assert by_ref["run_a:u5"]["update_row"][
        "rollout_policy_right_click_count"
    ] == 100 + 5 * 50
    assert by_ref["run_a:checkpoint"]["update_row"][
        "global_update_index"
    ] == 12
    # eval join: nearest at-or-before version
    assert by_ref["run_a:u10"]["eval_mean"] == 20.0
    assert by_ref["run_a:u10"]["eval_policy_version"] == 9

    # clamped effective value next to the raw one
    constants = {
        row["name"]: row for row in by_ref["run_a:u5"]["time_constants"]
    }
    assert constants["slow_beta"]["mean"] == pytest.approx(1.2)
    assert constants["slow_beta"]["effective_mean"] == pytest.approx(1.0)
    assert constants["fast_alpha"]["effective_mean"] <= 1.0

    history = data["history"]
    assert history["total_updates"] == 12
    assert history["stride"] == 3  # ceil(12 / 5)
    # strided rows plus the guaranteed final row
    assert history["series"]["global_update_index"] == [1, 4, 7, 10, 12]

    assert len(data["evals"]) == 2
    assert data["evals"][-1]["mean_reward"] == 20.0

    out_path = tmp_path / "out" / "run_data.json"
    written = write_run_data_json(
        "run_a", out_path=out_path, models_dir=tmp_path,
    )
    assert written == out_path
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["entries"][0]["ref"] == "run_a:u5"


def test_export_without_db_or_config(tmp_path):
    _write_run(tmp_path, "run_a", versions=[5])
    data = export_run_data("run_a", models_dir=tmp_path)
    assert data["history"] is None
    assert data["evals"] == []
    assert data["config"] == {}
    assert data["entries"][0]["update_row"] is None
    assert data["entries"][0]["eval_mean"] is None
    # must serialize to strict JSON even with everything missing
    json.dumps(data, allow_nan=False)


def test_json_sanitize_strips_nonfinite():
    dirty = {
        "a": float("nan"),
        "b": [1.0, float("inf"), float("-inf")],
        "c": {"d": 2.5},
    }
    assert _json_sanitize(dirty) == {
        "a": None,
        "b": [1.0, None, None],
        "c": {"d": 2.5},
    }
