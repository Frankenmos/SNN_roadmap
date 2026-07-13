from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import Utility.run_manifest as run_manifest


def _payload(run_name="run_a", lr=5e-5):
    return {
        "schema_version": run_manifest.RUN_MANIFEST_SCHEMA_VERSION,
        "kind": "snn-run-manifest",
        "run_name": run_name,
        "effective_config": {"ppo": {"lr": lr}},
    }


def test_manifest_is_written_once_and_never_overwritten(tmp_path):
    path, created = run_manifest.ensure_run_manifest(tmp_path, _payload(lr=5e-5))
    assert created is True
    original = path.read_bytes()

    same_path, created = run_manifest.ensure_run_manifest(
        tmp_path,
        _payload(lr=1e-4),
    )
    assert same_path == path
    assert created is False
    assert path.read_bytes() == original
    assert (
        json.loads(path.read_text(encoding="utf-8"))["effective_config"]["ppo"]["lr"]
        == 5e-5
    )


def test_existing_manifest_rejects_run_name_mismatch(tmp_path):
    run_manifest.ensure_run_manifest(tmp_path, _payload(run_name="run_a"))
    with pytest.raises(run_manifest.RunManifestError, match="name mismatch"):
        run_manifest.ensure_run_manifest(tmp_path, _payload(run_name="run_b"))


def test_resume_events_are_append_only_and_self_contained(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_manifest,
        "collect_source_state",
        lambda _root: {"git_commit": "abc", "git_dirty": False},
    )
    monkeypatch.setattr(
        run_manifest,
        "collect_runtime_state",
        lambda: {"python_version": "test"},
    )
    config_path = tmp_path / "source.yaml"
    config_path.write_text("answer: 42\n", encoding="utf-8")

    for event_type, lr in (("start", 5e-5), ("resume", 1e-4)):
        run_manifest.append_resume_event(
            tmp_path,
            event_type=event_type,
            effective_config={"ppo": {"lr": lr}},
            launch_mode="ray",
            repo_root=tmp_path,
            config_path=config_path,
            checkpoint_present=event_type == "resume",
            phase_id=0 if event_type == "start" else 1,
            argv=["train", "--api-token", "do-not-store"],
            resolved_launch={"num_rollout_actors": 10},
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / run_manifest.RESUME_EVENTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in rows] == ["start", "resume"]
    assert [row["phase_id"] for row in rows] == [0, 1]
    assert rows[0]["effective_config"]["ppo"]["lr"] == 5e-5
    assert rows[1]["effective_config"]["ppo"]["lr"] == 1e-4
    assert rows[1]["checkpoint_present"] is True
    assert rows[0]["launch"]["argv"][-1] == "<redacted>"


def test_manifest_payload_records_resolved_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_manifest,
        "collect_source_state",
        lambda _root: {"git_commit": "abc", "git_dirty": True},
    )
    monkeypatch.setattr(
        run_manifest,
        "collect_runtime_state",
        lambda: {"python_version": "test"},
    )
    config_path = tmp_path / "source.yaml"
    config_path.write_text("ppo:\n  lr: 0.1\n", encoding="utf-8")
    effective = {"run_name": "run_a", "distributed": {"num_rollout_actors": 7}}

    payload = run_manifest.build_manifest_payload(
        run_name="run_a",
        effective_config=effective,
        launch_mode="ray",
        repo_root=tmp_path,
        config_path=config_path,
        argv=["python", "-m", "distributed.ray_train"],
        resolved_launch={"num_rollout_actors": 7},
    )

    assert payload["run_name"] == "run_a"
    assert payload["launch"]["resolved"]["num_rollout_actors"] == 7
    assert payload["effective_config_sha256"] == run_manifest.json_sha256(effective)
    assert payload["config_source"]["sha256"] == run_manifest.file_sha256(config_path)
    assert payload["source"]["git_commit"] == "abc"


def test_effective_config_uses_resolved_distributed_overrides(monkeypatch):
    import train

    fake_cfg = SimpleNamespace(
        environment=SimpleNamespace(
            run_name="run_a",
            map_name="DefeatRoaches",
            steps_per_episode=3600,
            total_episodes=10000,
            reward_window=200,
            eval_frequency=50,
            eval_episodes=5,
            checkpoint_path="checkpoint.pth",
            best_checkpoint_path="best_checkpoint.pth",
            db_path="training_logs.db",
        ),
        distributed=SimpleNamespace(
            items=lambda: {
                "num_rollout_actors": 10,
                "fragment_steps": 256,
            }.items(),
        ),
    )
    monkeypatch.setattr(train, "cfg", fake_cfg)
    agent = SimpleNamespace(
        policy=SimpleNamespace(resolved_config=lambda: {"kind": "policy"}),
        ppo=SimpleNamespace(resolved_config=lambda: {"lr": 5e-5}),
        reward_function=SimpleNamespace(
            resolved_config=lambda: {"name": "defeat_roaches_v4"},
        ),
        reward_scale=1.0,
        total_updates_estimate=100,
    )

    payload = train.build_effective_config(
        agent,
        distributed_overrides={
            "num_rollout_actors": 3,
            "fragment_steps": 64,
            "global_rollout_steps": 512,
        },
    )

    assert payload["distributed"]["num_rollout_actors"] == 3
    assert payload["distributed"]["fragment_steps"] == 64
    assert payload["distributed"]["global_rollout_steps"] == 512


def test_initial_config_copy_is_immutable(tmp_path, monkeypatch):
    import train

    source = tmp_path / "source.yaml"
    destination = tmp_path / "run" / "config.yaml"
    destination.parent.mkdir()
    source.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setattr(train, "cfg", SimpleNamespace(config_path=source))
    monkeypatch.setattr(train, "_run_path", lambda _name: str(destination))

    train.save_initial_config()
    source.write_text("version: 2\n", encoding="utf-8")
    train.save_initial_config()

    assert destination.read_text(encoding="utf-8") == "version: 1\n"


def test_training_provenance_preserves_birth_and_appends_resume(tmp_path, monkeypatch):
    import train

    run_dir = tmp_path / "models" / "run_a"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: a\n", encoding="utf-8")
    fake_cfg = SimpleNamespace(
        config_path=config_path,
        environment=SimpleNamespace(
            run_name="run_a",
            map_name="DefeatRoaches",
            steps_per_episode=3600,
            total_episodes=10000,
            reward_window=200,
            eval_frequency=50,
            eval_episodes=5,
            checkpoint_path="checkpoint.pth",
            best_checkpoint_path="best_checkpoint.pth",
            db_path="training_logs.db",
        ),
        distributed=SimpleNamespace(items=lambda: {"num_rollout_actors": 10}.items()),
    )
    lr = {"value": 5e-5}
    agent = SimpleNamespace(
        policy=SimpleNamespace(resolved_config=lambda: {"kind": "policy"}),
        ppo=SimpleNamespace(resolved_config=lambda: {"lr": lr["value"]}),
        reward_function=SimpleNamespace(
            resolved_config=lambda: {"name": "defeat_roaches_v4"},
        ),
        reward_scale=1.0,
        total_updates_estimate=100,
    )
    monkeypatch.setattr(train, "cfg", fake_cfg)
    monkeypatch.setattr(train, "_run_dir", lambda: str(run_dir))
    monkeypatch.setattr(train, "_run_path", lambda name: str(run_dir / name))
    monkeypatch.setattr(
        run_manifest,
        "collect_source_state",
        lambda _root: {"git_commit": "abc", "git_dirty": False},
    )
    monkeypatch.setattr(
        run_manifest,
        "collect_runtime_state",
        lambda: {"python_version": "test"},
    )

    train.initialize_run_provenance(
        agent,
        launch_mode="ray",
        resolved_launch={"num_rollout_actors": 3},
        distributed_overrides={"num_rollout_actors": 3},
        argv=["ray_train"],
    )
    birth = (run_dir / run_manifest.RUN_MANIFEST_FILENAME).read_bytes()

    (run_dir / "checkpoint.pth").write_bytes(b"checkpoint")
    lr["value"] = 1e-4
    train.initialize_run_provenance(
        agent,
        launch_mode="ray",
        resolved_launch={"num_rollout_actors": 5},
        distributed_overrides={"num_rollout_actors": 5},
        argv=["ray_train", "--num-actors", "5"],
    )

    assert (run_dir / run_manifest.RUN_MANIFEST_FILENAME).read_bytes() == birth
    events = [
        json.loads(line)
        for line in (run_dir / run_manifest.RESUME_EVENTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in events] == ["start", "resume"]
    assert events[0]["effective_config"]["ppo"]["lr"] == 5e-5
    assert events[1]["effective_config"]["ppo"]["lr"] == 1e-4
    assert events[1]["launch"]["resolved"]["num_rollout_actors"] == 5
