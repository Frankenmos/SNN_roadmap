from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from envs.tiny_skirmish.real_snn_bridge import (
    build_real_policy,
    import_real_snn_modules,
)
from tools.representation_migration import (
    compare_artifacts,
    compare_replays,
    create_probe_bank_from_spec,
    create_tiny_probe_bank,
    linear_cka,
    load_policy_artifact,
    load_probe_bank,
    replay_policy,
    timeline_report,
)
from tools.representation_migration import (
    main as representation_main,
)
from Utility.checkpoint_snapshots import save_policy_snapshot


def _small_policy(seed: int):
    torch.manual_seed(seed)
    return build_real_policy(
        import_real_snn_modules(),
        device=torch.device("cpu"),
        small=True,
    )


def test_fixed_tiny_probe_bank_is_reproducible_and_replays(tmp_path):
    first = create_tiny_probe_bank(
        tmp_path / "first.pt",
        seeds=[3, 7],
        steps_per_seed=3,
    )
    second = create_tiny_probe_bank(
        tmp_path / "second.pt",
        seeds=[3, 7],
        steps_per_seed=3,
    )
    bank_a = load_probe_bank(first)
    bank_b = load_probe_bank(second)

    assert bank_a["probe_bank_sha256"] == bank_b["probe_bank_sha256"]
    assert len(bank_a["samples"]) == 6

    replay = replay_policy(_small_policy(0), bank_a)
    assert replay["latent"].shape[0] == 6
    assert replay["action_logits"].shape == (6, 3)
    assert replay["snn_syn"].shape[0] == 6
    assert linear_cka(replay["latent"], replay["latent"]) > 0.999999


def test_versioned_probe_spec_regenerates_expected_digest(tmp_path):
    spec_path = Path(__file__).resolve().parents[1] / "probes" / "tiny_skirmish_v1.json"
    bank_path = create_probe_bank_from_spec(spec_path, tmp_path / "bank.pt")
    bank = load_probe_bank(bank_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert bank["probe_bank_sha256"] == spec["expected_probe_bank_sha256"]
    assert len(bank["samples"]) == len(spec["seeds"]) * spec["steps_per_seed"]

    bank["samples"][0]["meta_vec"][0] += 1.0
    torch.save(bank, bank_path)
    try:
        load_probe_bank(bank_path)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered probe bank was accepted")


def test_representation_comparison_detects_identical_and_changed_outputs(tmp_path):
    bank = load_probe_bank(
        create_tiny_probe_bank(tmp_path / "bank.pt", seeds=[1, 2], steps_per_seed=3),
    )
    policy_a = _small_policy(0)
    policy_b = _small_policy(0)
    same = compare_replays(replay_policy(policy_a, bank), replay_policy(policy_b, bank))
    assert same["action"]["argmax_agreement"] == 1.0
    assert same["action"]["rmse"] == 0.0
    assert same["latent"]["cka"] > 0.999999

    with torch.no_grad():
        policy_b.actor_fc.bias.add_(torch.tensor([2.0, -1.0, -1.0]))
    changed = compare_replays(replay_policy(policy_a, bank), replay_policy(policy_b, bank))
    assert changed["action"]["rmse"] > 0.0
    assert changed["action"]["mean_js_divergence"] > 0.0
    assert changed["target_primary"]["distribution"]["compatible"] is True

    changed_shape = dict(replay_policy(policy_b, bank))
    changed_shape["latent"] = torch.cat(
        (changed_shape["latent"], torch.zeros(changed_shape["latent"].shape[0], 1)),
        dim=1,
    )
    changed_shape["target_primary"] = changed_shape["target_primary"][:, :-1]
    cross_shape = compare_replays(replay_policy(policy_a, bank), changed_shape)
    assert cross_shape["latent"]["compatible"] is False
    assert cross_shape["latent"]["cka"] is not None
    assert cross_shape["target_primary"]["distribution"]["compatible"] is False


def test_saved_extractor_state_participates_in_probe_replay(tmp_path):
    bank = load_probe_bank(
        create_tiny_probe_bank(tmp_path / "bank.pt", seeds=[8], steps_per_seed=2),
    )
    policy = _small_policy(0)
    raw = replay_policy(policy, bank)
    normalized = replay_policy(
        policy,
        bank,
        extractor_state={
            "entity_normalizer": {
                "count": 100.0,
                "mean": [10.0] * 15,
                "m2": [99.0] * 15,
            },
            "selection_normalizer": {
                "count": 100.0,
                "mean": [10.0] * 5,
                "m2": [99.0] * 5,
            },
        },
    )
    assert not torch.allclose(raw["latent"], normalized["latent"])


def test_saved_artifact_pair_report_includes_weight_and_snn_metrics(tmp_path):
    bank = load_probe_bank(
        create_tiny_probe_bank(tmp_path / "bank.pt", seeds=[4], steps_per_seed=3),
    )
    policy_a = _small_policy(0)
    policy_b = _small_policy(0)
    with torch.no_grad():
        policy_b.actor_fc.bias[0] += 0.5
    path_a = tmp_path / "a.pth"
    path_b = tmp_path / "b.pth"
    torch.save({"agent_state": policy_a.state_dict(), "policy_version": 1}, path_a)
    torch.save({"agent_state": policy_b.state_dict(), "policy_version": 2}, path_b)

    def loader(path):
        policy = _small_policy(99)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["agent_state"])
        return policy

    report = compare_artifacts(path_a, path_b, bank, policy_loader=loader)

    assert report["weights"]["total_l2"] == 0.5
    assert report["representations"]["action"]["rmse"] > 0.0
    assert report["representations"]["snn_syn"]["compatible"] is True
    assert report["representations"]["target_primary"]["compatible"] is True
    assert report["representations"]["target_primary"]["distribution"]["compatible"] is True
    json.dumps(report, allow_nan=False)

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, list | tuple):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(report)


def test_snapshot_embeds_constructor_config_and_timeline_uses_real_loader(tmp_path):
    bank_path = create_tiny_probe_bank(
        tmp_path / "bank.pt",
        seeds=[5],
        steps_per_seed=2,
    )
    bank = load_probe_bank(bank_path)
    run_dir = tmp_path / "timeline_run"
    run_dir.mkdir()

    for version, seed in ((1, 0), (2, 1)):
        policy = _small_policy(seed)
        agent = SimpleNamespace(
            policy=policy,
            extractor=SimpleNamespace(
                state_dict=lambda: {"probe_passthrough": True},
            ),
            ppo=SimpleNamespace(update_count=version),
        )
        assert save_policy_snapshot(
            agent,
            run_dir=run_dir,
            episode=version,
            run_name="timeline_run",
            phase_id=3,
        ) is not None

    first_object = next((run_dir / "model_git" / "objects").glob("*.pth"))
    payload = torch.load(first_object, map_location="cpu", weights_only=False)
    assert payload["policy_config"]["spatial_input_shape"] == [27, 84, 84]
    assert payload["policy_config"]["attention_embed_dim"] == 32
    loaded = load_policy_artifact(first_object)
    assert loaded.resolved_config()["attention_embed_dim"] == 32

    report = timeline_report("timeline_run", bank, models_dir=tmp_path)
    assert report["kind"] == "representation-migration-timeline"
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["policy_version_a"] == 1
    assert report["pairs"][0]["policy_version_b"] == 2
    assert report["pairs"][0]["representations"]["latent"]["cka"] is not None

    cli_out = tmp_path / "pairwise.json"
    assert representation_main(
        [
            "--models-dir",
            str(tmp_path),
            "compare",
            "timeline_run:u1",
            "timeline_run:u2",
            "--probe",
            str(bank_path),
            "--out",
            str(cli_out),
        ],
    ) == 0
    cli_report = json.loads(cli_out.read_text(encoding="utf-8"))
    assert cli_report["kind"] == "representation-migration-report"
    assert cli_report["probe_bank"]["samples"] == 2
