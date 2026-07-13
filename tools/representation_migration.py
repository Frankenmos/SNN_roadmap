"""Fixed-probe offline analysis of representation migration across snapshots.

Only compact, named tensors are retained (latent, recurrent state, and output
heads); this deliberately avoids dumping every intermediate activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from agent_core.policy_protocol import (
    META_AVAILABLE_ACTION_DIM,
    META_AVAILABLE_ACTION_OFFSET,
    POLICY_ACTION_RIGHT_CLICK,
    POLICY_INPUT_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    PolicyInputBatch,
)
from envs.tiny_skirmish.env import TinySkirmishEnv
from envs.tiny_skirmish.real_snn_bridge import (
    import_real_snn_modules,
    tiny_observation_to_real_batch,
)
from envs.tiny_skirmish.rollout import scripted_action
from tools.registry.core import (
    DEFAULT_MODELS_DIR,
    _run_dir_for,
    diff_entries,
    list_run_entries,
    resolve_ref,
)
from Utility.model_git import sha256_file

PROBE_BANK_SCHEMA_VERSION = 1


def _tensor_digest(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(f"{sample['sequence_id']}:{sample['step']}".encode())
        for name in (
            "spatial_obs",
            "entity_features",
            "entity_mask",
            "selection_features",
            "selection_mask",
            "action_feedback_tokens",
            "meta_vec",
        ):
            tensor = sample[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def create_tiny_probe_bank(
    path: str | Path,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3),
    steps_per_seed: int = 8,
) -> Path:
    """Create deterministic TinySkirmish observations with scripted actions."""
    seeds = [int(seed) for seed in seeds]
    samples = []
    real = import_real_snn_modules()
    for sequence_id, seed in enumerate(seeds):
        env = TinySkirmishEnv(seed=seed, max_steps=max(steps_per_seed, 1))
        observation = env.reset(seed=seed)
        for step in range(int(steps_per_seed)):
            batch = tiny_observation_to_real_batch(
                observation,
                real,
                device=torch.device("cpu"),
            )
            samples.append(
                {
                    "sequence_id": int(sequence_id),
                    "seed": int(seed),
                    "step": int(step),
                    "spatial_obs": batch.spatial_obs[0].cpu(),
                    "entity_features": batch.entity_features[0].cpu(),
                    "entity_mask": batch.entity_mask[0].cpu(),
                    "selection_features": batch.selection_features[0].cpu(),
                    "selection_mask": batch.selection_mask[0].cpu(),
                    "action_feedback_tokens": batch.action_feedback_tokens[0].cpu(),
                    "meta_vec": batch.meta_vec[0].cpu(),
                },
            )
            result = env.step(scripted_action(env))
            observation = result.observation
            if result.done or result.truncated:
                break
    payload = {
        "schema_version": PROBE_BANK_SCHEMA_VERSION,
        "kind": "snn-fixed-probe-bank",
        "name": "tiny_skirmish_scripted_v1",
        "source": "TinySkirmish deterministic scripted trajectories",
        "policy_protocol_version": POLICY_PROTOCOL_VERSION,
        "policy_input_schema": POLICY_INPUT_SCHEMA,
        "steps_per_seed": int(steps_per_seed),
        "seeds": seeds,
        "samples": samples,
    }
    payload["probe_bank_sha256"] = _tensor_digest(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def create_probe_bank_from_spec(spec_path: str | Path, out_path: str | Path) -> Path:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("kind") != "snn-probe-bank-spec":
        raise ValueError(f"Not a probe-bank spec: {spec_path}")
    path = create_tiny_probe_bank(
        out_path,
        seeds=spec["seeds"],
        steps_per_seed=int(spec["steps_per_seed"]),
    )
    bank = load_probe_bank(path)
    if bank["name"] != spec.get("name"):
        path.unlink(missing_ok=True)
        raise ValueError(f"Generated probe bank name differs from spec: {bank['name']}")
    expected = spec.get("expected_probe_bank_sha256")
    if expected and bank["probe_bank_sha256"] != expected:
        path.unlink(missing_ok=True)
        raise ValueError(
            "Generated probe bank differs from the versioned spec digest: "
            f"{bank['probe_bank_sha256']} != {expected}",
        )
    return path


def load_probe_bank(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != PROBE_BANK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported probe bank schema: {payload.get('schema_version')}")
    if payload.get("policy_input_schema") != POLICY_INPUT_SCHEMA:
        raise ValueError("Probe bank policy-input schema does not match this checkout")
    if int(payload.get("policy_protocol_version", -1)) != POLICY_PROTOCOL_VERSION:
        raise ValueError("Probe bank policy protocol does not match this checkout")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Probe bank contains no samples")
    ordering = [(int(row["sequence_id"]), int(row["step"])) for row in samples]
    if ordering != sorted(ordering):
        raise ValueError("Probe bank samples are not in stable sequence/step order")
    actual = _tensor_digest(samples)
    if actual != payload.get("probe_bank_sha256"):
        raise ValueError("Probe bank tensor digest mismatch")
    return payload


def _batch(sample: dict[str, Any], state) -> PolicyInputBatch:
    return PolicyInputBatch(
        spatial_obs=sample["spatial_obs"].float().unsqueeze(0),
        entity_features=sample["entity_features"].float().unsqueeze(0),
        entity_mask=sample["entity_mask"].bool().unsqueeze(0),
        selection_features=sample["selection_features"].float().unsqueeze(0),
        selection_mask=sample["selection_mask"].bool().unsqueeze(0),
        action_feedback_tokens=sample["action_feedback_tokens"].float().unsqueeze(0),
        meta_vec=sample["meta_vec"].float().unsqueeze(0),
        state_in=state,
    )


def _probe_normalizer(extractor_state: dict[str, Any] | None):
    if not extractor_state:
        return None
    from obs_space.obs_space_2 import ObservationExtractor

    extractor = ObservationExtractor()
    extractor.load_state_dict(extractor_state)
    return extractor


def _normalized_sample(sample: dict[str, Any], extractor) -> dict[str, Any]:
    if extractor is None:
        return sample
    normalized = dict(sample)
    for feature_key, mask_key, normalizer in (
        ("entity_features", "entity_mask", extractor.entity_normalizer),
        ("selection_features", "selection_mask", extractor.selection_normalizer),
    ):
        values = sample[feature_key].detach().cpu().numpy().copy()
        mask = sample[mask_key].detach().cpu().numpy().astype(bool)
        if mask.any():
            values[mask] = normalizer.normalize(values[mask])
        normalized[feature_key] = torch.as_tensor(values, dtype=torch.float32)
    return normalized


def replay_policy(
    policy,
    probe_bank: dict[str, Any],
    *,
    extractor_state: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Replay compact probes, resetting recurrent state between sequences."""
    device = torch.device(getattr(policy, "device", "cpu"))
    policy.eval()
    if extractor_state is None:
        extractor_state = getattr(policy, "_representation_extractor_state", None)
    extractor = _probe_normalizer(extractor_state)
    rows: dict[str, list[torch.Tensor]] = {
        "latent": [],
        "action_logits": [],
        "masked_action_logits": [],
        "action_probs": [],
        "value": [],
        "target_primary": [],
        "target_secondary": [],
        "snn_syn": [],
        "snn_mem": [],
    }
    state = None
    previous_sequence = None
    with torch.no_grad():
        for sample in probe_bank["samples"]:
            sample = _normalized_sample(sample, extractor)
            sequence = int(sample["sequence_id"])
            if state is None or sequence != previous_sequence:
                state = policy.init_concrete_state(batch_size=1, device=device)
            batch = _batch(sample, state).to(device)
            latent, value, next_state, spatial_context = policy.encode_step_tensors(
                spatial_obs=batch.spatial_obs,
                entity_features=batch.entity_features,
                entity_mask=batch.entity_mask,
                selection_features=batch.selection_features,
                selection_mask=batch.selection_mask,
                action_feedback_tokens=batch.action_feedback_tokens,
                meta_vec=batch.meta_vec,
                state_in=batch.state_in,
            )
            action_logits = policy.action_head(latent).float()
            available = batch.meta_vec[
                :,
                META_AVAILABLE_ACTION_OFFSET : META_AVAILABLE_ACTION_OFFSET
                + META_AVAILABLE_ACTION_DIM,
            ] > 0.5
            masked_action_logits = action_logits.masked_fill(~available, -1.0e9)
            target_action_ids = torch.full_like(
                masked_action_logits.argmax(dim=-1),
                POLICY_ACTION_RIGHT_CLICK,
            )
            target = policy.build_target_head(
                latent,
                spatial_context,
                target_action_ids,
            )
            rows["latent"].append(latent.cpu().float())
            rows["action_logits"].append(action_logits.cpu())
            rows["masked_action_logits"].append(masked_action_logits.cpu())
            rows["action_probs"].append(
                torch.softmax(masked_action_logits, dim=-1).cpu(),
            )
            rows["value"].append(value.reshape(1, -1).cpu().float())
            rows["target_primary"].append(target.primary_logits.reshape(1, -1).cpu().float())
            if target.secondary_logits is not None:
                rows["target_secondary"].append(
                    target.secondary_logits.reshape(1, -1).cpu().float(),
                )
            rows["snn_syn"].append(next_state[0].reshape(1, -1).cpu().float())
            rows["snn_mem"].append(next_state[1].reshape(1, -1).cpu().float())
            state = next_state
            previous_sequence = sequence
    return {
        name: torch.cat(values, dim=0) if values else torch.empty(0)
        for name, values in rows.items()
    }


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return None
    a = a.double() - a.double().mean(dim=0, keepdim=True)
    b = b.double() - b.double().mean(dim=0, keepdim=True)
    cross = torch.linalg.norm(a.T @ b).pow(2)
    denom = torch.linalg.norm(a.T @ a) * torch.linalg.norm(b.T @ b)
    return float((cross / denom.clamp_min(1e-12)).item())


def subspace_metrics(a: torch.Tensor, b: torch.Tensor, rank: int = 8) -> dict[str, Any]:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return {"rank": 0, "mean_principal_angle_deg": None, "max_principal_angle_deg": None}
    ua = torch.linalg.svd(a.double() - a.double().mean(0), full_matrices=False).U
    ub = torch.linalg.svd(b.double() - b.double().mean(0), full_matrices=False).U
    use_rank = min(int(rank), ua.shape[1], ub.shape[1], a.shape[0] - 1)
    singular = torch.linalg.svdvals(ua[:, :use_rank].T @ ub[:, :use_rank]).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(singular))
    return {
        "rank": int(use_rank),
        "mean_principal_angle_deg": float(angles.mean().item()),
        "max_principal_angle_deg": float(angles.max().item()),
    }


def _matrix_comparison(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    cka = linear_cka(a, b)
    if a.shape != b.shape or a.numel() == 0:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "compatible": False,
            "cka": cka,
        }
    delta = b.double() - a.double()
    row_cos = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)
    return {
        "compatible": True,
        "shape": list(a.shape),
        "rmse": float(delta.pow(2).mean().sqrt().item()),
        "mean_row_cosine": float(row_cos.mean().item()),
        "cka": linear_cka(a, b),
    }


def _distribution_comparison(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape or a.ndim != 2 or a.numel() == 0:
        return {
            "compatible": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    probs_a = torch.softmax(a.double(), dim=-1)
    probs_b = torch.softmax(b.double(), dim=-1)
    midpoint = (probs_a + probs_b) * 0.5
    js = 0.5 * (
        (
            probs_a
            * (
                probs_a.clamp_min(1e-12).log()
                - midpoint.clamp_min(1e-12).log()
            )
        ).sum(-1)
        + (
            probs_b
            * (
                probs_b.clamp_min(1e-12).log()
                - midpoint.clamp_min(1e-12).log()
            )
        ).sum(-1)
    )
    entropy_a = -(probs_a * probs_a.clamp_min(1e-12).log()).sum(-1)
    entropy_b = -(probs_b * probs_b.clamp_min(1e-12).log()).sum(-1)
    return {
        "compatible": True,
        "mean_js_divergence": float(js.mean().item()),
        "argmax_agreement": float(
            (probs_a.argmax(-1) == probs_b.argmax(-1)).double().mean().item()
        ),
        "mean_probability_l1": float((probs_b - probs_a).abs().sum(-1).mean().item()),
        "mean_entropy_a": float(entropy_a.mean().item()),
        "mean_entropy_b": float(entropy_b.mean().item()),
    }


def compare_replays(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    value_a, value_b = a["value"].reshape(-1), b["value"].reshape(-1)
    action_distribution = _distribution_comparison(
        a["masked_action_logits"],
        b["masked_action_logits"],
    )
    value_delta = value_b - value_a
    centered_a = value_a.double() - value_a.double().mean()
    centered_b = value_b.double() - value_b.double().mean()
    correlation_den = centered_a.norm() * centered_b.norm()
    value_correlation = (
        float((torch.dot(centered_a, centered_b) / correlation_den).item())
        if value_a.numel() > 1 and float(correlation_den.item()) > 1.0e-12
        else None
    )
    return {
        "latent": {
            **_matrix_comparison(a["latent"], b["latent"]),
            "subspace": subspace_metrics(a["latent"], b["latent"]),
        },
        "snn_syn": _matrix_comparison(a["snn_syn"], b["snn_syn"]),
        "snn_mem": _matrix_comparison(a["snn_mem"], b["snn_mem"]),
        "action": {
            **_matrix_comparison(a["action_logits"], b["action_logits"]),
            "masked_logits": _matrix_comparison(
                a["masked_action_logits"], b["masked_action_logits"],
            ),
            "distribution": action_distribution,
            "argmax_agreement": action_distribution.get("argmax_agreement"),
            "mean_js_divergence": action_distribution.get("mean_js_divergence"),
        },
        "value": {
            "mae": float(value_delta.abs().mean().item()),
            "rmse": float(value_delta.pow(2).mean().sqrt().item()),
            "correlation": value_correlation,
            "mean_a": float(value_a.mean().item()),
            "mean_b": float(value_b.mean().item()),
            "std_a": float(value_a.std(unbiased=False).item()),
            "std_b": float(value_b.std(unbiased=False).item()),
            "p10_delta": float(torch.quantile(value_delta, 0.10).item()),
            "p50_delta": float(torch.quantile(value_delta, 0.50).item()),
            "p90_delta": float(torch.quantile(value_delta, 0.90).item()),
        },
        "target_primary": {
            **_matrix_comparison(a["target_primary"], b["target_primary"]),
            "distribution": _distribution_comparison(
                a["target_primary"], b["target_primary"],
            ),
        },
        "target_secondary": {
            **_matrix_comparison(a["target_secondary"], b["target_secondary"]),
            "distribution": _distribution_comparison(
                a["target_secondary"], b["target_secondary"],
            ),
        },
    }


def load_policy_artifact(path: str | Path):
    """Build the historical run architecture, then load one policy object."""
    from agent import DefeatRoaches
    from agent_core.spiking_policy import PolicyNetwork
    from Utility.config import cfg

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy_config = checkpoint.get("policy_config")
    if isinstance(policy_config, dict) and policy_config.get("spatial_input_shape"):
        policy = PolicyNetwork(
            spatial_input_shape=tuple(policy_config["spatial_input_shape"]),
            vector_input_dim=int(policy_config["meta_input_dim"]),
            action_dim=int(policy_config["action_dim"]),
            num_steps=int(policy_config["num_steps"]),
            screen_size=int(policy_config["screen_size"]),
            fast_token_snn_alpha=float(policy_config["fast_token_snn_alpha"]),
            fast_token_snn_beta=float(policy_config["fast_token_snn_beta"]),
            slow_token_snn_alpha=float(policy_config["slow_token_snn_alpha"]),
            slow_token_snn_beta=float(policy_config["slow_token_snn_beta"]),
            temporal_combine_mode=str(policy_config["temporal_combine_mode"]),
            attention_embed_dim=int(policy_config["attention_embed_dim"]),
            attention_pool_size=int(policy_config["attention_pool_size"]),
            attention_beta=float(policy_config["attention_beta"]),
            spatial_head_type=str(policy_config["spatial_head_type"]),
            coarse_grid_size=int(policy_config["coarse_grid_size"]),
            local_grid_size=int(policy_config["local_grid_size"]),
            target_decode_mode=str(policy_config["target_decode_mode"]),
            fine_skip_connection=bool(policy_config["fine_skip_connection"]),
            fine_skip_dim=int(policy_config["fine_skip_dim"]),
            amp_dtype="fp32",
        )
    else:
        run_dir = _run_dir_for(path)
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            cfg.reload(str(config_path))
        agent = DefeatRoaches(
            spatial_input_shape=tuple(cfg.model.spatial_input_shape),
            vector_input_dim=int(cfg.model.vector_input_dim),
            action_dim=int(cfg.model.action_dim),
        )
        policy = agent.policy
    policy.load_state_dict(checkpoint["agent_state"])
    policy.to("cpu")
    policy.device = torch.device("cpu")
    policy.configure_amp("fp32")
    policy.eval()
    policy._representation_extractor_state = checkpoint.get("extractor_state")
    return policy


def compare_artifacts(
    path_a: str | Path,
    path_b: str | Path,
    probe_bank: dict[str, Any],
    *,
    policy_loader=load_policy_artifact,
) -> dict[str, Any]:
    path_a, path_b = Path(path_a), Path(path_b)
    policy_a = policy_loader(path_a)
    replay_a = replay_policy(policy_a, probe_bank)
    del policy_a
    policy_b = policy_loader(path_b)
    replay_b = replay_policy(policy_b, probe_bank)
    weights = diff_entries(path_a, path_b)
    return {
        "schema_version": 1,
        "kind": "representation-migration-report",
        "probe_bank": {
            "name": probe_bank["name"],
            "sha256": probe_bank["probe_bank_sha256"],
            "samples": len(probe_bank["samples"]),
        },
        "replay_contract": {
            "recurrent_state": "carried_within_sequence_reset_between_sequences",
            "extractor_state": "loaded_per_artifact_no_stat_updates",
            "action_distribution": "semantic_availability_masked",
            "target_distribution_condition": "RIGHT_CLICK",
            "retained_tensors": [
                "latent",
                "action_logits",
                "value",
                "target_primary",
                "target_secondary",
                "snn_syn",
                "snn_mem",
            ],
        },
        "artifact_a": {"path": str(path_a), "sha256": sha256_file(path_a)},
        "artifact_b": {"path": str(path_b), "sha256": sha256_file(path_b)},
        "weights": {
            "total_l2": weights["total_l2"],
            "shape_mismatches": weights["shape_mismatches"],
            "only_in_a": weights["only_in_a"],
            "only_in_b": weights["only_in_b"],
            "top_layers": weights["layers"][:25],
        },
        "representations": compare_replays(replay_a, replay_b),
    }


def _write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def timeline_report(
    run_name: str,
    probe_bank: dict[str, Any],
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    policy_loader=load_policy_artifact,
) -> dict[str, Any]:
    entries = [
        entry for entry in list_run_entries(run_name, models_dir=models_dir)
        if entry.kind == "snapshot"
    ]
    pairs = []
    for left, right in zip(entries, entries[1:], strict=False):
        report = compare_artifacts(
            left.path,
            right.path,
            probe_bank,
            policy_loader=policy_loader,
        )
        pairs.append(
            {
                "policy_version_a": left.policy_version,
                "policy_version_b": right.policy_version,
                "artifact_a": report["artifact_a"],
                "artifact_b": report["artifact_b"],
                "weights": report["weights"],
                "representations": report["representations"],
            },
        )
    return {
        "schema_version": 1,
        "kind": "representation-migration-timeline",
        "run": run_name,
        "probe_bank": {
            "name": probe_bank["name"],
            "sha256": probe_bank["probe_bank_sha256"],
        },
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("probe-create")
    create.add_argument("--out", default="probes/tiny_skirmish_v1.pt")
    create.add_argument("--spec", default="probes/tiny_skirmish_v1.json")
    create.add_argument("--seeds", default="0,1,2,3")
    create.add_argument("--steps", type=int, default=8)
    compare = commands.add_parser("compare")
    compare.add_argument("ref_a")
    compare.add_argument("ref_b")
    compare.add_argument("--probe", required=True)
    compare.add_argument("--out", required=True)
    timeline = commands.add_parser("timeline")
    timeline.add_argument("run")
    timeline.add_argument("--probe", required=True)
    timeline.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "probe-create":
        path = (
            create_probe_bank_from_spec(args.spec, args.out)
            if args.spec
            else create_tiny_probe_bank(
                args.out,
                seeds=[int(value) for value in args.seeds.split(",") if value],
                steps_per_seed=args.steps,
            )
        )
        print(f"Wrote fixed probe bank {path}")
        return 0
    bank = load_probe_bank(args.probe)
    if args.command == "compare":
        path_a = resolve_ref(args.ref_a, models_dir=args.models_dir)
        path_b = resolve_ref(args.ref_b, models_dir=args.models_dir)
        out = _write_json(args.out, compare_artifacts(path_a, path_b, bank))
    else:
        out = _write_json(
            args.out,
            timeline_report(args.run, bank, models_dir=args.models_dir),
        )
    print(f"Wrote representation report {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
