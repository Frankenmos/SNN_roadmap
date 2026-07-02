from __future__ import annotations

import argparse
from typing import Iterable

import torch

from agent_core.policy_protocol import (
    CURATED_FEATURE_UNIT_FIELDS,
    META_VECTOR_DIM,
    POLICY_ACTION_DIM,
    SPATIAL_OBS_SHAPE,
    PolicyInputBatch,
)
from agent_core.ppo_trainer import PPO
from agent_core.spiking_policy import PolicyNetwork

from .env import TinySkirmishEnv


def import_real_snn_modules():
    """Return the SNN classes/constants the bridge harnesses consume.

    TinySkirmish lives inside the SNN repo now, so these are plain imports;
    the dict shape is kept so the render/live/rollout harnesses need not change.
    """

    return {
        "CURATED_FEATURE_UNIT_FIELDS": CURATED_FEATURE_UNIT_FIELDS,
        "META_VECTOR_DIM": META_VECTOR_DIM,
        "POLICY_ACTION_DIM": POLICY_ACTION_DIM,
        "SPATIAL_OBS_SHAPE": SPATIAL_OBS_SHAPE,
        "PolicyInputBatch": PolicyInputBatch,
        "PolicyNetwork": PolicyNetwork,
        "PPO": PPO,
    }


def tiny_observation_to_real_batch(observation, real, *, device: torch.device):
    PolicyInputBatch = real["PolicyInputBatch"]
    batch = PolicyInputBatch(
        spatial_obs=torch.as_tensor(
            observation.spatial_obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        entity_features=torch.as_tensor(
            observation.entity_features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        entity_mask=torch.as_tensor(
            observation.entity_mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0),
        selection_features=torch.as_tensor(
            observation.selection_features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        selection_mask=torch.as_tensor(
            observation.selection_mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0),
        action_feedback_tokens=torch.as_tensor(
            observation.action_feedback_tokens,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        meta_vec=torch.as_tensor(
            observation.meta_vec,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
    )
    return batch


def build_real_policy(real, *, device: torch.device, small: bool):
    PolicyNetwork = real["PolicyNetwork"]
    embed_dim = 32 if small else 64
    policy = PolicyNetwork(
        real["SPATIAL_OBS_SHAPE"],
        real["META_VECTOR_DIM"],
        real["POLICY_ACTION_DIM"],
        num_steps=1,
        screen_size=84,
        fast_token_snn_alpha=0.55,
        fast_token_snn_beta=0.65,
        slow_token_snn_alpha=0.92,
        slow_token_snn_beta=0.97,
        temporal_combine_mode="mean",
        attention_embed_dim=embed_dim,
        attention_pool_size=7,
        attention_beta=0.5,
        spatial_head_type="coarse_to_fine",
        coarse_grid_size=7,
        local_grid_size=12,
        target_decode_mode="center",
        fine_skip_connection=True,
        fine_skip_dim=32,
        amp_dtype="auto",
    )
    policy.to(device)
    policy.device = device
    policy.configure_amp("fp32" if device.type == "cpu" else "auto")
    policy.eval()
    return policy


def run_real_forward_check(
    *,
    seed: int = 9,
    device_name: str = "cpu",
    small: bool = False,
) -> dict[str, object]:
    device = torch.device(device_name)
    real = import_real_snn_modules()
    env = TinySkirmishEnv(seed=seed)
    observation = env.reset()
    batch = tiny_observation_to_real_batch(observation, real, device=device)
    policy = build_real_policy(real, device=device, small=small)

    with torch.no_grad():
        batch = batch.with_state(policy.init_concrete_state(batch_size=1, device=device))
        action_logits, target_head_state, state_value, next_state = policy(batch)

    if action_logits.shape != (1, real["POLICY_ACTION_DIM"]):
        raise AssertionError(f"bad action logits shape: {tuple(action_logits.shape)}")
    if tuple(state_value.shape) not in {(1,), (1, 1)}:
        raise AssertionError(f"bad state value shape: {tuple(state_value.shape)}")
    if next_state is None or next_state[0].shape[0] != 1:
        raise AssertionError("bad next recurrent state")

    result = {
        "device": str(device),
        "small": bool(small),
        "policy_class_module": policy.__class__.__module__,
        "batch_class_module": batch.__class__.__module__,
        "action_logits_shape": tuple(action_logits.shape),
        "state_value_shape": tuple(state_value.shape),
        "target_head_type": getattr(target_head_state, "head_type", None),
        "next_state_shape": tuple(next_state[0].shape),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TinySkirmish observations against the real SNN architecture.",
    )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a smaller attention embedding for a faster smoke pass.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_real_forward_check(
        seed=args.seed,
        device_name=args.device,
        small=args.small,
    )
    print("Real SNN bridge check passed")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
