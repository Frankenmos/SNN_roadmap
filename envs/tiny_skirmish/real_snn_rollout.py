from __future__ import annotations

import argparse
from typing import Iterable

import torch

from .env import TinySkirmishEnv
from .protocol import SkirmishAction
from .real_snn_bridge import (
    build_real_policy,
    import_real_snn_modules,
    tiny_observation_to_real_batch,
)


def collect_real_snn_fragment(
    *,
    seed: int = 9,
    max_steps: int = 16,
    device_name: str = "cpu",
    small: bool = False,
    deterministic: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    device = torch.device(device_name)
    real = import_real_snn_modules()
    PPO = real["PPO"]
    policy = build_real_policy(real, device=device, small=small)
    ppo = PPO(
        policy,
        lr=1.0e-4,
        gamma=0.99,
        clip_epsilon=0.1,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        tbptt_window=8,
        rollout_cache_spatial_dtype="float32",
    )

    env = TinySkirmishEnv(seed=seed, max_steps=max_steps)
    observation = env.reset()
    state = policy.init_concrete_state(batch_size=1, device=device)
    total_reward = 0.0
    events: dict[str, int] = {}

    for _step in range(max_steps):
        batch = tiny_observation_to_real_batch(observation, real, device=device).with_state(state)
        sample = ppo.select_action(batch, deterministic=deterministic)
        result = env.step(SkirmishAction(sample.action_id, sample.x, sample.y))
        if verbose:
            print(
                f"{_step + 1:03d} action={sample.action_id} "
                f"target=({sample.x},{sample.y}) reward={result.reward.total:+.3f} "
                f"parts={result.reward.compact_parts()}",
            )
        total_reward += result.reward.total
        for event in result.reward.events:
            events[event] = events.get(event, 0) + 1

        done = torch.tensor(float(result.done), dtype=torch.float32, device=device)
        truncated = torch.tensor(float(result.truncated), dtype=torch.float32, device=device)
        episode_reset = torch.tensor(
            bool(result.done or result.truncated),
            dtype=torch.bool,
            device=device,
        )
        ppo.store_transition(
            batch,
            torch.tensor(sample.action_id, dtype=torch.long, device=device),
            torch.tensor(sample.x, dtype=torch.long, device=device),
            torch.tensor(sample.y, dtype=torch.long, device=device),
            torch.tensor(sample.log_prob, dtype=torch.float32, device=device),
            torch.tensor(result.reward.total, dtype=torch.float32, device=device),
            torch.tensor(sample.value, dtype=torch.float32, device=device),
            done,
            sample_mask=torch.tensor(1.0, dtype=torch.float32, device=device),
            truncated=truncated,
            episode_reset=episode_reset,
            target_index=(
                None
                if sample.target_index is None
                else torch.tensor(sample.target_index, dtype=torch.long, device=device)
            ),
            coarse_index=(
                None
                if sample.coarse_index is None
                else torch.tensor(sample.coarse_index, dtype=torch.long, device=device)
            ),
            fine_index=(
                None
                if sample.fine_index is None
                else torch.tensor(sample.fine_index, dtype=torch.long, device=device)
            ),
        )

        state = sample.next_state
        observation = result.observation
        next_batch = tiny_observation_to_real_batch(observation, real, device=device).with_state(state)
        ppo.set_final_next(next_batch)

        if result.done or result.truncated:
            break

    fragment = ppo.finalize_fragment(
        actor_id=0,
        fragment_id=0,
        policy_version=0,
        reward_component_summaries=tuple(
            transition["reward"].item() for transition in ppo.memory
        ),
    )
    if fragment is None:
        raise AssertionError("expected a rollout fragment")

    return {
        "device": str(device),
        "small": bool(small),
        "deterministic": bool(deterministic),
        "steps": int(fragment.num_steps),
        "learnable_steps": int(fragment.num_learnable_steps),
        "total_reward": round(float(total_reward), 6),
        "events": events,
        "fragment_spatial_shape": tuple(fragment.spatial_obs.shape),
        "fragment_feedback_shape": tuple(fragment.action_feedback_tokens.shape),
        "tail_next_batch": fragment.tail_next_policy_input is not None,
        "terminated": bool(fragment.terminated),
        "truncated": bool(fragment.truncated),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a TinySkirmish rollout through the real SNN PolicyNetwork/PPO seam.",
    )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = collect_real_snn_fragment(
        seed=args.seed,
        max_steps=args.max_steps,
        device_name=args.device,
        small=args.small,
        deterministic=args.deterministic,
        verbose=args.verbose,
    )
    print("Real SNN rollout fragment check passed")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
