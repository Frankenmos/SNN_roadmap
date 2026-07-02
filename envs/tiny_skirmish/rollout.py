from __future__ import annotations

import argparse
from collections.abc import Iterable

from .env import TinySkirmishEnv
from .protocol import ACTION_NAMES, SkirmishAction


def scripted_action(env: TinySkirmishEnv) -> SkirmishAction:
    x, y = env.nearest_enemy_screen_target()
    return SkirmishAction.right_click(x, y)


def run_episode(
    *,
    mode: str,
    seed: int,
    max_steps: int,
    render: bool,
) -> dict[str, object]:
    env = TinySkirmishEnv(seed=seed, max_steps=max_steps)
    env.reset()
    total = 0.0
    events: dict[str, int] = {}
    termination = "unknown"

    if render:
        print(env.render_text())

    for step_index in range(max_steps):
        if mode == "scripted":
            action = scripted_action(env)
        elif mode == "random":
            action = env.random_action()
        else:
            raise ValueError(f"unknown mode: {mode}")

        result = env.step(action)
        total += result.reward.total
        for event in result.reward.events:
            events[event] = events.get(event, 0) + 1

        parts = result.reward.compact_parts()
        target = result.info["target_grid"]
        print(
            f"{step_index + 1:03d} {ACTION_NAMES[action.action_id]:>11}"
            f" target={target} reward={result.reward.total:+.3f} parts={parts}",
        )

        if render:
            print(env.render_text())

        if result.done or result.truncated:
            termination = str(result.info.get("termination") or "done")
            break

    summary = {
        "mode": mode,
        "seed": seed,
        "total_reward": round(total, 6),
        "events": events,
        "termination": termination,
    }
    print(f"SUMMARY {summary}")
    return summary


def run_many(
    *,
    mode: str,
    episodes: int,
    seed: int,
    max_steps: int,
    render: bool,
) -> list[dict[str, object]]:
    summaries = []
    for episode in range(episodes):
        print(f"=== episode={episode} seed={seed + episode} mode={mode} ===")
        summaries.append(
            run_episode(
                mode=mode,
                seed=seed + episode,
                max_steps=max_steps,
                render=render,
            ),
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TinySkirmish protocol smoke rollouts.")
    parser.add_argument("--mode", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--render", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_many(
        mode=args.mode,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        render=args.render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
