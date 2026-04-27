from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from multiprocessing import Queue
from pathlib import Path
from typing import Iterable

import numpy as np

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
from distributed.learner import LearnerCoordinator
from distributed.protocol import EpisodeSummary, RolloutFragment
from distributed.ray_actor import RolloutActor
from Utility.config import cfg
from Utility.logger_utils import LogListener


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronous Ray rollout training for DefeatRoaches.",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=None,
        help="Number of rollout actors. Defaults to distributed.num_rollout_actors.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Stop after this many learner updates. Defaults to config.",
    )
    parser.add_argument(
        "--fragment-steps",
        type=int,
        default=None,
        help="Per-actor fragment target. Defaults to distributed.fragment_steps.",
    )
    parser.add_argument(
        "--global-rollout-steps",
        type=int,
        default=None,
        help="Aggregate on-policy rollout budget per learner update.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name. Defaults to environment.run_name or timestamp.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path. Defaults to SNN_CONFIG_PATH or repo config.yaml.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Start Ray in local_mode for debugging.",
    )
    return parser.parse_args()


def _distributed_value(name: str, fallback):
    distributed_cfg = getattr(cfg, "distributed", {})
    return getattr(distributed_cfg, name, fallback)


def _resolve_num_actors(args: argparse.Namespace) -> int:
    value = args.num_actors
    if value is None:
        value = _distributed_value("num_rollout_actors", 4)
    return max(1, int(value))


def _resolve_fragment_steps(args: argparse.Namespace) -> int:
    value = args.fragment_steps
    if value is None:
        value = _distributed_value(
            "fragment_steps",
            getattr(cfg.hyperparameters, "rollout_steps", 2048),
        )
    return max(1, int(value))


def _resolve_global_rollout_steps(args: argparse.Namespace) -> int:
    value = args.global_rollout_steps
    if value is None:
        value = _distributed_value(
            "global_rollout_steps",
            getattr(cfg.hyperparameters, "rollout_steps", 2048),
        )
    return max(1, int(value))


def _resolve_max_updates(
    args: argparse.Namespace,
    *,
    global_rollout_steps: int,
) -> int:
    value = args.max_updates
    if value is None:
        value = int(_distributed_value("max_updates", 0) or 0)
    if value and int(value) > 0:
        return int(value)

    total_steps = int(cfg.environment.total_episodes) * int(
        cfg.environment.steps_per_episode,
    )
    return max(1, math.ceil(total_steps / max(1, int(global_rollout_steps))))


def _object_store_memory_bytes() -> int | None:
    configured_gb = float(_distributed_value("object_store_memory_gb", 0.0) or 0.0)
    if configured_gb <= 0.0:
        return None
    return int(configured_gb * (1024**3))


def _iter_episode_summaries(
    fragments: Iterable[RolloutFragment],
) -> Iterable[EpisodeSummary]:
    for fragment in fragments:
        yield from fragment.episode_summaries


def _log_episode_summaries(
    *,
    log_queue,
    fragments: Iterable[RolloutFragment],
    episode_rewards: deque,
) -> int:
    logged = 0
    for summary in _iter_episode_summaries(fragments):
        internal_ep = int(summary.actor_id) * 1_000_000_000 + int(
            summary.episode_index,
        )
        episode_rewards.append(float(summary.total_reward))
        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        log_queue.put(
            {
                "type": "EPISODE_START",
                "internal_ep": internal_ep,
                "actor_id": int(summary.actor_id),
                "policy_version": summary.policy_version,
            },
        )
        log_queue.put(
            {
                "type": "EPISODE_END",
                "internal_ep": internal_ep,
                "total": float(summary.total_reward),
                "avg": float(avg_reward),
                "steps": int(summary.steps),
            },
        )
        logged += 1
    return logged


def _log_update(
    *,
    log_queue,
    stats: dict,
    episode_index: int,
) -> None:
    log_queue.put(
        {
            "type": "UPDATE",
            "internal_ep": -1,
            "episode_index": int(episode_index),
            "policy_version": int(stats.get("policy_version", 0) or 0),
            "policy_protocol_version": POLICY_PROTOCOL_VERSION,
            "policy_input_schema": POLICY_INPUT_SCHEMA,
            **stats,
        },
    )


def _ray_init_kwargs(args: argparse.Namespace) -> dict:
    local_mode = bool(args.local_mode or _distributed_value("ray_local_mode", False))
    kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": bool(_distributed_value("include_dashboard", False)),
        "local_mode": local_mode,
    }
    object_store_memory = _object_store_memory_bytes()
    if object_store_memory is not None:
        kwargs["object_store_memory"] = object_store_memory
    return kwargs


def _broadcast_weights(
    ray,
    actors,
    learner: LearnerCoordinator,
    *,
    include_extractor_state: bool,
) -> None:
    weights_ref = ray.put(
        learner.make_weight_payload(
            include_extractor_state=include_extractor_state,
        ),
    )
    ray.get([actor.set_weights.remote(weights_ref) for actor in actors])


def _collect_sync_fragments(
    ray,
    actors,
    *,
    fragment_steps: int,
    global_rollout_steps: int,
    policy_version: int,
) -> tuple[list[RolloutFragment], dict[str, float | int]]:
    fragments: list[RolloutFragment] = []
    collected_steps = 0
    stats: dict[str, float | int] = {
        "ray_get_wall_seconds": 0.0,
        "ray_submit_wall_seconds": 0.0,
        "rollout_collect_waves": 0,
        "rollout_empty_waves": 0,
        "rollout_steps_collected": 0,
    }
    empty_waves = 0
    while collected_steps < int(global_rollout_steps):
        submit_started = time.perf_counter()
        refs = [
            actor.collect_fragment.remote(fragment_steps, policy_version)
            for actor in actors
        ]
        stats["ray_submit_wall_seconds"] = float(
            stats["ray_submit_wall_seconds"],
        ) + (time.perf_counter() - submit_started)
        ray_get_started = time.perf_counter()
        wave = ray.get(refs)
        stats["ray_get_wall_seconds"] = float(
            stats["ray_get_wall_seconds"],
        ) + (time.perf_counter() - ray_get_started)
        stats["rollout_collect_waves"] = int(stats["rollout_collect_waves"]) + 1
        # Filter out None values (actor had no new fragment to contribute)
        wave = [f for f in wave if f is not None]
        if not wave:
            empty_waves += 1
            stats["rollout_empty_waves"] = int(empty_waves)
            if empty_waves >= 3:
                raise RuntimeError(
                    "Rollout collection made no progress for 3 consecutive "
                    "waves. All actors returned None; actor-local rollout "
                    "buffers may be stale or wedged.",
                )
            continue
        empty_waves = 0
        fragments.extend(wave)
        collected_steps = sum(fragment.num_steps for fragment in fragments)
    stats["rollout_steps_collected"] = int(collected_steps)
    return fragments, stats


def main() -> None:
    args = _parse_args()
    if args.config:
        os.environ["SNN_CONFIG_PATH"] = str(Path(args.config).expanduser().resolve())
    cfg.reload(args.config)
    if args.run_name:
        cfg.environment.run_name = args.run_name

    import ray
    from train import (
        _run_dir,
        _run_path,
        load_checkpoint,
        save_checkpoint,
        save_initial_config,
        write_effective_config,
    )

    num_actors = _resolve_num_actors(args)
    fragment_steps = _resolve_fragment_steps(args)
    global_rollout_steps = _resolve_global_rollout_steps(args)
    max_updates = _resolve_max_updates(
        args,
        global_rollout_steps=global_rollout_steps,
    )
    actor_cpus = float(_distributed_value("actor_cpus", 1) or 1)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(cfg.config_path).resolve()

    log_queue = Queue()
    run_dir = _run_dir()
    run_name = cfg.environment.run_name
    print(f"Ray run directory: {run_dir}")
    print(
        "Starting synchronous Ray PPO: "
        f"actors={num_actors}, fragment_steps={fragment_steps}, "
        f"global_rollout_steps={global_rollout_steps}, max_updates={max_updates}",
    )
    save_initial_config()

    actors = []
    learner = None
    ray_started = False
    interrupted = False
    db_listener = LogListener(log_queue, _run_path(cfg.environment.db_path))
    db_listener.start()
    try:
        ray.init(**_ray_init_kwargs(args))
        ray_started = True
        learner = LearnerCoordinator()
        start_episode, best_eval_reward, episode_rewards = load_checkpoint(
            learner.agent,
        )
        if not isinstance(episode_rewards, deque):
            episode_rewards = deque(
                episode_rewards,
                maxlen=cfg.environment.reward_window,
            )
        episode_index = int(start_episode)
        write_effective_config(learner.agent)

        RemoteRolloutActor = ray.remote(num_cpus=actor_cpus)(RolloutActor)
        actors = [
            RemoteRolloutActor.remote(
                actor_id=actor_id,
                repo_root=str(repo_root),
                config_path=str(config_path),
                run_name=run_name,
                visualize=False,
            )
            for actor_id in range(num_actors)
        ]
        _broadcast_weights(
            ray,
            actors,
            learner,
            include_extractor_state=True,
        )

        for update_index in range(max_updates):
            policy_version = int(learner.policy_version)
            rollout_started = time.perf_counter()
            fragments, rollout_stats = _collect_sync_fragments(
                ray,
                actors,
                fragment_steps=fragment_steps,
                global_rollout_steps=global_rollout_steps,
                policy_version=policy_version,
            )
            rollout_wall_seconds = time.perf_counter() - rollout_started

            stats = dict(learner.update_from_fragments(fragments))
            stats["rollout_wall_seconds"] = float(rollout_wall_seconds)
            stats["rollout_actor_count"] = int(num_actors)
            stats["rollout_fragments_collected"] = int(len(fragments))
            stats.update(rollout_stats)
            stats["rollout_collect_overhead_wall_seconds"] = max(
                0.0,
                float(rollout_wall_seconds)
                - float(rollout_stats.get("ray_get_wall_seconds", 0.0))
                - float(rollout_stats.get("ray_submit_wall_seconds", 0.0)),
            )

            episode_log_started = time.perf_counter()
            episodes_logged = _log_episode_summaries(
                log_queue=log_queue,
                fragments=fragments,
                episode_rewards=episode_rewards,
            )
            stats["episode_log_enqueue_wall_seconds"] = float(
                time.perf_counter() - episode_log_started,
            )
            stats["episodes_logged_in_update"] = int(episodes_logged)
            episode_index += episodes_logged

            checkpoint_frequency = max(1, int(cfg.environment.log_frequency))
            checkpoint_wall_seconds = 0.0
            if (update_index + 1) % checkpoint_frequency == 0:
                avg_reward = (
                    float(np.mean(episode_rewards)) if episode_rewards else None
                )
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    learner.agent,
                    episode_index,
                    best_eval_reward,
                    episode_rewards,
                    avg_reward=avg_reward,
                    require_rollout_clear=False,
                )
                checkpoint_wall_seconds = time.perf_counter() - checkpoint_started
            stats["checkpoint_wall_seconds"] = float(checkpoint_wall_seconds)

            update_log_started = time.perf_counter()
            _log_update(
                log_queue=log_queue,
                stats=stats,
                episode_index=episode_index,
            )
            update_log_wall_seconds = time.perf_counter() - update_log_started
            cuda_peak_gib = float(
                stats.get("cuda_peak_allocated_bytes", 0) or 0,
            ) / float(1024**3)

            print(
                f"Update {update_index + 1}/{max_updates} | "
                f"policy_version={learner.policy_version} | "
                f"steps={sum(fragment.num_steps for fragment in fragments)} | "
                f"episodes={episode_index} | "
                f"rollout={rollout_wall_seconds:.1f}s | "
                f"ray_get={stats.get('ray_get_wall_seconds', 0.0):.1f}s | "
                f"update={stats.get('update_wall_seconds', 0.0):.1f}s | "
                f"log={update_log_wall_seconds:.3f}s",
            )
            print(
                "  learner_detail | "
                f"transfer={stats.get('cpu_to_gpu_transfer_wall_seconds', 0.0):.1f}s | "
                f"pack={stats.get('chunk_pack_wall_seconds', 0.0):.1f}s | "
                f"replay={stats.get('replay_forward_wall_seconds', 0.0):.1f}s | "
                f"backward={stats.get('backward_optimizer_wall_seconds', 0.0):.1f}s | "
                f"fwd_calls={int(stats.get('tbptt_forward_calls', 0) or 0)} | "
                f"active_chunks={stats.get('tbptt_group_mean_active_chunks', 0.0):.2f} | "
                f"payload={stats.get('payload_total_mib', 0.0):.1f}MiB | "
                f"cuda_peak={cuda_peak_gib:.2f}GiB",
            )

            _broadcast_weights(
                ray,
                actors,
                learner,
                include_extractor_state=False,
            )

        if learner is not None and not interrupted:
            avg_reward = float(np.mean(episode_rewards)) if episode_rewards else None
            save_checkpoint(
                learner.agent,
                episode_index,
                best_eval_reward,
                episode_rewards,
                avg_reward=avg_reward,
                require_rollout_clear=False,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted by user; shutting down Ray actors and logger.")
    finally:
        if actors:
            try:
                ray.get([actor.close.remote() for actor in actors], timeout=30)
            except KeyboardInterrupt:
                print("Interrupted while closing actors; killing remaining Ray actors.")
                for actor in actors:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"Warning: failed to close one or more Ray actors cleanly: {exc}")
                for actor in actors:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
        try:
            log_queue.put({"type": "KILL"})
        except Exception as exc:
            print(f"Warning: failed to send logger shutdown event: {exc}")
        db_listener.join(timeout=10)
        if db_listener.is_alive():
            print("Warning: logger did not stop cleanly; terminating it.")
            db_listener.terminate()
            db_listener.join(timeout=5)
        if ray_started:
            ray.shutdown()


if __name__ == "__main__":
    main()
