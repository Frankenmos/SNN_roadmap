import json
import logging
import os
import shutil
import time
from collections import deque
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import torch
from absl import app, flags

from agent import DefeatRoaches
from agent_core.policy_protocol import (
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    POLICY_INPUT_SCHEMA,
    POLICY_PROTOCOL_VERSION,
)
from envs.setup_env import create_env
from obs_space.obs_space_2 import ObservationExtractor
from Utility.config import cfg
from Utility.logger_utils import LogListener
from Utility.run_manifest import (
    RUN_MANIFEST_FILENAME,
    append_resume_event,
    build_manifest_payload,
    ensure_run_manifest,
    next_phase_id,
)


def _run_dir() -> str:
    run_name = getattr(cfg.environment, "run_name", "") or ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        cfg.environment.run_name = run_name
    models_dir = getattr(cfg.environment, "models_dir", "models")
    run_dir = os.path.join(models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _run_path(filename: str) -> str:
    return os.path.join(_run_dir(), filename)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_ACTIVE_PHASE_ID = 0


def current_phase_id() -> int:
    return int(_ACTIVE_PHASE_ID)


class CheckpointProtocolMismatch(RuntimeError):
    pass


FLAGS = flags.FLAGS
if "run_name" not in FLAGS:
    flags.DEFINE_string(
        "run_name",
        None,
        "Run directory to resume from/write into. Overrides config.run_name. "
        "Pass 'latest' to reuse the most recently modified run under models_dir.",
    )


def reset_environment():
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=cfg.environment.visualize,
            use_action_printer=cfg.environment.use_action_printer,
            use_available_actions_diagnostics=getattr(
                cfg.environment,
                "use_available_actions_diagnostics",
                False,
            ),
            available_actions_diagnostics_output_path=getattr(
                cfg.environment,
                "available_actions_diagnostics_output_path",
                "analysis_results/available_actions_diagnostics.jsonl",
            ),
            available_actions_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "available_actions_diagnostics_every_n_steps",
                1,
            ),
            use_last_action_diagnostics=getattr(
                cfg.environment,
                "use_last_action_diagnostics",
                False,
            ),
            last_action_diagnostics_output_path=getattr(
                cfg.environment,
                "last_action_diagnostics_output_path",
                "analysis_results/last_action_diagnostics.jsonl",
            ),
            last_action_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "last_action_diagnostics_every_n_steps",
                1,
            ),
            use_score_diagnostics=getattr(
                cfg.environment,
                "use_score_diagnostics",
                False,
            ),
            score_diagnostics_output_path=getattr(
                cfg.environment,
                "score_diagnostics_output_path",
                "analysis_results/score_diagnostics.jsonl",
            ),
            score_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "score_diagnostics_every_n_steps",
                1,
            ),
            use_observation_inspector=getattr(
                cfg.environment,
                "use_observation_inspector",
                False,
            ),
            observation_inspector_output_path=getattr(
                cfg.environment,
                "observation_inspector_output_path",
                "analysis_results/observation_space.jsonl",
            ),
            observation_inspector_every_n_steps=getattr(
                cfg.environment,
                "observation_inspector_every_n_steps",
                10,
            ),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment,
                "observation_inspector_max_unit_samples",
                5,
            ),
            use_policy_input_diagnostics=getattr(
                cfg.environment,
                "use_policy_input_diagnostics",
                False,
            ),
            policy_input_diagnostics_output_path=getattr(
                cfg.environment,
                "policy_input_diagnostics_output_path",
                "analysis_results/policy_input_diagnostics.jsonl",
            ),
            policy_input_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "policy_input_diagnostics_every_n_steps",
                1,
            ),
            policy_input_diagnostics_max_entity_samples=getattr(
                cfg.environment,
                "policy_input_diagnostics_max_entity_samples",
                3,
            ),
            policy_input_diagnostics_max_selection_samples=getattr(
                cfg.environment,
                "policy_input_diagnostics_max_selection_samples",
                3,
            ),
        )
    except Exception as exc:
        print(f"Error during environment reset: {exc}")
        raise exc
    return env


def _latest_run_name() -> str:
    models_dir = getattr(cfg.environment, "models_dir", "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"models_dir does not exist, cannot resolve latest run: {models_dir}",
        )

    candidates = []
    for entry in os.scandir(models_dir):
        if entry.is_dir():
            candidates.append(entry)
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found under models_dir: {models_dir}",
        )
    return max(candidates, key=lambda entry: entry.stat().st_mtime).name


def _apply_runtime_flags():
    run_name = FLAGS.run_name
    if not run_name:
        return
    if run_name.lower() == "latest":
        run_name = _latest_run_name()
    cfg.environment.run_name = run_name
    logger.info("Using run_name from CLI: %s", run_name)


def save_checkpoint(
    agent,
    episode,
    best_eval_reward,
    episode_rewards,
    checkpoint_path=None,
    avg_reward=None,
    eval_reward=None,
    require_rollout_clear=True,
):
    saved = False
    if require_rollout_clear and agent.ppo.has_pending_rollout():
        print(
            "Skipping checkpoint save (PPO rollout cache not empty; "
            "waiting for update to preserve continuity).",
        )
        return saved
    if checkpoint_path is None:
        checkpoint_path = _run_path(cfg.environment.checkpoint_path)
    checkpoint = {
        "agent_state": agent.policy.state_dict(),
        "optimizer_state": agent.ppo.optimizer.state_dict(),
        "scheduler_state": (
            agent.ppo.scheduler.state_dict() if agent.ppo.scheduler is not None else None
        ),
        "episode": episode,
        "best_eval_reward": best_eval_reward,
        "best_avg_reward": avg_reward,
        "avg_reward_at_save": avg_reward,
        "eval_reward_at_save": eval_reward,
        "episode_rewards": list(episode_rewards),
        "extractor_state": agent.extractor.state_dict(),
        "global_update_index": int(getattr(agent.ppo, "update_count", 0)),
        "policy_version": int(getattr(agent.ppo, "update_count", 0)),
        "policy_protocol_version": POLICY_PROTOCOL_VERSION,
        "policy_input_schema": POLICY_INPUT_SCHEMA,
    }

    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)

    try:
        os.replace(temp_path, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} at episode {episode}.")
        saved = True
    except OSError as exc:
        print(f"Error saving checkpoint: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return saved


def maybe_save_best_checkpoint(
    agent,
    episode,
    avg_reward,
    eval_summary,
    best_eval_reward,
    episode_rewards,
):
    if not eval_summary:
        return best_eval_reward

    best_name = getattr(
        cfg.environment,
        "best_checkpoint_path",
        "best_checkpoint.pth",
    )
    best_path = _run_path(best_name)
    min_eps = getattr(cfg.environment, "best_min_episodes", 50)
    eval_reward = float(eval_summary["mean_reward"])
    if episode < min_eps:
        return best_eval_reward
    if eval_reward <= best_eval_reward:
        return best_eval_reward

    did_save = save_checkpoint(
        agent,
        episode,
        eval_reward,
        episode_rewards,
        checkpoint_path=best_path,
        avg_reward=avg_reward,
        eval_reward=eval_reward,
        require_rollout_clear=False,
    )
    if not did_save:
        return best_eval_reward

    print(
        f"  -> new best deterministic eval reward: {eval_reward:.2f} (was {best_eval_reward:.2f})",
    )
    return eval_reward


def load_checkpoint(agent, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = _run_path(cfg.environment.checkpoint_path)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
            )
            checkpoint_protocol = checkpoint.get("policy_protocol_version")
            if checkpoint_protocol != POLICY_PROTOCOL_VERSION:
                raise CheckpointProtocolMismatch(
                    "Checkpoint policy protocol mismatch: "
                    f"checkpoint={checkpoint_protocol!r}, "
                    f"current={POLICY_PROTOCOL_VERSION}. "
                    "Start a fresh run for stream action-feedback tokens.",
                )
            checkpoint_schema = checkpoint.get("policy_input_schema")
            if checkpoint_schema != POLICY_INPUT_SCHEMA:
                raise CheckpointProtocolMismatch(
                    "Checkpoint policy input schema mismatch: "
                    f"checkpoint={checkpoint_schema!r}, "
                    f"current={POLICY_INPUT_SCHEMA!r}. "
                    "Start a fresh run for stream action-feedback tokens.",
                )
            agent.policy.load_state_dict(checkpoint["agent_state"])
            agent.ppo.optimizer.load_state_dict(checkpoint["optimizer_state"])
            sched_state = checkpoint.get("scheduler_state")
            if sched_state is not None and agent.ppo.scheduler is not None:
                agent.ppo.scheduler.load_state_dict(sched_state)
            agent.ppo.update_count = int(
                checkpoint.get(
                    "global_update_index",
                    checkpoint.get("policy_version", 0),
                )
                or 0
            )
            agent.extractor.load_state_dict(checkpoint.get("extractor_state", {}))
            episode = checkpoint["episode"]
            best_eval_reward = checkpoint.get("best_eval_reward", float("-inf"))
            episode_rewards = deque(
                checkpoint["episode_rewards"],
                maxlen=cfg.environment.reward_window,
            )
            print(f"Checkpoint loaded from episode {episode}.")
            return episode, best_eval_reward, episode_rewards
        except CheckpointProtocolMismatch as exc:
            print(f"Skipping incompatible checkpoint '{checkpoint_path}': {exc}")
            return 0, float("-inf"), deque(maxlen=cfg.environment.reward_window)
        except (EOFError, RuntimeError, Exception) as exc:
            print(
                f"Warning: Failed to load checkpoint '{checkpoint_path}': {exc}",
            )
            print(
                "Starting from scratch. The corrupted checkpoint will be "
                "renamed to avoid future errors.",
            )
            try:
                os.rename(checkpoint_path, checkpoint_path + ".corrupted")
            except OSError:
                pass
            return 0, float("-inf"), deque(maxlen=cfg.environment.reward_window)
    return 0, float("-inf"), deque(maxlen=cfg.environment.reward_window)


def build_effective_config(agent, *, distributed_overrides=None):
    distributed_cfg = getattr(cfg, "distributed", None)
    distributed_payload = None
    if distributed_cfg is not None:
        try:
            distributed_payload = dict(distributed_cfg.items())
        except Exception:
            distributed_payload = None
    if distributed_overrides:
        if distributed_payload is None:
            distributed_payload = {}
        distributed_payload.update(dict(distributed_overrides))
    return {
        "run_name": getattr(cfg.environment, "run_name", ""),
        "environment": {
            "map_name": cfg.environment.map_name,
            "steps_per_episode": int(cfg.environment.steps_per_episode),
            "total_episodes": int(cfg.environment.total_episodes),
            "reward_window": int(cfg.environment.reward_window),
            "eval_frequency": int(getattr(cfg.environment, "eval_frequency", 0)),
            "eval_episodes": int(getattr(cfg.environment, "eval_episodes", 0)),
            "checkpoint_path": cfg.environment.checkpoint_path,
            "best_checkpoint_path": cfg.environment.best_checkpoint_path,
            "db_path": cfg.environment.db_path,
        },
        "model": agent.policy.resolved_config(),
        "ppo": agent.ppo.resolved_config(),
        "reward": (
            agent.reward_function.resolved_config()
            if hasattr(agent.reward_function, "resolved_config")
            else {"name": getattr(agent, "reward_name", "unknown")}
        ),
        "reward_scale": float(agent.reward_scale),
        "total_updates_estimate": int(agent.total_updates_estimate),
        "distributed": distributed_payload,
    }


def write_effective_config(agent, *, distributed_overrides=None):
    payload = build_effective_config(
        agent,
        distributed_overrides=distributed_overrides,
    )
    with open(_run_path("effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def initialize_run_provenance(
    agent,
    *,
    launch_mode,
    resolved_launch=None,
    distributed_overrides=None,
    argv=None,
):
    """Create the immutable run birth certificate and append this launch.

    ``effective_config.json`` remains for existing analysis tools, while the
    manifest preserves the first resolved configuration and the JSONL stream
    records every later resume/config/code phase.
    """
    global _ACTIVE_PHASE_ID

    run_dir = Path(_run_dir()).resolve()
    run_name = str(getattr(cfg.environment, "run_name", ""))
    config_path = Path(getattr(cfg, "config_path", Path(__file__).parent / "config.yaml"))
    checkpoint_path = run_dir / str(cfg.environment.checkpoint_path)
    checkpoint_present = checkpoint_path.exists()
    manifest_preexisting = (run_dir / RUN_MANIFEST_FILENAME).exists()
    effective_config = write_effective_config(
        agent,
        distributed_overrides=distributed_overrides,
    )
    manifest_payload = build_manifest_payload(
        run_name=run_name,
        effective_config=effective_config,
        launch_mode=launch_mode,
        repo_root=Path(__file__).resolve().parent,
        config_path=config_path,
        argv=argv,
        resolved_launch=resolved_launch,
        adopted_existing_run=bool(checkpoint_present and not manifest_preexisting),
    )
    manifest_path, manifest_created = ensure_run_manifest(run_dir, manifest_payload)
    if manifest_created and checkpoint_present:
        event_type = "legacy_adoption"
    elif manifest_created:
        event_type = "start"
    else:
        event_type = "resume"
    phase_id = next_phase_id(run_dir)
    events_path = append_resume_event(
        run_dir,
        event_type=event_type,
        effective_config=effective_config,
        launch_mode=launch_mode,
        repo_root=Path(__file__).resolve().parent,
        config_path=config_path,
        checkpoint_present=checkpoint_present,
        phase_id=phase_id,
        argv=argv,
        resolved_launch=resolved_launch,
    )
    print(
        f"Run provenance: {manifest_path.name} "
        f"({'created' if manifest_created else 'preserved'}); "
        f"appended {events_path.name} ({event_type}, phase={phase_id}).",
    )
    _ACTIVE_PHASE_ID = int(phase_id)
    return manifest_path, events_path, phase_id


def save_initial_config():
    """Save the birth config once; never rewrite it during resume."""
    src_config = Path(getattr(cfg, "config_path", Path(__file__).parent / "config.yaml"))
    if src_config.exists():
        dst_config = Path(_run_path("config.yaml"))
        if dst_config.exists():
            print(f"Initial config preserved at {dst_config}")
            return
        shutil.copy2(src_config, dst_config)
        print(f"Initial config saved to {dst_config}")
    else:
        print(f"Warning: config.yaml not found at {src_config}")


def run_eval_sweep(
    env,
    agent,
    episodes,
    steps_per_episode,
    deterministic=True,
    reset_lock=None,
):
    """Run `episodes` deterministic eval episodes and summarize native reward.

    `reset_lock`, when given, is a zero-arg callable returning a context manager
    that guards each `env.reset()`. Ray actors pass the cross-process SC2
    create-game lock so parallel eval resets stay serialized exactly like
    training resets; the single-process caller leaves it None.
    """
    rewards = []
    episode_results = []
    was_training = agent.policy.training
    agent.policy.eval()
    try:
        for episode_number in range(episodes):
            if reset_lock is not None:
                with reset_lock():
                    obs = env.reset()[0]
            else:
                obs = env.reset()[0]
            agent.reset()
            ep_reward = 0.0
            steps = 0
            action_counts = {
                "policy_no_op_count": 0,
                "policy_left_click_count": 0,
                "policy_right_click_count": 0,
            }
            while True:
                step_result = agent.step(obs, deterministic=deterministic)
                action_func, action_id = step_result[0], step_result[1]
                action_key = {
                    POLICY_ACTION_NO_OP: "policy_no_op_count",
                    POLICY_ACTION_LEFT_CLICK: "policy_left_click_count",
                    POLICY_ACTION_RIGHT_CLICK: "policy_right_click_count",
                }.get(None if action_id is None else int(action_id))
                if action_key is not None:
                    action_counts[action_key] += 1
                next_obs = env.step([action_func])[0]
                steps += 1
                ep_reward += float(next_obs.reward)
                obs = next_obs
                terminated = bool(next_obs.last())
                truncated = bool(steps >= steps_per_episode and not terminated)
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            episode_results.append(
                {
                    "episode_number": int(episode_number),
                    "native_reward": float(ep_reward),
                    "steps": int(steps),
                    "terminated": terminated,
                    "truncated": truncated,
                    **action_counts,
                }
            )
    finally:
        if was_training:
            agent.policy.train()

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return {
        "num_episodes": int(len(rewards)),
        "mean_reward": float(rewards_arr.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards_arr.std()) if len(rewards) else 0.0,
        "min_reward": float(rewards_arr.min()) if len(rewards) else 0.0,
        "max_reward": float(rewards_arr.max()) if len(rewards) else 0.0,
        "deterministic": bool(deterministic),
        "episode_results": episode_results,
    }


def maybe_run_policy_update(agent, log_queue, episode_index):
    if agent.ppo.memory:
        agent.ppo.finalize_fragment(
            actor_id=0,
            policy_version=int(getattr(agent.ppo, "update_count", 0)),
        )
    fragments = agent.ppo.consume_pending_fragments()
    if not fragments:
        return None
    stats = agent.update_policy(fragments=fragments)
    if stats is not None:
        policy_version = int(stats.get("policy_version", stats.get("global_update_index", 0)) or 0)
        log_queue.put(
            {
                "type": "UPDATE",
                "phase_id": current_phase_id(),
                "internal_ep": episode_index - 1,
                "episode_index": episode_index,
                "policy_version": policy_version,
                "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                "policy_input_schema": POLICY_INPUT_SCHEMA,
                **stats,
            }
        )
    return stats


def train_agent(env, agent, observation_extractor, log_queue):
    del observation_extractor

    episode_count = cfg.environment.total_episodes
    rollout_steps = int(getattr(cfg.hyperparameters, "rollout_steps", 2048) or 2048)
    reward_scale = float(getattr(cfg.hyperparameters, "reward_scale", 1.0))
    eval_frequency = int(getattr(cfg.environment, "eval_frequency", 0) or 0)
    eval_episodes = int(getattr(cfg.environment, "eval_episodes", 0) or 0)

    start_episode, best_eval_reward, episode_rewards = load_checkpoint(agent)

    steps_per_episode = cfg.environment.steps_per_episode
    learnable_steps = 0
    helper_steps = 0

    for episode in range(start_episode, episode_count):
        policy_version = int(getattr(agent.ppo, "update_count", 0))
        obs = env.reset()[0]
        agent.reset()
        agent.reward_function.calculate_reward(obs, None)

        episode_reward = 0.0
        episode_native_reward = 0.0
        episode_reward_components = {}
        episode_action_counts = {
            "policy_no_op_count": 0,
            "policy_left_click_count": 0,
            "policy_right_click_count": 0,
        }
        cumulative_reward = 0.0
        step_count = 0

        log_queue.put(
            {
                "type": "EPISODE_START",
                "phase_id": current_phase_id(),
                "internal_ep": episode,
                "actor_id": 0,
                "policy_version": policy_version,
            }
        )

        while True:
            (
                action_func,
                action_id,
                move_x,
                move_y,
                pre_step_state,
                log_prob,
                value,
                policy_input,
                learnable,
            ) = agent.step(obs)

            next_obs = env.step([action_func])[0]
            episode_native_reward += float(next_obs.reward)

            step_count += 1
            env_done = next_obs.last()
            time_cap = step_count >= steps_per_episode
            # Treat time_cap as truncation, not terminal:
            # - terminal (done) is only true game endings
            # - timeout continues bootstrapping through the cap
            terminal = env_done
            done = terminal

            raw_reward = agent.reward_function.calculate_reward(next_obs, None)
            raw_reward = float(
                raw_reward.item() if isinstance(raw_reward, torch.Tensor) else raw_reward
            )
            scaled_reward = raw_reward * reward_scale

            if policy_input is not None:
                action_sample = getattr(agent, "last_action_sample", None)
                agent.ppo.store_transition(
                    policy_input,
                    torch.tensor(action_id, device=agent.policy.device),
                    torch.tensor(move_x, device=agent.policy.device),
                    torch.tensor(move_y, device=agent.policy.device),
                    torch.tensor(log_prob, device=agent.policy.device),
                    torch.tensor(scaled_reward, device=agent.policy.device),
                    torch.tensor(value, device=agent.policy.device),
                    torch.tensor(
                        done,
                        dtype=torch.float32,
                        device=agent.policy.device,
                    ),
                    sample_mask=torch.tensor(
                        1.0 if learnable else 0.0,
                        dtype=torch.float32,
                        device=agent.policy.device,
                    ),
                    truncated=torch.tensor(
                        bool(time_cap and not terminal),
                        dtype=torch.float32,
                        device=agent.policy.device,
                    ),
                    episode_reset=torch.tensor(
                        bool(env_done or time_cap),
                        dtype=torch.bool,
                        device=agent.policy.device,
                    ),
                    target_index=(
                        None
                        if action_sample is None or action_sample.target_index is None
                        else torch.tensor(
                            action_sample.target_index,
                            device=agent.policy.device,
                        )
                    ),
                    coarse_index=(
                        None
                        if action_sample is None or action_sample.coarse_index is None
                        else torch.tensor(
                            action_sample.coarse_index,
                            device=agent.policy.device,
                        )
                    ),
                    fine_index=(
                        None
                        if action_sample is None or action_sample.fine_index is None
                        else torch.tensor(
                            action_sample.fine_index,
                            device=agent.policy.device,
                        )
                    ),
                )
                next_policy_input = agent.peek_observation(next_obs).with_state(
                    agent.snn_state,
                )
                agent.ppo.set_final_next(next_policy_input)
                if learnable:
                    learnable_steps += 1
                else:
                    helper_steps += 1
            else:
                helper_steps += 1

            episode_reward += raw_reward
            cumulative_reward += raw_reward
            action_key = {
                POLICY_ACTION_NO_OP: "policy_no_op_count",
                POLICY_ACTION_LEFT_CLICK: "policy_left_click_count",
                POLICY_ACTION_RIGHT_CLICK: "policy_right_click_count",
            }.get(None if action_id is None else int(action_id))
            if action_key is not None:
                episode_action_counts[action_key] += 1

            log_queue.put(
                {
                    "type": "STEP",
                    "internal_ep": episode,
                    "actor_id": 0,
                    "policy_version": int(getattr(agent.ppo, "update_count", 0)),
                    "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                    "policy_input_schema": POLICY_INPUT_SCHEMA,
                    "step": step_count,
                    "act": None if action_id is None else int(action_id),
                    "move_x": None if action_id is None else int(move_x),
                    "move_y": None if action_id is None else int(move_y),
                    "rew": float(raw_reward),
                    "cum_rew": float(cumulative_reward),
                }
            )

            reward_info = agent.reward_function.get_last_reward_components()
            if reward_info:
                for name, value in reward_info.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    episode_reward_components[name] = (
                        episode_reward_components.get(name, 0.0) + numeric
                    )
                log_queue.put(
                    {
                        "type": "REWARD_COMP",
                        "internal_ep": episode,
                        "actor_id": 0,
                        "step": step_count,
                        "h_rew": float(reward_info["health_reward"]),
                        "e_rew": float(reward_info["engagement_reward"]),
                        "p_rew": float(reward_info["positioning_reward"]),
                        "s_rew": float(reward_info["score_reward"]),
                        "b_rew": float(reward_info["bonus_reward"]),
                        "end_rew": float(reward_info["end_of_episode_reward"]),
                        "tot_rew": float(reward_info["total_reward"]),
                    }
                )

            if agent.ppo.pending_rollout_steps() >= rollout_steps:
                maybe_run_policy_update(agent, log_queue, episode + 1)

            # Exit on true terminal or time cap (but cap is truncation, not terminal)
            if env_done or time_cap:
                break
            obs = next_obs

        episode_rewards.append(episode_reward)
        avg_reward = float(np.mean(episode_rewards))

        log_queue.put(
            {
                "type": "EPISODE_END",
                "phase_id": current_phase_id(),
                "internal_ep": episode,
                "total": float(episode_reward),
                "shaped_reward": float(episode_reward),
                "native_reward": float(episode_native_reward),
                "avg": float(avg_reward),
                "steps": step_count,
                "terminated": bool(env_done),
                "truncated": bool(time_cap and not env_done),
                "reward_components": episode_reward_components,
                **episode_action_counts,
            }
        )

        if agent.ppo.memory:
            agent.ppo.finalize_fragment(
                actor_id=0,
                policy_version=int(getattr(agent.ppo, "update_count", 0)),
            )

        if agent.ppo.pending_rollout_steps(include_current=False) >= rollout_steps:
            maybe_run_policy_update(agent, log_queue, episode + 1)

        if eval_frequency > 0 and eval_episodes > 0 and (episode + 1) % eval_frequency == 0:
            eval_summary = run_eval_sweep(
                env=env,
                agent=agent,
                episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                deterministic=True,
            )
            log_queue.put(
                {
                    "type": "EVAL",
                    "phase_id": current_phase_id(),
                    "episode_index": episode + 1,
                    "policy_version": int(getattr(agent.ppo, "update_count", 0)),
                    "policy_protocol_version": POLICY_PROTOCOL_VERSION,
                    "policy_input_schema": POLICY_INPUT_SCHEMA,
                    **eval_summary,
                }
            )
            logger.info(
                "Eval @ ep %s | mean=%.2f std=%.2f min=%.2f max=%.2f",
                episode + 1,
                eval_summary["mean_reward"],
                eval_summary["std_reward"],
                eval_summary["min_reward"],
                eval_summary["max_reward"],
            )
            best_eval_reward = maybe_save_best_checkpoint(
                agent,
                episode + 1,
                avg_reward,
                eval_summary,
                best_eval_reward,
                episode_rewards,
            )

        if (episode + 1) % cfg.environment.log_frequency == 0:
            save_checkpoint(
                agent,
                episode + 1,
                best_eval_reward,
                episode_rewards,
                avg_reward=avg_reward,
                require_rollout_clear=False,
            )
            total_steps = learnable_steps + helper_steps
            helper_pct = (helper_steps / total_steps * 100) if total_steps else 0
            logger.info(
                "Ep %s | Avg: %.2f | Rollout: %s | Helper: %.1f%%",
                episode + 1,
                avg_reward,
                agent.ppo.pending_rollout_steps(),
                helper_pct,
            )

    if agent.ppo.memory:
        agent.ppo.finalize_fragment(
            actor_id=0,
            policy_version=int(getattr(agent.ppo, "update_count", 0)),
        )
    if agent.ppo.pending_fragments:
        maybe_run_policy_update(agent, log_queue, episode_count)

    return best_eval_reward


def main(argv):
    del argv

    _apply_runtime_flags()

    log_queue = Queue()

    db_listener = LogListener(log_queue, _run_path(cfg.environment.db_path))
    print(f"Run directory: {_run_dir()}")
    save_initial_config()
    db_listener.start()

    env = None
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=False,
            use_action_printer=cfg.environment.use_action_printer,
            use_available_actions_diagnostics=getattr(
                cfg.environment,
                "use_available_actions_diagnostics",
                False,
            ),
            available_actions_diagnostics_output_path=getattr(
                cfg.environment,
                "available_actions_diagnostics_output_path",
                "analysis_results/available_actions_diagnostics.jsonl",
            ),
            available_actions_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "available_actions_diagnostics_every_n_steps",
                1,
            ),
            use_last_action_diagnostics=getattr(
                cfg.environment,
                "use_last_action_diagnostics",
                False,
            ),
            last_action_diagnostics_output_path=getattr(
                cfg.environment,
                "last_action_diagnostics_output_path",
                "analysis_results/last_action_diagnostics.jsonl",
            ),
            last_action_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "last_action_diagnostics_every_n_steps",
                1,
            ),
            use_score_diagnostics=getattr(
                cfg.environment,
                "use_score_diagnostics",
                False,
            ),
            score_diagnostics_output_path=getattr(
                cfg.environment,
                "score_diagnostics_output_path",
                "analysis_results/score_diagnostics.jsonl",
            ),
            score_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "score_diagnostics_every_n_steps",
                1,
            ),
            use_observation_inspector=getattr(
                cfg.environment,
                "use_observation_inspector",
                False,
            ),
            observation_inspector_output_path=getattr(
                cfg.environment,
                "observation_inspector_output_path",
                "analysis_results/observation_space.jsonl",
            ),
            observation_inspector_every_n_steps=getattr(
                cfg.environment,
                "observation_inspector_every_n_steps",
                10,
            ),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment,
                "observation_inspector_max_unit_samples",
                5,
            ),
            use_policy_input_diagnostics=getattr(
                cfg.environment,
                "use_policy_input_diagnostics",
                False,
            ),
            policy_input_diagnostics_output_path=getattr(
                cfg.environment,
                "policy_input_diagnostics_output_path",
                "analysis_results/policy_input_diagnostics.jsonl",
            ),
            policy_input_diagnostics_every_n_steps=getattr(
                cfg.environment,
                "policy_input_diagnostics_every_n_steps",
                1,
            ),
            policy_input_diagnostics_max_entity_samples=getattr(
                cfg.environment,
                "policy_input_diagnostics_max_entity_samples",
                3,
            ),
            policy_input_diagnostics_max_selection_samples=getattr(
                cfg.environment,
                "policy_input_diagnostics_max_selection_samples",
                3,
            ),
        )
    except Exception as exc:
        print(f"Env setup failed: {exc}")
        db_listener.terminate()
        return

    try:
        observation_extractor = ObservationExtractor()
        initial_obs = env.reset()[0]
        spatial_shape, vector_dim = observation_extractor.get_observation_dimensions(
            initial_obs,
        )

        agent = DefeatRoaches(
            spatial_input_shape=spatial_shape,
            vector_input_dim=vector_dim,
            action_dim=cfg.model.action_dim,
        )
        initialize_run_provenance(
            agent,
            launch_mode="single_process",
            resolved_launch={"run_name": str(cfg.environment.run_name)},
        )

        print("Starting Async PPO training...")
        best_reward = train_agent(
            env=env,
            agent=agent,
            observation_extractor=observation_extractor,
            log_queue=log_queue,
        )

        print(f"Done! Best deterministic eval reward: {best_reward:.2f}")

    except Exception as exc:
        print(f"Critical: {exc}")
        raise exc
    finally:
        if env is not None:
            env.close()
        try:
            log_queue.put({"type": "KILL"})
        except Exception as exc:
            print(f"Warning: failed to send logger shutdown event: {exc}")
        db_listener.join(timeout=10)
        if db_listener.is_alive():
            print("Warning: logger did not stop cleanly; terminating it.")
            db_listener.terminate()
            db_listener.join(timeout=5)


if __name__ == "__main__":
    app.run(main)
