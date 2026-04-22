import json
import logging
import os
import time
from collections import deque
from multiprocessing import Manager

import numpy as np
import torch
from absl import app, flags

from agent import DefeatRoaches
from Utility.config import cfg
from Utility.logger_utils import LogListener
from envs.setup_env import create_env
from obs_space.obs_space_2 import ObservationExtractor


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
                cfg.environment, "use_available_actions_diagnostics", False,
            ),
            available_actions_diagnostics_output_path=getattr(
                cfg.environment,
                "available_actions_diagnostics_output_path",
                "analysis_results/available_actions_diagnostics.jsonl",
            ),
            available_actions_diagnostics_every_n_steps=getattr(
                cfg.environment, "available_actions_diagnostics_every_n_steps", 1,
            ),
            use_observation_inspector=getattr(
                cfg.environment, "use_observation_inspector", False,
            ),
            observation_inspector_output_path=getattr(
                cfg.environment,
                "observation_inspector_output_path",
                "analysis_results/observation_space.jsonl",
            ),
            observation_inspector_every_n_steps=getattr(
                cfg.environment, "observation_inspector_every_n_steps", 10,
            ),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment, "observation_inspector_max_unit_samples", 5,
            ),
            use_policy_input_diagnostics=getattr(
                cfg.environment, "use_policy_input_diagnostics", False,
            ),
            policy_input_diagnostics_output_path=getattr(
                cfg.environment,
                "policy_input_diagnostics_output_path",
                "analysis_results/policy_input_diagnostics.jsonl",
            ),
            policy_input_diagnostics_every_n_steps=getattr(
                cfg.environment, "policy_input_diagnostics_every_n_steps", 1,
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
    if require_rollout_clear and len(agent.ppo.memory) > 0:
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
            agent.ppo.scheduler.state_dict()
            if agent.ppo.scheduler is not None
            else None
        ),
        "episode": episode,
        "best_eval_reward": best_eval_reward,
        "best_avg_reward": avg_reward,
        "avg_reward_at_save": avg_reward,
        "eval_reward_at_save": eval_reward,
        "episode_rewards": list(episode_rewards),
        "extractor_state": agent.extractor.state_dict(),
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
        cfg.environment, "best_checkpoint_path", "best_checkpoint.pth",
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
    )
    if not did_save:
        return best_eval_reward

    print(
        f"  -> new best deterministic eval reward: {eval_reward:.2f} "
        f"(was {best_eval_reward:.2f})",
    )
    return eval_reward


def load_checkpoint(agent, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = _run_path(cfg.environment.checkpoint_path)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu"),
            )
            agent.policy.load_state_dict(checkpoint["agent_state"])
            agent.ppo.optimizer.load_state_dict(checkpoint["optimizer_state"])
            sched_state = checkpoint.get("scheduler_state")
            if sched_state is not None and agent.ppo.scheduler is not None:
                agent.ppo.scheduler.load_state_dict(sched_state)
            agent.extractor.load_state_dict(checkpoint.get("extractor_state", {}))
            episode = checkpoint["episode"]
            best_eval_reward = checkpoint.get("best_eval_reward", float("-inf"))
            episode_rewards = deque(
                checkpoint["episode_rewards"],
                maxlen=cfg.environment.reward_window,
            )
            print(f"Checkpoint loaded from episode {episode}.")
            return episode, best_eval_reward, episode_rewards
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


def write_effective_config(agent):
    payload = {
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
    }
    with open(_run_path("effective_config.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_eval_sweep(env, agent, episodes, steps_per_episode, deterministic=True):
    rewards = []
    was_training = agent.policy.training
    agent.policy.eval()
    try:
        for _ in range(episodes):
            obs = env.reset()[0]
            agent.reset()
            ep_reward = 0.0
            steps = 0
            while True:
                action_func = agent.step(obs, deterministic=deterministic)[0]
                next_obs = env.step([action_func])[0]
                steps += 1
                ep_reward += float(next_obs.reward)
                obs = next_obs
                if next_obs.last() or steps >= steps_per_episode:
                    break
            rewards.append(ep_reward)
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
    }


def maybe_run_policy_update(agent, log_queue, episode_index):
    if not agent.ppo.memory:
        return None
    stats = agent.update_policy()
    if stats is not None:
        log_queue.put(
            {
                "type": "UPDATE",
                "internal_ep": episode_index - 1,
                "episode_index": episode_index,
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
        obs = env.reset()[0]
        agent.reset()
        agent.reward_function.calculate_reward(obs, None)

        episode_reward = 0.0
        cumulative_reward = 0.0
        step_count = 0

        log_queue.put({"type": "EPISODE_START", "internal_ep": episode})

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

            step_count += 1
            env_done = next_obs.last()
            time_cap = step_count >= steps_per_episode
            done = env_done or time_cap

            raw_reward = agent.reward_function.calculate_reward(next_obs, None)
            raw_reward = float(
                raw_reward.item() if isinstance(raw_reward, torch.Tensor) else raw_reward
            )
            scaled_reward = raw_reward * reward_scale

            if policy_input is not None:
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

            log_queue.put(
                {
                    "type": "STEP",
                    "internal_ep": episode,
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
                log_queue.put(
                    {
                        "type": "REWARD_COMP",
                        "internal_ep": episode,
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

            if done:
                break
            obs = next_obs

        episode_rewards.append(episode_reward)
        avg_reward = float(np.mean(episode_rewards))

        log_queue.put(
            {
                "type": "EPISODE_END",
                "internal_ep": episode,
                "total": float(episode_reward),
                "avg": float(avg_reward),
                "steps": step_count,
            }
        )

        if len(agent.ppo.memory) >= rollout_steps:
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
                    "episode_index": episode + 1,
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
            )
            total_steps = learnable_steps + helper_steps
            helper_pct = (helper_steps / total_steps * 100) if total_steps else 0
            logger.info(
                "Ep %s | Avg: %.2f | Rollout: %s | Helper: %.1f%%",
                episode + 1,
                avg_reward,
                len(agent.ppo.memory),
                helper_pct,
            )

    if agent.ppo.memory:
        maybe_run_policy_update(agent, log_queue, episode_count)

    return best_eval_reward


def main(argv):
    del argv

    _apply_runtime_flags()

    manager = Manager()
    log_queue = manager.Queue()

    db_listener = LogListener(log_queue, _run_path(cfg.environment.db_path))
    print(f"Run directory: {_run_dir()}")
    db_listener.start()

    env = None
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=False,
            use_action_printer=cfg.environment.use_action_printer,
            use_available_actions_diagnostics=getattr(
                cfg.environment, "use_available_actions_diagnostics", False,
            ),
            available_actions_diagnostics_output_path=getattr(
                cfg.environment,
                "available_actions_diagnostics_output_path",
                "analysis_results/available_actions_diagnostics.jsonl",
            ),
            available_actions_diagnostics_every_n_steps=getattr(
                cfg.environment, "available_actions_diagnostics_every_n_steps", 1,
            ),
            use_observation_inspector=getattr(
                cfg.environment, "use_observation_inspector", False,
            ),
            observation_inspector_output_path=getattr(
                cfg.environment,
                "observation_inspector_output_path",
                "analysis_results/observation_space.jsonl",
            ),
            observation_inspector_every_n_steps=getattr(
                cfg.environment, "observation_inspector_every_n_steps", 10,
            ),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment, "observation_inspector_max_unit_samples", 5,
            ),
            use_policy_input_diagnostics=getattr(
                cfg.environment, "use_policy_input_diagnostics", False,
            ),
            policy_input_diagnostics_output_path=getattr(
                cfg.environment,
                "policy_input_diagnostics_output_path",
                "analysis_results/policy_input_diagnostics.jsonl",
            ),
            policy_input_diagnostics_every_n_steps=getattr(
                cfg.environment, "policy_input_diagnostics_every_n_steps", 1,
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
        write_effective_config(agent)

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
        log_queue.put({"type": "KILL"})
        db_listener.join()


if __name__ == "__main__":
    app.run(main)
