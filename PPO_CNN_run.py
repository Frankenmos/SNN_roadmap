import os
import time
import logging
from absl import app
from PPO_CNN_agent import DefeatRoaches
from obs_space.obs_space_2 import ObservationExtractor
from envs.setup_env import create_env
import numpy as np
import torch
from collections import deque
from Utility.logger_utils import LogListener
from multiprocessing import Manager
from Utility.config import cfg


def _run_dir() -> str:
    """Return the per-run directory ({models_dir}/{run_name}/), creating it
    if missing. Auto-generates a timestamped run_name when the config has
    an empty value, and writes the generated name back into cfg so every
    caller in this process sees the same name."""
    run_name = getattr(cfg.environment, "run_name", "") or ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        cfg.environment.run_name = run_name
    models_dir = getattr(cfg.environment, "models_dir", "models")
    run_dir = os.path.join(models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _run_path(filename: str) -> str:
    """Join a bare filename (from config) to the per-run directory."""
    return os.path.join(_run_dir(), filename)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_environment():
    """Resets the environment and ensures it is ready for a new training run."""
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=cfg.environment.visualize,
            use_action_printer=cfg.environment.use_action_printer,
            use_observation_inspector=getattr(cfg.environment, "use_observation_inspector", False),
            observation_inspector_output_path=getattr(
                cfg.environment, "observation_inspector_output_path", "analysis_results/observation_space.jsonl"
            ),
            observation_inspector_every_n_steps=getattr(cfg.environment, "observation_inspector_every_n_steps", 10),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment, "observation_inspector_max_unit_samples", 5
            ),
        )
    except Exception as e:
        print(f"Error during environment reset: {e}")
        raise e
    return env


def save_checkpoint(agent, episode, best_avg_reward, episode_rewards, checkpoint_path=None, avg_reward=None):
    """Save training checkpoint atomically."""
    if checkpoint_path is None:
        checkpoint_path = _run_path(cfg.environment.checkpoint_path)
    checkpoint = {
        'agent_state': agent.policy.state_dict(),
        'optimizer_state': agent.ppo.optimizer.state_dict(),
        'scheduler_state': (
            agent.ppo.scheduler.state_dict()
            if agent.ppo.scheduler is not None else None
        ),
        'episode': episode,
        'best_avg_reward': best_avg_reward,
        'avg_reward_at_save': avg_reward,
        'episode_rewards': list(episode_rewards),
    }

    # Save to a temporary file first
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)

    # Atomically rename to the final path
    try:
        os.replace(temp_path, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} at episode {episode}.")
    except OSError as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def maybe_save_best_checkpoint(agent, episode, avg_reward, best_avg_reward, episode_rewards):
    """Save a separate best-of-run checkpoint if avg_reward beats the prior best.

    Gated on a minimum episode count so the first few noisy rolling averages
    don't set an unbeatable floor. Returns the (possibly updated) best value.
    """
    best_name = getattr(cfg.environment, "best_checkpoint_path", "best_checkpoint.pth")
    best_path = _run_path(best_name)
    min_eps = getattr(cfg.environment, "best_min_episodes", 50)
    if episode < min_eps:
        return best_avg_reward
    if avg_reward <= best_avg_reward:
        return best_avg_reward

    save_checkpoint(
        agent, episode, avg_reward, episode_rewards,
        checkpoint_path=best_path, avg_reward=avg_reward,
    )
    print(f"  -> new best avg reward: {avg_reward:.2f} (was {best_avg_reward:.2f})")
    return float(avg_reward)


def load_checkpoint(agent, checkpoint_path=None):
    """Load training checkpoint with error handling."""
    if checkpoint_path is None:
        checkpoint_path = _run_path(cfg.environment.checkpoint_path)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            agent.policy.load_state_dict(checkpoint['agent_state'])
            agent.ppo.optimizer.load_state_dict(checkpoint['optimizer_state'])
            # Restore LR schedule if both sides have one.
            sched_state = checkpoint.get('scheduler_state')
            if sched_state is not None and agent.ppo.scheduler is not None:
                agent.ppo.scheduler.load_state_dict(sched_state)
            episode = checkpoint['episode']
            best_avg_reward = checkpoint['best_avg_reward']
            episode_rewards = deque(checkpoint['episode_rewards'], maxlen=cfg.environment.reward_window)
            print(f"Checkpoint loaded from episode {episode}.")
            return episode, best_avg_reward, episode_rewards
        except (EOFError, RuntimeError, Exception) as e:
            print(f"Warning: Failed to load checkpoint '{checkpoint_path}': {e}")
            print("Starting from scratch. The corrupted checkpoint will be renamed to avoid future errors.")
            try:
                os.rename(checkpoint_path, checkpoint_path + ".corrupted")
            except OSError:
                pass
            return 0, float('-inf'), deque(maxlen=cfg.environment.reward_window)
    return 0, float('-inf'), deque(maxlen=cfg.environment.reward_window)

def train_agent(env, agent, observation_extractor, log_queue):
    # Load config
    episode_count = cfg.environment.total_episodes
    update_frequency = cfg.environment.update_frequency # e.g., 10 episodes
    
    # Load Checkpoint
    start_episode, best_avg_reward, episode_rewards = load_checkpoint(agent)

    for episode in range(start_episode, episode_count):
        obs = env.reset()[0]
        episode_reward = 0
        cumulative_reward = 0
        step_count = 0
        
        # 1. Async Log: Start Episode (Pass simple int ID)
        log_queue.put({'type': 'EPISODE_START', 'internal_ep': episode})

        # --- FAST STEP LOOP ---
        # Because logging is async, this loop runs at full 5090 speed
        while True:
            # Agent Step
            (action_func, action_id, move_x, move_y, log_prob, value,
             spatial, vector, reward) = agent.step(obs)

            next_obs = env.step([action_func])[0]
            done = next_obs.last()

            # Store Data (RAM only, fast)
            agent.ppo.store_transition(
                spatial, vector,
                torch.tensor(action_id, device=agent.policy.device),
                torch.tensor(move_x, device=agent.policy.device),
                torch.tensor(move_y, device=agent.policy.device),
                torch.tensor(log_prob, device=agent.policy.device),
                torch.tensor(reward, device=agent.policy.device),
                torch.tensor(value, device=agent.policy.device),
                torch.tensor(done, device=agent.policy.device),
            )

            episode_reward += reward
            cumulative_reward += reward
            step_count += 1

            # NEW: hard cap from YAML
            if step_count >= cfg.environment.steps_per_episode:
                done = True

            # Async logging...
            # 2. Async Log: Step Data
            log_queue.put({
                'type': 'STEP', 'internal_ep': episode, 'step': step_count,
                'act': int(action_id),
                'move_x': int(move_x), 'move_y': int(move_y),
                'rew': float(reward), 'cum_rew': float(cumulative_reward)
            })
            
            # 3. Async Log: Components (Optional: Only log every 10 steps to save overhead?)
            reward_info = agent.reward_function.get_last_reward_components()
            if reward_info:
                log_queue.put({
                    'type': 'REWARD_COMP', 'internal_ep': episode, 'step': step_count,
                    'h_rew': float(reward_info['health_reward']),
                    'e_rew': float(reward_info['engagement_reward']),
                    'p_rew': float(reward_info['positioning_reward']),
                    's_rew': float(reward_info['score_reward']),
                    'b_rew': float(reward_info['bonus_reward']),
                    'end_rew': float(reward_info['end_of_episode_reward']),
                    'tot_rew': float(reward_info['total_reward'])
                })

            if done: break
            obs = next_obs

        # --- END OF EPISODE ---
        episode_rewards.append(episode_reward)
        avg_reward = float(np.mean(episode_rewards))
        best_avg_reward = maybe_save_best_checkpoint(
            agent, episode + 1, avg_reward, best_avg_reward, episode_rewards,
        )
        
        log_queue.put({
            'type': 'EPISODE_END', 'internal_ep': episode, 
            'total': float(episode_reward), 'avg': float(avg_reward), 'steps': step_count
        })

        # --- THE FREEZE FIX ---
        # This will still pause the game, but with batch_size=1024 it will take 1s instead of 15s.
        if (episode + 1) % update_frequency == 0:
            stats = agent.update_policy()  # Now uses Optimized Big-Batch Update
            if stats is not None:
                log_queue.put({
                    'type': 'UPDATE',
                    'internal_ep': episode,
                    **stats,
                })

        # Checkpointing
        if (episode + 1) % cfg.environment.log_frequency == 0:
            save_checkpoint(agent, episode + 1, best_avg_reward, episode_rewards)
            logger.info(f"Ep {episode+1} | Avg: {avg_reward:.2f}")

    return best_avg_reward

def main(argv):
    # 1. Setup Async Queue
    manager = Manager()
    log_queue = manager.Queue()
    
    # 2. Start Background Logger. _run_path joins under models/{run_name}/
    #    and creates the run dir if missing.
    db_listener = LogListener(log_queue, _run_path(cfg.environment.db_path))
    print(f"Run directory: {_run_dir()}")
    db_listener.start()
    
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=False, # Force False for 5090 stability
            use_action_printer=cfg.environment.use_action_printer,
            use_observation_inspector=getattr(cfg.environment, "use_observation_inspector", False),
            observation_inspector_output_path=getattr(
                cfg.environment, "observation_inspector_output_path", "analysis_results/observation_space.jsonl"
            ),
            observation_inspector_every_n_steps=getattr(cfg.environment, "observation_inspector_every_n_steps", 10),
            observation_inspector_max_unit_samples=getattr(
                cfg.environment, "observation_inspector_max_unit_samples", 5
            ),
        )
    except Exception as e:
        print(f"Env setup failed: {e}")
        db_listener.terminate()
        return

    try:
        observation_extractor = ObservationExtractor()
        initial_obs = env.reset()[0]
        spatial_shape, vector_dim = observation_extractor.get_observation_dimensions(initial_obs)

        agent = DefeatRoaches(
            spatial_input_shape=spatial_shape,
            vector_input_dim=vector_dim,
            action_dim=cfg.model.action_dim
        )

        print("Starting Async PPO training...")
        best_reward = train_agent(
            env=env,
            agent=agent,
            observation_extractor=observation_extractor,
            log_queue=log_queue # Pass queue instead of DB path
        )

        print(f"Done! Best reward: {best_reward:.2f}")

    except Exception as e:
        print(f"Critical: {e}")
        raise e
    finally:
        env.close()
        # Ensure logger finishes writing
        log_queue.put({'type': 'KILL'}) 
        db_listener.join()

if __name__ == '__main__':
    app.run(main)
