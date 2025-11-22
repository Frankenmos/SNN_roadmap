import os
import sqlite3
import logging
import json
from absl import app
from pysc2.env.run_loop import run_loop
from PPO_CNN_agent import DefeatRoaches
from obs_space.obs_space_2 import ObservationExtractor
from envs.setup_env import create_env
import numpy as np
import torch
from collections import deque
import time
from Utility.logger_utils import LogListener, initialize_db
from multiprocessing import Manager
from Utility.config import cfg
from PPO_CNN.PPO import PPO
from PPO_CNN.policy_network import PolicyNetwork

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_environment():
    """Resets the environment and ensures it is ready for a new training run."""
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=cfg.environment.visualize,
            use_action_printer=cfg.environment.use_action_printer
        )
    except Exception as e:
        print(f"Error during environment reset: {e}")
        raise e
    return env


def save_checkpoint(agent, episode, best_avg_reward, episode_rewards, checkpoint_path=None):
    """Save training checkpoint atomically."""
    if checkpoint_path is None:
        checkpoint_path = cfg.environment.checkpoint_path
    checkpoint = {
        'agent_state': agent.policy.state_dict(),
        'optimizer_state': agent.ppo.optimizer.state_dict(),
        'episode': episode,
        'best_avg_reward': best_avg_reward,
        'episode_rewards': list(episode_rewards),
    }
    
    # Save to a temporary file first
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)
    
    # Atomically rename to the final path
    try:
        os.replace(temp_path, checkpoint_path)
        print(f"Checkpoint saved at episode {episode}.")
    except OSError as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_checkpoint(agent, checkpoint_path=None):
    """Load training checkpoint with error handling."""
    if checkpoint_path is None:
        checkpoint_path = cfg.environment.checkpoint_path
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            agent.policy.load_state_dict(checkpoint['agent_state'])
            agent.ppo.optimizer.load_state_dict(checkpoint['optimizer_state'])
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
            action_func, action_id, log_prob, value, spatial, vector, reward = agent.step(obs)

            next_obs = env.step([action_func])[0]
            done = next_obs.last()

            # Store Data (RAM only, fast)
            agent.ppo.store_transition(
                spatial, vector,
                torch.tensor(action_id, device=agent.policy.device),
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
                'act': int(action_id), 'rew': float(reward), 'cum_rew': float(cumulative_reward)
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
        avg_reward = np.mean(episode_rewards)
        
        log_queue.put({
            'type': 'EPISODE_END', 'internal_ep': episode, 
            'total': float(episode_reward), 'avg': float(avg_reward), 'steps': step_count
        })

        # --- THE FREEZE FIX ---
        # This will still pause the game, but with batch_size=1024 it will take 1s instead of 15s.
        if (episode + 1) % update_frequency == 0:
            agent.update_policy() # Now uses Optimized Big-Batch Update

        # Checkpointing
        if (episode + 1) % cfg.environment.log_frequency == 0:
            save_checkpoint(agent, episode + 1, best_avg_reward, episode_rewards)
            logger.info(f"Ep {episode+1} | Avg: {avg_reward:.2f}")

    return best_avg_reward

def main(argv):
    # 1. Setup Async Queue
    manager = Manager()
    log_queue = manager.Queue()
    
    # 2. Start Background Logger
    db_listener = LogListener(log_queue, cfg.environment.db_path)
    db_listener.start()
    
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=False, # Force False for 5090 stability
            use_action_printer=cfg.environment.use_action_printer,
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