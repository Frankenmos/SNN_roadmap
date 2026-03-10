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
from Utility.config import cfg



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


def initialize_db(db_path):
    """Initialize database with all required tables."""
    conn = sqlite3.connect(db_path)

    with conn:
        # Episodes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_reward REAL,
                average_reward REAL,
                steps INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Steps table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                step_number INTEGER,
                action INTEGER,
                reward REAL,
                cumulative_reward REAL,
                FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
            )
        """)

        # Reward components table - Removed WITHOUT ROWID and changed PRIMARY KEY
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reward_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER,
                step INTEGER,
                health_reward REAL,
                engagement_reward REAL,
                positioning_reward REAL,
                score_reward REAL,
                bonus_reward REAL,
                end_of_episode_reward REAL,
                total_reward REAL
            )
        """)

        # Create index for faster queries but allow duplicates
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_step 
            ON reward_components(episode, step)
        """)

    return conn


def train_agent(
    env,
    agent,
    observation_extractor,
    episode_count=None,
    steps_per_episode=None,
    update_frequency=None,
    target_reward=None,
    reward_window=None,
    checkpoint_path=None,
    db_path=None
):
    """Train the agent using PPO with enhanced logging."""
    if episode_count is None:
        episode_count = cfg.environment.total_episodes
    if steps_per_episode is None:
        steps_per_episode = cfg.environment.steps_per_episode
    if update_frequency is None:
        update_frequency = cfg.environment.update_frequency
    if target_reward is None:
        target_reward = cfg.environment.target_reward
    if reward_window is None:
        reward_window = cfg.environment.reward_window
    if checkpoint_path is None:
        checkpoint_path = cfg.environment.checkpoint_path
    if db_path is None:
        db_path = cfg.environment.db_path

    episode_rewards = deque(maxlen=reward_window)
    best_avg_reward = float('-inf')

    # Initialize database
    conn = initialize_db(db_path)

    try:
        start_episode, best_avg_reward, episode_rewards = load_checkpoint(agent, checkpoint_path)

        for episode in range(start_episode, episode_count):
            try:
                obs = env.reset()[0]
                episode_reward = 0
                cumulative_reward = 0
                step_count = 0

                # Log episode start
                with conn:
                    cursor = conn.execute(
                        "INSERT INTO episodes (total_reward, average_reward, steps) VALUES (0, 0, 0)"
                    )
                    episode_id = cursor.lastrowid

                for step in range(steps_per_episode):
                    # Get agent action and info
                    action_func_call, action_id, log_prob, value, spatial_obs, vector_obs, reward = agent.step(obs)

                    # Step environment
                    next_obs = env.step([action_func_call])[0]
                    done = next_obs.last()

                    # Store transition - pass native Python values directly
                    agent.ppo.store_transition(
                        spatial_obs=spatial_obs,
                        vector_obs=vector_obs,
                        action=action_id,
                        log_prob=log_prob,
                        reward=reward,
                        value=value,
                        done=done
                    )

                    # For database logging, use the original values
                    episode_reward += reward
                    cumulative_reward += reward
                    step_count += 1

                    # Log step details - use original values for DB
                    with conn:
                        conn.execute(
                            "INSERT INTO steps (episode_id, step_number, action, reward, cumulative_reward) VALUES (?, ?, ?, ?, ?)",
                            (int(episode_id), int(step_count), int(action_id), float(reward), float(cumulative_reward))
                        )

                        # Log reward components
                        reward_info = agent.reward_function.get_last_reward_components()
                        if reward_info:
                            conn.execute(
                                """INSERT INTO reward_components 
                                   (episode, step, health_reward, engagement_reward, positioning_reward, 
                                    score_reward, bonus_reward, end_of_episode_reward, total_reward)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (int(episode), int(step_count),
                                 float(reward_info['health_reward']),
                                 float(reward_info['engagement_reward']),
                                 float(reward_info['positioning_reward']),
                                 float(reward_info['score_reward']),
                                 float(reward_info['bonus_reward']),
                                 float(reward_info['end_of_episode_reward']),
                                 float(reward_info['total_reward']))
                            )

                    if done:
                        break

                    obs = next_obs

                # Update episode statistics
                episode_rewards.append(episode_reward)
                avg_reward = np.mean(episode_rewards)

                # Update episode record - ensure all values are Python native types
                with conn:
                    conn.execute(
                        "UPDATE episodes SET total_reward=?, average_reward=?, steps=? WHERE episode_id=?",
                        (float(episode_reward), float(avg_reward), int(step_count), int(episode_id))
                    )

                # Rest of your existing training loop logic...
                if (episode + 1) % update_frequency == 0:
                    agent.update_policy()

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(agent.policy.state_dict(), 'best_model_CNN_Version2.pth')
                    logger.info(f"New best model saved with avg reward: {avg_reward:.2f}")

                if (episode + 1) % cfg.environment.log_frequency == 0:
                    save_checkpoint(agent, episode + 1, best_avg_reward, episode_rewards, checkpoint_path)
                    logger.info(f"Episode {episode + 1}/{episode_count} | Avg Reward: {avg_reward:.2f}")

            except Exception as e:
                logger.error(f"Error in episode {episode + 1}: {str(e)}")
                env.close()
                env = reset_environment()

    finally:
        conn.close()

    return best_avg_reward


def main(argv):
    """Main function to initialize and run training."""
    try:
        env = create_env(
            map_name=cfg.environment.map_name,
            visualize=cfg.environment.visualize,
            use_action_printer=cfg.environment.use_action_printer,
        )
    except Exception as e:
        print(f"Environment setup failed: {e}")
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

        print("Starting PPO training...")
        best_reward = train_agent(
            env=env,
            agent=agent,
            observation_extractor=observation_extractor
        )

        print(f"Training completed! Best average reward: {best_reward:.2f}")

        # Evaluation
        model_path = 'best_model_CNN_Version2.pth'
        if os.path.exists(model_path):
            agent.policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.policy.eval()
            print("Running evaluation episodes...")
            run_loop([agent], env, max_episodes=10)

    except Exception as e:
        print(f"Critical error: {e}")
        raise e
    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)