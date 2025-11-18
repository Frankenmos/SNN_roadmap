import json
from typing import Optional
import numpy as np


class RewardShaping:
    """Constants for reward shaping."""
    DAMAGE_DEALT_WEIGHT = 5.0  # High weight for damaging enemies
    DAMAGE_TAKEN_WEIGHT = -0.5  # Penalty for taking damage
    DISTANCE_WEIGHT = -1.0  # Penalize being far from enemies
    MOVEMENT_WEIGHT = 1.0  # Reward moving toward enemies
    VICTORY_REWARD = 10.0
    DEFEAT_PENALTY = -5.0
    OPTIMAL_DISTANCE = 3.0
    RANGE_REWARD_WEIGHT = 2.0  # Reward being within attack range

    # Observation vector indices
    AGENT_HEALTH_IDX = 0
    AGENT_POSITION_X_IDX = 1
    AGENT_POSITION_Y_IDX = 2
    ENEMY_HEALTH_IDX = 3
    ENEMY_POSITION_X_IDX = 4
    ENEMY_POSITION_Y_IDX = 5
    DISTANCE_TO_ENEMY_IDX = 6
    AUTO_ATTACK_RANGE_IDX = 7
    RELATIVE_DISTANCE_TO_RANGE_IDX = 8
    SCALAR_DIRECTION_IDX = 9


class RewardFunction:
    def __init__(self, steps_per_episode=600, log_file="reward_logs.json"):
        """Initialize the reward function."""
        self.previous_observation: Optional[np.ndarray] = None
        self.previous_enemy_health = None
        self.previous_enemy_position = None
        self.episode_step = 0
        self.steps_per_episode = steps_per_episode
        self.max_observed_score = 1.0
        self.log_file = log_file
        self.logs = []

    def update_max_score(self, score: float):
        """Update the maximum observed score."""
        self.max_observed_score = max(self.max_observed_score, score)

    def log_step(self, step_data):
        """Log data for the current step."""
        self.logs.append(step_data)

    def save_logs(self):
        """Save logs to a file."""
        with open(self.log_file, "w") as file:
            json.dump(self.logs, file, indent=4)

    def calculate_health_reward(self, current_agent_health: float, current_enemy_health: float,
                                current_enemy_position: tuple) -> float:
        """Reward based on reducing health of the same enemy."""
        if self.previous_observation is None or self.previous_enemy_position != current_enemy_position:
            self.previous_enemy_health = current_enemy_health
            self.previous_enemy_position = current_enemy_position
            return 0.0

        damage_dealt = max(0, self.previous_enemy_health - current_enemy_health)
        damage_taken = max(0, self.previous_observation[RewardShaping.AGENT_HEALTH_IDX] - current_agent_health)

        self.previous_enemy_health = current_enemy_health

        return (
            RewardShaping.DAMAGE_DEALT_WEIGHT * damage_dealt +
            RewardShaping.DAMAGE_TAKEN_WEIGHT * damage_taken
        ) / self.steps_per_episode

    def calculate_positioning_reward(self, relative_distance_to_range: float) -> float:
        """Reward for maintaining optimal attack range."""
        return RewardShaping.RANGE_REWARD_WEIGHT * max(0, 1 - abs(relative_distance_to_range)) / self.steps_per_episode

    def calculate_engagement_reward(self, distance_to_enemy: float, scalar_direction: float) -> float:
        """Reward engaging with enemies."""
        distance_reward = RewardShaping.DISTANCE_WEIGHT * abs(distance_to_enemy - RewardShaping.OPTIMAL_DISTANCE)
        engagement_reward = RewardShaping.MOVEMENT_WEIGHT * scalar_direction
        return (distance_reward + engagement_reward) / self.steps_per_episode

    def calculate_reward(self, obs, observation_vector: np.ndarray) -> float:
        """Calculate the total reward for the current step."""
        self.episode_step += 1

        # Extract relevant metrics
        agent_health = observation_vector[RewardShaping.AGENT_HEALTH_IDX]
        enemy_health = observation_vector[RewardShaping.ENEMY_HEALTH_IDX]
        enemy_position = (
            observation_vector[RewardShaping.ENEMY_POSITION_X_IDX],
            observation_vector[RewardShaping.ENEMY_POSITION_Y_IDX]
        )
        distance_to_enemy = observation_vector[RewardShaping.DISTANCE_TO_ENEMY_IDX]
        relative_distance_to_range = observation_vector[RewardShaping.RELATIVE_DISTANCE_TO_RANGE_IDX]
        scalar_direction = observation_vector[RewardShaping.SCALAR_DIRECTION_IDX]

        # Reward components
        health_reward = self.calculate_health_reward(agent_health, enemy_health, enemy_position)
        engagement_reward = self.calculate_engagement_reward(distance_to_enemy, scalar_direction)
        positioning_reward = self.calculate_positioning_reward(relative_distance_to_range)

        # Score-based and bonus rewards
        score_reward = 0.0
        bonus_reward = 0.0
        score_cumulative = getattr(obs.observation, 'score_cumulative', [0])[0]
        if score_cumulative > self.max_observed_score:
            bonus_reward = 1.0
            self.update_max_score(score_cumulative)
        score_reward = (score_cumulative / (self.max_observed_score + 1e-5)) * 0.5

        # End-of-episode reward
        end_of_episode_reward = (
            RewardShaping.VICTORY_REWARD if obs.reward > 0 else RewardShaping.DEFEAT_PENALTY
        ) / self.steps_per_episode if obs.last() else 0.0

        # Combine rewards
        total_reward = (
            0.1 * health_reward +
            0.1 * engagement_reward +
            0.1 * positioning_reward +
            7 * end_of_episode_reward +
            2.0 * score_reward +
            2.0 * bonus_reward
        )
        total_reward = np.clip(total_reward, -1, 1)

        # Log data
        log_data = {
            "step": self.episode_step,
            "health_reward": health_reward,
            "engagement_reward": engagement_reward,
            "positioning_reward": positioning_reward,
            "score_reward": score_reward,
            "bonus_reward": bonus_reward,
            "end_of_episode_reward": end_of_episode_reward,
            "total_reward": total_reward
        }
        self.log_step({k: v.item() if hasattr(v, 'item') else v for k, v in log_data.items()})

        self.previous_observation = observation_vector.clone()
        if obs.last():
            self.save_logs()

        # Store components for logging
        self.last_reward_components = {
            'health_reward': health_reward,
            'engagement_reward': engagement_reward,
            'positioning_reward': positioning_reward,
            'score_reward': score_reward,
            'bonus_reward': bonus_reward,
            'end_of_episode_reward': end_of_episode_reward,
            'total_reward': total_reward
        }

        return total_reward

    def get_last_reward_components(self):
        """Return the components of the last calculated reward."""
        return getattr(self, 'last_reward_components', None)

    def reset(self):
        """Reset the reward function for a new episode."""
        self.previous_observation = None
        self.previous_enemy_health = None
        self.previous_enemy_position = None
        self.episode_step = 0
        self.logs = []
        self.last_reward_components = None
