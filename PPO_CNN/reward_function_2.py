import numpy as np

class RewardFunctionV2:
    """
    An improved, event-driven reward function tailored for the 'Defeat Roaches'
    scenario.

    This version simplifies the reward calculation by focusing on the core
    objective of the scenario: destroying all enemy roaches. It provides clear,
    impactful signals to encourage effective combat behavior.

    Core Principles:
    - Primary Objective Focus: The largest rewards are tied to damaging and
      destroying enemy roaches.
    - Health Preservation: The agent is penalized for taking damage to encourage
      survival and effective micro-management.
    - Sparse Terminal Rewards: A large, decisive reward is given for winning
      the scenario, and a corresponding penalty is given for losing.
    """
    def __init__(self):
        """Initializes the reward function's state."""
        self.previous_agent_health = None
        self.previous_enemy_health = None
        self.enemy_unit_count = 0

    def reset(self):
        """Resets the health and unit tracking for a new episode."""
        self.previous_agent_health = None
        self.previous_enemy_health = None
        self.enemy_unit_count = 0

    def calculate_reward(self, obs, vector_observation):
        """
        Calculates the total reward for the current step based on game events
        in the 'Defeat Roaches' scenario.

        Args:
            obs: The raw observation object from the environment.
            vector_observation (np.ndarray): The processed vector observation.

        Returns:
            float: The total calculated reward for the current step.
        """
        # --- Extract Key Metrics from Observations ---
        current_agent_health = obs.observation.player[0]
        enemy_units = [u for u in obs.observation.feature_units if u.alliance == 4] # 4 for enemy
        current_enemy_health = sum(u.health for u in enemy_units)
        current_enemy_count = len(enemy_units)

        # Initialize health and unit count on the first step.
        if self.previous_agent_health is None:
            self.previous_agent_health = current_agent_health
        if self.previous_enemy_health is None:
            self.previous_enemy_health = current_enemy_health
            self.enemy_unit_count = current_enemy_count

        total_reward = 0

        # --- Combat Rewards (Continuous) ---
        # Reward for damaging roaches.
        damage_dealt = self.previous_enemy_health - current_enemy_health
        if damage_dealt > 0:
            total_reward += 0.5 * damage_dealt

        # Penalty for taking damage.
        damage_taken = self.previous_agent_health - current_agent_health
        if damage_taken > 0:
            total_reward -= 0.5 * damage_taken

        # --- Major Event Rewards (Sparse) ---
        # Large reward for killing a roach.
        if current_enemy_count < self.enemy_unit_count:
            total_reward += 10.0 * (self.enemy_unit_count - current_enemy_count)

        # Large terminal rewards for winning or losing.
        if obs.last():
            if obs.reward > 0:  # Win condition (all roaches defeated)
                total_reward += 20.0
            else:  # Lose condition (all friendly units defeated)
                total_reward -= 20.0

        # --- Update State for Next Step ---
        self.previous_agent_health = current_agent_health
        self.previous_enemy_health = current_enemy_health
        self.enemy_unit_count = current_enemy_count

        return total_reward
