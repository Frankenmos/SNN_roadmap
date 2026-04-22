import numpy as np

from obs_space.obs_space_2 import get_friendly_health


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
        self.last_reward_components = None

    def reset(self):
        """Resets the health and unit tracking for a new episode."""
        self.previous_agent_health = None
        self.previous_enemy_health = None
        self.enemy_unit_count = 0
        self.last_reward_components = None

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
        # Previously read obs.observation.player[0] as "agent health" —
        # that's actually player_id (constant within an episode), so the
        # health_reward term was numerically inert on every past run.
        # Now sourced from feature_units via the shared helper.
        current_agent_health = get_friendly_health(obs)
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
        
        # Initialize components for logging
        health_reward = 0.0
        damage_reward = 0.0
        kill_reward = 0.0
        win_loss_reward = 0.0

        # --- Combat Rewards (Continuous) ---
        # Slight engagement-leaning asymmetry (0.6 dealt vs 0.4 taken).
        # In DefeatRoaches the objective is kills, not survival, so
        # dying-while-fighting is OK. Previous 0.5/0.5 symmetric split
        # made the health penalty dominate for a policy that
        # over-explored movement — see run_20260416_233240
        # components plot.
        damage_dealt = self.previous_enemy_health - current_enemy_health
        if damage_dealt > 0:
            r = 0.6 * damage_dealt
            total_reward += r
            damage_reward += r

        # Penalty for taking damage.
        damage_taken = self.previous_agent_health - current_agent_health
        if damage_taken > 0:
            r = 0.4 * damage_taken
            total_reward -= r
            health_reward -= r # Negative reward for health loss

        # --- Major Event Rewards (Sparse) ---
        # Large reward for killing a roach.
        if current_enemy_count < self.enemy_unit_count:
            r = 10.0 * (self.enemy_unit_count - current_enemy_count)
            total_reward += r
            kill_reward += r

        # Large terminal rewards for winning or losing.
        if obs.last():
            if obs.reward > 0:  # Win condition (all roaches defeated)
                total_reward += 20.0
                win_loss_reward += 20.0
            else:  # Lose condition (all friendly units defeated)
                total_reward -= 20.0
                win_loss_reward -= 20.0

        # --- Update State for Next Step ---
        self.previous_agent_health = current_agent_health
        self.previous_enemy_health = current_enemy_health
        self.enemy_unit_count = current_enemy_count

        # Store components for logging
        # Mapping to the keys expected by PPO_CNN_run.py (reward_components table)
        # The table has: health_reward, engagement_reward, positioning_reward, score_reward, bonus_reward, end_of_episode_reward
        # We map our V2 components to these best-fit categories or 0.0 if not applicable.
        self.last_reward_components = {
            'health_reward': health_reward, # Maps to health loss penalty
            'engagement_reward': damage_reward, # Maps to damage dealt
            'positioning_reward': 0.0, # Not explicitly tracked in V2
            'score_reward': kill_reward, # Maps to kill reward
            'bonus_reward': 0.0,
            'end_of_episode_reward': win_loss_reward,
            'total_reward': total_reward
        }

        return total_reward

    def get_last_reward_components(self):
        """Return the components of the last calculated reward."""
        return getattr(self, 'last_reward_components', None)
