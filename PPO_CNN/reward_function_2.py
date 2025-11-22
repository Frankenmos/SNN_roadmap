import numpy as np

class RewardFunctionV2Half:
    """
    A simplified, lighter reward function for DefeatRoaches.
    Focuses only on:
    - Damage dealt to roaches
    - Health lost
    - Kill events
    - Win/loss outcome
    """

    def __init__(self):
        self.prev_agent_hp = None
        self.prev_enemy_hp = None
        self.prev_enemy_count = None
        self.last_reward_components = None

    def reset(self):
        self.prev_agent_hp = None
        self.prev_enemy_hp = None
        self.prev_enemy_count = None
        self.last_reward_components = None

    def calculate_reward(self, obs, vector_observation):
        # --- Extract raw data ---
        agent_hp = obs.observation.player[0]

        enemies = [u for u in obs.observation.feature_units if u.alliance == 4]
        enemy_hp = sum(u.health for u in enemies)
        enemy_count = len(enemies)

        # Initialize
        if self.prev_agent_hp is None:
            self.prev_agent_hp = agent_hp
            self.prev_enemy_hp = enemy_hp
            self.prev_enemy_count = enemy_count

        total_reward = 0.0
        dmg_reward = 0.0
        hp_penalty = 0.0
        kill_reward = 0.0
        terminal_reward = 0.0

        # --- Damage dealt reward ---
        dmg = self.prev_enemy_hp - enemy_hp
        if dmg > 0:
            r = 0.5 * dmg
            total_reward += r
            dmg_reward += r

        # --- Damage taken penalty ---
        taken = self.prev_agent_hp - agent_hp
        if taken > 0:
            r = 0.5 * taken
            total_reward -= r
            hp_penalty -= r

        # --- Kill reward ---
        if enemy_count < self.prev_enemy_count:
            k = self.prev_enemy_count - enemy_count
            r = 10.0 * k
            total_reward += r
            kill_reward += r

        # --- Terminal win/loss ---
        if obs.last():
            if obs.reward > 0:
                total_reward += 20.0
                terminal_reward += 20.0
            else:
                total_reward -= 20.0
                terminal_reward -= 20.0

        # Update state
        self.prev_agent_hp = agent_hp
        self.prev_enemy_hp = enemy_hp
        self.prev_enemy_count = enemy_count

        # Logging (mapped to existing DB schema)
        self.last_reward_components = {
            "health_reward": hp_penalty,
            "engagement_reward": dmg_reward,
            "positioning_reward": 0.0,
            "score_reward": kill_reward,
            "bonus_reward": 0.0,
            "end_of_episode_reward": terminal_reward,
            "total_reward": total_reward,
        }

        return total_reward

    def get_last_reward_components(self):
        return self.last_reward_components


class RewardFunctionV2(RewardFunctionV2Half):
    """
    An enhanced reward function for DefeatRoaches.
    Considers:
    - Damage dealt to roaches
    - Health lost
    - Kill events
    - Distance-based shaping (Kiting)
    - Win/loss outcome
    Inherits from RewardFunctionV2Half and extends it.
    """

    def calculate_reward(self, obs, vector_observation):
        # --- Extract raw data for shaping ---
        agent_units = [u for u in obs.observation.feature_units if u.alliance == 1]
        enemy_units = [u for u in obs.observation.feature_units if u.alliance == 4]
        
        # 1. Base Reward (HP, Kills)
        agent_hp = obs.observation.player[0]
        enemy_hp = sum(u.health for u in enemy_units) if enemy_units else 0
        enemy_count = len(enemy_units)

        if self.prev_agent_hp is None:
            self.prev_agent_hp = agent_hp
            self.prev_enemy_hp = enemy_hp
            self.prev_enemy_count = enemy_count
            self.prev_min_dist = 0 # Initialize

        total_reward = 0.0
        dmg_reward = 0.0
        hp_penalty = 0.0
        kill_reward = 0.0
        terminal_reward = 0.0
        dist_reward = 0.0
        
        # Damage Dealt (+0.5 per point)
        dmg = self.prev_enemy_hp - enemy_hp
        if dmg > 0:
            r = 0.5 * dmg
            total_reward += r
            dmg_reward += r

        # Damage Taken (-0.5 per point)
        taken = self.prev_agent_hp - agent_hp
        if taken > 0:
            r = 0.5 * taken
            total_reward -= r
            hp_penalty -= r

        # Kill Reward (Reduced to +3.0 to prevent suicide trades)
        if enemy_count < self.prev_enemy_count:
            kills = self.prev_enemy_count - enemy_count
            r = 3.0 * kills
            total_reward += r
            kill_reward += r

        # --- 2. Distance-Based Shaping (Manual Kiting Reward) ---
        current_min_dist = 0
        if agent_units and enemy_units:
            min_dists = []
            for m in agent_units:
                m_pos = np.array([m.x, m.y])
                dists = [np.linalg.norm(m_pos - np.array([e.x, e.y])) for e in enemy_units]
                min_dists.append(min(dists))
            current_min_dist = min(min_dists)
        
        if not hasattr(self, 'prev_min_dist'):
            self.prev_min_dist = current_min_dist

        # Shaping: +0.01 for every pixel we move away from the nearest enemy
        dist_delta = current_min_dist - self.prev_min_dist
        r_dist = 0.01 * dist_delta
        total_reward += r_dist
        dist_reward += r_dist

        self.prev_min_dist = current_min_dist

        # --- 3. Terminal States (Aggressive) ---
        if obs.last():
            if obs.reward > 0:  # Win
                r_term = 200.0
                total_reward += r_term
                terminal_reward += r_term
            else:  # Loss (Marine died)
                r_term = -100.0
                total_reward += r_term # Huge penalty for dying
                terminal_reward += r_term

        # Update state
        self.prev_agent_hp = agent_hp
        self.prev_enemy_hp = enemy_hp
        self.prev_enemy_count = enemy_count

        # Update logs - MUST INCLUDE ALL KEYS expected by PPO_CNN_run.py
        self.last_reward_components = {
            "health_reward": hp_penalty,
            "engagement_reward": dmg_reward,
            "positioning_reward": dist_reward, # Map distance shaping here
            "score_reward": kill_reward,
            "bonus_reward": 0.0,
            "end_of_episode_reward": terminal_reward,
            "total_reward": total_reward,
        }
        
        return total_reward