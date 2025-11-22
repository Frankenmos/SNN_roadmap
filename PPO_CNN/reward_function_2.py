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
        if self.prev_enemy_hp is None:
            self.prev_enemy_hp = enemy_hp
        if self.prev_enemy_count is None:
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
    - Positional advantages
    - Score changes
    - Bonus rewards
    - Win/loss outcome
    Inherits from RewardFunctionV2Half and extends it.
    """

    def calculate_reward(self, obs, vector_observation):
        # Call base reward calculation
        total_reward = super().calculate_reward(obs, vector_observation)

        # --- Positional advantage reward ---
        # Example: Reward being closer to the center of the map
        screen_size = obs.observation.feature_screen.shape[1:3]
        center_x, center_y = screen_size[0] // 2, screen_size[1] // 2

        agent_units = [u for u in obs.observation.feature_units if u.alliance == 1]
        if agent_units:
            avg_x = np.mean([u.x for u in agent_units])
            avg_y = np.mean([u.y for u in agent_units])
            dist_to_center = np.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
            max_dist = np.sqrt((center_x) ** 2 + (center_y) ** 2)
            pos_reward = (max_dist - dist_to_center) / max_dist * 5.0  # Scale factor
            total_reward += pos_reward

            # Update positional reward component
            self.last_reward_components["positioning_reward"] = pos_reward

        # --- Score change reward ---
        score = obs.observation.score_cumulative[0]
        if not hasattr(self, 'prev_score'):
            self.prev_score = score

        score_diff = score - self.prev_score
        if score_diff > 0:
            score_reward = 0.1 * score_diff
            total_reward += score_reward

            # Update score reward component
            self.last_reward_components["score_reward"] += score_reward

        self.prev_score = score

        return total_reward         