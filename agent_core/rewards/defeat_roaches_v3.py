import math

import numpy as np

from obs_space.obs_space_2 import get_friendly_health


class RewardFunctionV3:
    """
    First reward redesign after the wrapper-driven env inspection pass.

    Main intent:
    - make kills matter more than raw chip damage
    - fix terminal win/loss detection
    - give movement a reason to exist through a small positioning signal
    - add a tiny per-step time pressure so passive no-op loops are less safe
    """

    def __init__(
        self,
        damage_dealt_coef=0.10,
        damage_taken_coef=0.15,
        kill_reward_coef=30.0,
        win_reward=60.0,
        loss_penalty=30.0,
        step_penalty=0.02,
        target_distance=9.0,
        distance_band_low=7.0,
        distance_band_high=11.0,
        distance_reward_coef=0.20,
        distance_reward_clip=1.0,
        distance_hold_bonus=0.05,
        distance_gate=18.0,
    ):
        self.damage_dealt_coef = float(damage_dealt_coef)
        self.damage_taken_coef = float(damage_taken_coef)
        self.kill_reward_coef = float(kill_reward_coef)
        self.win_reward = float(win_reward)
        self.loss_penalty = float(loss_penalty)
        self.step_penalty = float(step_penalty)
        self.target_distance = float(target_distance)
        self.distance_band_low = float(distance_band_low)
        self.distance_band_high = float(distance_band_high)
        self.distance_reward_coef = float(distance_reward_coef)
        self.distance_reward_clip = float(distance_reward_clip)
        self.distance_hold_bonus = float(distance_hold_bonus)
        self.distance_gate = float(distance_gate)
        self.previous_agent_health = None
        self.previous_enemy_health = None
        self.previous_mean_distance = None
        self.enemy_unit_count = 0
        self.last_reward_components = None

    def resolved_config(self):
        return {
            "name": "defeat_roaches_v3",
            "damage_dealt_coef": self.damage_dealt_coef,
            "damage_taken_coef": self.damage_taken_coef,
            "kill_reward_coef": self.kill_reward_coef,
            "win_reward": self.win_reward,
            "loss_penalty": self.loss_penalty,
            "step_penalty": self.step_penalty,
            "target_distance": self.target_distance,
            "distance_band_low": self.distance_band_low,
            "distance_band_high": self.distance_band_high,
            "distance_reward_coef": self.distance_reward_coef,
            "distance_reward_clip": self.distance_reward_clip,
            "distance_hold_bonus": self.distance_hold_bonus,
            "distance_gate": self.distance_gate,
        }

    def reset(self):
        self.previous_agent_health = None
        self.previous_enemy_health = None
        self.previous_mean_distance = None
        self.enemy_unit_count = 0
        self.last_reward_components = None

    def _split_units(self, obs):
        feature_units = getattr(obs.observation, "feature_units", None)
        if feature_units is None:
            feature_units = []
        friendly_units = [
            unit for unit in feature_units if getattr(unit, "alliance", 0) == 1
        ]
        enemy_units = [
            unit for unit in feature_units if getattr(unit, "alliance", 0) == 4
        ]
        return friendly_units, enemy_units

    def _mean_closest_enemy_distance(self, friendly_units, enemy_units):
        if not friendly_units or not enemy_units:
            return None

        closest = []
        for friendly in friendly_units:
            fx = float(getattr(friendly, "x", 0.0))
            fy = float(getattr(friendly, "y", 0.0))
            nearest = min(
                math.hypot(
                    fx - float(getattr(enemy, "x", 0.0)),
                    fy - float(getattr(enemy, "y", 0.0)),
                )
                for enemy in enemy_units
            )
            closest.append(nearest)
        return float(np.mean(closest)) if closest else None

    def _positioning_delta_reward(self, current_mean_distance, enemy_count):
        if (
            enemy_count <= 0
            or self.previous_mean_distance is None
            or current_mean_distance is None
        ):
            return 0.0

        # Do not reward passive "the enemy walked toward me from across the map"
        # transitions; only shape once combat is in the local micro zone.
        if (
            self.previous_mean_distance > self.distance_gate
            and current_mean_distance > self.distance_gate
        ):
            return 0.0

        prev_error = abs(self.previous_mean_distance - self.target_distance)
        curr_error = abs(current_mean_distance - self.target_distance)
        reward = self.distance_reward_coef * (prev_error - curr_error)

        if self.distance_band_low <= current_mean_distance <= self.distance_band_high:
            reward += self.distance_hold_bonus

        return float(
            np.clip(
                reward,
                -self.distance_reward_clip,
                self.distance_reward_clip,
            ),
        )

    def calculate_reward(self, obs, vector_observation):
        del vector_observation

        friendly_units, enemy_units = self._split_units(obs)
        current_agent_health = get_friendly_health(obs)
        current_enemy_health = float(sum(float(unit.health) for unit in enemy_units))
        current_enemy_count = len(enemy_units)
        current_mean_distance = self._mean_closest_enemy_distance(
            friendly_units,
            enemy_units,
        )

        if self.previous_agent_health is None:
            self.previous_agent_health = current_agent_health
        if self.previous_enemy_health is None:
            self.previous_enemy_health = current_enemy_health
            self.enemy_unit_count = current_enemy_count
        if self.previous_mean_distance is None:
            self.previous_mean_distance = current_mean_distance

        total_reward = 0.0
        health_reward = 0.0
        damage_reward = 0.0
        positioning_reward = 0.0
        kill_reward = 0.0
        bonus_reward = 0.0
        win_loss_reward = 0.0

        if current_enemy_count > 0:
            bonus_reward -= self.step_penalty
            total_reward += bonus_reward

        damage_dealt = self.previous_enemy_health - current_enemy_health
        if damage_dealt > 0:
            r = self.damage_dealt_coef * damage_dealt
            total_reward += r
            damage_reward += r

        damage_taken = self.previous_agent_health - current_agent_health
        if damage_taken > 0:
            r = self.damage_taken_coef * damage_taken
            total_reward -= r
            health_reward -= r

        if current_enemy_count < self.enemy_unit_count:
            r = self.kill_reward_coef * (self.enemy_unit_count - current_enemy_count)
            total_reward += r
            kill_reward += r

        positioning_reward = self._positioning_delta_reward(
            current_mean_distance,
            current_enemy_count,
        )
        total_reward += positioning_reward

        if obs.last():
            if current_enemy_count == 0:
                total_reward += self.win_reward
                win_loss_reward += self.win_reward
            else:
                total_reward -= self.loss_penalty
                win_loss_reward -= self.loss_penalty

        self.previous_agent_health = current_agent_health
        self.previous_enemy_health = current_enemy_health
        self.previous_mean_distance = current_mean_distance
        self.enemy_unit_count = current_enemy_count

        self.last_reward_components = {
            "health_reward": health_reward,
            "engagement_reward": damage_reward,
            "positioning_reward": positioning_reward,
            "score_reward": kill_reward,
            "bonus_reward": bonus_reward,
            "end_of_episode_reward": win_loss_reward,
            "total_reward": total_reward,
        }
        return total_reward

    def get_last_reward_components(self):
        return getattr(self, "last_reward_components", None)
