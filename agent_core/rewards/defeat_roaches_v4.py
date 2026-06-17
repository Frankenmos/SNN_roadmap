import math

from agent_core.policy_protocol import (
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    SMART_SCREEN_FUNCTION_ID,
)
from agent_core.rewards.defeat_roaches_v3 import RewardFunctionV3
from obs_space.action_effects import extract_frame_snapshot
from obs_space.smart_outcome_detector import SmartOutcomeDetector


class RewardFunctionV4(RewardFunctionV3):
    """
    Action-aware DefeatRoaches shaping.

    V3 rewards combat outcomes but still lets passive no-op auto-attack look
    viable. V4 adds a small immediate signal for the command that caused the
    transition: target visible enemies with Smart_screen, and stop treating
    visible-enemy no-op as neutral.
    """

    def __init__(
        self,
        *args,
        smart_target_radius=6.0,
        smart_near_enemy_reward=0.08,
        smart_far_enemy_penalty=0.03,
        noop_visible_enemy_penalty=0.02,
        smart_outcome_window=5,
        smart_attack_likely_reward=0.12,
        smart_fired_likely_reward=0.06,
        smart_attack_intent_reward=0.02,
        smart_null_unclear_penalty=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.smart_target_radius = float(smart_target_radius)
        self.smart_near_enemy_reward = float(smart_near_enemy_reward)
        self.smart_far_enemy_penalty = float(smart_far_enemy_penalty)
        self.noop_visible_enemy_penalty = float(noop_visible_enemy_penalty)
        self.smart_attack_likely_reward = float(smart_attack_likely_reward)
        self.smart_fired_likely_reward = float(smart_fired_likely_reward)
        self.smart_attack_intent_reward = float(smart_attack_intent_reward)
        self.smart_null_unclear_penalty = float(smart_null_unclear_penalty)
        self.smart_outcome_detector = SmartOutcomeDetector(
            outcome_window=int(smart_outcome_window),
            near_enemy_threshold=float(smart_target_radius),
        )
        self._reward_step = 0
        self._last_action_context = None

    def resolved_config(self):
        config = super().resolved_config()
        config.update(
            {
                "name": "defeat_roaches_v4",
                "smart_target_radius": float(self.smart_target_radius),
                "smart_near_enemy_reward": float(self.smart_near_enemy_reward),
                "smart_far_enemy_penalty": float(self.smart_far_enemy_penalty),
                "noop_visible_enemy_penalty": float(
                    self.noop_visible_enemy_penalty,
                ),
                "smart_outcome_window": int(
                    self.smart_outcome_detector.outcome_window,
                ),
                "smart_attack_likely_reward": float(
                    self.smart_attack_likely_reward,
                ),
                "smart_fired_likely_reward": float(
                    self.smart_fired_likely_reward,
                ),
                "smart_attack_intent_reward": float(
                    self.smart_attack_intent_reward,
                ),
                "smart_null_unclear_penalty": float(
                    self.smart_null_unclear_penalty,
                ),
            },
        )
        return config

    def reset(self):
        super().reset()
        self.smart_outcome_detector.reset()
        self._reward_step = 0
        self._last_action_context = None

    def observe_action(self, action_id, target_x, target_y, obs, action_call=None):
        del action_call
        _friendly_units, enemy_units = self._split_units(obs)
        available_actions = getattr(obs.observation, "available_actions", None)
        available_set = set()
        if available_actions is not None:
            try:
                available_set = {int(action_id) for action_id in available_actions}
            except Exception:
                available_set = set()

        self._last_action_context = {
            "action_id": None if action_id is None else int(action_id),
            "target_x": int(target_x),
            "target_y": int(target_y),
            "enemy_positions": [
                (
                    float(getattr(unit, "x", 0.0)),
                    float(getattr(unit, "y", 0.0)),
                )
                for unit in enemy_units
            ],
            "smart_available": SMART_SCREEN_FUNCTION_ID in available_set,
        }
        if action_id is not None and int(action_id) == POLICY_ACTION_RIGHT_CLICK:
            self.smart_outcome_detector.observe_smart_click(
                previous_frame=extract_frame_snapshot(obs),
                target=(float(target_x), float(target_y)),
                click_step=int(self._reward_step),
                previous_feature_units=getattr(obs.observation, "feature_units", None),
            )

    def calculate_reward(self, obs, vector_observation):
        total_reward = float(super().calculate_reward(obs, vector_observation))
        action_reward = self._action_guidance_reward()
        outcome_reward = self._smart_outcome_reward(obs)
        self._last_action_context = None
        total_reward += action_reward + outcome_reward

        if self.last_reward_components is not None:
            self.last_reward_components["bonus_reward"] += (
                action_reward + outcome_reward
            )
            self.last_reward_components["total_reward"] = total_reward

        self._reward_step += 1
        return total_reward

    def _smart_outcome_reward(self, obs):
        current_frame = extract_frame_snapshot(obs)
        outcomes = self.smart_outcome_detector.resolve(
            current_frame=current_frame,
            resolution_step=int(self._reward_step) + 1,
            current_feature_units=getattr(obs.observation, "feature_units", None),
        )
        reward = 0.0
        for outcome in outcomes:
            if outcome.outcome_class == "attack_likely":
                reward += self.smart_attack_likely_reward
            elif outcome.outcome_class == "fired_likely":
                reward += self.smart_fired_likely_reward
            elif outcome.outcome_class == "attack_intent":
                reward += self.smart_attack_intent_reward
            elif outcome.outcome_class in {"null_unclear", "null_or_unclear"}:
                reward -= self.smart_null_unclear_penalty

        return float(
            max(
                -self.smart_null_unclear_penalty,
                min(self.smart_attack_likely_reward, reward),
            ),
        )

    def _action_guidance_reward(self):
        context = self._last_action_context
        if not context:
            return 0.0

        enemy_positions = context["enemy_positions"]
        if not enemy_positions:
            return 0.0

        action_id = context["action_id"]
        if action_id == POLICY_ACTION_NO_OP and context["smart_available"]:
            return -self.noop_visible_enemy_penalty

        if action_id != POLICY_ACTION_RIGHT_CLICK:
            return 0.0

        target_x = float(context["target_x"])
        target_y = float(context["target_y"])
        nearest = min(
            math.hypot(target_x - enemy_x, target_y - enemy_y)
            for enemy_x, enemy_y in enemy_positions
        )
        radius = max(1.0e-6, self.smart_target_radius)
        if nearest <= radius:
            proximity = 1.0 - (nearest / radius)
            return self.smart_near_enemy_reward * (0.5 + 0.5 * proximity)
        return -self.smart_far_enemy_penalty
