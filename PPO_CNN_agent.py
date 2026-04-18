import math
import random

import numpy as np
import torch
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import colors as _colors

from PPO_CNN.PPO import PPO
from PPO_CNN.policy_network import PolicyNetwork
from PPO_CNN.reward_function_2 import RewardFunctionV2
from Utility.config import cfg
from action_space.action_space import ActionSpace
from obs_space.obs_space_2 import ObservationExtractor


def _shuffled_hue_fixed(scale):
    palette = list(_colors.smooth_hue_palette(scale))
    random_keys = [random.random() for _ in palette]
    palette = [x for _, x in sorted(zip(random_keys, palette))]
    return np.array(palette)


_colors.shuffled_hue = _shuffled_hue_fixed

_PLAYER_FRIENDLY = 1


class DefeatRoaches(base_agent.BaseAgent):
    def __init__(
        self,
        spatial_input_shape=None,
        vector_input_dim=None,
        action_dim=None,
        lr=None,
        gamma=None,
        clip_eps=None,
        snn_state=None,
    ):
        super(DefeatRoaches, self).__init__()
        self.steps = 0
        self.extractor = ObservationExtractor()

        if spatial_input_shape is None:
            spatial_input_shape = tuple(cfg.model.spatial_input_shape)
        if vector_input_dim is None:
            vector_input_dim = cfg.model.vector_input_dim
        if action_dim is None:
            action_dim = cfg.model.action_dim

        assert spatial_input_shape == tuple(cfg.model.spatial_input_shape), (
            f"Invalid spatial_input_shape: {spatial_input_shape}"
        )
        assert vector_input_dim == cfg.model.vector_input_dim
        assert action_dim == cfg.model.action_dim

        screen_size = int(getattr(cfg.model, "screen_size", spatial_input_shape[-1]))
        self.action_space = ActionSpace(screen_size=screen_size)
        self.reward_function = RewardFunctionV2()
        self.reward_scale = float(getattr(cfg.hyperparameters, "reward_scale", 1.0))
        self.snn_state = snn_state

        self.policy = PolicyNetwork(
            spatial_input_shape,
            vector_input_dim,
            action_dim,
            num_steps=getattr(cfg.model, "num_steps", 1),
            screen_size=screen_size,
            token_snn_alpha=getattr(cfg.model, "token_snn_alpha", 0.8),
            token_snn_beta=getattr(cfg.model, "token_snn_beta", 0.9),
            attention_embed_dim=getattr(cfg.model, "attention_embed_dim", 64),
            attention_pool_size=getattr(cfg.model, "attention_pool_size", 7),
            attention_beta=getattr(cfg.model, "attention_beta", 0.5),
        )
        self.policy.to(self.policy.device)

        if self.snn_state is None:
            self.snn_state = self.policy.init_concrete_state(batch_size=1)

        total_eps = int(getattr(cfg.environment, "total_episodes", 0))
        steps_per_episode = int(getattr(cfg.environment, "steps_per_episode", 1) or 1)
        rollout_steps = int(getattr(cfg.hyperparameters, "rollout_steps", steps_per_episode) or 1)
        self.total_updates_estimate = max(
            0,
            math.ceil(total_eps * steps_per_episode / rollout_steps),
        )
        lr_min = float(getattr(cfg.hyperparameters, "lr_min", 0.0))

        self.ppo = PPO(
            policy_net=self.policy,
            lr=lr if lr is not None else cfg.hyperparameters.lr,
            gamma=gamma if gamma is not None else cfg.hyperparameters.gamma,
            clip_epsilon=clip_eps if clip_eps is not None else cfg.hyperparameters.clip_eps,
            critic_loss_coef=getattr(cfg.hyperparameters, "critic_loss_coef", 0.5),
            entropy_coef=getattr(cfg.hyperparameters, "entropy_coef", 0.01),
            total_updates=self.total_updates_estimate,
            lr_min=lr_min,
            target_kl=getattr(cfg.hyperparameters, "target_kl", None),
        )

        self.selected_armies = []

    def effective_config(self):
        return {
            "policy": self.policy.resolved_config(),
            "ppo": self.ppo.resolved_config(),
            "reward_scale": float(self.reward_scale),
            "total_updates_estimate": int(self.total_updates_estimate),
        }

    def peek_observation(self, obs):
        return self.extractor.peek_observation(obs)

    def step(self, obs, deterministic: bool = False):
        super(DefeatRoaches, self).step(obs)
        self.steps += 1

        spatial_observation, vector_observation = self.extractor.extract_observation(obs)

        pre_step_state = self.snn_state
        action, move_x, move_y, log_prob, value, self.snn_state = self.ppo.select_action(
            (spatial_observation, vector_observation),
            state=pre_step_state,
            deterministic=deterministic,
        )

        player_relative = obs.observation.feature_screen.player_relative
        self.selected_armies = self.action_space.find_units(
            player_relative, _PLAYER_FRIENDLY,
        )

        action_func = actions.FUNCTIONS.no_op()
        learnable = True

        can_attack = (
            actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions
        )
        can_move = (
            actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions
        )
        can_select_army = (
            actions.FUNCTIONS.select_army.id in obs.observation.available_actions
        )

        if action == 0:
            if can_attack:
                target_position = self.action_space.nearest_enemy_unit_center(obs)
                if target_position is not None:
                    action_func = self.action_space.attack(obs, target_position)
            elif can_select_army:
                action_func = actions.FUNCTIONS.select_army("select")
                learnable = False
        elif action == 1:
            if can_move and self.selected_armies:
                action_func = self.action_space.move(obs, move_x, move_y)
            elif can_select_army:
                action_func = actions.FUNCTIONS.select_army("select")
                learnable = False

        return (
            action_func,
            action,
            move_x,
            move_y,
            pre_step_state,
            float(log_prob),
            float(value),
            spatial_observation,
            vector_observation,
            learnable,
        )

    def reset(self):
        super(DefeatRoaches, self).reset()
        self.snn_state = self.policy.init_concrete_state(batch_size=1)
        self.extractor.reset()
        self.reward_function.reset()
        self.selected_armies = []

    def update_policy(self):
        _, stats = self.ppo.update_policy(
            batch_size=cfg.hyperparameters.batch_size,
            epochs=cfg.hyperparameters.epochs,
        )
        return stats
