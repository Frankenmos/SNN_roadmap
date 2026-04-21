import math
import random

import numpy as np
import torch
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import colors as _colors

from agent_core.policy_protocol import (
    POLICY_ACTION_ATTACK,
    POLICY_ACTION_MOVE,
    POLICY_ACTION_NO_OP,
)
from agent_core.ppo_trainer import PPO
from agent_core.rewards import build_reward_function
from agent_core.spiking_policy import PolicyNetwork
from Utility.config import cfg
from action_space.action_space import ActionSpace
from obs_space.obs_space_2 import ObservationExtractor


def _shuffled_hue_fixed(scale):
    palette = list(_colors.smooth_hue_palette(scale))
    random_keys = [random.random() for _ in palette]
    palette = [x for _, x in sorted(zip(random_keys, palette))]
    return np.array(palette)


_colors.shuffled_hue = _shuffled_hue_fixed


def _reward_config_from_cfg():
    reward_cfg = getattr(cfg, "reward", None)
    if reward_cfg is None:
        return "defeat_roaches_v2", {}

    try:
        items = dict(reward_cfg.items())
    except Exception:
        items = {}
    name = items.pop("name", "defeat_roaches_v2")
    return str(name), items


def _matches_function_call(action_call, target_function) -> bool:
    function_id = getattr(action_call, "function", None)
    if function_id is None:
        function_id = getattr(action_call, "id", None)
    if function_id is not None:
        return int(function_id) == int(target_function.id)

    function_name = getattr(action_call, "name", None)
    target_name = getattr(target_function, "name", None)
    return function_name is not None and function_name == target_name


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
        self.reward_name, self.reward_kwargs = _reward_config_from_cfg()
        self.reward_function = build_reward_function(
            self.reward_name,
            **self.reward_kwargs,
        )
        self.reward_scale = float(getattr(cfg.hyperparameters, "reward_scale", 1.0))
        self.snn_state = snn_state

        self.policy = PolicyNetwork(
            spatial_input_shape,
            vector_input_dim,
            action_dim,
            num_steps=getattr(cfg.model, "num_steps", 1),
            screen_size=screen_size,
            fast_token_snn_alpha=getattr(
                cfg.model,
                "fast_token_snn_alpha",
                getattr(cfg.model, "token_snn_alpha", 0.8),
            ),
            fast_token_snn_beta=getattr(
                cfg.model,
                "fast_token_snn_beta",
                getattr(cfg.model, "token_snn_beta", 0.9),
            ),
            slow_token_snn_alpha=getattr(
                cfg.model,
                "slow_token_snn_alpha",
                0.92,
            ),
            slow_token_snn_beta=getattr(
                cfg.model,
                "slow_token_snn_beta",
                0.97,
            ),
            temporal_combine_mode=getattr(
                cfg.model,
                "temporal_combine_mode",
                "mean",
            ),
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
            tbptt_window=getattr(cfg.hyperparameters, "tbptt_window", 32),
        )

        self.bootstrap_pending = True
        self.last_action_token = self.action_space.get_last_token()

    def effective_config(self):
        return {
            "policy": self.policy.resolved_config(),
            "ppo": self.ppo.resolved_config(),
            "reward": (
                self.reward_function.resolved_config()
                if hasattr(self.reward_function, "resolved_config")
                else {
                    "name": self.reward_name,
                    **self.reward_kwargs,
                }
            ),
            "reward_scale": float(self.reward_scale),
            "total_updates_estimate": int(self.total_updates_estimate),
        }

    def peek_observation(self, obs):
        return self.extractor.peek_observation(
            obs,
            last_action_token=self.last_action_token,
        )

    def step(self, obs, deterministic: bool = False):
        super(DefeatRoaches, self).step(obs)
        self.steps += 1

        policy_input = self.extractor.extract_observation(
            obs,
            update_stats=not deterministic,
            last_action_token=self.last_action_token,
        )

        can_attack = (
            actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions
        )
        can_move = (
            actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions
        )
        can_select_army = (
            actions.FUNCTIONS.select_army.id in obs.observation.available_actions
        )

        if self.bootstrap_pending and can_select_army and not (can_move or can_attack):
            self.bootstrap_pending = False
            self.action_space.reset()
            action_func = self.action_space.bootstrap_select_army(obs)
            self.last_action_token = self.action_space.get_last_token()
            return (
                action_func,
                None,
                0,
                0,
                self.snn_state,
                0.0,
                0.0,
                None,
                False,
            )

        self.bootstrap_pending = False
        pre_step_state = self.snn_state
        policy_input = policy_input.with_state(pre_step_state)
        action, move_x, move_y, log_prob, value, self.snn_state = (
            self.ppo.select_action(
                policy_input,
                deterministic=deterministic,
            )
        )

        action_func = self.action_space.no_op()
        learnable = True

        if action == POLICY_ACTION_ATTACK:
            action_func = self.action_space.attack(obs, (move_x, move_y))
            learnable = _matches_function_call(
                action_func,
                actions.FUNCTIONS.Attack_screen,
            )
        elif action == POLICY_ACTION_MOVE:
            action_func = self.action_space.move(obs, move_x, move_y)
            learnable = _matches_function_call(
                action_func,
                actions.FUNCTIONS.Move_screen,
            )
        elif action == POLICY_ACTION_NO_OP:
            action_func = self.action_space.no_op()
        else:
            raise ValueError(f"Unknown policy action id: {action}")

        self.last_action_token = self.action_space.get_last_token()

        return (
            action_func,
            action,
            move_x,
            move_y,
            pre_step_state,
            float(log_prob),
            float(value),
            policy_input,
            learnable,
        )

    def reset(self):
        super(DefeatRoaches, self).reset()
        self.snn_state = self.policy.init_concrete_state(batch_size=1)
        self.extractor.reset()
        self.reward_function.reset()
        self.bootstrap_pending = True
        self.action_space.reset()
        self.last_action_token = self.action_space.get_last_token()

    def update_policy(self):
        _, stats = self.ppo.update_policy(
            batch_size=cfg.hyperparameters.batch_size,
            epochs=cfg.hyperparameters.epochs,
        )
        return stats
