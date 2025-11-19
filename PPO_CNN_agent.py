import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import torch

from PPO_CNN.policy_network import PolicyNetwork
from PPO_CNN.PPO import PPO
from obs_space.obs_space_2 import ObservationExtractor
from action_space.action_space import ActionSpace
from PPO_CNN.reward_function_2 import RewardFunctionV2
from Utility.config import cfg

_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4


class DefeatRoaches(base_agent.BaseAgent):
    def __init__(self, spatial_input_shape=None, vector_input_dim=None, action_dim=None, lr=None, gamma=None, clip_eps=None):
        super(DefeatRoaches, self).__init__()
        self.steps = 0
        self.extractor = ObservationExtractor()
        self.action_space = ActionSpace()
        self.reward_function = RewardFunctionV2()

        if spatial_input_shape is None:
            spatial_input_shape = tuple(cfg.model.spatial_input_shape)
        if vector_input_dim is None:
            vector_input_dim = cfg.model.vector_input_dim
        if action_dim is None:
            action_dim = cfg.model.action_dim

        # Validate input shapes
        assert spatial_input_shape == tuple(cfg.model.spatial_input_shape), f"Invalid spatial_input_shape: {spatial_input_shape}"
        assert vector_input_dim == cfg.model.vector_input_dim, f"Invalid vector_input_dim: {vector_input_dim}"
        assert action_dim == cfg.model.action_dim, f"Invalid action_dim: {action_dim}. Expected 3 (attack, move, no_op)."

        # Initialize policy and PPO
        self.policy = PolicyNetwork(spatial_input_shape, vector_input_dim, action_dim)
        self.ppo = PPO(
            policy_net=self.policy,
            lr=lr if lr is not None else cfg.hyperparameters.lr,
            gamma=gamma if gamma is not None else cfg.hyperparameters.gamma,
            clip_epsilon=clip_eps if clip_eps is not None else cfg.hyperparameters.clip_eps
        )

        self.selected_armies = []

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)
        self.steps += 1

        # Extract observations
        spatial_observation, vector_observation = self.extractor.extract_observation(obs)
        
        # Get action through PPO's select_action method
        action, angle, log_prob, value = self.ppo.select_action((spatial_observation, vector_observation))
        
        # Execute action
        # Execute action
        player_relative = obs.observation.feature_screen.player_relative
        
        # Update selected armies (friendly units)
        self.selected_armies = self.action_space.find_units(player_relative, _PLAYER_FRIENDLY)
        
        action_func = actions.FUNCTIONS.no_op()
        
        # Check action availability
        can_attack = actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions
        can_move = actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions
        can_select_army = actions.FUNCTIONS.select_army.id in obs.observation.available_actions

        if action == 0:  # Attack
            if can_attack:
                enemy_units = self.action_space.find_units(player_relative, _PLAYER_ENEMY)
                if enemy_units:
                    action_func = self.action_space.attack(obs, target_position=enemy_units[0])
            elif can_select_army:
                # If we want to attack but can't, try selecting the army
                action_func = actions.FUNCTIONS.select_army("select")
                
        elif action == 1:  # Move
            if can_move and self.selected_armies:
                agent_position = self.selected_armies[0]
                action_func = self.action_space.move(obs, agent_position, angle)
            elif can_select_army:
                # If we want to move but can't, try selecting the army
                action_func = actions.FUNCTIONS.select_army("select")

        # Calculate reward
        reward = self.reward_function.calculate_reward(obs, vector_observation)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.policy.device)

        # Return all necessary information for the training loop
        return action_func, action, log_prob, value, spatial_observation, vector_observation, reward.item()

    def reset(self):
        """Reset the agent's internal state."""
        super(DefeatRoaches, self).reset()
        self.extractor.reset()
        self.reward_function.reset()
        self.selected_armies = []

    def update_policy(self):
        """Train the PPO policy."""
        self.ppo.update_policy(
            batch_size=cfg.hyperparameters.batch_size,
            epochs=cfg.hyperparameters.epochs
        )
