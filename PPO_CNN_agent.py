import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import torch

from PPO_CNN.policy_network import PolicyNetwork
from PPO_CNN.PPO import PPO
from obs_space.obs_space_2 import ObservationExtractor
from action_space.action_space import ActionSpace
from PPO_CNN.reward_function import RewardFunction

_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4


class DefeatRoaches(base_agent.BaseAgent):
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim, lr=1e-4, gamma=0.99, clip_eps=0.18):
        super(DefeatRoaches, self).__init__()
        self.steps = 0
        self.extractor = ObservationExtractor()
        self.action_space = ActionSpace()
        self.reward_function = RewardFunction()

        # Validate input shapes
        assert spatial_input_shape == (27, 84, 84), f"Invalid spatial_input_shape: {spatial_input_shape}"
        assert vector_input_dim == 100, f"Invalid vector_input_dim: {vector_input_dim}"
        assert action_dim == 3, f"Invalid action_dim: {action_dim}. Expected 3 (attack, move, no_op)."

        # Initialize policy and PPO
        self.policy = PolicyNetwork(spatial_input_shape, vector_input_dim, action_dim)
        self.ppo = PPO(policy_net=self.policy, lr=lr, gamma=gamma, clip_epsilon=clip_eps)

        self.selected_armies = []

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)
        self.steps += 1

        # Extract observations
        spatial_observation, vector_observation = self.extractor.extract_observation(obs)
        
        # Get action through PPO's select_action method
        action, angle, log_prob, value = self.ppo.select_action((spatial_observation, vector_observation))
        
        # Execute action
        player_relative = obs.observation.feature_screen.player_relative
        action_func = actions.FUNCTIONS.no_op()

        if action == 0:  # Attack
            enemy_units = self.action_space.find_units(player_relative, _PLAYER_ENEMY)
            if enemy_units:
                action_func = self.action_space.attack(obs, target_position=enemy_units[0])
        elif action == 1:  # Move
            if self.selected_armies:
                agent_position = self.selected_armies[0]
                action_func = self.action_space.move(obs, agent_position, angle)

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
        self.ppo.update_policy(batch_size=128, epochs=20)
