import sys
from unittest.mock import MagicMock
import unittest
import numpy as np
import torch
import types
import os

# --- MOCK pysc2 library ---
# To create a fast and isolated unit test, we mock the entire pysc2 library.
pysc2_mock = MagicMock()

# 1. Mock the BaseAgent class.
pysc2_mock.agents.base_agent.BaseAgent = type('BaseAgent', (object,), {'step': lambda self, obs: None})

# 2. Mock the FunctionCall class and the specific FUNCTIONS.
pysc2_mock.lib.actions.FunctionCall = type('FunctionCall', (), {})
pysc2_mock.lib.actions.FUNCTIONS.no_op.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.select_army.id = 7
pysc2_mock.lib.actions.FUNCTIONS.select_army.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id = 1
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Move_screen.id = 331
pysc2_mock.lib.actions.FUNCTIONS.Move_screen.return_value = pysc2_mock.lib.actions.FunctionCall()

# 3. Inject the mock into sys.modules.
sys.modules['pysc2'] = pysc2_mock
sys.modules['pysc2.agents'] = pysc2_mock.agents
sys.modules['pysc2.agents.base_agent'] = pysc2_mock.agents.base_agent
sys.modules['pysc2.lib'] = pysc2_mock.lib
sys.modules['pysc2.lib.actions'] = pysc2_mock.lib.actions
sys.modules['pysc2.lib.features'] = pysc2_mock.lib.features

# --- Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO_CNN_agent import DefeatRoaches

# Mock Unit class for feature_units
class MockUnit:
    def __init__(self, alliance, health, x, y, unit_type=0):
        self.alliance = alliance
        self.health = health
        self.x = x
        self.y = y
        self.unit_type = unit_type
        self.attack_range = 5.0

# Mock Feature Screen
class MockFeatureScreen(np.ndarray):
    def __new__(cls, input_array, player_relative=None):
        obj = np.asarray(input_array).view(cls)
        obj.player_relative = player_relative
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.player_relative = getattr(obj, 'player_relative', None)

class TestAgent(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.spatial_dims = (27, 84, 84)
        self.vector_dim = 100
        self.action_dim = 3

        # Force device to CPU for consistency in tests
        self.agent = DefeatRoaches(
            spatial_input_shape=self.spatial_dims,
            vector_input_dim=self.vector_dim,
            action_dim=self.action_dim
        )
        # Hack to force CPU if the machine has GPU but we want simple tests
        # self.agent.policy.to('cpu')
        # But let's trust the agent's logic for now.

    def _create_mock_obs(self):
        """
        Creates a mock observation object that mimics the structure of the real
        PySC2 observation.
        """
        obs = types.SimpleNamespace()
        obs.observation = types.SimpleNamespace()

        # Mock spatial features
        screen_data = np.random.randint(0, 256, size=self.spatial_dims, dtype=np.uint8)
        player_relative_data = np.random.randint(
            0, 5, size=(self.spatial_dims[1], self.spatial_dims[2]), dtype=np.uint8
        )
        obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)

        # Mock vector features and other attributes
        obs.observation.player = [100, 100, 0, 0] # [minerals, vespene, supply_used, supply_cap]

        # Mock feature_units (Critical for RewardFunctionV2)
        # Create one friendly unit (alliance=1) and one enemy unit (alliance=4)
        obs.observation.feature_units = [
            MockUnit(alliance=1, health=100, x=10, y=10, unit_type=48), # Marine
            MockUnit(alliance=4, health=50, x=20, y=20)   # Roach
        ]

        obs.observation.score_cumulative = [0]
        obs.observation.available_actions = {
            pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id,
            pysc2_mock.lib.actions.FUNCTIONS.select_army.id
        }

        # Mock the reward and the `last()` method
        obs.reward = 0
        obs.last = lambda: False

        return obs

    def test_agent_step_returns_valid_action(self):
        """
        Test that the agent's step function returns a valid action and
        doesn't crash with a mock observation.
        """
        mock_obs = self._create_mock_obs()

        try:
            # The agent's step_with_data returns 8 values now
            action_func, action_id, log_prob, value, spatial_obs, vector_obs, reward, xy_raw = self.agent.step_with_data(mock_obs)

            # action_func: Mock FunctionCall
            self.assertIsInstance(action_func, pysc2_mock.lib.actions.FunctionCall)

            # action_id: Tensor (scalar)
            self.assertIsInstance(action_id, torch.Tensor)
            # self.assertIn(action_id.item(), list(range(self.action_dim))) # Check int value

            # log_prob: Tensor
            self.assertIsInstance(log_prob, torch.Tensor)

            # value: Tensor
            self.assertIsInstance(value, torch.Tensor)

            # spatial_obs: Tensor
            self.assertIsInstance(spatial_obs, torch.Tensor)
            self.assertEqual(spatial_obs.shape, self.spatial_dims)

            # vector_obs: Tensor
            self.assertIsInstance(vector_obs, torch.Tensor)
            self.assertEqual(vector_obs.shape, (self.vector_dim,))

            # reward: float (Scalar) - RewardFunction returns float
            self.assertIsInstance(reward, float)

            # xy_raw: Tensor
            self.assertIsInstance(xy_raw, torch.Tensor)

        except Exception as e:
            self.fail(f"Agent.step() raised an exception unexpectedly: {e}")

    def test_agent_backward_pass(self):
        """
        Test that a backward pass can be performed without errors.
        """
        mock_obs = self._create_mock_obs()

        try:
            spatial_obs, vector_obs = self.agent.extractor.extract_observation(mock_obs)
            spatial_tensor = spatial_obs.unsqueeze(0)
            vector_tensor = vector_obs.unsqueeze(0)

            # Forward pass: returns (logits, xy_mean, value, next_state)
            action_logits, xy_mean, state_value, next_state = self.agent.policy(spatial_tensor, vector_tensor)

            # --- Actor Loss ---
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = torch.tensor([0], device=self.agent.policy.device)
            log_prob = action_dist.log_prob(action)

            # --- Critic Loss ---
            critic_loss = state_value.mean()

            loss = critic_loss - log_prob.mean()
            self.agent.ppo.optimizer.zero_grad()
            loss.backward()

            for name, param in self.agent.policy.named_parameters():
                 if param.grad is not None:
                     pass
                 # Note: Some parameters (like angle heads if unused) might not have grads depending on architecture.
                 # But we verify at least some do.

            self.assertTrue(any(p.grad is not None for p in self.agent.policy.parameters()))

        except Exception as e:
            self.fail(f"Agent backward pass raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
