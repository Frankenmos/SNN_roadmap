import sys
from unittest.mock import MagicMock
import unittest
import numpy as np
import torch
import types

# --- MOCK pysc2 library ---
pysc2_mock = MagicMock()

# 1. Mock the BaseAgent class and provide a mock 'step' method.
pysc2_mock.agents.base_agent.BaseAgent = type('BaseAgent', (object,), {'step': lambda self, obs: None})

# 2. Mock the FunctionCall class and the specific FUNCTIONS used by the agent.
pysc2_mock.lib.actions.FunctionCall = type('FunctionCall', (), {})
pysc2_mock.lib.actions.FUNCTIONS.no_op.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id = 1
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.return_value = pysc2_mock.lib.actions.FunctionCall()

# 3. Inject the mock into sys.modules.
sys.modules['pysc2'] = pysc2_mock
sys.modules['pysc2.agents'] = pysc2_mock.agents
sys.modules['pysc2.agents.base_agent'] = pysc2_mock.agents.base_agent
sys.modules['pysc2.lib'] = pysc2_mock.lib
sys.modules['pysc2.lib.actions'] = pysc2_mock.lib.actions
sys.modules['pysc2.lib.features'] = pysc2_mock.lib.features

# --- Imports ---
from PPO_CNN_agent import DefeatRoaches

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

        self.agent = DefeatRoaches(
            spatial_input_shape=self.spatial_dims,
            vector_input_dim=self.vector_dim,
            action_dim=self.action_dim
        )

    def _create_mock_obs(self):
        """Creates a mock observation object for the agent."""
        obs = types.SimpleNamespace()
        obs.observation = types.SimpleNamespace()

        screen_data = np.random.randint(0, 256, size=self.spatial_dims, dtype=np.uint8)
        player_relative_data = np.random.randint(
            0, 5, size=(self.spatial_dims[1], self.spatial_dims[2]), dtype=np.uint8
        )
        obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)

        obs.observation.player = [100, 100, 0, 0]
        obs.observation.feature_units = []
        obs.observation.score_cumulative = [0]
        obs.observation.available_actions = {pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id}

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
            result = self.agent.step(mock_obs)
            action_func, action, log_prob, value, spatial_obs, vector_obs, reward = result

            self.assertIsInstance(action_func, pysc2_mock.lib.actions.FunctionCall)
            self.assertIsInstance(action, int)
            self.assertIn(action, list(range(self.action_dim)))
            self.assertIsInstance(log_prob, float)
            self.assertIsInstance(value, float)
            self.assertIsInstance(spatial_obs, torch.Tensor)
            self.assertEqual(spatial_obs.shape, self.spatial_dims)
            self.assertIsInstance(vector_obs, torch.Tensor)
            self.assertEqual(vector_obs.shape, (self.vector_dim,))
            self.assertIsInstance(reward, float)

        except Exception as e:
            self.fail(f"Agent.step() raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
