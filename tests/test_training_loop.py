import sys
from unittest.mock import MagicMock
import unittest
import numpy as np
import torch
import types
import os

# --- MOCK pysc2 library ---
# To create a fast and isolated unit test, we mock the entire pysc2 library.
# This allows us to test the agent's logic without needing to install or run
# the full StarCraft II game environment, which is slow and resource-intensive.
pysc2_mock = MagicMock()

# 1. Mock the BaseAgent class. Our agent inherits from this, so we provide a
#    simple object with a mock `step` method to satisfy the super() call.
pysc2_mock.agents.base_agent.BaseAgent = type('BaseAgent', (object,), {'step': lambda self, obs: None})

# 2. Mock the FunctionCall class and the specific FUNCTIONS used by the agent.
#    The agent's `step` method returns an instance of `FunctionCall`. We also
#    mock the `.id` attribute for the actions that are checked in the agent's
#    `action_space`.
pysc2_mock.lib.actions.FunctionCall = type('FunctionCall', (), {})
pysc2_mock.lib.actions.FUNCTIONS.no_op.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id = 1
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.return_value = pysc2_mock.lib.actions.FunctionCall()

# 3. Inject the mock into sys.modules. Any subsequent import of pysc2 or its
#    submodules will now use our mock instead of the real library.
sys.modules['pysc2'] = pysc2_mock
sys.modules['pysc2.agents'] = pysc2_mock.agents
sys.modules['pysc2.agents.base_agent'] = pysc2_mock.agents.base_agent
sys.modules['pysc2.lib'] = pysc2_mock.lib
sys.modules['pysc2.lib.actions'] = pysc2_mock.lib.actions
sys.modules['pysc2.lib.features'] = pysc2_mock.lib.features

# --- Imports ---
# Tell Python to look in the parent directory (upstairs) for code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ... NOW you can import your agent
from PPO_CNN_agent import DefeatRoaches

# A mock class to simulate the pysc2 feature screen object (a NamedNumpyArray).
# This allows the object to be treated as a numpy array while also having custom attributes.
class MockFeatureScreen(np.ndarray):
    def __new__(cls, input_array, player_relative=None):
        obj = np.asarray(input_array).view(cls)
        obj.player_relative = player_relative
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.player_relative = getattr(obj, 'player_relative', None)

class TestTrainingLoop(unittest.TestCase):

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
        """
        Creates a mock observation object that mimics the structure of the real
        PySC2 observation, providing just enough data for the agent to run.
        """
        obs = types.SimpleNamespace()
        obs.observation = types.SimpleNamespace()

        # Mock spatial features using our custom ndarray subclass.
        screen_data = np.random.randint(0, 256, size=self.spatial_dims, dtype=np.uint8)
        player_relative_data = np.random.randint(
            0, 5, size=(self.spatial_dims[1], self.spatial_dims[2]), dtype=np.uint8
        )
        obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)

        # Mock vector features and other required attributes.
        obs.observation.player = [100, 100, 0, 0]
        obs.observation.feature_units = []
        obs.observation.score_cumulative = [0]
        obs.observation.available_actions = {pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id}

        # Mock the reward and the `last()` method.
        obs.reward = 0
        obs.last = lambda: False

        return obs

    def test_endurance_training_loop(self):
        """
        Tests a simulated training loop to ensure that the agent can collect
        data, perform a PPO update, and that the loss changes as expected.
        """
        try:
            # 1. Data Collection Loop
            for i in range(100):
                mock_obs = self._create_mock_obs()
                # Inject a 'Done' signal at step 50.
                if i == 50:
                    mock_obs.last = lambda: True

                _, action, log_prob, value, spatial_obs, vector_obs, reward = self.agent.step_with_data(mock_obs)

                # Convert scalars to tensors for storage.
                action_tensor = torch.tensor(action, device=self.agent.policy.device)
                log_prob_tensor = torch.tensor(log_prob, device=self.agent.policy.device)
                reward_tensor = torch.tensor(reward, device=self.agent.policy.device)
                value_tensor = torch.tensor(value, device=self.agent.policy.device)
                done_tensor = torch.tensor(mock_obs.last(), device=self.agent.policy.device)

                self.agent.ppo.store_transition(
                    spatial_obs, vector_obs, action_tensor, log_prob_tensor,
                    reward_tensor, value_tensor, done_tensor
                )

                # Check Detachment: Ensure stored tensors do not have a gradient function.
                # This is crucial because these values should be treated as fixed data points
                # for the PPO update, not as part of the computation graph.
                stored_transition = self.agent.ppo.memory[-1]
                self.assertIsNone(stored_transition['log_prob'].grad_fn)
                self.assertIsNone(stored_transition['value'].grad_fn)

            # Assert that the 'Done' signal was correctly recorded.
            self.assertTrue(self.agent.ppo.memory[50]['done'])
            self.assertFalse(self.agent.ppo.memory[49]['done'])
            self.assertFalse(self.agent.ppo.memory[51]['done'])

            # 2. PPO Update
            losses = self.agent.ppo.update_policy(batch_size=10, epochs=5)

            # 3. Verification
            self.assertIsInstance(losses, list)
            self.assertGreater(len(losses), 0, "Loss list should not be empty")
            # In a real scenario, loss should decrease, but for this test, we just check that it changes.
            self.assertNotEqual(losses[0], losses[-1], "Loss should change during training")
            self.assertEqual(len(self.agent.ppo.memory), 0, "Memory buffer should be empty after update")

        except Exception as e:
            self.fail(f"Endurance test raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
