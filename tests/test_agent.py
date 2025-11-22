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
        """
        Creates a mock observation object that mimics the structure of the real
        PySC2 observation, providing just enough data for the agent to run.
        """
        obs = types.SimpleNamespace()
        obs.observation = types.SimpleNamespace()

        # Mock spatial features using our custom ndarray subclass. This allows
        # the object to be treated as an array while also having the required
        # `.player_relative` attribute.
        screen_data = np.random.randint(0, 256, size=self.spatial_dims, dtype=np.uint8)
        player_relative_data = np.random.randint(
            0, 5, size=(self.spatial_dims[1], self.spatial_dims[2]), dtype=np.uint8
        )
        obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)

        # Mock vector features and other attributes required by the agent's logic.
        obs.observation.player = [100, 100, 0, 0]
        obs.observation.feature_units = []
        obs.observation.score_cumulative = [0]
        obs.observation.available_actions = {pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id}

        # Mock the reward and the `last()` method for the end-of-episode logic.
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
            # The agent's step method returns a tuple of values that are used
            # for training. We verify the types and shapes of these values.
            # Use the agent helper that returns the full training tuple
            action_func, action, log_prob, value, spatial_obs, vector_obs, reward = self.agent.step_with_data(mock_obs)

            # action_func: The action to be executed in the game (a mock FunctionCall).
            self.assertIsInstance(action_func, pysc2_mock.lib.actions.FunctionCall)
            # action: The integer representation of the action (e.g., 0 for attack).
            self.assertIsInstance(action, int)
            self.assertIn(action, list(range(self.action_dim)))
            # log_prob: The log probability of the selected action.
            self.assertIsInstance(log_prob, float)
            # value: The value of the state, as estimated by the critic.
            self.assertIsInstance(value, float)
            # spatial_obs: The processed spatial observation tensor.
            self.assertIsInstance(spatial_obs, torch.Tensor)
            self.assertEqual(spatial_obs.shape, self.spatial_dims)
            # vector_obs: The processed vector observation tensor.
            self.assertIsInstance(vector_obs, torch.Tensor)
            self.assertEqual(vector_obs.shape, (self.vector_dim,))
            # reward: The calculated reward for the step.
            self.assertIsInstance(reward, float)

        except Exception as e:
            self.fail(f"Agent.step() raised an exception unexpectedly: {e}")

    def test_agent_backward_pass(self):
        """
        Test that a backward pass can be performed without errors, ensuring
        that gradients flow correctly through the policy network.
        """
        mock_obs = self._create_mock_obs()

        try:
            # The observation extractor returns tensors. We unsqueeze to add a batch dimension.
            spatial_obs, vector_obs = self.agent.extractor.extract_observation(mock_obs)
            spatial_tensor = spatial_obs.unsqueeze(0)
            vector_tensor = vector_obs.unsqueeze(0)

            # Perform a forward pass through the policy network.
            action_logits, _, state_value = self.agent.policy(spatial_tensor, vector_tensor)

            # --- Actor Loss Component ---
            # Calculate a sample log probability using a fixed action to ensure the
            # computation graph is not broken by a random `.sample()` call.
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = torch.tensor([0], device=self.agent.policy.device) # Dummy action
            log_prob = action_dist.log_prob(action)

            # --- Critic Loss Component ---
            critic_loss = state_value.mean()

            # Combine the losses and perform the backward pass.
            # We negate the actor loss because the optimizer performs gradient descent,
            # but we want to perform gradient ascent on the policy.
            loss = critic_loss - log_prob.mean()
            self.agent.ppo.optimizer.zero_grad()
            loss.backward()

            # Verify that gradients have been computed for all relevant parameters.
            for name, param in self.agent.policy.named_parameters():
                # The 'angle_fc' head is not used in our dummy loss, so it won't have a gradient.
                if 'angle_fc' not in name:
                    self.assertIsNotNone(param.grad, f"Gradient for {name} is None")

        except Exception as e:
            self.fail(f"Agent backward pass raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
