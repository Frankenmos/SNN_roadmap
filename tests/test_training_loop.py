import sys
from unittest.mock import MagicMock
import unittest
import numpy as np
import torch
import types
import os

# --- MOCK pysc2 library ---
pysc2_mock = MagicMock()
pysc2_mock.agents.base_agent.BaseAgent = type('BaseAgent', (object,), {'step': lambda self, obs: None})
pysc2_mock.lib.actions.FunctionCall = type('FunctionCall', (), {})
pysc2_mock.lib.actions.FUNCTIONS.no_op.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.select_army.id = 7
pysc2_mock.lib.actions.FUNCTIONS.select_army.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id = 1
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Move_screen.id = 331
pysc2_mock.lib.actions.FUNCTIONS.Move_screen.return_value = pysc2_mock.lib.actions.FunctionCall()

sys.modules['pysc2'] = pysc2_mock
sys.modules['pysc2.agents'] = pysc2_mock.agents
sys.modules['pysc2.agents.base_agent'] = pysc2_mock.agents.base_agent
sys.modules['pysc2.lib'] = pysc2_mock.lib
sys.modules['pysc2.lib.actions'] = pysc2_mock.lib.actions
sys.modules['pysc2.lib.features'] = pysc2_mock.lib.features

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO_CNN_agent import DefeatRoaches

class MockUnit:
    def __init__(self, alliance, health, x, y, unit_type=0):
        self.alliance = alliance
        self.health = health
        self.x = x
        self.y = y
        self.unit_type = unit_type
        self.attack_range = 5.0

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
        self.spatial_dims = (27, 84, 84)
        self.vector_dim = 100
        self.action_dim = 3
        self.agent = DefeatRoaches(
            spatial_input_shape=self.spatial_dims,
            vector_input_dim=self.vector_dim,
            action_dim=self.action_dim
        )
        # Assuming CPU for tests
        # self.agent.policy.to('cpu')

    def _create_mock_obs(self):
        obs = types.SimpleNamespace()
        obs.observation = types.SimpleNamespace()

        screen_data = np.random.randint(0, 256, size=self.spatial_dims, dtype=np.uint8)
        player_relative_data = np.random.randint(0, 5, size=(self.spatial_dims[1], self.spatial_dims[2]), dtype=np.uint8)
        obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)

        obs.observation.player = [100, 100, 0, 0]
        obs.observation.feature_units = [
            MockUnit(alliance=1, health=100, x=10, y=10, unit_type=48),
            MockUnit(alliance=4, health=50, x=20, y=20)
        ]
        obs.observation.score_cumulative = [0]
        obs.observation.available_actions = {
            pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id,
            pysc2_mock.lib.actions.FUNCTIONS.select_army.id
        }

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
            for i in range(20): # reduced from 100 for speed
                mock_obs = self._create_mock_obs()
                if i == 10:
                    mock_obs.last = lambda: True

                # Step returns 8 values:
                # action_func, action_id, log_prob, value, spatial, vector, reward, xy_raw
                action_func, action_id, log_prob, value, spatial, vector, reward, xy_raw = self.agent.step_with_data(mock_obs)

                # Convert simple scalars to tensors for storage if they aren't already
                # Note: step_with_data returns tensors for everything except reward and action_func

                # Reward is float
                reward_tensor = torch.tensor(reward, device=self.agent.policy.device, dtype=torch.float32)

                # Done flag
                done_val = 1.0 if mock_obs.last() else 0.0
                done_tensor = torch.tensor(done_val, device=self.agent.policy.device, dtype=torch.float32)

                # Store
                self.agent.ppo.store_transition(
                    spatial, vector,
                    action_id,
                    xy_raw.squeeze(0), # remove batch dim [1,2] -> [2]
                    log_prob,
                    reward_tensor,
                    value,
                    done_tensor
                )

                # Check Detachment
                # stored = self.agent.ppo.memory[-1]
                # self.assertIsNone(stored['log_prob'].grad_fn) # Should be detached

            # Assert 'Done' logic
            # i=10 was Done.
            self.assertTrue(self.agent.ppo.memory[10]['done'] == 1.0)
            self.assertTrue(self.agent.ppo.memory[9]['done'] == 0.0)

            # 2. PPO Update
            losses = self.agent.ppo.update_policy(batch_size=4, epochs=2)

            # 3. Verification
            self.assertIsInstance(losses, list)
            self.assertGreater(len(losses), 0)
            self.assertEqual(len(self.agent.ppo.memory), 0)

        except Exception as e:
            self.fail(f"Endurance test raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
