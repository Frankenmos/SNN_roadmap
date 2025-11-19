
import sys
import os
import torch
import numpy as np
import types
from unittest.mock import MagicMock

# Mock pysc2
pysc2_mock = MagicMock()
pysc2_mock.agents.base_agent.BaseAgent = type('BaseAgent', (object,), {'step': lambda self, obs: None})
pysc2_mock.lib.actions.FunctionCall = type('FunctionCall', (), {})
pysc2_mock.lib.actions.FUNCTIONS.no_op.return_value = pysc2_mock.lib.actions.FunctionCall()
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.id = 1
pysc2_mock.lib.actions.FUNCTIONS.Attack_screen.return_value = pysc2_mock.lib.actions.FunctionCall()

sys.modules['pysc2'] = pysc2_mock
sys.modules['pysc2.agents'] = pysc2_mock.agents
sys.modules['pysc2.agents.base_agent'] = pysc2_mock.agents.base_agent
sys.modules['pysc2.lib'] = pysc2_mock.lib
sys.modules['pysc2.lib.actions'] = pysc2_mock.lib.actions
sys.modules['pysc2.lib.features'] = pysc2_mock.lib.features

# Import agent
from PPO_CNN_agent import DefeatRoaches

class MockFeatureScreen(np.ndarray):
    def __new__(cls, input_array, player_relative=None):
        obj = np.asarray(input_array).view(cls)
        obj.player_relative = player_relative
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.player_relative = getattr(obj, 'player_relative', None)

def create_mock_obs(spatial_dims):
    obs = types.SimpleNamespace()
    obs.observation = types.SimpleNamespace()
    screen_data = np.random.randint(0, 256, size=spatial_dims, dtype=np.uint8)
    player_relative_data = np.random.randint(0, 5, size=(spatial_dims[1], spatial_dims[2]), dtype=np.uint8)
    obs.observation.feature_screen = MockFeatureScreen(screen_data, player_relative=player_relative_data)
    obs.observation.player = [100, 100, 0, 0]
    obs.observation.feature_units = []
    obs.observation.score_cumulative = [0]
    obs.observation.available_actions = {1}
    obs.reward = 0
    obs.last = lambda: False
    return obs

def main():
    print("Initializing agent...")
    try:
        agent = DefeatRoaches(
            spatial_input_shape=(27, 84, 84),
            vector_input_dim=100,
            action_dim=3
        )
        print(f"Agent initialized. Device: {agent.policy.device}")
        
        obs = create_mock_obs((27, 84, 84))
        print("Calling step...")
        result = agent.step(obs)
        print("Step successful!")
    except Exception as e:
        print("Caught exception:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
