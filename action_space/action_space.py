import numpy as np
from pysc2.lib import actions

_PLAYER_ENEMY = 4  # PlayerRelative.ENEMY

from pysc2.lib import features  # Make sure this import is at the top of the file

class ActionSpace:
    def __init__(self, max_step_size=10):
        self.max_step_size = max_step_size

    def find_units(self, feature_layer, condition):
        """
        Find units in a feature layer based on a condition.
        Example: condition = (feature_layer == features.PlayerRelative.ENEMY)
        """
        units = np.argwhere(feature_layer == condition)  # Use the correct condition
        return [tuple(unit) for unit in units]  # Convert to a list of tuples (y, x)

    def move(self, obs, xy_norm):
        """
        Move the agent to a specific screen location.
        Args:
            obs: Current observation.
            xy_norm: (x, y) normalized coordinates in [0, 1].
        """
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            screen_size = 84 # Hardcoded to match feature dimensions
            x = int(xy_norm[0] * screen_size)
            y = int(xy_norm[1] * screen_size)
            # Clamp
            x = np.clip(x, 0, screen_size - 1)
            y = np.clip(y, 0, screen_size - 1)
            return actions.FUNCTIONS.Move_screen("now", [x, y])

        return actions.FUNCTIONS.no_op()

    def attack(self, obs, xy_norm):
        """
        Attack a specific screen location.
        Args:
            obs: Current observation.
            xy_norm: (x, y) normalized coordinates in [0, 1].
        """
        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            screen_size = 84
            x = int(xy_norm[0] * screen_size)
            y = int(xy_norm[1] * screen_size)
            # Clamp
            x = np.clip(x, 0, screen_size - 1)
            y = np.clip(y, 0, screen_size - 1)
            return actions.FUNCTIONS.Attack_screen("now", [x, y])
        return actions.FUNCTIONS.no_op()




