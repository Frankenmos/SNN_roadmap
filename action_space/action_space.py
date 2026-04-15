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

    def move(self, obs, target_x, target_y, screen_size=84):
        """
        Move the selected units to an absolute screen coordinate.
        target_x, target_y are ints in [0, screen_size - 1].
        """
        max_coord = screen_size - 1
        target_x = int(np.clip(target_x, 0, max_coord))
        target_y = int(np.clip(target_y, 0, max_coord))

        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # PySC2 expects (x, y).
            return actions.FUNCTIONS.Move_screen("now", [target_x, target_y])

        return actions.FUNCTIONS.no_op()

    def attack(self, obs, target_position):
        """
        Execute an attack on the given target position.
        Args:
            obs: Current observation.
            target_position: (x, y) coordinates on the screen.
        Returns:
            A valid SC2 action or no-op if the command is not available.
        """
        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            if target_position and len(target_position) == 2:
                return actions.FUNCTIONS.Attack_screen("now", [target_position[1], target_position[0]])
        return actions.FUNCTIONS.no_op()




