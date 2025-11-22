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

    def move(self, obs, agent_position, angle, magnitude=10):
        """
        Move the agent based on a vector (angle, fixed magnitude).
        Corrects for Numpy (y, x) vs Cartesian (x, y) mismatch.
        """
        # 1. Calculate the Shift
        # dx is change in Column (Index 1)
        # dy is change in Row    (Index 0)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        # 2. Apply to Numpy Coordinates (y, x)
        # agent_position is (y, x) from np.argwhere
        current_y, current_x = agent_position
        
        target_y = int(current_y + dy) # Apply Sine to Row
        target_x = int(current_x + dx) # Apply Cosine to Column

        # 3. Clamp to Screen Bounds (Safe for any resolution)
        # usually 64 or 84 depending on your feature_screen_size
        max_coord = 83 # Or self.screen_size - 1
        target_y = np.clip(target_y, 0, max_coord)
        target_x = np.clip(target_x, 0, max_coord)

        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # 4. Final Flip for PySC2 Action
            # PySC2 expects (x, y) for the action command
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




