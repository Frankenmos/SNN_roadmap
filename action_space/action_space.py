import numpy as np
from pysc2.lib import actions

from agent_core.policy_protocol import (
    BRIDGE_ACTION_BOOTSTRAP_SELECT,
    BRIDGE_ACTION_LEFT_CLICK,
    BRIDGE_ACTION_NO_OP,
    BRIDGE_ACTION_RIGHT_CLICK,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
)


_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4


class ActionSpace:
    def __init__(self, max_step_size=10, screen_size=84):
        self.max_step_size = max_step_size
        self.screen_size = screen_size
        self.last_token = self._token(BRIDGE_ACTION_NO_OP, 0, 0, 0)

    def _clip_coords(self, target_x, target_y, screen_size=None):
        screen_size = screen_size or self.screen_size
        max_coord = screen_size - 1
        target_x = int(np.clip(target_x, 0, max_coord))
        target_y = int(np.clip(target_y, 0, max_coord))
        return target_x, target_y

    @staticmethod
    def _token(type_id, target_x, target_y, extra):
        return np.asarray(
            [type_id, target_x, target_y, extra],
            dtype=np.int32,
        )

    def _set_token(self, type_id, target_x=0, target_y=0, extra=0):
        self.last_token = self._token(type_id, target_x, target_y, extra)

    def get_last_token(self):
        return self.last_token.copy()

    def reset(self):
        self._set_token(BRIDGE_ACTION_NO_OP, 0, 0, 0)

    def no_op(self):
        self._set_token(BRIDGE_ACTION_NO_OP, 0, 0, 0)
        return actions.FUNCTIONS.no_op()

    def bootstrap_select_army(self, obs):
        if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
            self._set_token(BRIDGE_ACTION_BOOTSTRAP_SELECT, 0, 0, 0)
            return actions.FUNCTIONS.select_army("select")
        return self.no_op()

    def find_units(self, feature_layer, condition):
        units = np.argwhere(feature_layer == condition)
        return [tuple(unit) for unit in units]

    def left_click(self, obs, target_x, target_y, screen_size=None):
        del obs, target_x, target_y, screen_size
        return self.no_op()

    def right_click(self, obs, target_x, target_y, screen_size=None):
        target_x, target_y = self._clip_coords(target_x, target_y, screen_size)
        if actions.FUNCTIONS.Smart_screen.id in obs.observation.available_actions:
            self._set_token(BRIDGE_ACTION_RIGHT_CLICK, target_x, target_y, 0)
            return actions.FUNCTIONS.Smart_screen("now", [target_x, target_y])
        return self.no_op()

    def smart(self, obs, target_x, target_y, screen_size=None):
        return self.right_click(obs, target_x, target_y, screen_size=screen_size)

    def dispatch(self, action_id, target_x, target_y, obs):
        if int(action_id) == POLICY_ACTION_NO_OP:
            return self.no_op()
        if int(action_id) == POLICY_ACTION_LEFT_CLICK:
            self._set_token(BRIDGE_ACTION_LEFT_CLICK, target_x, target_y, 0)
            return self.left_click(obs, target_x, target_y)
        if int(action_id) == POLICY_ACTION_RIGHT_CLICK:
            return self.right_click(obs, target_x, target_y)
        raise ValueError(f"Unknown semantic action id: {action_id}")
