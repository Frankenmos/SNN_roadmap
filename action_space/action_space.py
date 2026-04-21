import numpy as np
from pysc2.lib import actions

from agent_core.policy_protocol import (
    BRIDGE_ACTION_ATTACK,
    BRIDGE_ACTION_BOOTSTRAP_SELECT,
    BRIDGE_ACTION_MOVE,
    BRIDGE_ACTION_NO_OP,
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

    def move(self, obs, target_x, target_y, screen_size=None):
        target_x, target_y = self._clip_coords(target_x, target_y, screen_size)
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            self._set_token(BRIDGE_ACTION_MOVE, target_x, target_y, 0)
            return actions.FUNCTIONS.Move_screen("now", [target_x, target_y])
        return self.no_op()

    def nearest_enemy_unit_center(self, obs):
        feature_units = getattr(obs.observation, "feature_units", None)
        if feature_units is None or len(feature_units) == 0:
            return None

        friendlies = [
            unit for unit in feature_units if unit.alliance == _PLAYER_FRIENDLY
        ]
        enemies = [
            unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY
        ]
        if not enemies:
            return None

        if friendlies:
            center_x = float(np.mean([unit.x for unit in friendlies]))
            center_y = float(np.mean([unit.y for unit in friendlies]))
            target = min(
                enemies,
                key=lambda unit: (unit.x - center_x) ** 2 + (unit.y - center_y) ** 2,
            )
        else:
            target = enemies[0]

        return (int(target.x), int(target.y))

    def attack(self, obs, target_position):
        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            if target_position and len(target_position) == 2:
                target_x, target_y = self._clip_coords(
                    target_position[0],
                    target_position[1],
                )
                self._set_token(BRIDGE_ACTION_ATTACK, target_x, target_y, 0)
                return actions.FUNCTIONS.Attack_screen("now", [target_x, target_y])
        return self.no_op()
