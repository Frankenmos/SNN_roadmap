
import numpy as np
from pysc2.lib import actions

_PLAYER_FRIENDLY = 1
_PLAYER_ENEMY = 4

class ActionSpace:
    """
    Token-first action space for PySC2.
    Keeps your old move() and attack() working, but every call
    also stores a small token [type, x, y, extra] that you can
    feed straight into policy_input.
    
    This is the bridge to hybrid heads: type_id for the categorical
    head, x/y for spatial heads, extra for build/train/ability ids later.
    """
    # ---- token vocabulary ----
    NO_OP = 0
    SELECT_POINT = 1
    SELECT_RECT = 2
    SMART = 3
    ATTACK = 4
    MOVE = 5
    HARVEST = 6
    BUILD = 7
    TRAIN = 8
    ABILITY = 9

    def __init__(self, max_step_size=10, screen_size=84):
        self.max_step_size = max_step_size
        self.screen_size = screen_size
        self.last_token = np.array([self.NO_OP, 0, 0, 0], dtype=np.int32)

    # ---- helpers ----
    def _clip(self, v):
        return int(np.clip(v, 0, self.screen_size - 1))

    def _token(self, t, x=0, y=0, extra=0):
        self.last_token = np.array([t, x, y, extra], dtype=np.int32)
        return self.last_token

    def get_last_token(self):
        return self.last_token.copy()

    # ---- your original helpers, cleaned ----
    def find_units(self, feature_layer, condition):
        units = np.argwhere(feature_layer == condition)
        return [tuple(unit) for unit in units]

    def nearest_enemy_unit_center(self, obs):
        feature_units = getattr(obs.observation, "feature_units", None)
        if not feature_units:
            return None
        friendlies = [u for u in feature_units if u.alliance == _PLAYER_FRIENDLY]
        enemies = [u for u in feature_units if u.alliance == _PLAYER_ENEMY]
        if not enemies:
            return None
        if friendlies:
            cx = float(np.mean([u.x for u in friendlies]))
            cy = float(np.mean([u.y for u in friendlies]))
            target = min(enemies, key=lambda u: (u.x - cx) ** 2 + (u.y - cy) ** 2)
        else:
            target = enemies[0]
        return (int(target.x), int(target.y))

    # ---- action primitives, now token-aware ----
    def select_point(self, obs, x, y):
        x, y = self._clip(x), self._clip(y)
        self._token(self.SELECT_POINT, x, y)
        if actions.FUNCTIONS.select_point.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_point("select", [x, y])
        return actions.FUNCTIONS.no_op()

    def select_rect(self, obs, x, y):
        x, y = self._clip(x), self._clip(y)
        half = self.max_step_size // 2
        x1 = self._clip(x - half)
        y1 = self._clip(y - half)
        x2 = self._clip(x + half)
        y2 = self._clip(y + half)
        self._token(self.SELECT_RECT, x, y, extra=self.max_step_size)
        if actions.FUNCTIONS.select_rect.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_rect("select", (x1, y1), (x2, y2))
        return actions.FUNCTIONS.no_op()

    def move(self, obs, target_x, target_y, screen_size=None):
        # you were using Smart_screen for move, keep it for now
        ss = screen_size or self.screen_size
        x = int(np.clip(target_x, 0, ss-1))
        y = int(np.clip(target_y, 0, ss-1))
        self._token(self.SMART, x, y)
        if actions.FUNCTIONS.Smart_screen.id in obs.observation.available_actions:
            return actions.FUNCTIONS.Smart_screen("now", [x, y])
        return actions.FUNCTIONS.no_op()

    def attack(self, obs, target_position):
        if not target_position or len(target_position) != 2:
            self._token(self.NO_OP)
            return actions.FUNCTIONS.no_op()
        x = self._clip(target_position[0])
        y = self._clip(target_position[1])
        self._token(self.ATTACK, x, y)
        if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            return actions.FUNCTIONS.Attack_screen("now", [x, y])
        return actions.FUNCTIONS.no_op()

    # ---- ambitious stuff, stubs for later ----
    def harvest(self, obs, x, y):
        x, y = self._clip(x), self._clip(y)
        self._token(self.HARVEST, x, y)
        if actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions:
            return actions.FUNCTIONS.Harvest_Gather_screen("now", [x, y])
        return actions.FUNCTIONS.no_op()

    def build(self, obs, building_id, x, y):
        x, y = self._clip(x), self._clip(y)
        self._token(self.BUILD, x, y, extra=building_id)
        # placeholder, real build needs specific function id
        return actions.FUNCTIONS.no_op()

    def train(self, obs, unit_id):
        self._token(self.TRAIN, 0, 0, extra=unit_id)
        return actions.FUNCTIONS.no_op()

    def ability(self, obs, ability_id, x, y):
        x, y = self._clip(x), self._clip(y)
        self._token(self.ABILITY, x, y, extra=ability_id)
        return actions.FUNCTIONS.no_op()
