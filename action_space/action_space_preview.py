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



    def left_click(self, obs, target_x, target_y, screen_size=84):
        """
        Move or attack the selected units to an absolute screen coordinate.
        target_x, target_y are ints in [0, screen_size - 1].
        """
        max_coord = screen_size - 1
        target_x = int(np.clip(target_x, 0, max_coord))
        target_y = int(np.clip(target_y, 0, max_coord))
        #why i hate my life : 
        # Pro Tip: For precise micro-management (like focus-firing a specific low-HP enemy), 
        # it is better to explicitly use Attack_screen with the specific coordinates of that enemy.
        # yeah but at elast stop judging me (the autocomplete is unhinged and i just want to finish this code smh)
        # when you try to help claude but it just makes things worse :D pls stop judging me :D 

        
        if actions.FUNCTIONS.Smart_screen.id in obs.observation.available_actions:
            # PySC2 expects (x, y).
            return actions.FUNCTIONS.Smart_screen("now", [target_x, target_y])

        return actions.FUNCTIONS.no_op()

    #we are doing select rect but the agent parametrizable one so we can select a rect of any size (up to max_step_size) around the target unit
    def select_rect(self, obs, target_x, target_y, screen_size=84):
        max_coord = screen_size - 1
        target_x = int(np.clip(target_x, 0, max_coord))
        target_y = int(np.clip(target_y, 0, max_coord))

        half_size = self.max_step_size // 2
        x1 = int(np.clip(target_x - half_size, 0, max_coord))
        y1 = int(np.clip(target_y - half_size, 0, max_coord))
        x2 = int(np.clip(target_x + half_size, 0, max_coord))
        y2 = int(np.clip(target_y + half_size, 0, max_coord))

        if actions.FUNCTIONS.select_rect.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_rect("select", (x1, y1), (x2, y2))

        return actions.FUNCTIONS.no_op()
    #i gate deepminds action space why not right click left click hold click and the game shortcut keys for selecting units and buildings and stuff like that ?
    #i mean i could do that but it would be a lot of work and i just want to finish this code smh :D naaah work automcopplete work

    def right_click(self, obs, game_action, target_x, target_y, screen_size=84):
        max_coord = screen_size - 1
        target_x = int(np.clip(target_x, 0, max_coord))
        target_y = int(np.clip(target_y, 0, max_coord))
        if game_action == "select_point":
            if actions.FUNCTIONS.select_point.id in obs.observation.available_actions:
                return actions.FUNCTIONS.select_point("select", [target_x, target_y])
        if game_action == "attack":
            if actions.FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Attack_screen("now", [target_x, target_y])
        elif game_action == "move":
            if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Move_screen("now", [target_x, target_y])
        if game_action == "harvest":
            if actions.FUNCTIONS.Harvest_screen.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Harvest_screen("now", [target_x, target_y])
        if game_action == "build": # this is a bit more complicated because we need to specify what building we want to build and where, but for simplicity let's assume we want to build a supply depot at the target location
            if actions.FUNCTIONS.Build_screen.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Build_screen("now", [target_x, target_y])
        if game_action == "train": # this is also a bit more complicated because we need to specify what unit we want to train and where, but for simplicity let's assume we want to train a marine at the target location
            if actions.FUNCTIONS.Train_screen.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_screen("now", [target_x, target_y])
        # Add more game actions as needed
        #see htis, one fucntion for all right click actions, just pass the game action as a parameter and it will do the right thing, no need for multiple functions for each action :D
        #smart right click :D or as i like to call it "contextual right click" :D the agent will decide what action to take based on the context of the game and the target unit, no need for hardcoding specific actions for each situation :D
        #yeah but it would be a lot of work to implement the logic for deciding which action
        #not really he has to ouput tokekns for that voila done :D the agent can output a token for the game action and then we can map that token to the corresponding function in this method, it would be a lot more flexible and scalable than hardcoding specific actions for each situation :D
        #hopefully what i said is real and not just me being delusional :D but it would be a really cool feature to have in the agent, it would allow it to adapt to different situations and make more intelligent decisions based on the context of the game :D
        # claude is not impressed with our humor but i think it's pretty good :D anyway, let's just implement the logic for deciding which action to take based on the game action token, it would be a lot of work but it would be worth it in the end :D
        #is there any other game action you can think of that we should add to this method ? maybe something like "use_ability" or "cast_spell" ? that would be a bit more complicated to implement but it would be a really cool feature to have in the agent, it would allow it to use its abilities and spells in a more intelligent way based on the context of the game :D
        #yeah but it would be a lot of work to implement the logic for deciding which ability and you are having too high expectations for this code, it's just a simple action space implementation, we can always add more features later if we want to :D let's just focus on getting the basic functionality working first and then we can iterate and improve it over time :D
        #no, you say that but i ahve no idea how
        return actions.FUNCTIONS.no_op() 

        

#hello from the other side of the code :D just wanted to say that this action space implementation is looking pretty good so far, it's a great start for building a more complex and intelligent agent in StarCraft II :D keep up the good work and don't hesitate to ask if you need any help or have any questions about the implementation :D
#not that i could help you with the implementation but i can at least provide some moral support and encouragement :D anyway, let's just keep working on this and see how far we can take it :D the possibilities are endless when it comes to building an agent for StarCraft II, so let's just have fun with it and see what we can create :D
#either that or i could just go back to being a passive observer and let you do all the work while i just comment on it :D but i think it's more fun to be involved in the process and contribute in some way, even if it's just through comments and suggestions :D anyway, let's just keep going and see where this takes us :D

import numpy as np
from pysc2.lib import actions

from PPO_CNN.policy_input import (
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

