import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeFunctionCall:
    def __init__(self, name, *args):
        self.name = name
        self.args = args


class FakeFunction:
    def __init__(self, name, function_id):
        self.name = name
        self.id = function_id

    def __call__(self, *args):
        return FakeFunctionCall(self.name, *args)


class MockFeatureScreen(np.ndarray):
    def __new__(cls, input_array, player_relative=None):
        obj = np.asarray(input_array).view(cls)
        obj.player_relative = player_relative
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.player_relative = getattr(obj, "player_relative", None)


def _install_fake_pysc2():
    for name in list(sys.modules):
        if name == "pysc2" or name.startswith("pysc2."):
            del sys.modules[name]

    pysc2_mod = types.ModuleType("pysc2")

    agents_mod = types.ModuleType("pysc2.agents")
    base_agent_mod = types.ModuleType("pysc2.agents.base_agent")

    class BaseAgent:
        def step(self, obs):
            return None

        def reset(self):
            return None

    base_agent_mod.BaseAgent = BaseAgent
    agents_mod.base_agent = base_agent_mod

    lib_mod = types.ModuleType("pysc2.lib")
    actions_mod = types.ModuleType("pysc2.lib.actions")
    actions_mod.FunctionCall = FakeFunctionCall
    actions_mod.FUNCTIONS = SimpleNamespace(
        no_op=FakeFunction("no_op", 0),
        Attack_screen=FakeFunction("Attack_screen", 1),
        Move_screen=FakeFunction("Move_screen", 2),
        select_army=FakeFunction("select_army", 3),
    )

    colors_mod = types.ModuleType("pysc2.lib.colors")
    colors_mod.smooth_hue_palette = lambda scale: [
        (idx, idx, idx) for idx in range(scale)
    ]

    features_mod = types.ModuleType("pysc2.lib.features")

    class Dimensions:
        def __init__(self, screen, minimap):
            self.screen = screen
            self.minimap = minimap

    class AgentInterfaceFormat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    features_mod.Dimensions = Dimensions
    features_mod.AgentInterfaceFormat = AgentInterfaceFormat

    lib_mod.actions = actions_mod
    lib_mod.colors = colors_mod
    lib_mod.features = features_mod

    env_mod = types.ModuleType("pysc2.env")
    sc2_env_mod = types.ModuleType("pysc2.env.sc2_env")
    base_env_wrapper_mod = types.ModuleType("pysc2.env.base_env_wrapper")

    class BaseEnvWrapper:
        def __init__(self, env):
            self._env = env

        def __getattr__(self, name):
            return getattr(self._env, name)

        def reset(self, *args, **kwargs):
            return self._env.reset(*args, **kwargs)

        def step(self, *args, **kwargs):
            return self._env.step(*args, **kwargs)

    class Race:
        terran = "terran"
        zerg = "zerg"

    class Difficulty:
        hard = "hard"

    class AgentSpec:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class BotSpec:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class SC2Env:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def reset(self):
            return [None]

        def step(self, actions):
            return [None]

        def close(self):
            return None

    sc2_env_mod.Race = Race
    sc2_env_mod.Difficulty = Difficulty
    sc2_env_mod.Agent = AgentSpec
    sc2_env_mod.Bot = BotSpec
    sc2_env_mod.SC2Env = SC2Env
    env_mod.sc2_env = sc2_env_mod
    base_env_wrapper_mod.BaseEnvWrapper = BaseEnvWrapper
    env_mod.base_env_wrapper = base_env_wrapper_mod

    pysc2_mod.agents = agents_mod
    pysc2_mod.lib = lib_mod
    pysc2_mod.env = env_mod

    sys.modules["pysc2"] = pysc2_mod
    sys.modules["pysc2.agents"] = agents_mod
    sys.modules["pysc2.agents.base_agent"] = base_agent_mod
    sys.modules["pysc2.lib"] = lib_mod
    sys.modules["pysc2.lib.actions"] = actions_mod
    sys.modules["pysc2.lib.colors"] = colors_mod
    sys.modules["pysc2.lib.features"] = features_mod
    sys.modules["pysc2.env"] = env_mod
    sys.modules["pysc2.env.base_env_wrapper"] = base_env_wrapper_mod
    sys.modules["pysc2.env.sc2_env"] = sc2_env_mod


_install_fake_pysc2()


@pytest.fixture
def fake_actions():
    from pysc2.lib import actions

    return actions.FUNCTIONS


@pytest.fixture
def make_obs(fake_actions):
    def _make_obs(
        spatial_shape=(27, 84, 84),
        friendly_positions=None,
        enemy_positions=None,
        friendly_health=100,
        enemy_health=45,
        available_actions=None,
        last=False,
        reward=0.0,
    ):
        if friendly_positions is None:
            friendly_positions = [(10, 10)]
        if enemy_positions is None:
            enemy_positions = [(20, 20)]
        if available_actions is None:
            available_actions = {
                fake_actions.Attack_screen.id,
                fake_actions.Move_screen.id,
                fake_actions.select_army.id,
            }

        player_relative = np.zeros(
            (spatial_shape[1], spatial_shape[2]), dtype=np.uint8,
        )
        for x, y in friendly_positions:
            player_relative[y, x] = 1
        for x, y in enemy_positions:
            player_relative[y, x] = 4

        screen_data = np.zeros(spatial_shape, dtype=np.uint8)
        feature_screen = MockFeatureScreen(
            screen_data, player_relative=player_relative,
        )

        friendlies = [
            SimpleNamespace(
                alliance=1,
                health=friendly_health,
                x=x,
                y=y,
                unit_type=48,
                attack_range=5,
            )
            for x, y in friendly_positions
        ]
        enemies = [
            SimpleNamespace(
                alliance=4,
                health=enemy_health,
                x=x,
                y=y,
                unit_type=110,
                attack_range=5,
            )
            for x, y in enemy_positions
        ]

        return SimpleNamespace(
            observation=SimpleNamespace(
                feature_screen=feature_screen,
                feature_units=friendlies + enemies,
                player=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                score_cumulative=[0] * 13,
                available_actions=set(available_actions),
            ),
            reward=reward,
            last=lambda: last,
        )

    return _make_obs
