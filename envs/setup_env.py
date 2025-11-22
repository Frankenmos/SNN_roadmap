from pysc2.env import sc2_env
from pysc2.lib import features, actions
from Utility.available_actions_wrapper import AvailableActionsPrinter  # Ensure correct import

def create_env(map_name="DefeatZerglingsAndBanelings", visualize=False, use_action_printer=False):
    """Create and return a PySC2 environment, optionally with wrappers."""
    env = sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard),
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True,
            use_raw_units = True,
        ),
        step_mul=6,
        realtime=False,
        game_steps_per_episode=0,
        visualize=visualize
    )
    if use_action_printer:
        env = AvailableActionsPrinter(env)  # Conditionally wrap environment
    return env

