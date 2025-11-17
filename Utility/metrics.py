from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np

class MetricsWrapper:
    def __init__(self, env):
        self.env = env
        self.metrics = {
            'total_reward': 0,
            'episode_steps': 0,
            'episode_count': 0,
            'total_episodes': 0
        }

    def reset(self):
        """Reset the environment and metrics."""
        self.metrics['total_reward'] = 0
        self.metrics['episode_steps'] = 0
        self.metrics['episode_count'] += 1
        self.metrics['total_episodes'] += 1
        return self.env.reset()

    def step(self, action):
        """Execute the step and update metrics."""
        obs = self.env.step(action)
        self.metrics['episode_steps'] += 1
        self.metrics['total_reward'] += obs.reward
        return obs

    def get_metrics(self):
        """Retrieve the current metrics."""
        return self.metrics
