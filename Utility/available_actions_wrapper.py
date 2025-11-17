"""An env wrapper to print the available actions."""

from pysc2.env import base_env_wrapper


class AvailableActionsPrinter(base_env_wrapper.BaseEnvWrapper):
  """An env wrapper to print the available actions."""

  def __init__(self, env):
    super(AvailableActionsPrinter, self).__init__(env)
    self._seen = set()
    self._action_spec = self.action_spec()[0]

  def step(self, *args, **kwargs):
    all_obs = super(AvailableActionsPrinter, self).step(*args, **kwargs)
    for obs in all_obs:
      for avail in obs.observation["available_actions"]:
        if avail not in self._seen:
          self._seen.add(avail)
          self._print(self._action_spec.functions[avail].str(True))
    return all_obs

  def _print(self, s):
    print(s)

