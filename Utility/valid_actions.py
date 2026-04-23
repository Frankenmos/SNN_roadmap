"""Print the valid actions."""

from absl import app
from absl import flags

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import point_flag

FLAGS = flags.FLAGS
point_flag.DEFINE_point("screen_size", "84", "Resolution for screen actions.")
point_flag.DEFINE_point("minimap_size", "64", "Resolution for minimap actions.")
flags.DEFINE_bool("hide_specific", False, "Hide the specific actions")


def main(unused_argv):
  """Print the valid actions."""
  feats = features.Features(
      # Actually irrelevant whether it's feature or rgb size.
      features.AgentInterfaceFormat(
          feature_dimensions=features.Dimensions(
              screen=FLAGS.screen_size,
              minimap=FLAGS.minimap_size)))
  action_spec = feats.action_spec()
  flattened = 0
  count = 0
  for func in action_spec.functions:
    if FLAGS.hide_specific and actions.FUNCTIONS[func.id].general_id != 0:
      continue
    count += 1
    act_flat = 1
    for arg in func.args:
      for size in arg.sizes:
        act_flat *= size
    flattened += act_flat
    print(func.str(True))
  print("Total base actions:", count)
  print("Total possible actions (flattened):", flattened)


if __name__ == "__main__":
  app.run(main)