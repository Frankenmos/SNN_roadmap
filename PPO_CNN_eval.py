"""Inference-only runner for DefeatRoaches."""

import logging
import os

import torch
from absl import app, flags

from PPO_CNN_agent import DefeatRoaches
from Utility.config import cfg
from envs.setup_env import create_env
from obs_space.obs_space_2 import ObservationExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint", None, "Path to a .pth file. Overrides --run_name / --best.",
)
flags.DEFINE_string(
    "run_name", None, "Run directory to load from (joins models_dir/run_name/).",
)
flags.DEFINE_bool(
    "best", False, "Prefer best_checkpoint.pth over checkpoint.pth.",
)
flags.DEFINE_integer("episodes", 5, "Number of episodes to play.")
flags.DEFINE_bool(
    "visualize", True, "Show the SC2 renderer. Disable with --novisualize.",
)
flags.DEFINE_bool(
    "deterministic",
    True,
    "Use argmax actions instead of sampling. Disable with --nodeterministic.",
)
flags.DEFINE_bool(
    "inspect",
    False,
    "Enable the ObservationInspectorWrapper to log obs schema/stats.",
)
flags.DEFINE_string(
    "inspect_output",
    None,
    "Path for inspector JSONL. Defaults to "
    "analysis_results/<run_name>/eval_observation_space.jsonl when "
    "--run_name is known, otherwise analysis_results/eval_observation_space.jsonl.",
)


def _locate_checkpoint(explicit_path, run_name, prefer_best):
    if explicit_path:
        return explicit_path

    models_dir = getattr(cfg.environment, "models_dir", "models")
    name = run_name or getattr(cfg.environment, "run_name", "")
    if not name:
        raise FileNotFoundError(
            "No --checkpoint given and config has no run_name. "
            "Pass --checkpoint or --run_name.",
        )

    filename = (
        getattr(cfg.environment, "best_checkpoint_path", "best_checkpoint.pth")
        if prefer_best
        else getattr(cfg.environment, "checkpoint_path", "checkpoint.pth")
    )
    return os.path.join(models_dir, name, filename)


def play(
    checkpoint_path,
    episodes,
    visualize,
    deterministic,
    inspect=False,
    inspect_output_path=None,
):
    env = create_env(
        map_name=cfg.environment.map_name,
        visualize=visualize,
        use_action_printer=False,
        use_observation_inspector=inspect,
        observation_inspector_output_path=(
            inspect_output_path
            or getattr(
                cfg.environment,
                "observation_inspector_output_path",
                "analysis_results/eval_observation_space.jsonl",
            )
        ),
        observation_inspector_every_n_steps=getattr(
            cfg.environment, "observation_inspector_every_n_steps", 10,
        ),
        observation_inspector_max_unit_samples=getattr(
            cfg.environment, "observation_inspector_max_unit_samples", 5,
        ),
    )
    try:
        obs_ext = ObservationExtractor()
        initial_obs = env.reset()[0]
        spatial_shape, vector_dim = obs_ext.get_observation_dimensions(initial_obs)

        agent = DefeatRoaches(
            spatial_input_shape=spatial_shape,
            vector_input_dim=vector_dim,
            action_dim=cfg.model.action_dim,
        )

        state = torch.load(checkpoint_path, map_location=agent.policy.device)
        agent.policy.load_state_dict(state["agent_state"])
        agent.policy.eval()
        ckpt_ep = state.get("episode", "?")
        logger.info(
            "Loaded checkpoint from %s (trained to episode %s)",
            checkpoint_path,
            ckpt_ep,
        )

        steps_cap = cfg.environment.steps_per_episode
        rewards = []
        for ep in range(episodes):
            obs = env.reset()[0]
            agent.reset()
            ep_reward = 0.0
            steps = 0
            while True:
                action_func = agent.step(obs, deterministic=deterministic)[0]
                next_obs = env.step([action_func])[0]
                steps += 1
                ep_reward += float(next_obs.reward)
                obs = next_obs
                if next_obs.last() or steps >= steps_cap:
                    break
            rewards.append(ep_reward)
            logger.info(
                "Episode %s/%s: reward=%.2f, steps=%s",
                ep + 1,
                episodes,
                ep_reward,
                steps,
            )

        if rewards:
            avg = sum(rewards) / len(rewards)
            logger.info(
                "Mean reward over %s episodes: %.2f (min=%.2f, max=%.2f) | deterministic=%s",
                len(rewards),
                avg,
                min(rewards),
                max(rewards),
                deterministic,
            )
    finally:
        env.close()


def main(argv):
    del argv

    checkpoint_path = _locate_checkpoint(
        FLAGS.checkpoint, FLAGS.run_name, FLAGS.best,
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inspect_output_path = FLAGS.inspect_output
    if FLAGS.inspect and inspect_output_path is None:
        analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
        name = FLAGS.run_name or getattr(cfg.environment, "run_name", "")
        if name:
            inspect_output_path = os.path.join(
                analysis_dir, name, "eval_observation_space.jsonl",
            )
        else:
            inspect_output_path = os.path.join(
                analysis_dir, "eval_observation_space.jsonl",
            )

    play(
        checkpoint_path=checkpoint_path,
        episodes=FLAGS.episodes,
        visualize=FLAGS.visualize,
        deterministic=FLAGS.deterministic,
        inspect=FLAGS.inspect,
        inspect_output_path=inspect_output_path,
    )


if __name__ == "__main__":
    app.run(main)
