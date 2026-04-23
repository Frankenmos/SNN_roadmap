"""Inference-only runner for DefeatRoaches."""

import logging
import os

import torch
from absl import app, flags

from agent import DefeatRoaches
from Utility.config import cfg
from Utility.eval_trace import EpisodeTraceRecorder
from envs.setup_env import create_env
from obs_space.obs_space_2 import ObservationExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
if "checkpoint" not in FLAGS:
    flags.DEFINE_string(
        "checkpoint", None, "Path to a .pth file. Overrides --run_name / --best.",
    )
if "run_name" not in FLAGS:
    flags.DEFINE_string(
        "run_name", None, "Run directory to load from (joins models_dir/run_name/).",
    )
if "best" not in FLAGS:
    flags.DEFINE_bool(
        "best", False, "Prefer best_checkpoint.pth over checkpoint.pth.",
    )
if "episodes" not in FLAGS:
    flags.DEFINE_integer("episodes", 5, "Number of episodes to play.")
if "visualize" not in FLAGS:
    flags.DEFINE_bool(
        "visualize", True, "Show the SC2 renderer. Disable with --novisualize.",
    )
if "deterministic" not in FLAGS:
    flags.DEFINE_bool(
        "deterministic",
        True,
        "Use argmax actions instead of sampling. Disable with --nodeterministic.",
    )
if "inspect" not in FLAGS:
    flags.DEFINE_bool(
        "inspect",
        False,
        "Enable the ObservationInspectorWrapper to log obs schema/stats.",
    )
if "inspect_output" not in FLAGS:
    flags.DEFINE_string(
        "inspect_output",
        None,
        "Path for inspector JSONL. Defaults to "
        "analysis_results/<run_name>/eval_observation_space.jsonl when "
        "--run_name is known, otherwise analysis_results/eval_observation_space.jsonl.",
    )
if "inspect_policy_input" not in FLAGS:
    flags.DEFINE_bool(
        "inspect_policy_input",
        False,
        "Enable PolicyInputDiagnosticsWrapper to log raw obs + extracted batch summaries.",
    )
if "inspect_actions" not in FLAGS:
    flags.DEFINE_bool(
        "inspect_actions",
        False,
        "Enable AvailableActionsDiagnosticsWrapper to log per-step action availability and dispatched calls.",
    )
if "policy_input_output" not in FLAGS:
    flags.DEFINE_string(
        "policy_input_output",
        None,
        "Path for policy-input diagnostics JSONL. Defaults to "
        "analysis_results/<run_name>/policy_input_diagnostics.jsonl when "
        "--run_name is known, otherwise analysis_results/policy_input_diagnostics.jsonl.",
    )
if "policy_input_every" not in FLAGS:
    flags.DEFINE_integer(
        "policy_input_every",
        1,
        "Log every N env steps when --inspect_policy_input is enabled.",
    )
if "actions_output" not in FLAGS:
    flags.DEFINE_string(
        "actions_output",
        None,
        "Path for action-space diagnostics JSONL. Defaults to "
        "analysis_results/<run_name>/available_actions_diagnostics.jsonl when "
        "--run_name is known, otherwise analysis_results/available_actions_diagnostics.jsonl.",
    )
if "actions_every" not in FLAGS:
    flags.DEFINE_integer(
        "actions_every",
        1,
        "Log every N env steps when --inspect_actions is enabled.",
    )
if "trace_episodes" not in FLAGS:
    flags.DEFINE_integer(
        "trace_episodes",
        0,
        "Number of eval episodes to save as per-step trace files. 0 disables tracing.",
    )
if "trace_output_dir" not in FLAGS:
    flags.DEFINE_string(
        "trace_output_dir",
        None,
        "Directory for per-episode eval trace .pt files. Defaults to "
        "analysis_results/<run_name>/episode_traces when --run_name is known, "
        "otherwise analysis_results/episode_traces.",
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


def _load_checkpoint_state(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)


def _default_trace_output_dir(run_name):
    analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
    name = run_name or getattr(cfg.environment, "run_name", "")
    if name:
        return os.path.join(analysis_dir, name, "episode_traces")
    return os.path.join(analysis_dir, "episode_traces")


def play(
    checkpoint_path,
    episodes,
    visualize,
    deterministic,
    inspect=False,
    inspect_output_path=None,
    inspect_policy_input=False,
    policy_input_output_path=None,
    policy_input_every=1,
    inspect_actions=False,
    actions_output_path=None,
    actions_every=1,
    trace_episodes=0,
    trace_output_dir=None,
    run_name=None,
):
    env = create_env(
        map_name=cfg.environment.map_name,
        visualize=visualize,
        use_action_printer=False,
        use_available_actions_diagnostics=inspect_actions,
        available_actions_diagnostics_output_path=(
            actions_output_path
            or getattr(
                cfg.environment,
                "available_actions_diagnostics_output_path",
                "analysis_results/available_actions_diagnostics.jsonl",
            )
        ),
        available_actions_diagnostics_every_n_steps=actions_every,
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
        use_policy_input_diagnostics=inspect_policy_input,
        policy_input_diagnostics_output_path=(
            policy_input_output_path
            or getattr(
                cfg.environment,
                "policy_input_diagnostics_output_path",
                "analysis_results/policy_input_diagnostics.jsonl",
            )
        ),
        policy_input_diagnostics_every_n_steps=policy_input_every,
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

        state = _load_checkpoint_state(checkpoint_path, agent.policy.device)
        agent.policy.load_state_dict(state["agent_state"])
        agent.extractor.load_state_dict(state.get("extractor_state", {}))
        agent.policy.eval()
        ckpt_ep = state.get("episode", "?")
        logger.info(
            "Loaded checkpoint from %s (trained to episode %s)",
            checkpoint_path,
            ckpt_ep,
        )

        steps_cap = cfg.environment.steps_per_episode
        rewards = []
        traced_episode_count = max(0, min(int(trace_episodes), int(episodes)))
        if traced_episode_count > 0 and trace_output_dir is None:
            trace_output_dir = _default_trace_output_dir(run_name)
        for ep in range(episodes):
            obs = env.reset()[0]
            agent.reset()
            ep_reward = 0.0
            steps = 0
            trace_recorder = None
            if ep < traced_episode_count:
                trace_recorder = EpisodeTraceRecorder(
                    output_dir=trace_output_dir,
                    run_name=run_name,
                    checkpoint_path=checkpoint_path,
                    checkpoint_episode=ckpt_ep,
                    deterministic=deterministic,
                )
            while True:
                (
                    action_func,
                    action,
                    move_x,
                    move_y,
                    _pre_step_state,
                    log_prob,
                    value,
                    policy_input,
                    learnable,
                ) = agent.step(obs, deterministic=deterministic)
                next_obs = env.step([action_func])[0]
                steps += 1
                ep_reward += float(next_obs.reward)
                reached_cap = steps >= steps_cap
                done = bool(next_obs.last() or reached_cap)
                if trace_recorder is not None:
                    trace_recorder.add_step(
                        step_index=steps - 1,
                        action_func=action_func,
                        action=action,
                        move_x=move_x,
                        move_y=move_y,
                        log_prob=log_prob,
                        value=value,
                        reward=float(next_obs.reward),
                        cumulative_reward=ep_reward,
                        done=done,
                        learnable=learnable,
                        policy_input=policy_input,
                    )
                obs = next_obs
                if done:
                    break
            rewards.append(ep_reward)
            if trace_recorder is not None:
                saved_trace = trace_recorder.save(
                    episode_index=ep + 1,
                    total_reward=ep_reward,
                    steps=steps,
                )
                logger.info(
                    "Saved eval trace for episode %s to %s",
                    ep + 1,
                    saved_trace,
                )
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

    policy_input_output_path = FLAGS.policy_input_output
    if FLAGS.inspect_policy_input and policy_input_output_path is None:
        analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
        name = FLAGS.run_name or getattr(cfg.environment, "run_name", "")
        if name:
            policy_input_output_path = os.path.join(
                analysis_dir, name, "policy_input_diagnostics.jsonl",
            )
        else:
            policy_input_output_path = os.path.join(
                analysis_dir, "policy_input_diagnostics.jsonl",
            )

    actions_output_path = FLAGS.actions_output
    if FLAGS.inspect_actions and actions_output_path is None:
        analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
        name = FLAGS.run_name or getattr(cfg.environment, "run_name", "")
        if name:
            actions_output_path = os.path.join(
                analysis_dir, name, "available_actions_diagnostics.jsonl",
            )
        else:
            actions_output_path = os.path.join(
                analysis_dir, "available_actions_diagnostics.jsonl",
            )

    trace_output_dir = FLAGS.trace_output_dir
    if FLAGS.trace_episodes > 0 and trace_output_dir is None:
        trace_output_dir = _default_trace_output_dir(
            FLAGS.run_name or getattr(cfg.environment, "run_name", ""),
        )

    play(
        checkpoint_path=checkpoint_path,
        episodes=FLAGS.episodes,
        visualize=FLAGS.visualize,
        deterministic=FLAGS.deterministic,
        inspect=FLAGS.inspect,
        inspect_output_path=inspect_output_path,
        inspect_policy_input=FLAGS.inspect_policy_input,
        policy_input_output_path=policy_input_output_path,
        policy_input_every=FLAGS.policy_input_every,
        inspect_actions=FLAGS.inspect_actions,
        actions_output_path=actions_output_path,
        actions_every=FLAGS.actions_every,
        trace_episodes=FLAGS.trace_episodes,
        trace_output_dir=trace_output_dir,
        run_name=FLAGS.run_name or getattr(cfg.environment, "run_name", ""),
    )


if __name__ == "__main__":
    app.run(main)
