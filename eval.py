"""Inference-only runner for DefeatRoaches."""

import logging
import os

import torch
from absl import app, flags

from agent import DefeatRoaches
from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
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
if "split_diagnostics_by_mode" not in FLAGS:
    flags.DEFINE_bool(
        "split_diagnostics_by_mode",
        True,
        "Append _det or _stoch to eval diagnostic JSONL paths so deterministic "
        "and sampled eval traces do not append into the same files. Disable "
        "with --nosplit_diagnostics_by_mode.",
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
if "inspect_last_action" not in FLAGS:
    flags.DEFINE_bool(
        "inspect_last_action",
        False,
        "Enable LastActionDiagnosticsWrapper to log last_actions, action_result, alerts, and dispatched-action matching.",
    )
if "inspect_score" not in FLAGS:
    flags.DEFINE_bool(
        "inspect_score",
        False,
        "Enable ScoreDiagnosticsWrapper to log score_cumulative and score deltas.",
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
if "last_action_output" not in FLAGS:
    flags.DEFINE_string(
        "last_action_output",
        None,
        "Path for last-action feedback diagnostics JSONL. Defaults to "
        "analysis_results/<run_name>/last_action_diagnostics.jsonl when "
        "--run_name is known, otherwise analysis_results/last_action_diagnostics.jsonl.",
    )
if "last_action_every" not in FLAGS:
    flags.DEFINE_integer(
        "last_action_every",
        1,
        "Log every N env steps when --inspect_last_action is enabled.",
    )
if "score_output" not in FLAGS:
    flags.DEFINE_string(
        "score_output",
        None,
        "Path for score diagnostics JSONL. Defaults to "
        "analysis_results/<run_name>/score_diagnostics.jsonl when "
        "--run_name is known, otherwise analysis_results/score_diagnostics.jsonl.",
    )
if "score_every" not in FLAGS:
    flags.DEFINE_integer(
        "score_every",
        1,
        "Log every N env steps when --inspect_score is enabled.",
    )
if "inspect_smart_outcome" not in FLAGS:
    flags.DEFINE_bool(
        "inspect_smart_outcome",
        False,
        "Enable SmartOutcomeDiagnosticsWrapper to log short-window outcome "
        "classifications for dispatched Smart_screen calls.",
    )
if "smart_outcome_output" not in FLAGS:
    flags.DEFINE_string(
        "smart_outcome_output",
        None,
        "Path for smart-outcome diagnostics JSONL. Defaults to "
        "analysis_results/<run_name>/smart_outcome_diagnostics.jsonl when "
        "--run_name is known, otherwise "
        "analysis_results/smart_outcome_diagnostics.jsonl.",
    )
if "smart_outcome_every" not in FLAGS:
    flags.DEFINE_integer(
        "smart_outcome_every",
        1,
        "Log every N env steps when --inspect_smart_outcome is enabled.",
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


def _mode_suffix(deterministic):
    return "det" if deterministic else "stoch"


def _split_jsonl_path_by_mode(path, deterministic, enabled=True):
    if not path or not enabled:
        return path

    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".jsonl"

    mode_suffix = _mode_suffix(deterministic)
    basename = os.path.basename(root).lower()
    existing_suffixes = (
        "_det",
        "_stoch",
        "_deterministic",
        "_stochastic",
        "_sampled",
    )
    if basename.endswith(existing_suffixes):
        return root + ext
    return f"{root}_{mode_suffix}{ext}"


def _default_jsonl_path(filename, run_name):
    analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
    name = run_name or getattr(cfg.environment, "run_name", "")
    if name:
        return os.path.join(analysis_dir, name, filename).replace(os.sep, "/")
    return os.path.join(analysis_dir, filename).replace(os.sep, "/")


def _resolve_eval_jsonl_path(
    explicit_path,
    *,
    enabled,
    filename,
    run_name,
    deterministic,
    split_by_mode,
):
    if not enabled and explicit_path is None:
        return None
    path = explicit_path or _default_jsonl_path(filename, run_name)
    return _split_jsonl_path_by_mode(
        path,
        deterministic=deterministic,
        enabled=split_by_mode,
    )


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
    inspect_last_action=False,
    last_action_output_path=None,
    last_action_every=1,
    inspect_score=False,
    score_output_path=None,
    score_every=1,
    inspect_smart_outcome=False,
    smart_outcome_output_path=None,
    smart_outcome_every=1,
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
        use_last_action_diagnostics=inspect_last_action,
        last_action_diagnostics_output_path=(
            last_action_output_path
            or getattr(
                cfg.environment,
                "last_action_diagnostics_output_path",
                "analysis_results/last_action_diagnostics.jsonl",
            )
        ),
        last_action_diagnostics_every_n_steps=last_action_every,
        use_score_diagnostics=inspect_score,
        score_diagnostics_output_path=(
            score_output_path
            or getattr(
                cfg.environment,
                "score_diagnostics_output_path",
                "analysis_results/score_diagnostics.jsonl",
            )
        ),
        score_diagnostics_every_n_steps=score_every,
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
        use_smart_outcome_diagnostics=inspect_smart_outcome,
        smart_outcome_diagnostics_output_path=(
            smart_outcome_output_path
            or getattr(
                cfg.environment,
                "smart_outcome_diagnostics_output_path",
                "analysis_results/smart_outcome_diagnostics.jsonl",
            )
        ),
        smart_outcome_diagnostics_every_n_steps=smart_outcome_every,
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
        checkpoint_protocol = state.get("policy_protocol_version")
        if checkpoint_protocol != POLICY_PROTOCOL_VERSION:
            raise RuntimeError(
                "Checkpoint policy protocol mismatch: "
                f"checkpoint={checkpoint_protocol!r}, "
                f"current={POLICY_PROTOCOL_VERSION}. "
                "This evaluator expects stream action-feedback token checkpoints.",
            )
        checkpoint_schema = state.get("policy_input_schema")
        if checkpoint_schema != POLICY_INPUT_SCHEMA:
            raise RuntimeError(
                "Checkpoint policy input schema mismatch: "
                f"checkpoint={checkpoint_schema!r}, "
                f"current={POLICY_INPUT_SCHEMA!r}. "
                "This evaluator expects stream action-feedback token checkpoints.",
            )
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

    run_name = FLAGS.run_name or getattr(cfg.environment, "run_name", "")
    checkpoint_path = _locate_checkpoint(
        FLAGS.checkpoint, FLAGS.run_name, FLAGS.best,
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_by_mode = bool(FLAGS.split_diagnostics_by_mode)
    inspect_output_path = _resolve_eval_jsonl_path(
        FLAGS.inspect_output,
        enabled=FLAGS.inspect,
        filename="eval_observation_space.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )
    policy_input_output_path = _resolve_eval_jsonl_path(
        FLAGS.policy_input_output,
        enabled=FLAGS.inspect_policy_input,
        filename="policy_input_diagnostics.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )
    actions_output_path = _resolve_eval_jsonl_path(
        FLAGS.actions_output,
        enabled=FLAGS.inspect_actions,
        filename="available_actions_diagnostics.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )
    last_action_output_path = _resolve_eval_jsonl_path(
        FLAGS.last_action_output,
        enabled=FLAGS.inspect_last_action,
        filename="last_action_diagnostics.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )
    score_output_path = _resolve_eval_jsonl_path(
        FLAGS.score_output,
        enabled=FLAGS.inspect_score,
        filename="score_diagnostics.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )
    smart_outcome_output_path = _resolve_eval_jsonl_path(
        FLAGS.smart_outcome_output,
        enabled=FLAGS.inspect_smart_outcome,
        filename="smart_outcome_diagnostics.jsonl",
        run_name=run_name,
        deterministic=FLAGS.deterministic,
        split_by_mode=split_by_mode,
    )

    trace_output_dir = FLAGS.trace_output_dir
    if FLAGS.trace_episodes > 0 and trace_output_dir is None:
        trace_output_dir = _default_trace_output_dir(
            run_name,
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
        inspect_last_action=FLAGS.inspect_last_action,
        last_action_output_path=last_action_output_path,
        last_action_every=FLAGS.last_action_every,
        inspect_score=FLAGS.inspect_score,
        score_output_path=score_output_path,
        score_every=FLAGS.score_every,
        inspect_smart_outcome=FLAGS.inspect_smart_outcome,
        smart_outcome_output_path=smart_outcome_output_path,
        smart_outcome_every=FLAGS.smart_outcome_every,
        trace_episodes=FLAGS.trace_episodes,
        trace_output_dir=trace_output_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    app.run(main)
