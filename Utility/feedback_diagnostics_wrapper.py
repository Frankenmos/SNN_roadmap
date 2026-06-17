"""Env wrappers for action-feedback and score diagnostics."""

import json
import os
from collections.abc import Mapping, Sequence

import numpy as np
from pysc2.env import base_env_wrapper

from obs_space._numeric import safe_float


SCORE_CUMULATIVE_NAMES = (
    "score",
    "idle_production_time",
    "idle_worker_time",
    "total_value_units",
    "total_value_structures",
    "killed_value_units",
    "killed_value_structures",
    "collected_minerals",
    "collected_vespene",
    "collection_rate_minerals",
    "collection_rate_vespene",
    "spent_minerals",
    "spent_vespene",
)


class LastActionDiagnosticsWrapper(base_env_wrapper.BaseEnvWrapper):
    """
    Structured JSONL logger for the post-action feedback fields.

    This is deliberately separate from AvailableActionsDiagnosticsWrapper:
    that wrapper asks "what could I do and what did I dispatch?", while this
    wrapper asks "what did the next observation say happened?"
    """

    def __init__(self, env, output_path, log_every_n_steps=1):
        super(LastActionDiagnosticsWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.episode_index = 0
        self.step_index = 0
        self._prev_feedback = []

        try:
            self._action_spec = self.action_spec()[0]
        except Exception:
            self._action_spec = None

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(LastActionDiagnosticsWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self._prev_feedback = []
        self._process_timesteps(
            timesteps,
            event="reset",
            dispatched_actions=None,
            should_log=True,
        )
        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        dispatched_actions = _normalize_actions(args, kwargs)
        timesteps = super(LastActionDiagnosticsWrapper, self).step(*args, **kwargs)
        self.step_index += 1
        should_log = self.step_index % self.log_every_n_steps == 0
        self._process_timesteps(
            timesteps,
            event="step",
            dispatched_actions=dispatched_actions,
            should_log=should_log,
        )
        return timesteps

    def _process_timesteps(self, timesteps, event, dispatched_actions, should_log):
        current_feedback = []
        for agent_idx, timestep in enumerate(timesteps):
            feedback = self._read_feedback(timestep.observation)
            current_feedback.append(feedback)
            if should_log:
                previous = (
                    self._prev_feedback[agent_idx]
                    if agent_idx < len(self._prev_feedback)
                    else _empty_feedback()
                )
                dispatched_action = (
                    dispatched_actions[agent_idx]
                    if dispatched_actions is not None
                    and agent_idx < len(dispatched_actions)
                    else None
                )
                record = self._build_record(
                    event=event,
                    agent_idx=agent_idx,
                    previous=previous,
                    current=feedback,
                    dispatched_action=dispatched_action,
                )
                _append_jsonl(self.output_path, record)
        self._prev_feedback = current_feedback

    def _read_feedback(self, obs):
        return {
            "available_action_ids": _as_int_list(_read_field(obs, "available_actions")),
            "last_action_ids": _as_int_list(_read_field(obs, "last_actions")),
            "action_result": _as_int_list(_read_field(obs, "action_result")),
            "alerts": _as_int_list(_read_field(obs, "alerts")),
            "game_loop": _as_int_list(_read_field(obs, "game_loop")),
        }

    def _build_record(self, event, agent_idx, previous, current, dispatched_action):
        dispatched = self._serialize_action_call(dispatched_action)
        current_last_actions = current["last_action_ids"]
        dispatched_function_id = (
            None if dispatched is None else dispatched.get("function_id")
        )
        dispatched_seen_in_last_actions = None
        dispatched_available = None
        if dispatched_function_id is not None:
            dispatched_seen_in_last_actions = (
                int(dispatched_function_id) in set(current_last_actions)
            )
            dispatched_available = (
                int(dispatched_function_id) in set(previous["available_action_ids"])
            )

        return {
            "event": event,
            "episode": self.episode_index,
            "step": self.step_index,
            "agent_index": agent_idx,
            "previous_frame": self._decorate_feedback(previous),
            "current_frame": self._decorate_feedback(current),
            "dispatched_action": dispatched,
            "dispatched_action_in_previous_available": dispatched_available,
            "feedback_summary": {
                "has_last_actions": len(current_last_actions) > 0,
                "has_action_result": len(current["action_result"]) > 0,
                "has_alerts": len(current["alerts"]) > 0,
                "dispatched_function_seen_in_current_last_actions": (
                    dispatched_seen_in_last_actions
                ),
            },
        }

    def _decorate_feedback(self, feedback):
        return {
            "available_action_ids": feedback["available_action_ids"],
            "last_action_ids": feedback["last_action_ids"],
            "last_action_names": [
                self._decode_action_name(action_id)
                for action_id in feedback["last_action_ids"]
            ],
            "action_result": feedback["action_result"],
            "alerts": feedback["alerts"],
            "game_loop": feedback["game_loop"],
        }

    def _decode_action_name(self, function_id):
        if self._action_spec is None:
            return None
        try:
            fn = self._action_spec.functions[int(function_id)]
        except Exception:
            return None
        if hasattr(fn, "name"):
            return fn.name
        try:
            return fn.str(True)
        except Exception:
            return str(fn)

    def _serialize_action_call(self, action_call):
        if action_call is None:
            return None
        function_id = getattr(action_call, "function", None)
        if function_id is None:
            function_id = getattr(action_call, "id", None)
        function_name = getattr(action_call, "name", None)
        if function_name is None and function_id is not None:
            function_name = self._decode_action_name(function_id)

        arguments = getattr(action_call, "arguments", None)
        if arguments is None:
            arguments = getattr(action_call, "args", None)
        return {
            "function_id": None if function_id is None else int(function_id),
            "function_name": function_name,
            "arguments": _to_serializable(arguments),
        }


class ScoreDiagnosticsWrapper(base_env_wrapper.BaseEnvWrapper):
    """Structured JSONL logger for reward, score_cumulative, and score deltas."""

    def __init__(self, env, output_path, log_every_n_steps=1):
        super(ScoreDiagnosticsWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.episode_index = 0
        self.step_index = 0
        self._prev_scores = []
        self._episode_rewards = []

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(ScoreDiagnosticsWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self._prev_scores = []
        self._episode_rewards = [0.0 for _ in timesteps]
        self._process_timesteps(timesteps, event="reset", should_log=True)
        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        timesteps = super(ScoreDiagnosticsWrapper, self).step(*args, **kwargs)
        self.step_index += 1
        should_log = self.step_index % self.log_every_n_steps == 0
        self._process_timesteps(timesteps, event="step", should_log=should_log)
        return timesteps

    def _process_timesteps(self, timesteps, event, should_log):
        current_scores = []
        self._ensure_episode_reward_slots(len(timesteps))
        for agent_idx, timestep in enumerate(timesteps):
            obs = timestep.observation
            score = _as_float_list(_read_field(obs, "score_cumulative"))
            current_scores.append(score)
            reward = safe_float(getattr(timestep, "reward", 0.0))
            if event == "step":
                self._episode_rewards[agent_idx] += reward

            previous_score = (
                self._prev_scores[agent_idx]
                if agent_idx < len(self._prev_scores)
                else None
            )
            if should_log:
                delta = _score_delta(score, previous_score)
                record = self._build_record(
                    timestep=timestep,
                    event=event,
                    agent_idx=agent_idx,
                    score=score,
                    delta=delta,
                    reward=reward,
                    episode_reward=self._episode_rewards[agent_idx],
                )
                _append_jsonl(self.output_path, record)
        self._prev_scores = current_scores

    def _ensure_episode_reward_slots(self, count):
        while len(self._episode_rewards) < count:
            self._episode_rewards.append(0.0)

    def _build_record(
        self,
        timestep,
        event,
        agent_idx,
        score,
        delta,
        reward,
        episode_reward,
    ):
        obs = timestep.observation
        score_named = _named_values(score, SCORE_CUMULATIVE_NAMES)
        delta_named = (
            None if delta is None else _named_values(delta, SCORE_CUMULATIVE_NAMES)
        )
        nonzero_delta_indices = (
            []
            if delta is None
            else [
                idx
                for idx, value in enumerate(delta)
                if value is not None and abs(float(value)) > 1.0e-9
            ]
        )
        return {
            "event": event,
            "episode": self.episode_index,
            "step": self.step_index,
            "agent_index": agent_idx,
            "reward": reward,
            "episode_reward": episode_reward,
            "last": _safe_last(timestep),
            "game_loop": _as_int_list(_read_field(obs, "game_loop")),
            "alerts": _as_int_list(_read_field(obs, "alerts")),
            "score_cumulative": score,
            "score_cumulative_named": score_named,
            "score_delta": delta,
            "score_delta_named": delta_named,
            "score_total": score[0] if score else None,
            "score_total_delta": None if delta is None or not delta else delta[0],
            "score_nonzero_delta_indices": nonzero_delta_indices,
        }


def _empty_feedback():
    return {
        "available_action_ids": [],
        "last_action_ids": [],
        "action_result": [],
        "alerts": [],
        "game_loop": [],
    }


def _normalize_actions(args, kwargs):
    del kwargs
    if not args:
        return None
    action_list = args[0]
    if isinstance(action_list, Sequence) and not isinstance(action_list, (str, bytes)):
        return list(action_list)
    return [action_list]


def _read_field(obs, key):
    try:
        if isinstance(obs, Mapping):
            return obs.get(key)
        return getattr(obs, key, None)
    except Exception:
        return None


def _as_int_list(value):
    if value is None:
        return []
    try:
        arr = np.asarray(value).reshape(-1)
        return [int(item) for item in arr.tolist()]
    except Exception:
        try:
            return [int(item) for item in list(value)]
        except Exception:
            return []


def _as_float_list(value):
    if value is None:
        return []
    try:
        arr = np.asarray(value).reshape(-1)
        return [float(item) for item in arr.tolist()]
    except Exception:
        try:
            return [float(item) for item in list(value)]
        except Exception:
            return []


def _score_delta(current, previous):
    if previous is None:
        return None
    max_len = max(len(current), len(previous))
    delta = []
    for idx in range(max_len):
        cur = current[idx] if idx < len(current) else 0.0
        prev = previous[idx] if idx < len(previous) else 0.0
        delta.append(float(cur) - float(prev))
    return delta


def _named_values(values, names):
    return {
        names[idx] if idx < len(names) else f"field_{idx}": value
        for idx, value in enumerate(values)
    }


def _safe_last(timestep):
    last = getattr(timestep, "last", None)
    if callable(last):
        try:
            return bool(last())
        except Exception:
            return None
    return None


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_serializable(item) for item in value]
    return value


def _append_jsonl(output_path, record):
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
