"""Env wrappers for inspecting available PySC2 actions."""

import json
import os
from collections.abc import Mapping, Sequence

import numpy as np
from pysc2.env import base_env_wrapper


class AvailableActionsPrinter(base_env_wrapper.BaseEnvWrapper):
    """Print each newly seen available action once."""

    def __init__(self, env):
        super(AvailableActionsPrinter, self).__init__(env)
        self._seen = set()
        self._action_spec = self.action_spec()[0]

    def step(self, *args, **kwargs):
        all_obs = super(AvailableActionsPrinter, self).step(*args, **kwargs)
        for obs in all_obs:
            for avail in self._read_available_actions(obs):
                if avail not in self._seen:
                    self._seen.add(avail)
                    self._print(self._action_spec.functions[avail].str(True))
        return all_obs

    @staticmethod
    def _read_available_actions(obs):
        observation = getattr(obs, "observation", obs)
        if isinstance(observation, Mapping):
            available = observation.get("available_actions")
        else:
            available = getattr(observation, "available_actions", [])
        return np.asarray(available).reshape(-1).tolist()

    def _print(self, s):
        print(s)


class AvailableActionsDiagnosticsWrapper(base_env_wrapper.BaseEnvWrapper):
    """
    Structured JSONL logger for action-space design work.

    Logs, per reset/step:
    - previous-step available_actions ids/names
    - current-step available_actions ids/names
    - dispatched PySC2 function id/name/args
    - whether the dispatched function id was available on the previous frame
    - small state breadcrumbs (selection counts, feature unit count, last_actions)
    """

    def __init__(self, env, output_path, log_every_n_steps=1):
        super(AvailableActionsDiagnosticsWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.episode_index = 0
        self.step_index = 0
        self._prev_available_actions = []

        try:
            self._action_spec = self.action_spec()[0]
        except Exception:
            self._action_spec = None

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(AvailableActionsDiagnosticsWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self._process_timesteps(
            timesteps,
            event="reset",
            dispatched_actions=None,
            should_log=True,
        )
        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        dispatched_actions = self._normalize_actions(args, kwargs)
        timesteps = super(AvailableActionsDiagnosticsWrapper, self).step(*args, **kwargs)
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
        current_available = []
        for agent_idx, timestep in enumerate(timesteps):
            available_ids = self._as_int_list(
                self._read_field(timestep.observation, "available_actions"),
            )
            current_available.append(available_ids)
            if should_log:
                prev_available = (
                    self._prev_available_actions[agent_idx]
                    if agent_idx < len(self._prev_available_actions)
                    else []
                )
                dispatched_action = (
                    dispatched_actions[agent_idx]
                    if dispatched_actions is not None
                    and agent_idx < len(dispatched_actions)
                    else None
                )
                record = self._build_record(
                    timestep=timestep,
                    event=event,
                    agent_idx=agent_idx,
                    prev_available=prev_available,
                    current_available=available_ids,
                    dispatched_action=dispatched_action,
                )
                self._append_jsonl(record)
        self._prev_available_actions = current_available

    def _build_record(
        self,
        timestep,
        event,
        agent_idx,
        prev_available,
        current_available,
        dispatched_action,
    ):
        obs = timestep.observation
        feature_units = self._read_field(obs, "feature_units")
        multi_select = self._read_field(obs, "multi_select")
        single_select = self._read_field(obs, "single_select")
        last_actions = self._as_int_list(self._read_field(obs, "last_actions"))
        dispatched = self._serialize_action_call(dispatched_action)
        dispatched_available = None
        if dispatched is not None and dispatched.get("function_id") is not None:
            dispatched_available = (
                int(dispatched["function_id"]) in set(prev_available)
            )

        return {
            "event": event,
            "episode": self.episode_index,
            "step": self.step_index,
            "agent_index": agent_idx,
            "previous_frame": {
                "available_action_ids": prev_available,
                "available_action_names": [
                    self._decode_action_name(action_id) for action_id in prev_available
                ],
                "available_action_count": len(prev_available),
            },
            "current_frame": {
                "available_action_ids": current_available,
                "available_action_names": [
                    self._decode_action_name(action_id)
                    for action_id in current_available
                ],
                "available_action_count": len(current_available),
                "last_action_ids": last_actions,
                "feature_unit_count": self._safe_len(feature_units),
                "multi_select_count": self._safe_len(multi_select),
                "single_select_count": self._safe_len(single_select),
                "action_result": self._as_int_list(
                    self._read_field(obs, "action_result"),
                ),
            },
            "dispatched_action": dispatched,
            "dispatched_action_in_previous_available": dispatched_available,
        }

    def _normalize_actions(self, args, kwargs):
        del kwargs
        if not args:
            return None
        action_list = args[0]
        if isinstance(action_list, Sequence) and not isinstance(action_list, (str, bytes)):
            return list(action_list)
        return [action_list]

    @staticmethod
    def _read_field(obs, key):
        if isinstance(obs, Mapping):
            return obs.get(key)
        return getattr(obs, key, None)

    @staticmethod
    def _safe_len(value):
        try:
            return int(len(value))
        except Exception:
            return 0

    @staticmethod
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
            "function_id": (
                None if function_id is None else int(function_id)
            ),
            "function_name": function_name,
            "arguments": self._to_serializable(arguments),
        }

    def _to_serializable(self, value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {
                str(key): self._to_serializable(item) for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._to_serializable(item) for item in value]
        return value

    def _append_jsonl(self, record):
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
