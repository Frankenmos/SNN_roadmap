"""Env wrapper for diagnostics-only Smart_screen outcome logging."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

import numpy as np
from pysc2.env import base_env_wrapper

from agent_core.policy_protocol import SMART_SCREEN_FUNCTION_ID
from obs_space._numeric import safe_float
from obs_space.action_effects import FrameSnapshot, extract_frame_snapshot
from obs_space.smart_outcome_detector import SmartOutcomeDetector


class SmartOutcomeDiagnosticsWrapper(base_env_wrapper.BaseEnvWrapper):
    """
    Log short-window classifications for dispatched PySC2 Smart_screen calls.

    The wrapper owns PySC2 action parsing and previous/current frame timing. The
    detector owns attribution and classification.
    """

    def __init__(
        self,
        env,
        output_path,
        *,
        outcome_window: int = 5,
        near_enemy_threshold: float = 8.0,
        log_every_n_steps: int = 1,
    ):
        super(SmartOutcomeDiagnosticsWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.episode_index = 0
        self.step_index = 0
        self._detector_kwargs = {
            "outcome_window": int(outcome_window),
            "near_enemy_threshold": float(near_enemy_threshold),
        }
        self._detectors: list[SmartOutcomeDetector] = []
        self._previous_frames: list[FrameSnapshot | None] = []
        self._previous_feature_units: list[object | None] = []

        try:
            self._action_spec = self.action_spec()[0]
        except Exception:
            self._action_spec = None

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(SmartOutcomeDiagnosticsWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self._ensure_agent_slots(len(timesteps))
        for detector in self._detectors:
            detector.reset()

        for agent_idx, timestep in enumerate(timesteps):
            obs = timestep.observation
            self._previous_frames[agent_idx] = extract_frame_snapshot(obs)
            self._previous_feature_units[agent_idx] = _read_field(obs, "feature_units")
            self._log_record(
                event="reset",
                agent_idx=agent_idx,
                observation=obs,
                outcomes=[],
                dispatched_action=None,
                smart_target=None,
            )

        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        actions = _normalize_actions(args, kwargs)
        self._ensure_agent_slots(len(actions))

        smart_targets: list[tuple[float, float] | None] = []
        for agent_idx, action in enumerate(actions):
            target = self._extract_smart_target(action)
            smart_targets.append(target)
            if target is None:
                continue
            previous_frame = self._previous_frames[agent_idx]
            if previous_frame is None:
                continue
            self._detectors[agent_idx].observe_smart_click(
                previous_frame=previous_frame,
                target=target,
                click_step=self.step_index,
                previous_feature_units=self._previous_feature_units[agent_idx],
            )

        timesteps = super(SmartOutcomeDiagnosticsWrapper, self).step(*args, **kwargs)
        self.step_index += 1
        self._ensure_agent_slots(len(timesteps))
        should_log = self.step_index % self.log_every_n_steps == 0

        for agent_idx, timestep in enumerate(timesteps):
            obs = timestep.observation
            current_frame = extract_frame_snapshot(obs)
            current_feature_units = _read_field(obs, "feature_units")
            outcomes = self._detectors[agent_idx].resolve(
                current_frame=current_frame,
                resolution_step=self.step_index,
                current_feature_units=current_feature_units,
            )
            self._previous_frames[agent_idx] = current_frame
            self._previous_feature_units[agent_idx] = current_feature_units

            dispatched_action = actions[agent_idx] if agent_idx < len(actions) else None
            smart_target = smart_targets[agent_idx] if agent_idx < len(smart_targets) else None
            if should_log or outcomes:
                self._log_record(
                    event="step",
                    agent_idx=agent_idx,
                    observation=obs,
                    outcomes=[outcome.to_dict() for outcome in outcomes],
                    dispatched_action=dispatched_action,
                    smart_target=smart_target,
                )

        return timesteps

    def _ensure_agent_slots(self, count: int) -> None:
        while len(self._detectors) < count:
            self._detectors.append(SmartOutcomeDetector(**self._detector_kwargs))
            self._previous_frames.append(None)
            self._previous_feature_units.append(None)

    def _is_smart_screen_call(self, action) -> bool:
        return self._action_function_id(action) == SMART_SCREEN_FUNCTION_ID

    def _extract_smart_target(self, action) -> tuple[float, float] | None:
        if not self._is_smart_screen_call(action):
            return None

        if isinstance(action, Mapping):
            if "target" in action:
                return _target_pair(action["target"])
            if "x" in action and "y" in action:
                return (safe_float(action["x"]), safe_float(action["y"]))

        arguments = getattr(action, "arguments", None)
        if arguments is None:
            arguments = getattr(action, "args", None)
        if arguments is None and isinstance(action, Mapping):
            arguments = action.get("arguments") or action.get("args")
        if arguments is None:
            return None

        try:
            args = list(arguments)
        except TypeError:
            return None
        if len(args) < 2:
            return None
        return _target_pair(args[1])

    def _action_function_id(self, action) -> int | None:
        if action is None:
            return None
        if isinstance(action, Mapping):
            for key in ("function", "function_id", "id", "action_id"):
                if key in action:
                    function_id = _safe_int(action[key])
                    if function_id is not None:
                        return function_id
        for attr in ("function", "id", "action_id"):
            function_id = _safe_int(getattr(action, attr, None))
            if function_id is not None:
                return function_id
        name = getattr(action, "name", None)
        if name is None and isinstance(action, Mapping):
            name = action.get("name") or action.get("function_name")
        if name and "smart_screen" in str(name).lower():
            return SMART_SCREEN_FUNCTION_ID
        return None

    def _decode_action_name(self, function_id):
        if self._action_spec is None or function_id is None:
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
        function_id = self._action_function_id(action_call)
        function_name = getattr(action_call, "name", None)
        if function_name is None and isinstance(action_call, Mapping):
            function_name = action_call.get("name") or action_call.get("function_name")
        if function_name is None:
            function_name = self._decode_action_name(function_id)

        arguments = getattr(action_call, "arguments", None)
        if arguments is None:
            arguments = getattr(action_call, "args", None)
        if arguments is None and isinstance(action_call, Mapping):
            arguments = action_call.get("arguments") or action_call.get("args")

        return {
            "function_id": function_id,
            "function_name": function_name,
            "arguments": _to_serializable(arguments),
        }

    def _log_record(
        self,
        *,
        event: str,
        agent_idx: int,
        observation,
        outcomes: list[dict],
        dispatched_action,
        smart_target: tuple[float, float] | None,
    ) -> None:
        obs = getattr(observation, "observation", observation)
        record = {
            "event": event,
            "episode": self.episode_index,
            "step": self.step_index,
            "agent_index": agent_idx,
            "was_smart": smart_target is not None,
            "smart_target": None if smart_target is None else list(smart_target),
            "num_pending_clicks": self._detectors[agent_idx].pending_count,
            "num_resolved_outcomes": len(outcomes),
            "outcomes": outcomes,
            "dispatched_action": self._serialize_action_call(dispatched_action),
            "game_loop": _as_int_list(_read_field(obs, "game_loop")),
        }
        _append_jsonl(self.output_path, record)


def _normalize_actions(args, kwargs) -> list:
    del kwargs
    if not args:
        return []
    action_list = args[0]
    if isinstance(action_list, Sequence) and not isinstance(action_list, (str, bytes)):
        return list(action_list)
    return [action_list]


def _target_pair(value) -> tuple[float, float] | None:
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) == 1 and isinstance(items[0], Sequence) and not isinstance(items[0], (str, bytes)):
        try:
            items = list(items[0])
        except TypeError:
            return None
    if len(items) < 2:
        return None
    return (safe_float(items[0]), safe_float(items[1]))


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_field(obs, key):
    try:
        if isinstance(obs, Mapping):
            return obs.get(key)
        return getattr(obs, key, None)
    except Exception:
        return None


def _as_int_list(value) -> list[int]:
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


def _append_jsonl(output_path, record) -> None:
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
