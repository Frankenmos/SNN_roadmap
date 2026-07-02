"""Env wrapper to inspect and log PySC2 observation space statistics."""

import json
import os
from collections.abc import Mapping

import numpy as np
from pysc2.env import base_env_wrapper


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ObservationInspectorWrapper(base_env_wrapper.BaseEnvWrapper):
    """Logs observation schema and basic stats to a JSONL file."""

    def __init__(self, env, output_path, log_every_n_steps=10, max_unit_samples=5):
        super(ObservationInspectorWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.max_unit_samples = max(1, int(max_unit_samples))
        self.episode_index = 0
        self.step_index = 0

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(ObservationInspectorWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self._log_timesteps(timesteps, event="reset")
        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        timesteps = super(ObservationInspectorWrapper, self).step(*args, **kwargs)
        self.step_index += 1
        if self.step_index % self.log_every_n_steps == 0:
            self._log_timesteps(timesteps, event="step")
        return timesteps

    def _log_timesteps(self, timesteps, event):
        for agent_idx, timestep in enumerate(timesteps):
            obs = timestep.observation
            record = {
                "event": event,
                "episode": self.episode_index,
                "step": self.step_index,
                "agent_index": agent_idx,
                "keys": [],
                "fields": {},
                "feature_units": {},
                "available_actions_count": None,
            }

            if isinstance(obs, Mapping):
                keys = list(obs.keys())
            else:
                keys = [k for k in dir(obs) if not k.startswith("_")]
            record["keys"] = sorted([str(k) for k in keys])

            for key in keys:
                try:
                    value = obs[key] if isinstance(obs, Mapping) else getattr(obs, key)
                except Exception:
                    continue

                field_info = self._summarize_field(value)
                if field_info is not None:
                    record["fields"][str(key)] = field_info

            feature_units = self._read_field(obs, "feature_units")
            if feature_units is not None:
                record["feature_units"] = self._summarize_feature_units(feature_units)

            available_actions = self._read_field(obs, "available_actions")
            if available_actions is not None:
                try:
                    record["available_actions_count"] = int(len(available_actions))
                except Exception:
                    pass

            self._append_jsonl(record)

    def _read_field(self, obs, key):
        try:
            if isinstance(obs, Mapping):
                return obs.get(key)
            return getattr(obs, key, None)
        except Exception:
            return None

    def _summarize_field(self, value):
        try:
            if isinstance(value, np.ndarray):
                info = {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                if value.size > 0 and np.issubdtype(value.dtype, np.number):
                    info["min"] = _to_serializable(np.min(value))
                    info["max"] = _to_serializable(np.max(value))
                return info

            if isinstance(value, (list, tuple)):
                info = {
                    "type": type(value).__name__,
                    "len": len(value),
                }
                if len(value) > 0 and hasattr(value[0], "dtype"):
                    info["first_item_type"] = type(value[0]).__name__
                return info

            if np.isscalar(value):
                return {
                    "type": type(value).__name__,
                    "value": _to_serializable(value),
                }

            return {
                "type": type(value).__name__,
            }
        except Exception:
            return {
                "type": type(value).__name__,
                "summary_error": True,
            }

    def _summarize_feature_units(self, feature_units):
        summary = {"count": 0, "sample": []}
        try:
            summary["count"] = int(len(feature_units))
            for unit in list(feature_units)[: self.max_unit_samples]:
                sample_item = {
                    "unit_type": _to_serializable(getattr(unit, "unit_type", None)),
                    "alliance": _to_serializable(getattr(unit, "alliance", None)),
                    "x": _to_serializable(getattr(unit, "x", None)),
                    "y": _to_serializable(getattr(unit, "y", None)),
                    "health": _to_serializable(getattr(unit, "health", None)),
                    "health_max": _to_serializable(getattr(unit, "health_max", None)),
                }
                summary["sample"].append(sample_item)
        except Exception:
            summary["summary_error"] = True
        return summary

    def _append_jsonl(self, record):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
