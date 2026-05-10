"""Env wrapper to log raw SC2 obs plus extracted PolicyInputBatch summaries."""

import json
import os
from collections.abc import Mapping

import numpy as np
from pysc2.env import base_env_wrapper

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET,
    ACTION_FEEDBACK_FIELD_NAMES,
    ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET,
    ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET,
    ACTION_FEEDBACK_TOKEN_DIM,
    BRIDGE_ACTION_RIGHT_CLICK,
    CURATED_FEATURE_UNIT_FIELDS,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_DIM,
    META_AVAILABLE_ACTION_OFFSET,
    META_LAST_ACTION_INDEX_OFFSET,
    META_PLAYER_FEATURE_DIM,
    SELECTION_FEATURE_NAMES,
)
from obs_space.action_effects import classify_action_effect
from obs_space.obs_space_2 import ObservationExtractor


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class PolicyInputDiagnosticsWrapper(base_env_wrapper.BaseEnvWrapper):
    def __init__(
        self,
        env,
        output_path,
        log_every_n_steps=1,
        max_entity_samples=3,
        max_selection_samples=3,
    ):
        super(PolicyInputDiagnosticsWrapper, self).__init__(env)
        self.output_path = output_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.max_entity_samples = max(1, int(max_entity_samples))
        self.max_selection_samples = max(1, int(max_selection_samples))
        self.episode_index = 0
        self.step_index = 0
        self.extractor = ObservationExtractor()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def reset(self, *args, **kwargs):
        timesteps = super(PolicyInputDiagnosticsWrapper, self).reset(*args, **kwargs)
        self.step_index = 0
        self.extractor.reset()
        self._process_timesteps(timesteps, event="reset", should_log=True)
        self.episode_index += 1
        return timesteps

    def step(self, *args, **kwargs):
        timesteps = super(PolicyInputDiagnosticsWrapper, self).step(*args, **kwargs)
        self.step_index += 1
        should_log = self.step_index % self.log_every_n_steps == 0
        self._process_timesteps(timesteps, event="step", should_log=should_log)
        return timesteps

    def _process_timesteps(self, timesteps, event, should_log):
        for agent_idx, timestep in enumerate(timesteps):
            batch = self.extractor.extract_observation(timestep, update_stats=True)
            if should_log:
                record = self._build_record(
                    timestep=timestep,
                    batch=batch,
                    agent_idx=agent_idx,
                    event=event,
                )
                self._append_jsonl(record)

    def _build_record(self, timestep, batch, agent_idx, event):
        obs = timestep.observation
        available_actions = self._read_field(obs, "available_actions")
        available_action_ids = self._as_int_list(available_actions)
        last_action_ids = self._as_int_list(self._read_field(obs, "last_actions"))
        feature_units = self._read_field(obs, "feature_units")
        multi_select = self._read_field(obs, "multi_select")
        single_select = self._read_field(obs, "single_select")

        raw_entity_count = self._safe_len(feature_units)
        raw_multi_count = self._safe_len(multi_select)
        raw_single_count = self._safe_len(single_select)
        raw_selection_count = raw_multi_count if raw_multi_count > 0 else raw_single_count
        selection_source = "multi_select" if raw_multi_count > 0 else (
            "single_select" if raw_single_count > 0 else "none"
        )

        entity_count = int(batch.entity_mask[0].sum().item())
        selection_count = int(batch.selection_mask[0].sum().item())
        meta_vec = batch.meta_vec[0].detach().cpu().float()
        avail_slice = meta_vec[
            META_AVAILABLE_ACTION_OFFSET : META_AVAILABLE_ACTION_OFFSET
            + META_AVAILABLE_ACTION_DIM
        ]
        action_feedback_token = (
            batch.action_feedback_tokens[0, 0].detach().cpu().float()
        )

        return {
            "event": event,
            "episode": self.episode_index,
            "step": self.step_index,
            "agent_index": agent_idx,
            "raw": {
                "available_action_ids": available_action_ids,
                "available_action_count": len(available_action_ids),
                "last_action_ids": last_action_ids,
                "feature_unit_count": raw_entity_count,
                "feature_unit_truncated": raw_entity_count > MAX_ENTITY_TOKENS,
                "multi_select_count": raw_multi_count,
                "single_select_count": raw_single_count,
                "selection_source": selection_source,
                "selection_count": raw_selection_count,
                "selection_truncated": raw_selection_count > MAX_SELECTION_TOKENS,
                "feature_unit_sample": self._sample_rows(
                    feature_units,
                    field_names=CURATED_FEATURE_UNIT_FIELDS,
                    max_rows=self.max_entity_samples,
                ),
                "selection_sample": self._sample_rows(
                    multi_select if raw_multi_count > 0 else single_select,
                    field_names=SELECTION_FEATURE_NAMES,
                    max_rows=self.max_selection_samples,
                ),
            },
            "batch": {
                "spatial_shape": list(batch.spatial_obs.shape),
                "entity_count": entity_count,
                "entity_mask_utilization": entity_count / MAX_ENTITY_TOKENS,
                "selection_count": selection_count,
                "selection_mask_utilization": selection_count / MAX_SELECTION_TOKENS,
                "meta_dim": int(batch.meta_vec.shape[-1]),
                "meta_last_action_index": int(
                    round(float(meta_vec[META_LAST_ACTION_INDEX_OFFSET].item()))
                ),
                "meta_available_action_mask_active": int(round(float(avail_slice.sum().item()))),
                "action_feedback_token_dim": ACTION_FEEDBACK_TOKEN_DIM,
                "action_feedback_token": action_feedback_token.tolist(),
                "action_feedback_named": {
                    name: float(action_feedback_token[idx].item())
                    for idx, name in enumerate(ACTION_FEEDBACK_FIELD_NAMES)
                },
                "action_feedback_effect_class": self._effect_class(
                    action_feedback_token,
                ),
                "entity_feature_sample": batch.entity_features[
                    0, : min(entity_count, self.max_entity_samples)
                ].detach().cpu().tolist(),
                "selection_feature_sample": batch.selection_features[
                    0, : min(selection_count, self.max_selection_samples)
                ].detach().cpu().tolist(),
            },
            "normalizer": {
                "entity_count_seen": float(self.extractor.entity_normalizer.count),
                "selection_count_seen": float(self.extractor.selection_normalizer.count),
            },
        }

    def _read_field(self, obs, key):
        try:
            if isinstance(obs, Mapping):
                return obs.get(key)
            return getattr(obs, key, None)
        except Exception:
            return None

    def _as_int_list(self, value):
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

    def _safe_len(self, value):
        try:
            return int(len(value))
        except Exception:
            return 0

    def _sample_rows(self, rows, field_names, max_rows):
        count = self._safe_len(rows)
        if count == 0:
            return []

        try:
            arr = np.asarray(rows)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.dtype != object:
                samples = []
                for row in arr[:max_rows]:
                    samples.append(
                        {
                            field_name: _to_serializable(row[idx])
                            for idx, field_name in enumerate(field_names)
                            if idx < row.shape[0]
                        }
                    )
                return samples
        except Exception:
            pass

        samples = []
        for row in list(rows)[:max_rows]:
            samples.append(
                {
                    field_name: _to_serializable(getattr(row, field_name, None))
                    for field_name in field_names
                }
            )
        return samples

    def _effect_class(self, action_feedback_token):
        bridge_type = int(
            round(float(action_feedback_token[ACTION_FEEDBACK_BRIDGE_TYPE_OFFSET].item())),
        )
        moved = (
            float(
                action_feedback_token[
                    ACTION_FEEDBACK_MOVED_TOWARD_TARGET_OFFSET
                ].item(),
            )
            > 0.5
        )
        enemy_drop = float(
            action_feedback_token[ACTION_FEEDBACK_ENEMY_HEALTH_DROP_OFFSET].item(),
        )
        return classify_action_effect(
            is_smart=bridge_type == BRIDGE_ACTION_RIGHT_CLICK,
            moved_toward_target=moved,
            enemy_health_drop_norm=enemy_drop,
        )

    def _append_jsonl(self, record):
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
