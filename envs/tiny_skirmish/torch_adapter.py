from __future__ import annotations

import numpy as np
import torch

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_COUNT,
    ACTION_FEEDBACK_TOKEN_DIM,
    CURATED_FEATURE_UNIT_FIELDS,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_VECTOR_DIM,
    PolicyInputBatch,
    SELECTION_FEATURE_DIM,
    SPATIAL_OBS_SHAPE,
)

from .protocol import ObservationBatch


def observation_to_policy_input(
    observation: ObservationBatch,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PolicyInputBatch:
    """Convert a single TinySkirmish numpy observation to a Torch policy batch."""

    observation.validate()
    batch = PolicyInputBatch(
        spatial_obs=_float_tensor(observation.spatial_obs, device=device, dtype=dtype).unsqueeze(0),
        entity_features=_float_tensor(
            observation.entity_features,
            device=device,
            dtype=dtype,
        ).unsqueeze(0),
        entity_mask=_bool_tensor(observation.entity_mask, device=device).unsqueeze(0),
        selection_features=_float_tensor(
            observation.selection_features,
            device=device,
            dtype=dtype,
        ).unsqueeze(0),
        selection_mask=_bool_tensor(observation.selection_mask, device=device).unsqueeze(0),
        action_feedback_tokens=_float_tensor(
            observation.action_feedback_tokens,
            device=device,
            dtype=dtype,
        ).unsqueeze(0),
        meta_vec=_float_tensor(observation.meta_vec, device=device, dtype=dtype).unsqueeze(0),
    )
    batch_shape_assertions(batch)
    return batch


def batch_shape_assertions(batch: PolicyInputBatch) -> None:
    """Small, explicit integration sanity check against the SNN policy protocol."""

    if tuple(batch.spatial_obs.shape[1:]) != SPATIAL_OBS_SHAPE:
        raise ValueError(f"bad spatial shape: {tuple(batch.spatial_obs.shape)}")
    if tuple(batch.entity_features.shape[1:]) != (
        MAX_ENTITY_TOKENS,
        len(CURATED_FEATURE_UNIT_FIELDS),
    ):
        raise ValueError(f"bad entity shape: {tuple(batch.entity_features.shape)}")
    if tuple(batch.selection_features.shape[1:]) != (
        MAX_SELECTION_TOKENS,
        SELECTION_FEATURE_DIM,
    ):
        raise ValueError(f"bad selection shape: {tuple(batch.selection_features.shape)}")
    if tuple(batch.action_feedback_tokens.shape[1:]) != (
        ACTION_FEEDBACK_TOKEN_COUNT,
        ACTION_FEEDBACK_TOKEN_DIM,
    ):
        raise ValueError(f"bad feedback shape: {tuple(batch.action_feedback_tokens.shape)}")
    if tuple(batch.meta_vec.shape[1:]) != (META_VECTOR_DIM,):
        raise ValueError(f"bad meta shape: {tuple(batch.meta_vec.shape)}")


def _float_tensor(
    array: np.ndarray,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _bool_tensor(array: np.ndarray, *, device: torch.device | str | None) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.bool, device=device)
