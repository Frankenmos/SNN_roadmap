import torch
import pytest

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_COUNT,
    ACTION_FEEDBACK_TOKEN_DIM,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_VECTOR_DIM,
    PolicyInputBatch,
    SELECTION_FEATURE_DIM,
    SPATIAL_OBS_SHAPE,
)


def _make_batch(
    batch_size: int = 3,
    feature_unit_dim: int = 20,
    meta_dim: int = META_VECTOR_DIM,
):
    state_shape = (batch_size, 49, 64)
    return PolicyInputBatch(
        spatial_obs=torch.randn(batch_size, *SPATIAL_OBS_SHAPE),
        entity_features=torch.randn(batch_size, MAX_ENTITY_TOKENS, feature_unit_dim),
        entity_mask=torch.ones(batch_size, MAX_ENTITY_TOKENS, dtype=torch.bool),
        selection_features=torch.randn(
            batch_size,
            MAX_SELECTION_TOKENS,
            SELECTION_FEATURE_DIM,
        ),
        selection_mask=torch.ones(batch_size, MAX_SELECTION_TOKENS, dtype=torch.bool),
        action_feedback_tokens=torch.randn(
            batch_size,
            ACTION_FEEDBACK_TOKEN_COUNT,
            ACTION_FEEDBACK_TOKEN_DIM,
        ),
        meta_vec=torch.randn(batch_size, meta_dim),
        state_in=(torch.randn(*state_shape), torch.randn(*state_shape)),
    )


def test_policy_input_batch_accepts_fix3_protocol_shapes():
    batch = _make_batch()

    assert batch.batch_size == 3
    assert batch.feature_unit_dim == 20
    assert batch.meta_dim == META_VECTOR_DIM
    assert tuple(batch.spatial_obs.shape) == (3, *SPATIAL_OBS_SHAPE)
    assert tuple(batch.entity_features.shape) == (3, MAX_ENTITY_TOKENS, 20)
    assert tuple(batch.selection_features.shape) == (
        3,
        MAX_SELECTION_TOKENS,
        SELECTION_FEATURE_DIM,
    )
    assert tuple(batch.action_feedback_tokens.shape) == (
        3,
        ACTION_FEEDBACK_TOKEN_COUNT,
        ACTION_FEEDBACK_TOKEN_DIM,
    )


def test_policy_input_batch_index_select_slices_all_fields_consistently():
    batch = _make_batch(batch_size=4)

    subset = batch.index_select(torch.tensor([3, 1]))

    assert subset.batch_size == 2
    assert tuple(subset.spatial_obs.shape) == (2, *SPATIAL_OBS_SHAPE)
    assert tuple(subset.entity_mask.shape) == (2, MAX_ENTITY_TOKENS)
    assert tuple(subset.selection_mask.shape) == (2, MAX_SELECTION_TOKENS)
    assert subset.state_in is not None
    assert subset.state_in[0].shape[0] == 2
    assert torch.equal(subset.entity_features[0], batch.entity_features[3])
    assert torch.equal(
        subset.action_feedback_tokens[1],
        batch.action_feedback_tokens[1],
    )
    assert torch.equal(subset.meta_vec[1], batch.meta_vec[1])


def test_policy_input_batch_to_and_detach_preserve_mask_protocol():
    batch = _make_batch(batch_size=2)

    moved = batch.to(device="cpu", dtype=torch.float16)
    detached = moved.detach()

    assert moved.spatial_obs.dtype == torch.float16
    assert moved.entity_features.dtype == torch.float16
    assert moved.selection_features.dtype == torch.float16
    assert moved.action_feedback_tokens.dtype == torch.float16
    assert moved.meta_vec.dtype == torch.float16
    assert moved.entity_mask.dtype == torch.bool
    assert moved.selection_mask.dtype == torch.bool
    assert detached.spatial_obs.requires_grad is False


def test_policy_input_batch_rejects_non_bool_masks():
    with pytest.raises(TypeError, match="entity_mask must use torch.bool"):
        PolicyInputBatch(
            spatial_obs=torch.randn(1, *SPATIAL_OBS_SHAPE),
            entity_features=torch.randn(1, MAX_ENTITY_TOKENS, 20),
            entity_mask=torch.ones(1, MAX_ENTITY_TOKENS, dtype=torch.float32),
            selection_features=torch.randn(
                1,
                MAX_SELECTION_TOKENS,
                SELECTION_FEATURE_DIM,
            ),
            selection_mask=torch.ones(1, MAX_SELECTION_TOKENS, dtype=torch.bool),
            action_feedback_tokens=torch.randn(
                1,
                ACTION_FEEDBACK_TOKEN_COUNT,
                ACTION_FEEDBACK_TOKEN_DIM,
            ),
            meta_vec=torch.randn(1, META_VECTOR_DIM),
        )


def test_policy_input_batch_rejects_rank2_recurrent_state():
    with pytest.raises(
        ValueError,
        match="state_in tensors must be rank-3 legacy or rank-4 multi-timescale",
    ):
        PolicyInputBatch(
            spatial_obs=torch.randn(1, *SPATIAL_OBS_SHAPE),
            entity_features=torch.randn(1, MAX_ENTITY_TOKENS, 20),
            entity_mask=torch.ones(1, MAX_ENTITY_TOKENS, dtype=torch.bool),
            selection_features=torch.randn(
                1,
                MAX_SELECTION_TOKENS,
                SELECTION_FEATURE_DIM,
            ),
            selection_mask=torch.ones(1, MAX_SELECTION_TOKENS, dtype=torch.bool),
            action_feedback_tokens=torch.randn(
                1,
                ACTION_FEEDBACK_TOKEN_COUNT,
                ACTION_FEEDBACK_TOKEN_DIM,
            ),
            meta_vec=torch.randn(1, META_VECTOR_DIM),
            state_in=(
                torch.randn(1, 64),
                torch.randn(1, 64),
            ),
        )
