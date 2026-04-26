import torch

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_COUNT,
    ACTION_FEEDBACK_TOKEN_DIM,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_DIM,
    META_AVAILABLE_ACTION_OFFSET,
    META_LAST_ACTION_INDEX_OFFSET,
    META_VECTOR_DIM,
    PolicyInputBatch,
    SELECTION_FEATURE_DIM,
    SPATIAL_OBS_SHAPE,
    TOTAL_TOKEN_COUNT,
)


def make_dummy_state(batch_size=1, token_count=1, embed_dim=1, fill_value=0.0):
    return (
        torch.full(
            (batch_size, token_count, embed_dim),
            fill_value,
            dtype=torch.float32,
        ),
        torch.full(
            (batch_size, token_count, embed_dim),
            fill_value,
            dtype=torch.float32,
        ),
    )


def make_dummy_state_from_shape(state_shape, fill_value=0.0):
    return (
        torch.full(state_shape, fill_value, dtype=torch.float32),
        torch.full(state_shape, fill_value, dtype=torch.float32),
    )


def make_policy_batch(
    batch_size=1,
    spatial_shape=SPATIAL_OBS_SHAPE,
    meta_dim=META_VECTOR_DIM,
    entity_feature_dim=21,
    with_state=False,
    state_shape=None,
    zeros=False,
):
    tensor_factory = torch.zeros if zeros else torch.randn
    meta_vec = tensor_factory(batch_size, meta_dim)
    if meta_dim >= META_AVAILABLE_ACTION_OFFSET + META_AVAILABLE_ACTION_DIM:
        meta_vec[
            :,
            META_AVAILABLE_ACTION_OFFSET : META_AVAILABLE_ACTION_OFFSET
            + META_AVAILABLE_ACTION_DIM,
        ] = 1.0
    if meta_dim > META_LAST_ACTION_INDEX_OFFSET:
        meta_vec[:, META_LAST_ACTION_INDEX_OFFSET] = 0.0

    state_in = None
    if with_state:
        if state_shape is None:
            state_shape = (batch_size, TOTAL_TOKEN_COUNT, 64)
        fill_value = 0.0 if zeros else 1.0
        if len(state_shape) == 3:
            state_in = make_dummy_state(
                batch_size=state_shape[0],
                token_count=state_shape[1],
                embed_dim=state_shape[2],
                fill_value=fill_value,
            )
        else:
            state_in = make_dummy_state_from_shape(
                state_shape=state_shape,
                fill_value=fill_value,
            )

    return PolicyInputBatch(
        spatial_obs=tensor_factory(batch_size, *spatial_shape),
        entity_features=tensor_factory(
            batch_size, MAX_ENTITY_TOKENS, entity_feature_dim,
        ),
        entity_mask=torch.ones(batch_size, MAX_ENTITY_TOKENS, dtype=torch.bool),
        selection_features=tensor_factory(
            batch_size,
            MAX_SELECTION_TOKENS,
            SELECTION_FEATURE_DIM,
        ),
        selection_mask=torch.ones(batch_size, MAX_SELECTION_TOKENS, dtype=torch.bool),
        action_feedback_tokens=tensor_factory(
            batch_size,
            ACTION_FEEDBACK_TOKEN_COUNT,
            ACTION_FEEDBACK_TOKEN_DIM,
        ),
        meta_vec=meta_vec,
        state_in=state_in,
    )
