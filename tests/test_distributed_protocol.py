import pytest
import torch

from MockedEnv.policy_batch import make_dummy_state_from_shape, make_policy_batch
from agent_core.policy_protocol import (
    POLICY_INPUT_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    TOTAL_TOKEN_COUNT,
)
from distributed.protocol import RolloutFragment, validate_policy_protocol


def _make_fragment(
    length=3,
    sample_mask=None,
    dones=None,
    truncateds=None,
    rewards=None,
):
    batch = make_policy_batch(
        batch_size=length,
        with_state=False,
        zeros=True,
    )
    state = make_dummy_state_from_shape(
        (length, 2, TOTAL_TOKEN_COUNT, 64),
        fill_value=0.0,
    )
    tail = make_policy_batch(batch_size=1, with_state=True, zeros=True)
    if sample_mask is None:
        sample_mask = torch.ones(length, dtype=torch.float32)
    if dones is None:
        dones = torch.zeros(length, dtype=torch.float32)
    if truncateds is None:
        truncateds = torch.zeros(length, dtype=torch.float32)
    if rewards is None:
        rewards = torch.ones(length, dtype=torch.float32)

    return RolloutFragment(
        actor_id=2,
        fragment_id=5,
        policy_version=7,
        spatial_obs=batch.spatial_obs,
        entity_features=batch.entity_features,
        entity_mask=batch.entity_mask,
        selection_features=batch.selection_features,
        selection_mask=batch.selection_mask,
        action_feedback_tokens=batch.action_feedback_tokens,
        meta_vec=batch.meta_vec,
        actions=torch.zeros(length, dtype=torch.long),
        move_x=torch.zeros(length, dtype=torch.long),
        move_y=torch.zeros(length, dtype=torch.long),
        target_index=torch.full((length,), -1, dtype=torch.long),
        coarse_index=torch.full((length,), -1, dtype=torch.long),
        fine_index=torch.full((length,), -1, dtype=torch.long),
        old_log_probs=torch.zeros(length, dtype=torch.float32),
        values=torch.zeros(length, dtype=torch.float32),
        rewards=rewards,
        dones=dones,
        truncateds=truncateds,
        episode_reset_mask=truncateds > 0.5,
        sample_mask=sample_mask,
        pre_step_snn_state=state,
        tail_next_policy_input=tail,
        tail_next_snn_state=tail.state_in,
    )


def test_validate_policy_protocol_accepts_current_schema():
    validate_policy_protocol(
        policy_protocol_version=POLICY_PROTOCOL_VERSION,
        policy_input_schema=POLICY_INPUT_SCHEMA,
    )


def test_validate_policy_protocol_rejects_mismatches():
    with pytest.raises(ValueError, match="policy protocol mismatch"):
        validate_policy_protocol(
            policy_protocol_version=POLICY_PROTOCOL_VERSION + 1,
            policy_input_schema=POLICY_INPUT_SCHEMA,
        )
    with pytest.raises(ValueError, match="policy input schema mismatch"):
        validate_policy_protocol(
            policy_protocol_version=POLICY_PROTOCOL_VERSION,
            policy_input_schema="old_schema",
        )


def test_rollout_fragment_preserves_current_policy_input_contract():
    fragment = _make_fragment(length=3)

    assert fragment.num_steps == 3
    assert fragment.num_learnable_steps == 3
    assert not fragment.terminated
    assert not fragment.truncated

    batch = fragment.as_policy_input_batch()
    assert batch.spatial_obs.shape == (3, 27, 84, 84)
    assert batch.action_feedback_tokens.shape == (3, 1, 9)
    assert batch.meta_vec.shape == (3, 15)


def test_rollout_fragment_keeps_truncation_separate_from_done():
    fragment = _make_fragment(
        length=2,
        dones=torch.tensor([0.0, 0.0]),
        truncateds=torch.tensor([0.0, 1.0]),
    )

    assert not fragment.terminated
    assert fragment.truncated
    assert fragment.episode_reset_mask.tolist() == [False, True]


def test_rollout_fragment_rejects_length_mismatch():
    with pytest.raises(ValueError, match="rewards first dimension"):
        _make_fragment(length=3, rewards=torch.ones(2))
