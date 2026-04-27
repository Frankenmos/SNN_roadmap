from collections import deque

import pytest
import torch

from MockedEnv.policy_batch import make_dummy_state_from_shape, make_policy_batch
from agent_core.policy_protocol import TOTAL_TOKEN_COUNT
from distributed.learner import cpu_state_dict, validate_fragment_batch
from distributed.protocol import EpisodeSummary, RolloutFragment
from distributed.ray_train import _log_episode_summaries


def _make_fragment(length=2, policy_version=3):
    batch = make_policy_batch(batch_size=length, with_state=False, zeros=True)
    state = make_dummy_state_from_shape(
        (length, 2, TOTAL_TOKEN_COUNT, 64),
        fill_value=0.0,
    )
    tail = make_policy_batch(batch_size=1, with_state=True, zeros=True)
    return RolloutFragment(
        actor_id=1,
        fragment_id=0,
        policy_version=policy_version,
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
        rewards=torch.ones(length, dtype=torch.float32),
        dones=torch.zeros(length, dtype=torch.float32),
        truncateds=torch.zeros(length, dtype=torch.float32),
        episode_reset_mask=torch.zeros(length, dtype=torch.bool),
        sample_mask=torch.ones(length, dtype=torch.float32),
        pre_step_snn_state=state,
        tail_next_policy_input=tail,
        tail_next_snn_state=tail.state_in,
    )


class DummyQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_validate_fragment_batch_accepts_matching_policy_version():
    fragment = _make_fragment(policy_version=3)

    checked = validate_fragment_batch([fragment], expected_policy_version=3)

    assert checked == [fragment]


def test_validate_fragment_batch_rejects_empty_or_stale_batch():
    with pytest.raises(ValueError, match="empty fragment batch"):
        validate_fragment_batch([], expected_policy_version=3)

    with pytest.raises(ValueError, match="stale rollout fragment"):
        validate_fragment_batch(
            [_make_fragment(policy_version=2)],
            expected_policy_version=3,
        )


def test_cpu_state_dict_returns_detached_cpu_clones():
    module = torch.nn.Linear(2, 1)
    payload = cpu_state_dict(module)

    assert set(payload) == set(module.state_dict())
    for key, value in payload.items():
        source = module.state_dict()[key]
        assert value.device.type == "cpu"
        assert not value.requires_grad
        assert value.data_ptr() != source.data_ptr()


def test_log_episode_summaries_uses_stable_actor_episode_key():
    queue = DummyQueue()
    rewards = deque(maxlen=10)
    fragment = _make_fragment(policy_version=7)
    summary = EpisodeSummary(
        actor_id=2,
        episode_index=3,
        total_reward=11.0,
        steps=45,
        terminated=False,
        truncated=True,
        policy_version=7,
    )
    object.__setattr__(fragment, "episode_summaries", (summary,))

    count = _log_episode_summaries(
        log_queue=queue,
        fragments=[fragment],
        episode_rewards=rewards,
    )

    assert count == 1
    assert [item["type"] for item in queue.items] == ["EPISODE_START", "EPISODE_END"]
    assert queue.items[0]["internal_ep"] == 2_000_000_003
    assert queue.items[0]["actor_id"] == 2
    assert queue.items[0]["policy_version"] == 7
    assert queue.items[1]["total"] == 11.0
    assert queue.items[1]["avg"] == 11.0
