"""Resume safety for train.load_checkpoint.

The contract under test: a checkpoint that cannot be applied safely must
abort the launch (CheckpointResumeError) BEFORE mutating the agent, and
the checkpoint file must never be renamed or deleted. Fresh-run state is
returned only when no file exists or on an explicit protocol mismatch.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import train as run_mod


class _PolicyParamBeta(nn.Module):
    """Stand-in for the current policy: beta is a trainable parameter."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.beta = nn.Parameter(torch.tensor(0.5))


class _PolicyBufferBeta(nn.Module):
    """Same state_dict keys, but beta became a buffer (one fewer
    optimizer param) — the learn_beta=False migration scenario."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("beta", torch.tensor(0.5))


class _PolicyWideFc(nn.Module):
    """Same state_dict keys as _PolicyParamBeta, different fc shape."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.beta = nn.Parameter(torch.tensor(0.5))


class _FakeExtractor:
    def __init__(self):
        self.state = {}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def _make_agent(policy):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    ppo = SimpleNamespace(optimizer=optimizer, scheduler=None, update_count=0)
    return SimpleNamespace(policy=policy, ppo=ppo, extractor=_FakeExtractor())


def _step_optimizer(agent):
    loss = sum(p.sum() for p in agent.policy.parameters())
    loss.backward()
    agent.ppo.optimizer.step()
    agent.ppo.optimizer.zero_grad()


def _checkpoint_payload(agent, episode=7):
    return {
        "agent_state": agent.policy.state_dict(),
        "optimizer_state": agent.ppo.optimizer.state_dict(),
        "scheduler_state": None,
        "episode": episode,
        "best_eval_reward": 1.5,
        "episode_rewards": [0.5, 1.0],
        "extractor_state": {"count": 3},
        "global_update_index": 11,
        "policy_version": 11,
        "policy_protocol_version": run_mod.POLICY_PROTOCOL_VERSION,
        "policy_input_schema": run_mod.POLICY_INPUT_SCHEMA,
    }


def _save(tmp_path, payload):
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, str(path))
    return path


def _snapshot_weights(agent):
    return {
        key: value.clone()
        for key, value in agent.policy.state_dict().items()
    }


def _assert_untouched(agent, weights_before):
    after = agent.policy.state_dict()
    for key, before in weights_before.items():
        assert torch.equal(after[key], before), f"{key} was mutated"
    assert len(agent.ppo.optimizer.state) == 0
    assert agent.ppo.update_count == 0
    assert agent.extractor.state == {}


def _assert_not_renamed(path):
    assert path.exists()
    assert not os.path.exists(str(path) + ".corrupted")


def test_clean_resume_roundtrip(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    _step_optimizer(saver)
    path = _save(tmp_path, _checkpoint_payload(saver))

    loader = _make_agent(_PolicyParamBeta())
    episode, best_eval_reward, episode_rewards = run_mod.load_checkpoint(
        loader,
        checkpoint_path=str(path),
    )

    assert episode == 7
    assert best_eval_reward == 1.5
    assert list(episode_rewards) == [0.5, 1.0]
    assert loader.ppo.update_count == 11
    assert len(loader.ppo.optimizer.state) > 0
    assert loader.extractor.state == {"count": 3}
    saved = saver.policy.state_dict()
    for key, value in loader.policy.state_dict().items():
        assert torch.equal(value, saved[key])


def test_optimizer_mismatch_fails_clearly_and_atomically(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    _step_optimizer(saver)
    path = _save(tmp_path, _checkpoint_payload(saver))

    loader = _make_agent(_PolicyBufferBeta())
    weights_before = _snapshot_weights(loader)

    with pytest.raises(
        run_mod.CheckpointResumeError,
        match="Resume with weights only",
    ):
        run_mod.load_checkpoint(loader, checkpoint_path=str(path))

    _assert_untouched(loader, weights_before)
    _assert_not_renamed(path)


def test_weights_only_resume_starts_fresh_optimizer(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    _step_optimizer(saver)
    path = _save(tmp_path, _checkpoint_payload(saver))

    loader = _make_agent(_PolicyBufferBeta())
    episode, best_eval_reward, episode_rewards = run_mod.load_checkpoint(
        loader,
        checkpoint_path=str(path),
        weights_only=True,
    )

    assert episode == 7
    assert best_eval_reward == 1.5
    assert list(episode_rewards) == [0.5, 1.0]
    assert loader.ppo.update_count == 11
    assert len(loader.ppo.optimizer.state) == 0
    saved = saver.policy.state_dict()
    for key, value in loader.policy.state_dict().items():
        assert torch.equal(value, saved[key])


def test_weights_only_resume_tolerates_missing_optimizer_state(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    payload = _checkpoint_payload(saver)
    del payload["optimizer_state"]
    path = _save(tmp_path, payload)

    loader = _make_agent(_PolicyParamBeta())
    with pytest.raises(
        run_mod.CheckpointResumeError,
        match="missing required entries",
    ):
        run_mod.load_checkpoint(loader, checkpoint_path=str(path))
    _assert_not_renamed(path)

    episode, _, _ = run_mod.load_checkpoint(
        loader,
        checkpoint_path=str(path),
        weights_only=True,
    )
    assert episode == 7


def test_shape_mismatch_fails_before_any_mutation(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    _step_optimizer(saver)
    path = _save(tmp_path, _checkpoint_payload(saver))

    loader = _make_agent(_PolicyWideFc())
    weights_before = _snapshot_weights(loader)

    with pytest.raises(
        run_mod.CheckpointResumeError,
        match="shapes do not match",
    ):
        run_mod.load_checkpoint(loader, checkpoint_path=str(path))

    _assert_untouched(loader, weights_before)
    _assert_not_renamed(path)


def test_unreadable_checkpoint_raises_and_is_left_in_place(tmp_path):
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"not a checkpoint")

    loader = _make_agent(_PolicyParamBeta())
    weights_before = _snapshot_weights(loader)

    with pytest.raises(
        run_mod.CheckpointResumeError,
        match="could not be deserialized",
    ):
        run_mod.load_checkpoint(loader, checkpoint_path=str(path))

    _assert_untouched(loader, weights_before)
    _assert_not_renamed(path)


def test_protocol_mismatch_still_starts_fresh_without_rename(tmp_path):
    saver = _make_agent(_PolicyParamBeta())
    payload = _checkpoint_payload(saver)
    payload["policy_protocol_version"] = -999
    path = _save(tmp_path, payload)

    loader = _make_agent(_PolicyParamBeta())
    weights_before = _snapshot_weights(loader)

    episode, best_eval_reward, episode_rewards = run_mod.load_checkpoint(
        loader,
        checkpoint_path=str(path),
    )

    assert episode == 0
    assert best_eval_reward == float("-inf")
    assert len(episode_rewards) == 0
    _assert_untouched(loader, weights_before)
    _assert_not_renamed(path)


def test_missing_file_starts_fresh(tmp_path):
    loader = _make_agent(_PolicyParamBeta())
    episode, best_eval_reward, episode_rewards = run_mod.load_checkpoint(
        loader,
        checkpoint_path=str(tmp_path / "does_not_exist.pth"),
    )
    assert episode == 0
    assert best_eval_reward == float("-inf")
    assert len(episode_rewards) == 0
