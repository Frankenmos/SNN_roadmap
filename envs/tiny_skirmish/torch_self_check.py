from __future__ import annotations

import torch

from agent_core.policy_protocol import PolicyInputBatch, TOTAL_TOKEN_COUNT

from .env import TinySkirmishEnv
from .torch_adapter import observation_to_policy_input
from .torch_harness import (
    CyclingActor,
    ScriptedActor,
    collect_episode,
    make_zero_state,
    push_transition_to_ppo,
)


class _RecorderPPO:
    def __init__(self) -> None:
        self.calls = []

    def store_transition(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def _assert_adapter_shapes() -> None:
    env = TinySkirmishEnv(seed=5)
    obs = env.reset()
    batch = observation_to_policy_input(obs)
    assert isinstance(batch, PolicyInputBatch)
    assert batch.spatial_obs.shape == (1, 27, 84, 84)
    assert batch.entity_features.shape == (1, 24, 21)
    assert batch.selection_features.shape == (1, 20, 7)
    assert batch.action_feedback_tokens.shape == (1, 1, 12)
    assert batch.meta_vec.shape == (1, 15)


def _assert_policy_batch_ops() -> None:
    env = TinySkirmishEnv(seed=6)
    batch = observation_to_policy_input(env.reset())
    state = make_zero_state(embed_dim=8)
    with_state = batch.with_state(state)
    stacked = PolicyInputBatch.stack([with_state, with_state])
    assert stacked.batch_size == 2
    assert stacked.state_in is not None
    assert stacked.state_in[0].shape == (2, TOTAL_TOKEN_COUNT, 8)
    selected = stacked.index_select([1])
    assert selected.batch_size == 1
    detached = selected.detach()
    assert detached.spatial_obs.requires_grad is False
    moved = detached.to(dtype=torch.float32)
    assert moved.spatial_obs.dtype is torch.float32


def _assert_harness_scripted_episode() -> None:
    episode = collect_episode(actor=ScriptedActor(), seed=9, max_steps=40)
    assert episode.terminated
    assert not episode.truncated
    assert episode.events.get("win") == 1
    assert episode.events.get("damage_dealt", 0) > 0
    assert len(episode.transitions) > 0
    for transition in episode.transitions:
        transition.observation_batch._validate()
        transition.reward_info.validate()


def _assert_harness_can_push_to_ppo_style_store() -> None:
    episode = collect_episode(
        actor=CyclingActor(),
        seed=1,
        max_steps=3,
        initial_state=make_zero_state(embed_dim=4),
    )
    recorder = _RecorderPPO()
    push_transition_to_ppo(recorder, episode.transitions[0])
    assert len(recorder.calls) == 1
    args, kwargs = recorder.calls[0]
    assert isinstance(args[0], PolicyInputBatch)
    assert kwargs["sample_mask"].item() == 1.0


def main() -> int:
    checks = [
        _assert_adapter_shapes,
        _assert_policy_batch_ops,
        _assert_harness_scripted_episode,
        _assert_harness_can_push_to_ppo_style_store,
    ]
    for check in checks:
        check()
        print(f"ok {check.__name__}")
    print("TinySkirmish Torch integration self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
