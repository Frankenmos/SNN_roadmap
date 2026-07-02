from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from agent_core.policy_protocol import ActionSample, PolicyInputBatch, SNNState, TOTAL_TOKEN_COUNT

from .env import TinySkirmishEnv
from .protocol import ACTION_LEFT_CLICK, ACTION_NO_OP, ACTION_RIGHT_CLICK, RewardInfo, SkirmishAction
from .rollout import scripted_action
from .torch_adapter import observation_to_policy_input


class PPOStyleActor(Protocol):
    def select_action(
        self,
        observations: PolicyInputBatch,
        deterministic: bool = False,
    ) -> ActionSample | SkirmishAction | tuple[Any, ...]:
        ...


@dataclass(slots=True)
class TinyTransition:
    observation_batch: PolicyInputBatch
    action: torch.Tensor
    move_x: torch.Tensor
    move_y: torch.Tensor
    old_log_prob: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    episode_reset: torch.Tensor
    sample_mask: torch.Tensor
    reward_info: RewardInfo
    info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TinyEpisode:
    transitions: list[TinyTransition]
    total_reward: float
    terminated: bool
    truncated: bool
    events: dict[str, int]


class ScriptedActor:
    """Policy-like actor for smoke testing the Torch harness."""

    def select_action_for_env(self, env: TinySkirmishEnv) -> ActionSample:
        action = scripted_action(env)
        return _sample_from_action(action)


class RandomActor:
    """Policy-like random actor for smoke testing the Torch harness."""

    def select_action_for_env(self, env: TinySkirmishEnv) -> ActionSample:
        return _sample_from_action(env.random_action())


class CyclingActor:
    """Deterministic dummy actor that exercises all semantic action IDs."""

    def __init__(self) -> None:
        self._idx = 0

    def select_action(self, observations: PolicyInputBatch, deterministic: bool = False) -> ActionSample:
        del deterministic
        targets = [(0, 0), (20, 20), (70, 42)]
        action_ids = [ACTION_NO_OP, ACTION_LEFT_CLICK, ACTION_RIGHT_CLICK]
        action_id = action_ids[self._idx % len(action_ids)]
        x, y = targets[self._idx % len(targets)]
        self._idx += 1
        return ActionSample(
            action_id=action_id,
            x=x,
            y=y,
            target_index=None,
            coarse_index=None,
            fine_index=None,
            log_prob=0.0,
            value=0.0,
            next_state=_zero_state_like(observations.state_in),
        )


def collect_episode(
    *,
    env: TinySkirmishEnv | None = None,
    actor: PPOStyleActor | ScriptedActor | RandomActor | None = None,
    mode: str = "scripted",
    seed: int = 0,
    max_steps: int = 80,
    device: torch.device | str | None = None,
    deterministic: bool = False,
    initial_state: SNNState | None = None,
) -> TinyEpisode:
    """Collect a Torch-backed episode using a PPO-style actor seam."""

    env = env or TinySkirmishEnv(seed=seed, max_steps=max_steps)
    observation = env.reset(seed=seed)
    state_in = initial_state
    transitions: list[TinyTransition] = []
    total_reward = 0.0
    events: dict[str, int] = {}
    terminated = False
    was_truncated = False

    for _step in range(max_steps):
        batch = observation_to_policy_input(observation, device=device).with_state(state_in)
        sample = _select_action(actor=actor, mode=mode, env=env, batch=batch, deterministic=deterministic)
        action = SkirmishAction(sample.action_id, sample.x, sample.y)
        result = env.step(action)

        transition = TinyTransition(
            observation_batch=batch,
            action=_scalar(sample.action_id, dtype=torch.long),
            move_x=_scalar(sample.x, dtype=torch.long),
            move_y=_scalar(sample.y, dtype=torch.long),
            old_log_prob=_scalar(sample.log_prob),
            reward=_scalar(result.reward.total),
            value=_scalar(sample.value),
            done=_scalar(float(result.done)),
            truncated=_scalar(float(result.truncated)),
            episode_reset=_scalar(float(result.done or result.truncated)),
            sample_mask=_scalar(1.0),
            reward_info=result.reward,
            info=result.info,
        )
        transitions.append(transition)
        total_reward += result.reward.total
        for event in result.reward.events:
            events[event] = events.get(event, 0) + 1

        state_in = sample.next_state
        observation = result.observation
        if result.done or result.truncated:
            terminated = result.done
            was_truncated = result.truncated
            break

    return TinyEpisode(
        transitions=transitions,
        total_reward=float(total_reward),
        terminated=terminated,
        truncated=was_truncated,
        events=events,
    )


def push_transition_to_ppo(ppo: Any, transition: TinyTransition) -> None:
    """Feed one TinySkirmish transition to an SNN-style PPO object."""

    if transition.observation_batch.state_in is None:
        raise ValueError("PPO.store_transition requires pre-step recurrent state")
    ppo.store_transition(
        transition.observation_batch,
        transition.action,
        transition.move_x,
        transition.move_y,
        transition.old_log_prob,
        transition.reward,
        transition.value,
        transition.done,
        sample_mask=transition.sample_mask,
        truncated=transition.truncated,
        episode_reset=transition.episode_reset,
    )


def make_zero_state(
    *,
    batch_size: int = 1,
    token_count: int = TOTAL_TOKEN_COUNT,
    embed_dim: int = 64,
    device: torch.device | str | None = None,
) -> SNNState:
    syn = torch.zeros((batch_size, token_count, embed_dim), dtype=torch.float32, device=device)
    mem = torch.zeros((batch_size, token_count, embed_dim), dtype=torch.float32, device=device)
    return syn, mem


def _select_action(
    *,
    actor: PPOStyleActor | ScriptedActor | RandomActor | None,
    mode: str,
    env: TinySkirmishEnv,
    batch: PolicyInputBatch,
    deterministic: bool,
) -> ActionSample:
    if actor is not None and hasattr(actor, "select_action_for_env"):
        return _coerce_sample(actor.select_action_for_env(env))
    if actor is not None:
        return _coerce_sample(actor.select_action(batch, deterministic=deterministic))
    if mode == "scripted":
        return _sample_from_action(scripted_action(env))
    if mode == "random":
        return _sample_from_action(env.random_action())
    raise ValueError(f"unknown mode: {mode}")


def _coerce_sample(value: ActionSample | SkirmishAction | tuple[Any, ...]) -> ActionSample:
    if isinstance(value, ActionSample):
        return value
    if isinstance(value, SkirmishAction):
        return _sample_from_action(value)
    if isinstance(value, tuple) and len(value) == 6:
        action_id, x, y, log_prob, value_estimate, next_state = value
        return ActionSample(
            action_id=int(action_id),
            x=int(x),
            y=int(y),
            target_index=None,
            coarse_index=None,
            fine_index=None,
            log_prob=float(log_prob),
            value=float(value_estimate),
            next_state=next_state,
        )
    if isinstance(value, tuple) and len(value) == 3:
        action_id, x, y = value
        return ActionSample(
            action_id=int(action_id),
            x=int(x),
            y=int(y),
            target_index=None,
            coarse_index=None,
            fine_index=None,
            log_prob=0.0,
            value=0.0,
            next_state=None,
        )
    raise TypeError(f"unsupported actor return value: {type(value)!r}")


def _sample_from_action(action: SkirmishAction) -> ActionSample:
    return ActionSample(
        action_id=int(action.action_id),
        x=int(action.x),
        y=int(action.y),
        target_index=None,
        coarse_index=None,
        fine_index=None,
        log_prob=0.0,
        value=0.0,
        next_state=None,
    )


def _zero_state_like(state: SNNState | None) -> SNNState | None:
    if state is None:
        return None
    syn, mem = state
    return torch.zeros_like(syn), torch.zeros_like(mem)


def _scalar(value: float | int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype)
