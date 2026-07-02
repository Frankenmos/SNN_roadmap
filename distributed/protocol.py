from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch

from agent_core.policy_protocol import (
    POLICY_INPUT_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    PolicyInputBatch,
    SNNState,
)


def validate_policy_protocol(
    *,
    policy_protocol_version: int,
    policy_input_schema: str,
) -> None:
    if int(policy_protocol_version) != int(POLICY_PROTOCOL_VERSION):
        raise ValueError(
            "policy protocol mismatch: "
            f"{policy_protocol_version!r} != {POLICY_PROTOCOL_VERSION!r}",
        )
    if str(policy_input_schema) != str(POLICY_INPUT_SCHEMA):
        raise ValueError(
            "policy input schema mismatch: "
            f"{policy_input_schema!r} != {POLICY_INPUT_SCHEMA!r}",
        )


@dataclass(slots=True, frozen=True)
class EpisodeSummary:
    actor_id: int
    episode_index: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    policy_version: int | None = None


@dataclass(slots=True, frozen=True)
class TransitionRecord:
    observation_batch: PolicyInputBatch
    action: torch.Tensor
    move_x: torch.Tensor
    move_y: torch.Tensor
    target_index: torch.Tensor
    coarse_index: torch.Tensor
    fine_index: torch.Tensor
    old_log_prob: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    episode_reset: torch.Tensor
    sample_mask: torch.Tensor


@dataclass(slots=True, frozen=True)
class WeightSnapshot:
    policy_version: int
    state_dict: Mapping[str, torch.Tensor]
    policy_protocol_version: int = POLICY_PROTOCOL_VERSION
    policy_input_schema: str = POLICY_INPUT_SCHEMA

    def __post_init__(self) -> None:
        validate_policy_protocol(
            policy_protocol_version=self.policy_protocol_version,
            policy_input_schema=self.policy_input_schema,
        )


@dataclass(slots=True, frozen=True)
class UpdateSummary:
    global_update_index: int
    policy_version: int
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RolloutFragment:
    actor_id: int
    fragment_id: int
    policy_version: int
    spatial_obs: torch.Tensor
    entity_features: torch.Tensor
    entity_mask: torch.Tensor
    selection_features: torch.Tensor
    selection_mask: torch.Tensor
    action_feedback_tokens: torch.Tensor
    meta_vec: torch.Tensor
    actions: torch.Tensor
    move_x: torch.Tensor
    move_y: torch.Tensor
    target_index: torch.Tensor
    coarse_index: torch.Tensor
    fine_index: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    truncateds: torch.Tensor
    episode_reset_mask: torch.Tensor
    sample_mask: torch.Tensor
    pre_step_snn_state: SNNState | None
    tail_next_policy_input: PolicyInputBatch | None
    tail_next_snn_state: SNNState | None = None
    episode_summaries: tuple[EpisodeSummary, ...] = ()
    reward_component_summaries: tuple[Mapping[str, Any], ...] = ()
    step_counters: Mapping[str, int] = field(default_factory=dict)
    policy_protocol_version: int = POLICY_PROTOCOL_VERSION
    policy_input_schema: str = POLICY_INPUT_SCHEMA

    def __post_init__(self) -> None:
        validate_policy_protocol(
            policy_protocol_version=self.policy_protocol_version,
            policy_input_schema=self.policy_input_schema,
        )
        self._validate()

    @property
    def num_steps(self) -> int:
        return int(self.spatial_obs.shape[0])

    @property
    def num_learnable_steps(self) -> int:
        return int((self.sample_mask > 0.0).sum().item())

    @property
    def terminated(self) -> bool:
        return bool((self.dones > 0.5).any().item())

    @property
    def truncated(self) -> bool:
        return bool((self.truncateds > 0.5).any().item())

    def as_policy_input_batch(self, state_in: SNNState | None = None) -> PolicyInputBatch:
        return PolicyInputBatch(
            spatial_obs=self.spatial_obs,
            entity_features=self.entity_features,
            entity_mask=self.entity_mask,
            selection_features=self.selection_features,
            selection_mask=self.selection_mask,
            action_feedback_tokens=self.action_feedback_tokens,
            meta_vec=self.meta_vec,
            state_in=state_in,
        )

    def immutable_step_counters(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.step_counters))

    def _validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("RolloutFragment must contain at least one step")
        expected = self.num_steps
        per_step_tensors = {
            "entity_features": self.entity_features,
            "entity_mask": self.entity_mask,
            "selection_features": self.selection_features,
            "selection_mask": self.selection_mask,
            "action_feedback_tokens": self.action_feedback_tokens,
            "meta_vec": self.meta_vec,
            "actions": self.actions,
            "move_x": self.move_x,
            "move_y": self.move_y,
            "target_index": self.target_index,
            "coarse_index": self.coarse_index,
            "fine_index": self.fine_index,
            "old_log_probs": self.old_log_probs,
            "values": self.values,
            "rewards": self.rewards,
            "dones": self.dones,
            "truncateds": self.truncateds,
            "episode_reset_mask": self.episode_reset_mask,
            "sample_mask": self.sample_mask,
        }
        for name, tensor in per_step_tensors.items():
            self._require_tensor(name, tensor)
            if int(tensor.shape[0]) != expected:
                raise ValueError(
                    f"{name} first dimension must match spatial_obs: "
                    f"{int(tensor.shape[0])} != {expected}",
                )

        self.as_policy_input_batch(state_in=None)

        if self.pre_step_snn_state is not None:
            self._validate_state("pre_step_snn_state", self.pre_step_snn_state, expected)
        if self.tail_next_snn_state is not None:
            self._validate_state("tail_next_snn_state", self.tail_next_snn_state, 1)
        if (
            self.tail_next_policy_input is not None
            and self.tail_next_policy_input.batch_size != 1
        ):
            raise ValueError(
                "tail_next_policy_input must have batch_size=1, "
                f"got {self.tail_next_policy_input.batch_size}",
            )

    @staticmethod
    def _require_tensor(name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 1:
            raise ValueError(f"{name} must have a leading step dimension")

    @staticmethod
    def _validate_state(
        name: str,
        state: SNNState,
        expected_batch: int,
    ) -> None:
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError(f"{name} must be a (syn, mem) tensor tuple")
        syn, mem = state
        if not isinstance(syn, torch.Tensor) or not isinstance(mem, torch.Tensor):
            raise TypeError(f"{name} must contain tensors")
        if syn.shape != mem.shape:
            raise ValueError(
                f"{name} tensors must share shape, got {syn.shape} and {mem.shape}",
            )
        if syn.ndim not in (3, 4):
            raise ValueError(
                f"{name} tensors must be rank-3 or rank-4, got ndim={syn.ndim}",
            )
        if int(syn.shape[0]) != int(expected_batch):
            raise ValueError(
                f"{name} batch dimension must be {expected_batch}, "
                f"got {int(syn.shape[0])}",
            )
