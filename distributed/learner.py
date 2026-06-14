from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
from distributed.protocol import RolloutFragment, validate_policy_protocol
from Utility.config import cfg

if TYPE_CHECKING:
    from agent import DefeatRoaches


def resolve_torch_device(device_name: str | torch.device | None) -> torch.device:
    requested = torch.device(device_name or "cpu")
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def move_agent_to_device(
    agent: "DefeatRoaches",
    device_name: str | torch.device | None,
) -> torch.device:
    device = resolve_torch_device(device_name)
    agent.policy.to(device)
    agent.policy.device = device
    agent.ppo.device = device
    agent.policy.configure_amp(getattr(cfg.model, "amp_dtype", "auto"))
    _move_optimizer_state(agent.ppo.optimizer, device)
    agent.extractor.device = device
    agent.snn_state = agent.policy.init_concrete_state(batch_size=1, device=device)
    return device


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu").clone()
        for key, value in module.state_dict().items()
    }


def validate_fragment_batch(
    fragments: Iterable[RolloutFragment],
    *,
    expected_policy_version: int,
) -> list[RolloutFragment]:
    checked = list(fragments)
    if not checked:
        raise ValueError("Cannot update learner from an empty fragment batch")

    for fragment in checked:
        validate_policy_protocol(
            policy_protocol_version=fragment.policy_protocol_version,
            policy_input_schema=fragment.policy_input_schema,
        )
        if int(fragment.policy_version) != int(expected_policy_version):
            raise ValueError(
                "stale rollout fragment policy_version: "
                f"{fragment.policy_version} != {expected_policy_version}",
            )
    return checked


class LearnerCoordinator:
    """Owns the trainable policy and consumes synchronous rollout fragments."""

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        from agent import DefeatRoaches

        self.agent = DefeatRoaches()
        if device is None:
            device = getattr(getattr(cfg, "distributed", {}), "learner_device", "cuda")
        self.device = move_agent_to_device(self.agent, device)

    @property
    def policy_version(self) -> int:
        return int(getattr(self.agent.ppo, "update_count", 0))

    def make_weight_payload(
        self,
        *,
        include_extractor_state: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "policy_version": int(self.policy_version),
            "state_dict": cpu_state_dict(self.agent.policy),
            "policy_protocol_version": POLICY_PROTOCOL_VERSION,
            "policy_input_schema": POLICY_INPUT_SCHEMA,
        }
        if include_extractor_state:
            payload["extractor_state"] = self.agent.extractor.state_dict()
        return payload

    def update_from_fragments(
        self,
        fragments: Iterable[RolloutFragment],
    ) -> Mapping[str, Any]:
        validation_started = time.perf_counter()
        checked = validate_fragment_batch(
            fragments,
            expected_policy_version=self.policy_version,
        )
        validation_wall_seconds = time.perf_counter() - validation_started
        update_started = time.perf_counter()
        stats = self.agent.update_policy(fragments=checked)
        learner_wall_seconds = time.perf_counter() - update_started
        summary = dict(stats or {})
        summary.setdefault("global_update_index", int(self.policy_version))
        summary["policy_version"] = int(self.policy_version)
        summary.setdefault("fragments_in_update", int(len(checked)))
        summary["fragment_validation_wall_seconds"] = float(validation_wall_seconds)
        summary["learner_update_from_fragments_wall_seconds"] = float(
            learner_wall_seconds,
        )
        return summary
