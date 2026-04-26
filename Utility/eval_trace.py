from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from pysc2.lib import actions as sc2_actions

from agent_core.policy_protocol import PolicyInputBatch


TRACE_FORMAT_VERSION = 1


def _to_cpu_tensor(
    tensor: torch.Tensor,
    *,
    float_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    out = tensor.detach().to(device="cpu")
    if out.is_floating_point():
        out = out.to(dtype=float_dtype)
    return out.contiguous()


def serialize_policy_input_batch(
    batch: PolicyInputBatch,
    *,
    float_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    if not isinstance(batch, PolicyInputBatch):
        raise TypeError(
            f"serialize_policy_input_batch expects PolicyInputBatch, got {type(batch)!r}",
        )
    if int(batch.batch_size) != 1:
        raise ValueError(
            f"Eval traces currently expect batch_size=1, got {batch.batch_size}",
        )

    return {
        "spatial_obs": _to_cpu_tensor(batch.spatial_obs[0], float_dtype=float_dtype),
        "entity_features": _to_cpu_tensor(
            batch.entity_features[0],
            float_dtype=float_dtype,
        ),
        "entity_mask": batch.entity_mask[0].detach().to(device="cpu").contiguous(),
        "selection_features": _to_cpu_tensor(
            batch.selection_features[0],
            float_dtype=float_dtype,
        ),
        "selection_mask": batch.selection_mask[0].detach().to(
            device="cpu",
        ).contiguous(),
        "action_feedback_tokens": _to_cpu_tensor(
            batch.action_feedback_tokens[0],
            float_dtype=float_dtype,
        ),
        "meta_vec": _to_cpu_tensor(batch.meta_vec[0], float_dtype=float_dtype),
    }


def _normalize_argument(value: Any):
    if isinstance(value, (list, tuple)):
        return [_normalize_argument(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    return str(value)


def describe_function_call(action_func: Any) -> dict[str, Any]:
    if action_func is None:
        return {
            "function_id": None,
            "function_name": None,
            "arguments": None,
        }

    function_id = getattr(action_func, "function", None)
    if function_id is None:
        function_id = getattr(action_func, "id", None)
    try:
        function_id = None if function_id is None else int(function_id)
    except (TypeError, ValueError):
        function_id = None

    function_name = None
    if function_id is not None:
        try:
            function_name = sc2_actions.FUNCTIONS[function_id].name
        except Exception:
            function_name = None
    if function_name is None:
        raw_name = getattr(action_func, "name", None)
        function_name = str(raw_name) if raw_name is not None else None

    arguments = getattr(action_func, "arguments", None)
    if arguments is None:
        arguments = getattr(action_func, "args", None)
    if arguments is not None:
        arguments = _normalize_argument(arguments)

    return {
        "function_id": function_id,
        "function_name": function_name,
        "arguments": arguments,
    }


@dataclass(slots=True)
class EpisodeTraceRecorder:
    output_dir: str | Path
    run_name: str | None = None
    checkpoint_path: str | None = None
    checkpoint_episode: int | str | None = None
    deterministic: bool = True
    float_dtype: torch.dtype = torch.float16
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_step(
        self,
        *,
        step_index: int,
        action_func: Any,
        action: int | None,
        move_x: int,
        move_y: int,
        log_prob: float,
        value: float,
        reward: float,
        cumulative_reward: float,
        done: bool,
        learnable: bool,
        policy_input: PolicyInputBatch | None,
    ) -> None:
        record = {
            "step_index": int(step_index),
            "policy_step": bool(policy_input is not None),
            "learnable": bool(learnable),
            "action": None if action is None else int(action),
            "move_x": int(move_x),
            "move_y": int(move_y),
            "log_prob": float(log_prob),
            "value": float(value),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "done": bool(done),
            "dispatched_action": describe_function_call(action_func),
            "policy_input": (
                None
                if policy_input is None
                else serialize_policy_input_batch(
                    policy_input,
                    float_dtype=self.float_dtype,
                )
            ),
        }
        self.records.append(record)

    def save(
        self,
        *,
        episode_index: int,
        total_reward: float,
        steps: int,
    ) -> Path:
        output_root = Path(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        mode_tag = "det" if self.deterministic else "stoch"
        path = output_root / f"episode_{int(episode_index):04d}_{mode_tag}.pt"
        payload = {
            "format_version": TRACE_FORMAT_VERSION,
            "run_name": self.run_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_episode": self.checkpoint_episode,
            "deterministic": bool(self.deterministic),
            "episode_index": int(episode_index),
            "total_reward": float(total_reward),
            "steps": int(steps),
            "float_dtype": str(self.float_dtype),
            "records": self.records,
        }
        torch.save(payload, path)
        return path
