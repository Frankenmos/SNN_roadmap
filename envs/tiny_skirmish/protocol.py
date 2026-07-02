from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np


SPATIAL_CHANNELS: Final[int] = 27
SCREEN_SIZE: Final[int] = 84
SPATIAL_OBS_SHAPE: Final[tuple[int, int, int]] = (
    SPATIAL_CHANNELS,
    SCREEN_SIZE,
    SCREEN_SIZE,
)
MAX_ENTITY_TOKENS: Final[int] = 24
ENTITY_FEATURE_DIM: Final[int] = 21
MAX_SELECTION_TOKENS: Final[int] = 20
SELECTION_FEATURE_DIM: Final[int] = 7
ACTION_FEEDBACK_TOKEN_COUNT: Final[int] = 1
ACTION_FEEDBACK_TOKEN_DIM: Final[int] = 12
META_VECTOR_DIM: Final[int] = 15

ACTION_NO_OP: Final[int] = 0
ACTION_LEFT_CLICK: Final[int] = 1
ACTION_RIGHT_CLICK: Final[int] = 2
ACTION_NAMES: Final[dict[int, str]] = {
    ACTION_NO_OP: "NO_OP",
    ACTION_LEFT_CLICK: "LEFT_CLICK",
    ACTION_RIGHT_CLICK: "RIGHT_CLICK",
}

BRIDGE_NO_OP: Final[int] = 0
BRIDGE_LEFT_CLICK: Final[int] = 1
BRIDGE_RIGHT_CLICK: Final[int] = 2

ENTITY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "unit_type",
    "alliance",
    "health",
    "health_ratio",
    "shield",
    "shield_ratio",
    "energy",
    "energy_ratio",
    "weapon_cooldown",
    "x",
    "y",
    "radius",
    "build_progress",
    "order_id_0",
    "order_id_1",
    "is_selected",
    "is_in_cargo",
    "assigned_harvesters",
    "ideal_harvesters",
    "active",
    "hallucination",
)

SELECTION_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "unit_type",
    "player_relative",
    "health",
    "shields",
    "energy",
    "transport_slots_taken",
    "build_progress",
)

ACTION_FEEDBACK_FIELD_NAMES: Final[tuple[str, ...]] = (
    "bridge_type",
    "x_norm",
    "y_norm",
    "executed_smart",
    "any_executed",
    "score_delta_norm",
    "kill_delta_norm",
    "score_penalty_bit",
    "target_near_enemy",
    "friendly_moved_toward_target",
    "enemy_health_drop_norm",
    "friendly_health_drop_norm",
)


@dataclass(frozen=True, slots=True)
class SkirmishAction:
    """Semantic action plus optional screen-space target."""

    action_id: int
    x: int = 0
    y: int = 0

    @classmethod
    def no_op(cls) -> "SkirmishAction":
        return cls(ACTION_NO_OP, 0, 0)

    @classmethod
    def left_click(cls, x: int, y: int) -> "SkirmishAction":
        return cls(ACTION_LEFT_CLICK, int(x), int(y))

    @classmethod
    def right_click(cls, x: int, y: int) -> "SkirmishAction":
        return cls(ACTION_RIGHT_CLICK, int(x), int(y))

    @property
    def name(self) -> str:
        return ACTION_NAMES.get(int(self.action_id), f"UNKNOWN_{self.action_id}")


@dataclass(slots=True)
class ObservationBatch:
    """Numpy mirror of the SNN repo's policy input envelope."""

    spatial_obs: np.ndarray
    entity_features: np.ndarray
    entity_mask: np.ndarray
    selection_features: np.ndarray
    selection_mask: np.ndarray
    action_feedback_tokens: np.ndarray
    meta_vec: np.ndarray

    def validate(self) -> None:
        expected = {
            "spatial_obs": (SPATIAL_CHANNELS, SCREEN_SIZE, SCREEN_SIZE),
            "entity_features": (MAX_ENTITY_TOKENS, ENTITY_FEATURE_DIM),
            "entity_mask": (MAX_ENTITY_TOKENS,),
            "selection_features": (MAX_SELECTION_TOKENS, SELECTION_FEATURE_DIM),
            "selection_mask": (MAX_SELECTION_TOKENS,),
            "action_feedback_tokens": (
                ACTION_FEEDBACK_TOKEN_COUNT,
                ACTION_FEEDBACK_TOKEN_DIM,
            ),
            "meta_vec": (META_VECTOR_DIM,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape must be {shape}, got {value.shape}")


@dataclass(slots=True)
class RewardInfo:
    total: float
    parts: dict[str, float] = field(default_factory=dict)
    events: tuple[str, ...] = ()

    def validate(self, tolerance: float = 1.0e-6) -> None:
        summed = float(sum(self.parts.values()))
        if abs(float(self.total) - summed) > tolerance:
            raise ValueError(
                f"reward total {self.total:.6f} does not match parts {summed:.6f}",
            )
        if abs(float(self.total)) > tolerance and not self.parts:
            raise ValueError("nonzero reward must include named reward parts")

    def compact_parts(self) -> dict[str, float]:
        return {
            name: round(float(value), 6)
            for name, value in self.parts.items()
            if abs(float(value)) > 1.0e-9
        }


@dataclass(slots=True)
class StepResult:
    observation: ObservationBatch
    reward: RewardInfo
    done: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)
