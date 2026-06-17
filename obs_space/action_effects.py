from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from pysc2.lib import features

from agent_core.policy_protocol import (
    AGENT_ACTION_TOKEN_DIM,
    BRIDGE_ACTION_NO_OP,
    BRIDGE_ACTION_RIGHT_CLICK,
)
from obs_space._numeric import coerce_numeric_rows, safe_float


FRIENDLY_ALLIANCE: Final[int] = 1
ENEMY_ALLIANCE: Final[int] = 4
NEAR_ENEMY_RADIUS: Final[float] = 6.0
MOVEMENT_EPSILON: Final[float] = 0.25
HEALTH_NORM: Final[float] = 100.0

_FEATURE_UNIT_ALLIANCE_INDEX: Final[int] = int(features.FeatureUnit.alliance)
_FEATURE_UNIT_HEALTH_INDEX: Final[int] = int(features.FeatureUnit.health)
_FEATURE_UNIT_X_INDEX: Final[int] = int(features.FeatureUnit.x)
_FEATURE_UNIT_Y_INDEX: Final[int] = int(features.FeatureUnit.y)
_FEATURE_UNIT_TAG_INDEX: Final[int] = int(features.FeatureUnit.tag)


@dataclass(slots=True, frozen=True)
class UnitSnapshot:
    alliance: int
    health: float
    x: float
    y: float
    tag: int | None = None


@dataclass(slots=True, frozen=True)
class FrameSnapshot:
    friendlies: tuple[UnitSnapshot, ...]
    enemies: tuple[UnitSnapshot, ...]

    @property
    def friendly_health(self) -> float:
        return float(sum(unit.health for unit in self.friendlies))

    @property
    def enemy_health(self) -> float:
        return float(sum(unit.health for unit in self.enemies))


def zero_action_effect(effect_class: str = "not_smart") -> dict[str, float | str]:
    return {
        "target_near_enemy": 0.0,
        "friendly_moved_toward_target": 0.0,
        "enemy_health_drop_norm": 0.0,
        "friendly_health_drop_norm": 0.0,
        "effect_class": effect_class,
    }


def classify_action_effect(
    *,
    is_smart: bool,
    moved_toward_target: bool,
    enemy_health_drop_norm: float,
) -> str:
    if not is_smart:
        return "not_smart"
    damaged = float(enemy_health_drop_norm) > 0.0
    if moved_toward_target and damaged:
        return "move_and_damage"
    if moved_toward_target:
        return "move_like"
    if damaged:
        return "damage_like"
    return "null_or_unclear"


class ActionEffectTracker:
    def __init__(
        self,
        *,
        near_enemy_radius: float = NEAR_ENEMY_RADIUS,
        movement_epsilon: float = MOVEMENT_EPSILON,
        health_norm: float = HEALTH_NORM,
    ):
        self.near_enemy_radius = float(near_enemy_radius)
        self.movement_epsilon = float(movement_epsilon)
        self.health_norm = float(health_norm)
        self._previous_frame: FrameSnapshot | None = None

    def reset(self) -> None:
        self._previous_frame = None

    def update(self, obs) -> None:
        self._previous_frame = extract_frame_snapshot(obs)

    def compute(self, obs, last_action_token=None) -> dict[str, float | str]:
        current_frame = extract_frame_snapshot(obs)
        previous_frame = self._previous_frame
        action_token = normalize_action_token(last_action_token)
        is_smart = int(round(float(action_token[0]))) == BRIDGE_ACTION_RIGHT_CLICK

        if previous_frame is None:
            return zero_action_effect(
                "null_or_unclear" if is_smart else "not_smart",
            )

        enemy_drop = self._health_drop_norm(
            previous_frame.enemy_health,
            current_frame.enemy_health,
        )
        friendly_drop = self._health_drop_norm(
            previous_frame.friendly_health,
            current_frame.friendly_health,
        )

        target_near_enemy = 0.0
        moved_toward_target = 0.0
        if is_smart:
            target = (float(action_token[1]), float(action_token[2]))
            target_near_enemy = float(
                self._target_near_enemy(previous_frame.enemies, target),
            )
            moved_toward_target = float(
                self._friendly_moved_toward_target(
                    previous_frame.friendlies,
                    current_frame.friendlies,
                    target,
                ),
            )

        return {
            "target_near_enemy": target_near_enemy,
            "friendly_moved_toward_target": moved_toward_target,
            "enemy_health_drop_norm": enemy_drop,
            "friendly_health_drop_norm": friendly_drop,
            "effect_class": classify_action_effect(
                is_smart=is_smart,
                moved_toward_target=bool(moved_toward_target > 0.5),
                enemy_health_drop_norm=enemy_drop,
            ),
        }

    def compute_and_maybe_update(
        self,
        obs,
        last_action_token=None,
        *,
        update_state: bool,
    ) -> dict[str, float | str]:
        effect = self.compute(obs, last_action_token=last_action_token)
        if update_state:
            self.update(obs)
        return effect

    def _health_drop_norm(self, previous_health: float, current_health: float) -> float:
        norm = max(float(self.health_norm), 1.0e-6)
        drop = float(previous_health) - float(current_health)
        return float(np.clip(drop, 0.0, norm) / norm)

    def _target_near_enemy(
        self,
        enemies: tuple[UnitSnapshot, ...],
        target: tuple[float, float],
    ) -> bool:
        if not enemies:
            return False
        target_arr = np.asarray(target, dtype=np.float32)
        enemy_positions = np.asarray(
            [(unit.x, unit.y) for unit in enemies],
            dtype=np.float32,
        )
        distances = np.linalg.norm(enemy_positions - target_arr.reshape(1, 2), axis=1)
        return bool(np.any(distances <= self.near_enemy_radius))

    def _friendly_moved_toward_target(
        self,
        previous_friendlies: tuple[UnitSnapshot, ...],
        current_friendlies: tuple[UnitSnapshot, ...],
        target: tuple[float, float],
    ) -> bool:
        target_arr = np.asarray(target, dtype=np.float32)
        prev_by_tag = {
            unit.tag: unit
            for unit in previous_friendlies
            if unit.tag is not None and int(unit.tag) > 0
        }
        curr_by_tag = {
            unit.tag: unit
            for unit in current_friendlies
            if unit.tag is not None and int(unit.tag) > 0
        }
        shared_tags = sorted(set(prev_by_tag).intersection(curr_by_tag))
        if shared_tags:
            prev_positions = np.asarray(
                [(prev_by_tag[tag].x, prev_by_tag[tag].y) for tag in shared_tags],
                dtype=np.float32,
            )
            curr_positions = np.asarray(
                [(curr_by_tag[tag].x, curr_by_tag[tag].y) for tag in shared_tags],
                dtype=np.float32,
            )
            improvement = _mean_distance_improvement(
                prev_positions,
                curr_positions,
                target_arr,
            )
            return bool(improvement > self.movement_epsilon)

        if len(previous_friendlies) != len(current_friendlies):
            return False
        if not previous_friendlies:
            return False

        prev_median = _median_position(previous_friendlies)
        curr_median = _median_position(current_friendlies)
        improvement = _mean_distance_improvement(
            prev_median.reshape(1, 2),
            curr_median.reshape(1, 2),
            target_arr,
        )
        return bool(improvement > self.movement_epsilon)


def normalize_action_token(last_action_token=None) -> np.ndarray:
    if last_action_token is None:
        token = np.asarray([BRIDGE_ACTION_NO_OP, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        token = np.asarray(last_action_token, dtype=np.float32).reshape(-1)
    if token.size < AGENT_ACTION_TOKEN_DIM:
        token = np.pad(token, (0, AGENT_ACTION_TOKEN_DIM - token.size))
    return token[:AGENT_ACTION_TOKEN_DIM].astype(np.float32, copy=False)


def extract_frame_snapshot(obs) -> FrameSnapshot:
    observation = getattr(obs, "observation", obs)
    feature_units = getattr(observation, "feature_units", None)
    units = tuple(_iter_unit_snapshots(feature_units))
    friendlies = tuple(unit for unit in units if unit.alliance == FRIENDLY_ALLIANCE)
    enemies = tuple(unit for unit in units if unit.alliance == ENEMY_ALLIANCE)
    return FrameSnapshot(friendlies=friendlies, enemies=enemies)


def _iter_unit_snapshots(feature_units):
    if feature_units is None:
        return
    numeric_rows = coerce_numeric_rows(feature_units)
    if numeric_rows is not None:
        for row in numeric_rows:
            yield _unit_snapshot_from_numeric_row(row)
        return
    for unit in list(feature_units):
        snapshot = _unit_snapshot_from_object(unit)
        if snapshot is not None:
            yield snapshot


def _unit_snapshot_from_numeric_row(row) -> UnitSnapshot:
    row = np.asarray(row).reshape(-1)
    tag = None
    if row.size > _FEATURE_UNIT_TAG_INDEX:
        raw_tag = int(safe_float(row[_FEATURE_UNIT_TAG_INDEX]))
        tag = raw_tag if raw_tag > 0 else None
    return UnitSnapshot(
        alliance=int(safe_float(row[_FEATURE_UNIT_ALLIANCE_INDEX]))
        if row.size > _FEATURE_UNIT_ALLIANCE_INDEX
        else 0,
        health=float(safe_float(row[_FEATURE_UNIT_HEALTH_INDEX]))
        if row.size > _FEATURE_UNIT_HEALTH_INDEX
        else 0.0,
        x=float(safe_float(row[_FEATURE_UNIT_X_INDEX]))
        if row.size > _FEATURE_UNIT_X_INDEX
        else 0.0,
        y=float(safe_float(row[_FEATURE_UNIT_Y_INDEX]))
        if row.size > _FEATURE_UNIT_Y_INDEX
        else 0.0,
        tag=tag,
    )


def _unit_snapshot_from_object(unit) -> UnitSnapshot | None:
    if unit is None:
        return None
    raw_tag = getattr(unit, "tag", None)
    tag = None
    if raw_tag is not None:
        try:
            parsed_tag = int(raw_tag)
            tag = parsed_tag if parsed_tag > 0 else None
        except (TypeError, ValueError):
            tag = None
    return UnitSnapshot(
        alliance=int(safe_float(getattr(unit, "alliance", 0.0))),
        health=float(safe_float(getattr(unit, "health", 0.0))),
        x=float(safe_float(getattr(unit, "x", 0.0))),
        y=float(safe_float(getattr(unit, "y", 0.0))),
        tag=tag,
    )


def _median_position(units: tuple[UnitSnapshot, ...]) -> np.ndarray:
    positions = np.asarray([(unit.x, unit.y) for unit in units], dtype=np.float32)
    return np.median(positions, axis=0).astype(np.float32, copy=False)


def _mean_distance_improvement(
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    target: np.ndarray,
) -> float:
    previous_distances = np.linalg.norm(previous_positions - target.reshape(1, 2), axis=1)
    current_distances = np.linalg.norm(current_positions - target.reshape(1, 2), axis=1)
    return float(np.mean(previous_distances - current_distances))
