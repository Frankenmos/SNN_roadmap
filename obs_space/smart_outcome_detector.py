"""Diagnostics-first classification for PySC2 ``Smart_screen`` outcomes.

The detector deliberately stays outside the policy input path. It attributes a
dispatched Smart click to a short-window outcome so eval traces can answer the
question "did this click look like an attack, movement, or noise?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from pysc2.lib import features

from agent_core.policy_protocol import BRIDGE_ACTION_RIGHT_CLICK
from obs_space._numeric import coerce_numeric_rows, safe_float
from obs_space.action_effects import (
    FRIENDLY_ALLIANCE,
    FrameSnapshot,
    UnitSnapshot,
    extract_frame_snapshot,
    normalize_action_token,
)


# Derive feature_unit column indices from the live PySC2 enum rather than
# hardcoding them — the literals drift across PySC2 versions (weapon_cooldown
# in particular was wrongly pinned to 8, which is shield_ratio, silently
# killing the fired_likely outcome on the numeric path). Mirrors the dynamic
# lookup obs_space_2 already uses.
_FEATURE_UNIT_ALLIANCE: Final[int] = int(features.FeatureUnit.alliance)
_FEATURE_UNIT_HEALTH: Final[int] = int(features.FeatureUnit.health)
_FEATURE_UNIT_WEAPON_COOLDOWN: Final[int] = int(features.FeatureUnit.weapon_cooldown)
_FEATURE_UNIT_X: Final[int] = int(features.FeatureUnit.x)
_FEATURE_UNIT_Y: Final[int] = int(features.FeatureUnit.y)
_FEATURE_UNIT_TAG: Final[int] = int(features.FeatureUnit.tag)


@dataclass(slots=True)
class CooldownSnapshot:
    """Weapon-cooldown state for one friendly unit."""

    unit_tag: int
    cooldown: float
    health: float
    x: float
    y: float


@dataclass(slots=True)
class PendingSmartClick:
    """State captured from the pre-action frame for one Smart click."""

    click_step: int
    target_x: float
    target_y: float
    enemy_health_snapshot: float
    enemy_snapshots: tuple[UnitSnapshot, ...]
    friendly_cooldowns: tuple[CooldownSnapshot, ...]
    friendly_positions: tuple[tuple[int, float, float], ...]

    @property
    def target(self) -> tuple[float, float]:
        return (self.target_x, self.target_y)


@dataclass(slots=True)
class SmartOutcome:
    """Classification result for one resolved Smart click."""

    outcome_class: str
    click_step: int
    resolution_step: int
    window_steps: int
    target_x: float
    target_y: float
    enemy_health_delta: float
    nearest_enemy_distance: float
    any_cooldown_fired: bool
    friendly_moved_toward: bool
    target_was_near_enemy: bool
    resolution_reason: str

    def to_dict(self) -> dict[str, bool | float | int | str]:
        return {
            "outcome_class": self.outcome_class,
            "click_step": self.click_step,
            "resolution_step": self.resolution_step,
            "window_steps": self.window_steps,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "enemy_health_delta": self.enemy_health_delta,
            "nearest_enemy_distance": self.nearest_enemy_distance,
            "any_cooldown_fired": self.any_cooldown_fired,
            "friendly_moved_toward": self.friendly_moved_toward,
            "target_was_near_enemy": self.target_was_near_enemy,
            "resolution_reason": self.resolution_reason,
        }


class SmartOutcomeDetector:
    """Classify Smart click outcomes from explicit previous/current frames."""

    def __init__(
        self,
        *,
        outcome_window: int = 5,
        near_enemy_threshold: float = 8.0,
        cooldown_ready_threshold: float = 2.0,
        cooldown_fire_delta: float = 3.0,
        health_drop_threshold: float = 0.1,
        movement_epsilon: float = 0.25,
    ):
        self.outcome_window = max(1, int(outcome_window))
        self.near_enemy_threshold = float(near_enemy_threshold)
        self.cooldown_ready_threshold = float(cooldown_ready_threshold)
        self.cooldown_fire_delta = float(cooldown_fire_delta)
        self.health_drop_threshold = float(health_drop_threshold)
        self.movement_epsilon = float(movement_epsilon)
        self._pending_clicks: list[PendingSmartClick] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending_clicks)

    def reset(self) -> None:
        self._pending_clicks.clear()

    def on_step(self, obs, step: int, last_action_token=None) -> list[SmartOutcome]:
        """
        Backward-compatible helper for old unit code.

        New integration code should call ``observe_smart_click`` with the
        pre-action frame, then ``resolve`` with the post-action frame.
        """
        action_token = normalize_action_token(last_action_token)
        is_smart = int(round(float(action_token[0]))) == BRIDGE_ACTION_RIGHT_CLICK
        current_frame = extract_frame_snapshot(obs)
        if is_smart:
            self.observe_smart_click(
                previous_frame=current_frame,
                target=(float(action_token[1]), float(action_token[2])),
                click_step=step,
                previous_feature_units=_read_feature_units(obs),
            )
        return self.resolve(
            current_frame=current_frame,
            resolution_step=step,
            current_feature_units=_read_feature_units(obs),
        )

    def observe_smart_click(
        self,
        *,
        previous_frame: FrameSnapshot,
        target: tuple[float, float],
        click_step: int,
        previous_feature_units=None,
    ) -> PendingSmartClick:
        pending = PendingSmartClick(
            click_step=int(click_step),
            target_x=float(target[0]),
            target_y=float(target[1]),
            enemy_health_snapshot=float(previous_frame.enemy_health),
            enemy_snapshots=tuple(previous_frame.enemies),
            friendly_cooldowns=extract_cooldown_snapshots_from_feature_units(
                previous_feature_units,
            ),
            friendly_positions=_friendly_positions(previous_frame.friendlies),
        )
        if not pending.friendly_cooldowns:
            pending.friendly_cooldowns = _cooldown_snapshots_from_frame(
                previous_frame.friendlies,
            )
        self._pending_clicks.append(pending)
        return pending

    def process_transition(
        self,
        *,
        previous_frame: FrameSnapshot,
        current_frame: FrameSnapshot,
        click_step: int,
        resolution_step: int,
        smart_target: tuple[float, float] | None = None,
        previous_feature_units=None,
        current_feature_units=None,
    ) -> list[SmartOutcome]:
        if smart_target is not None:
            self.observe_smart_click(
                previous_frame=previous_frame,
                target=smart_target,
                click_step=click_step,
                previous_feature_units=previous_feature_units,
            )
        return self.resolve(
            current_frame=current_frame,
            resolution_step=resolution_step,
            current_feature_units=current_feature_units,
        )

    def resolve(
        self,
        *,
        current_frame: FrameSnapshot,
        resolution_step: int,
        current_feature_units=None,
    ) -> list[SmartOutcome]:
        outcomes: list[SmartOutcome] = []
        current_cooldowns = extract_cooldown_snapshots_from_feature_units(
            current_feature_units,
        )

        attack_idx = self._select_attack_attribution(current_frame)
        consumed_damage = attack_idx is not None

        remaining: list[PendingSmartClick] = []
        for idx, pending in enumerate(self._pending_clicks):
            age = int(resolution_step) - pending.click_step
            if idx == attack_idx:
                outcomes.append(
                    self._classify_outcome(
                        pending=pending,
                        current_frame=current_frame,
                        resolution_step=resolution_step,
                        current_cooldowns=current_cooldowns,
                        resolution_reason="enemy_health_drop",
                    ),
                )
                continue

            if consumed_damage:
                pending.enemy_health_snapshot = float(current_frame.enemy_health)
                if age >= self.outcome_window:
                    outcomes.append(
                        self._classify_outcome(
                            pending=pending,
                            current_frame=current_frame,
                            resolution_step=resolution_step,
                            current_cooldowns=(),
                            resolution_reason="window_expired",
                        ),
                    )
                else:
                    remaining.append(pending)
                continue

            if self._should_resolve_fired(pending, current_cooldowns):
                outcomes.append(
                    self._classify_outcome(
                        pending=pending,
                        current_frame=current_frame,
                        resolution_step=resolution_step,
                        current_cooldowns=current_cooldowns,
                        resolution_reason="cooldown_fired",
                    ),
                )
            elif age >= self.outcome_window:
                outcomes.append(
                    self._classify_outcome(
                        pending=pending,
                        current_frame=current_frame,
                        resolution_step=resolution_step,
                        current_cooldowns=current_cooldowns,
                        resolution_reason="window_expired",
                    ),
                )
            else:
                remaining.append(pending)

        self._pending_clicks = remaining
        return outcomes

    def _select_attack_attribution(self, current_frame: FrameSnapshot) -> int | None:
        candidates = []
        for idx, pending in enumerate(self._pending_clicks):
            delta = pending.enemy_health_snapshot - current_frame.enemy_health
            if delta <= self.health_drop_threshold:
                continue
            nearest = nearest_enemy_distance(pending.enemy_snapshots, pending.target)
            candidates.append(
                (
                    0 if nearest <= self.near_enemy_threshold else 1,
                    nearest,
                    -pending.click_step,
                    idx,
                ),
            )
        if not candidates:
            return None
        return min(candidates)[3]

    def _should_resolve_fired(
        self,
        pending: PendingSmartClick,
        current_cooldowns: tuple[CooldownSnapshot, ...],
    ) -> bool:
        if nearest_enemy_distance(pending.enemy_snapshots, pending.target) > self.near_enemy_threshold:
            return False
        return check_cooldowns_fired(
            pending.friendly_cooldowns,
            current_cooldowns,
            ready_threshold=self.cooldown_ready_threshold,
            fire_delta_threshold=self.cooldown_fire_delta,
        )

    def _classify_outcome(
        self,
        *,
        pending: PendingSmartClick,
        current_frame: FrameSnapshot,
        resolution_step: int,
        current_cooldowns: tuple[CooldownSnapshot, ...],
        resolution_reason: str,
    ) -> SmartOutcome:
        enemy_health_delta = max(
            0.0,
            float(pending.enemy_health_snapshot) - float(current_frame.enemy_health),
        )
        nearest_distance = nearest_enemy_distance(pending.enemy_snapshots, pending.target)
        target_was_near_enemy = nearest_distance <= self.near_enemy_threshold
        any_cooldown_fired = check_cooldowns_fired(
            pending.friendly_cooldowns,
            current_cooldowns,
            ready_threshold=self.cooldown_ready_threshold,
            fire_delta_threshold=self.cooldown_fire_delta,
        )
        friendly_moved_toward = self._friendly_moved_toward_target(
            pending.friendly_positions,
            current_frame,
            pending.target,
        )

        outcome_class = "null_unclear"
        if enemy_health_delta > self.health_drop_threshold:
            outcome_class = "attack_likely"
        elif target_was_near_enemy and any_cooldown_fired:
            outcome_class = "fired_likely"
        elif target_was_near_enemy:
            outcome_class = "attack_intent"
        elif friendly_moved_toward:
            outcome_class = "move_like"

        return SmartOutcome(
            outcome_class=outcome_class,
            click_step=pending.click_step,
            resolution_step=int(resolution_step),
            window_steps=int(resolution_step) - pending.click_step,
            target_x=pending.target_x,
            target_y=pending.target_y,
            enemy_health_delta=enemy_health_delta,
            nearest_enemy_distance=nearest_distance,
            any_cooldown_fired=any_cooldown_fired,
            friendly_moved_toward=friendly_moved_toward,
            target_was_near_enemy=target_was_near_enemy,
            resolution_reason=resolution_reason,
        )

    def _nearest_enemy_distance(
        self,
        enemies: tuple[UnitSnapshot, ...],
        target: tuple[float, float],
    ) -> float:
        return nearest_enemy_distance(enemies, target)

    def _friendly_moved_toward_target(
        self,
        previous_positions: tuple[tuple[int, float, float], ...],
        current_frame: FrameSnapshot,
        target: tuple[float, float],
    ) -> bool:
        if not previous_positions:
            return False

        current_by_tag = {
            int(unit.tag): (float(unit.x), float(unit.y))
            for unit in current_frame.friendlies
            if unit.tag is not None and int(unit.tag) > 0
        }
        target_arr = np.asarray(target, dtype=np.float32)
        improvements = []
        for tag, prev_x, prev_y in previous_positions:
            if tag not in current_by_tag:
                continue
            curr_x, curr_y = current_by_tag[tag]
            prev_pos = np.asarray([prev_x, prev_y], dtype=np.float32)
            curr_pos = np.asarray([curr_x, curr_y], dtype=np.float32)
            improvements.append(
                float(np.linalg.norm(prev_pos - target_arr))
                - float(np.linalg.norm(curr_pos - target_arr)),
            )
        return bool(improvements and float(np.mean(improvements)) > self.movement_epsilon)


def nearest_enemy_distance(
    enemies: tuple[UnitSnapshot, ...],
    target: tuple[float, float],
) -> float:
    if not enemies:
        return float("inf")
    target_arr = np.asarray(target, dtype=np.float32)
    enemy_positions = np.asarray([(unit.x, unit.y) for unit in enemies], dtype=np.float32)
    distances = np.linalg.norm(enemy_positions - target_arr.reshape(1, 2), axis=1)
    return float(np.min(distances))


def extract_cooldown_snapshots_from_feature_units(
    feature_units,
    friendly_alliance: int = FRIENDLY_ALLIANCE,
) -> tuple[CooldownSnapshot, ...]:
    """Extract friendly weapon cooldown snapshots from raw or object feature units."""
    if feature_units is None:
        return ()

    numeric_rows = coerce_numeric_rows(feature_units)
    if numeric_rows is not None:
        snapshots = []
        for row in numeric_rows:
            row = np.asarray(row).reshape(-1)
            if row.size <= _FEATURE_UNIT_TAG:
                continue
            alliance = int(safe_float(row[_FEATURE_UNIT_ALLIANCE]))
            tag = int(safe_float(row[_FEATURE_UNIT_TAG]))
            if alliance != friendly_alliance or tag <= 0:
                continue
            snapshots.append(
                CooldownSnapshot(
                    unit_tag=tag,
                    cooldown=float(safe_float(row[_FEATURE_UNIT_WEAPON_COOLDOWN])),
                    health=float(safe_float(row[_FEATURE_UNIT_HEALTH])),
                    x=float(safe_float(row[_FEATURE_UNIT_X])),
                    y=float(safe_float(row[_FEATURE_UNIT_Y])),
                ),
            )
        return tuple(snapshots)

    snapshots = []
    for unit in list(feature_units):
        alliance = int(safe_float(getattr(unit, "alliance", 0)))
        tag = int(safe_float(getattr(unit, "tag", 0)))
        if alliance != friendly_alliance or tag <= 0:
            continue
        snapshots.append(
            CooldownSnapshot(
                unit_tag=tag,
                cooldown=float(safe_float(getattr(unit, "weapon_cooldown", 0.0))),
                health=float(safe_float(getattr(unit, "health", 0.0))),
                x=float(safe_float(getattr(unit, "x", 0.0))),
                y=float(safe_float(getattr(unit, "y", 0.0))),
            ),
        )
    return tuple(snapshots)


def check_cooldowns_fired(
    previous_cooldowns: tuple[CooldownSnapshot, ...],
    current_cooldowns: tuple[CooldownSnapshot, ...],
    *,
    ready_threshold: float = 2.0,
    fire_delta_threshold: float = 3.0,
) -> bool:
    if not previous_cooldowns or not current_cooldowns:
        return False
    current_by_tag = {snapshot.unit_tag: snapshot for snapshot in current_cooldowns}
    for previous in previous_cooldowns:
        current = current_by_tag.get(previous.unit_tag)
        if current is None:
            continue
        if (
            previous.cooldown <= ready_threshold
            and current.cooldown >= previous.cooldown + fire_delta_threshold
        ):
            return True
    return False


def check_cooldowns_fired_production(
    previous_cooldowns: tuple[CooldownSnapshot, ...],
    current_feature_units,
) -> bool:
    """Compatibility wrapper for older callers/tests."""
    return check_cooldowns_fired(
        previous_cooldowns,
        extract_cooldown_snapshots_from_feature_units(current_feature_units),
    )


def _friendly_positions(
    friendlies: tuple[UnitSnapshot, ...],
) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (int(unit.tag), float(unit.x), float(unit.y))
        for unit in friendlies
        if unit.tag is not None and int(unit.tag) > 0
    )


def _cooldown_snapshots_from_frame(
    friendlies: tuple[UnitSnapshot, ...],
) -> tuple[CooldownSnapshot, ...]:
    return tuple(
        CooldownSnapshot(
            unit_tag=int(unit.tag),
            cooldown=0.0,
            health=float(unit.health),
            x=float(unit.x),
            y=float(unit.y),
        )
        for unit in friendlies
        if unit.tag is not None and int(unit.tag) > 0
    )


def _read_feature_units(obs):
    observation = getattr(obs, "observation", obs)
    return getattr(observation, "feature_units", None)

