from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .protocol import (
    ACTION_FEEDBACK_TOKEN_DIM,
    ACTION_LEFT_CLICK,
    ACTION_NO_OP,
    ACTION_RIGHT_CLICK,
    BRIDGE_LEFT_CLICK,
    BRIDGE_NO_OP,
    BRIDGE_RIGHT_CLICK,
    ENTITY_FEATURE_DIM,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_VECTOR_DIM,
    SCREEN_SIZE,
    SELECTION_FEATURE_DIM,
    SPATIAL_CHANNELS,
    ObservationBatch,
    RewardInfo,
    SkirmishAction,
    StepResult,
)


FRIENDLY_ALLIANCE = 1.0
ENEMY_ALLIANCE = 4.0
FRIENDLY_UNIT_TYPE = 1.0
ENEMY_UNIT_TYPE = 2.0


@dataclass(slots=True)
class Entity:
    entity_id: int
    kind: str
    x: int
    y: int
    health: int
    max_health: int
    attack_damage: int
    attack_range: int = 1
    selected: bool = False

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def alliance(self) -> float:
        return FRIENDLY_ALLIANCE if self.kind == "friendly" else ENEMY_ALLIANCE

    @property
    def unit_type(self) -> float:
        return FRIENDLY_UNIT_TYPE if self.kind == "friendly" else ENEMY_UNIT_TYPE


@dataclass(slots=True)
class TinySkirmishState:
    step_count: int
    entities: list[Entity]
    walls: set[tuple[int, int]]
    last_action: SkirmishAction
    last_feedback: np.ndarray
    last_target_grid: tuple[int, int] | None = None


class TinySkirmishEnv:
    """Small deterministic skirmish task with StarCraft-shaped observations."""

    def __init__(
        self,
        *,
        grid_size: int = 12,
        screen_size: int = SCREEN_SIZE,
        max_steps: int = 80,
        seed: int | None = None,
    ) -> None:
        if screen_size != SCREEN_SIZE:
            raise ValueError("MVP protocol currently expects screen_size=84")
        if SCREEN_SIZE % grid_size != 0:
            raise ValueError("grid_size must divide 84 for clean text-to-screen cells")
        self.grid_size = int(grid_size)
        self.screen_size = int(screen_size)
        self.max_steps = int(max_steps)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.state: TinySkirmishState | None = None
        self._damage_dealt_total = 0.0
        self._damage_taken_total = 0.0
        self._kill_total = 0.0

    def reset(self, seed: int | None = None) -> ObservationBatch:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        elif self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

        enemies = [(9, 4), (9, 7), (8, 9), (10, 6)]
        self.rng.shuffle(enemies)
        enemy_a, enemy_b = enemies[:2]

        walls = {
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 7),
            (5, 8),
            (5, 9),
            (6, 5),
            (6, 6),
        }
        entities = [
            Entity(1, "friendly", 2, 6, 14, 14, attack_damage=3, selected=True),
            Entity(101, "enemy", enemy_a[0], enemy_a[1], 8, 8, attack_damage=2),
            Entity(102, "enemy", enemy_b[0], enemy_b[1], 8, 8, attack_damage=2),
        ]
        self.state = TinySkirmishState(
            step_count=0,
            entities=entities,
            walls=walls,
            last_action=SkirmishAction.no_op(),
            last_feedback=np.zeros((1, ACTION_FEEDBACK_TOKEN_DIM), dtype=np.float32),
        )
        self._damage_dealt_total = 0.0
        self._damage_taken_total = 0.0
        self._kill_total = 0.0
        return self.observe()

    def observe(self) -> ObservationBatch:
        state = self._require_state()
        entity_features, entity_mask = self._build_entity_features(state)
        selection_features, selection_mask = self._build_selection_features(state)
        observation = ObservationBatch(
            spatial_obs=self._build_spatial_obs(state),
            entity_features=entity_features,
            entity_mask=entity_mask,
            selection_features=selection_features,
            selection_mask=selection_mask,
            action_feedback_tokens=state.last_feedback.copy(),
            meta_vec=self._build_meta_vec(state),
        )
        observation.validate()
        return observation

    def step(self, action: SkirmishAction) -> StepResult:
        state = self._require_state()
        if int(action.action_id) not in {
            ACTION_NO_OP,
            ACTION_LEFT_CLICK,
            ACTION_RIGHT_CLICK,
        }:
            raise ValueError(f"unknown action id: {action.action_id}")

        state.step_count += 1
        parts: dict[str, float] = {"step": -0.01}
        events: list[str] = []
        feedback = np.zeros((1, ACTION_FEEDBACK_TOKEN_DIM), dtype=np.float32)

        selected = self.selected_friendly()
        target_grid = self.screen_to_grid(action.x, action.y)
        state.last_target_grid = target_grid

        damage_dealt = 0
        damage_taken = 0
        kills = 0
        moved_toward = False
        target_near_enemy = False
        executed_smart = False

        if selected is not None and selected.alive:
            before_distance = self._distance_to_nearest_enemy(selected)
            if int(action.action_id) == ACTION_RIGHT_CLICK:
                executed_smart = True
                parts["smart_used"] = 0.0
                target_enemy, target_distance = self._nearest_enemy_to(target_grid)
                target_near_enemy = target_enemy is not None and target_distance <= 1.5
                if target_near_enemy:
                    parts["target_near_enemy"] = 0.04
                    events.append("target_near_enemy")
                    damage_dealt, kills = self._attack_or_approach(selected, target_enemy)
                else:
                    parts["target_far_enemy"] = -0.03
                    moved_toward = self._move_entity_toward(selected, target_grid)
            elif int(action.action_id) == ACTION_LEFT_CLICK:
                parts["left_click_placeholder"] = -0.005
                events.append("left_click_placeholder")
            elif self.enemies_alive():
                parts["noop_visible_enemy"] = -0.02

            after_distance = self._distance_to_nearest_enemy(selected)
            moved_toward = moved_toward or after_distance < before_distance
            if moved_toward:
                parts["moved_toward_enemy"] = 0.02
                events.append("moved_toward_enemy")

        enemy_damage = self._enemy_turn()
        damage_taken += enemy_damage

        if damage_dealt:
            reward = float(damage_dealt) * 0.10
            parts["damage_dealt"] = reward
            events.append("damage_dealt")
            self._damage_dealt_total += float(damage_dealt)
        if damage_taken:
            penalty = -float(damage_taken) * 0.12
            parts["damage_taken"] = penalty
            events.append("damage_taken")
            self._damage_taken_total += float(damage_taken)
        if kills:
            parts["kill"] = float(kills) * 1.5
            events.append("kill")
            self._kill_total += float(kills)

        done = False
        truncated = False
        termination = None
        if not self.enemies_alive():
            parts["win"] = 3.0
            events.append("win")
            done = True
            termination = "win"
        elif not self.friendlies_alive():
            parts["loss"] = -3.0
            events.append("loss")
            done = True
            termination = "loss"
        elif state.step_count >= self.max_steps:
            truncated = True
            termination = "max_steps"

        total = float(sum(parts.values()))
        reward_info = RewardInfo(total=total, parts=parts, events=tuple(events))
        reward_info.validate()

        feedback[0, 0] = self._bridge_type(action)
        feedback[0, 1] = self._norm_screen(action.x)
        feedback[0, 2] = self._norm_screen(action.y)
        feedback[0, 3] = 1.0 if executed_smart else 0.0
        feedback[0, 4] = 1.0
        feedback[0, 5] = float(np.clip(total / 3.0, -1.0, 1.0))
        feedback[0, 6] = float(np.clip(kills, 0, 1))
        feedback[0, 7] = 1.0 if total < 0.0 else 0.0
        feedback[0, 8] = 1.0 if target_near_enemy else 0.0
        feedback[0, 9] = 1.0 if moved_toward else 0.0
        feedback[0, 10] = float(np.clip(damage_dealt / 8.0, 0.0, 1.0))
        feedback[0, 11] = float(np.clip(damage_taken / 14.0, 0.0, 1.0))
        state.last_action = action
        state.last_feedback = feedback

        return StepResult(
            observation=self.observe(),
            reward=reward_info,
            done=done,
            truncated=truncated,
            info={
                "step": state.step_count,
                "action": action.name,
                "target_grid": target_grid,
                "termination": termination,
            },
        )

    def render_text(self) -> str:
        state = self._require_state()
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x, y in state.walls:
            grid[y][x] = "#"
        if state.last_target_grid is not None:
            tx, ty = state.last_target_grid
            if self._in_bounds(tx, ty) and grid[ty][tx] == ".":
                grid[ty][tx] = "*"
        for entity in state.entities:
            if not entity.alive:
                continue
            marker = "F" if entity.kind == "friendly" else "E"
            grid[entity.y][entity.x] = marker
        lines = [
            f"step={state.step_count} friendly_hp={self._friendly_hp()} enemy_hp={self._enemy_hp()}",
        ]
        lines.extend("".join(row) for row in grid)
        return "\n".join(lines)

    def selected_friendly(self) -> Entity | None:
        for entity in self._require_state().entities:
            if entity.kind == "friendly" and entity.selected and entity.alive:
                return entity
        return None

    def enemies_alive(self) -> list[Entity]:
        return [
            entity
            for entity in self._require_state().entities
            if entity.kind == "enemy" and entity.alive
        ]

    def friendlies_alive(self) -> list[Entity]:
        return [
            entity
            for entity in self._require_state().entities
            if entity.kind == "friendly" and entity.alive
        ]

    def nearest_enemy_screen_target(self) -> tuple[int, int]:
        selected = self.selected_friendly()
        enemies = self.enemies_alive()
        if selected is None or not enemies:
            return (self.screen_size // 2, self.screen_size // 2)
        enemy = min(
            enemies,
            key=lambda candidate: self._manhattan(
                (selected.x, selected.y),
                (candidate.x, candidate.y),
            ),
        )
        return self.grid_to_screen(enemy.x, enemy.y)

    def screen_to_grid(self, x: int, y: int) -> tuple[int, int]:
        gx = int(np.clip(x, 0, self.screen_size - 1) * self.grid_size // self.screen_size)
        gy = int(np.clip(y, 0, self.screen_size - 1) * self.grid_size // self.screen_size)
        return gx, gy

    def grid_to_screen(self, x: int, y: int) -> tuple[int, int]:
        cell = self.screen_size / self.grid_size
        sx = int(np.clip(round((x + 0.5) * cell), 0, self.screen_size - 1))
        sy = int(np.clip(round((y + 0.5) * cell), 0, self.screen_size - 1))
        return sx, sy

    def random_action(self) -> SkirmishAction:
        action_id = int(
            self.rng.choice(
                [ACTION_NO_OP, ACTION_LEFT_CLICK, ACTION_RIGHT_CLICK],
                p=[0.2, 0.1, 0.7],
            ),
        )
        if action_id == ACTION_NO_OP:
            return SkirmishAction.no_op()
        x = int(self.rng.integers(0, self.screen_size))
        y = int(self.rng.integers(0, self.screen_size))
        if action_id == ACTION_LEFT_CLICK:
            return SkirmishAction.left_click(x, y)
        return SkirmishAction.right_click(x, y)

    def _require_state(self) -> TinySkirmishState:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return self.state

    def _build_spatial_obs(self, state: TinySkirmishState) -> np.ndarray:
        obs = np.zeros((SPATIAL_CHANNELS, self.screen_size, self.screen_size), dtype=np.float32)
        for x, y in state.walls:
            self._paint_cell(obs[0], x, y, 1.0)
        selected = self.selected_friendly()
        for entity in state.entities:
            if not entity.alive:
                continue
            if entity.kind == "friendly":
                self._paint_cell(obs[1], entity.x, entity.y, 1.0)
                self._paint_cell(obs[4], entity.x, entity.y, entity.health / entity.max_health)
                if entity.selected:
                    self._paint_cell(obs[3], entity.x, entity.y, 1.0)
            else:
                self._paint_cell(obs[2], entity.x, entity.y, 1.0)
                self._paint_cell(obs[5], entity.x, entity.y, entity.health / entity.max_health)
                for nx, ny in self._cells_in_range(entity.x, entity.y, entity.attack_range):
                    self._paint_cell(obs[10], nx, ny, 1.0)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) not in state.walls:
                    self._paint_cell(obs[6], x, y, 1.0)
                if selected is not None:
                    dist_enemy = self._distance_from_cell_to_nearest_enemy((x, y))
                    if np.isfinite(dist_enemy):
                        self._paint_cell(obs[11], x, y, 1.0 / (1.0 + dist_enemy))
                    dist_selected = self._manhattan((x, y), (selected.x, selected.y))
                    self._paint_cell(obs[12], x, y, 1.0 / (1.0 + dist_selected))
        if selected is not None:
            for nx, ny in self._cells_in_range(selected.x, selected.y, selected.attack_range):
                self._paint_cell(obs[7], nx, ny, 1.0)
        if state.last_target_grid is not None:
            self._paint_cell(obs[8], state.last_target_grid[0], state.last_target_grid[1], 1.0)
        obs[13, :, :] = 1.0
        return obs

    def _paint_cell(self, plane: np.ndarray, grid_x: int, grid_y: int, value: float) -> None:
        if not self._in_bounds(grid_x, grid_y):
            return
        cell = self.screen_size // self.grid_size
        x0 = grid_x * cell
        y0 = grid_y * cell
        plane[y0 : y0 + cell, x0 : x0 + cell] = float(value)

    def _build_entity_features(self, state: TinySkirmishState) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((MAX_ENTITY_TOKENS, ENTITY_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((MAX_ENTITY_TOKENS,), dtype=np.bool_)
        alive = [entity for entity in state.entities if entity.alive]
        alive.sort(key=lambda entity: (0 if entity.kind == "enemy" else 1, entity.entity_id))
        for idx, entity in enumerate(alive[:MAX_ENTITY_TOKENS]):
            sx, sy = self.grid_to_screen(entity.x, entity.y)
            features[idx] = np.asarray(
                [
                    entity.unit_type,
                    entity.alliance,
                    float(entity.health),
                    float(entity.health / entity.max_health),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(sx),
                    float(sy),
                    1.0,
                    100.0,
                    0.0,
                    0.0,
                    1.0 if entity.selected else 0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            mask[idx] = True
        return features, mask

    def _build_selection_features(self, state: TinySkirmishState) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((MAX_SELECTION_TOKENS, SELECTION_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((MAX_SELECTION_TOKENS,), dtype=np.bool_)
        selected = [
            entity
            for entity in state.entities
            if entity.kind == "friendly" and entity.selected and entity.alive
        ]
        for idx, entity in enumerate(selected[:MAX_SELECTION_TOKENS]):
            features[idx] = np.asarray(
                [
                    entity.unit_type,
                    FRIENDLY_ALLIANCE,
                    float(entity.health),
                    0.0,
                    0.0,
                    0.0,
                    100.0,
                ],
                dtype=np.float32,
            )
            mask[idx] = True
        return features, mask

    def _build_meta_vec(self, state: TinySkirmishState) -> np.ndarray:
        selected = self.selected_friendly()
        selected_x = 0.0
        selected_y = 0.0
        if selected is not None:
            sx, sy = self.grid_to_screen(selected.x, selected.y)
            selected_x = self._norm_screen(sx)
            selected_y = self._norm_screen(sy)
        meta = np.zeros((META_VECTOR_DIM,), dtype=np.float32)
        meta[0] = state.step_count / max(1, self.max_steps)
        meta[1] = float(len(self.friendlies_alive()))
        meta[2] = float(len(self.enemies_alive()))
        meta[3] = self._friendly_hp() / 14.0
        meta[4] = self._enemy_hp() / 16.0
        meta[5] = selected_x
        meta[6] = selected_y
        meta[7] = 1.0 - meta[0]
        meta[8] = self._damage_dealt_total / 16.0
        meta[9] = self._damage_taken_total / 14.0
        meta[10] = self._kill_total / 2.0
        meta[11] = 1.0
        meta[12] = 0.0
        meta[13] = 1.0
        meta[14] = self._last_action_index(state.last_action)
        return meta

    def _attack_or_approach(self, friendly: Entity, enemy: Entity | None) -> tuple[int, int]:
        if enemy is None or not enemy.alive:
            return 0, 0
        if self._chebyshev((friendly.x, friendly.y), (enemy.x, enemy.y)) <= friendly.attack_range:
            before = enemy.health
            enemy.health = max(0, enemy.health - friendly.attack_damage)
            damage = before - enemy.health
            kills = 1 if enemy.health == 0 else 0
            return damage, kills
        self._move_entity_toward(friendly, (enemy.x, enemy.y))
        return 0, 0

    def _enemy_turn(self) -> int:
        damage_taken = 0
        friendlies = self.friendlies_alive()
        if not friendlies:
            return 0
        friendly = friendlies[0]
        for enemy in self.enemies_alive():
            if self._chebyshev((enemy.x, enemy.y), (friendly.x, friendly.y)) <= enemy.attack_range:
                before = friendly.health
                friendly.health = max(0, friendly.health - enemy.attack_damage)
                damage_taken += before - friendly.health
            elif self._manhattan((enemy.x, enemy.y), (friendly.x, friendly.y)) <= 6:
                self._move_entity_toward(enemy, (friendly.x, friendly.y))
        return damage_taken

    def _move_entity_toward(self, entity: Entity, target: tuple[int, int]) -> bool:
        start = (entity.x, entity.y)
        if start == target:
            return False

        queue = deque([(start, None, 0)])
        seen = {start}
        best = (self._manhattan(start, target), 0, None)

        while queue:
            cell, first_step, path_len = queue.popleft()
            distance = self._manhattan(cell, target)
            if first_step is not None and (distance, path_len) < (best[0], best[1]):
                best = (distance, path_len, first_step)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_cell = (cell[0] + dx, cell[1] + dy)
                if next_cell in seen:
                    continue
                if self._is_blocked(next_cell[0], next_cell[1], moving_entity=entity):
                    continue
                seen.add(next_cell)
                queue.append(
                    (
                        next_cell,
                        next_cell if first_step is None else first_step,
                        path_len + 1,
                    ),
                )

        if best[2] is None:
            return False
        entity.x, entity.y = best[2]
        return True

    def _is_blocked(self, x: int, y: int, *, moving_entity: Entity) -> bool:
        if not self._in_bounds(x, y) or (x, y) in self._require_state().walls:
            return True
        for entity in self._require_state().entities:
            if entity is moving_entity or not entity.alive:
                continue
            if entity.x == x and entity.y == y:
                return True
        return False

    def _nearest_enemy_to(self, target: tuple[int, int]) -> tuple[Entity | None, float]:
        enemies = self.enemies_alive()
        if not enemies:
            return None, float("inf")
        enemy = min(enemies, key=lambda candidate: self._euclidean(target, (candidate.x, candidate.y)))
        return enemy, self._euclidean(target, (enemy.x, enemy.y))

    def _distance_to_nearest_enemy(self, entity: Entity) -> float:
        return self._distance_from_cell_to_nearest_enemy((entity.x, entity.y))

    def _distance_from_cell_to_nearest_enemy(self, cell: tuple[int, int]) -> float:
        enemies = self.enemies_alive()
        if not enemies:
            return float("inf")
        return min(self._manhattan(cell, (enemy.x, enemy.y)) for enemy in enemies)

    def _friendly_hp(self) -> float:
        return float(sum(entity.health for entity in self.friendlies_alive()))

    def _enemy_hp(self) -> float:
        return float(sum(entity.health for entity in self.enemies_alive()))

    def _cells_in_range(self, x: int, y: int, radius: int) -> list[tuple[int, int]]:
        cells = []
        for ny in range(y - radius, y + radius + 1):
            for nx in range(x - radius, x + radius + 1):
                if self._in_bounds(nx, ny) and self._chebyshev((x, y), (nx, ny)) <= radius:
                    cells.append((nx, ny))
        return cells

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _norm_screen(self, value: int) -> float:
        return float(np.clip(value, 0, self.screen_size - 1) / (self.screen_size - 1))

    @staticmethod
    def _bridge_type(action: SkirmishAction) -> float:
        if int(action.action_id) == ACTION_RIGHT_CLICK:
            return float(BRIDGE_RIGHT_CLICK)
        if int(action.action_id) == ACTION_LEFT_CLICK:
            return float(BRIDGE_LEFT_CLICK)
        return float(BRIDGE_NO_OP)

    @staticmethod
    def _last_action_index(action: SkirmishAction) -> float:
        if int(action.action_id) == ACTION_NO_OP:
            return 1.0
        if int(action.action_id) == ACTION_LEFT_CLICK:
            return 2.0
        if int(action.action_id) == ACTION_RIGHT_CLICK:
            return 3.0
        return 4.0
