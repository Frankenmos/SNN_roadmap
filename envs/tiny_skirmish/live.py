from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field
import numpy as np
import pygame

from .env import TinySkirmishEnv
from .protocol import (
    ACTION_NAMES,
    SPATIAL_CHANNELS,
    ObservationBatch,
    SkirmishAction,
)
from .render import (
    CHANNEL_COLORS,
    CORE_CHANNEL_IDS,
    _build_real_snn_context,
    _choose_action,
    channel_name,
    colorize_channel,
)


WINDOW_SIZE = (1280, 820)
BOARD_RECT = pygame.Rect(24, 88, 576, 576)
CHANNEL_RECT = pygame.Rect(632, 88, 624, 576)
HUD_RECT = pygame.Rect(24, 682, 1232, 112)
BACKGROUND = (12, 17, 22)
PANEL = (18, 25, 31)
PANEL_BORDER = (55, 66, 76)
TEXT = (232, 238, 244)
MUTED = (150, 164, 176)
GRID_LINE = (47, 58, 66)


@dataclass(slots=True)
class LiveSnapshot:
    action_name: str = "RESET"
    action_id: int | None = None
    target: tuple[int, int] | None = None
    reward: float = 0.0
    reward_parts: dict[str, float] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    termination: str | None = None
    total_reward: float = 0.0


@dataclass(slots=True)
class LiveSession:
    mode: str
    seed: int
    steps: int
    real_snn: bool
    device: str
    small: bool
    deterministic: bool
    env: TinySkirmishEnv = field(init=False)
    observation: ObservationBatch = field(init=False)
    snapshot: LiveSnapshot = field(default_factory=LiveSnapshot)
    real_context: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.env = TinySkirmishEnv(seed=self.seed, max_steps=self.steps)
        self.reset()

    def reset(self) -> None:
        self.observation = self.env.reset(self.seed)
        if self.real_snn:
            self.real_context = _build_real_snn_context(
                device=self.device,
                small=self.small,
            )
        self.snapshot = LiveSnapshot()

    def step(self, manual_action: SkirmishAction | None = None) -> None:
        if self.snapshot.termination is not None:
            return
        action = manual_action or self._next_action()
        result = self.env.step(action)
        self.observation = result.observation
        total_reward = self.snapshot.total_reward + result.reward.total
        self.snapshot = LiveSnapshot(
            action_name=ACTION_NAMES.get(action.action_id, str(action.action_id)),
            action_id=action.action_id,
            target=result.info.get("target_grid"),
            reward=result.reward.total,
            reward_parts=result.reward.compact_parts(),
            events=result.reward.events,
            termination=result.info.get("termination"),
            total_reward=total_reward,
        )

    def _next_action(self) -> SkirmishAction:
        if self.real_context is not None and self.mode != "manual":
            return _choose_action(
                env=self.env,
                mode=self.mode,
                real_context=self.real_context,
                deterministic=self.deterministic,
            )
        if self.mode == "scripted":
            return _choose_action(
                env=self.env,
                mode="scripted",
                real_context=None,
                deterministic=self.deterministic,
            )
        if self.mode == "random":
            return _choose_action(
                env=self.env,
                mode="random",
                real_context=None,
                deterministic=self.deterministic,
            )
        return SkirmishAction.no_op()


@dataclass(slots=True)
class LiveView:
    channels: str
    channel_page: int = 0
    paused: bool = False
    fps: float = 4.0

    def toggle_channels(self) -> None:
        self.channels = "all" if self.channels == "core" else "core"
        self.channel_page = 0

    def channel_ids(self) -> tuple[int, ...]:
        if self.channels == "core":
            return CORE_CHANNEL_IDS
        page_size = 12
        start = self.channel_page * page_size
        return tuple(range(SPATIAL_CHANNELS))[start : start + page_size]

    def page_count(self) -> int:
        if self.channels == "core":
            return 1
        return int(np.ceil(SPATIAL_CHANNELS / 12))

    def next_page(self) -> None:
        self.channel_page = (self.channel_page + 1) % self.page_count()


def manual_action_from_board(
    position: tuple[int, int],
    button: int,
    *,
    env: TinySkirmishEnv,
    board_rect: pygame.Rect = BOARD_RECT,
) -> SkirmishAction | None:
    if not board_rect.collidepoint(position):
        return None
    rel_x = float(position[0] - board_rect.left) / float(board_rect.width)
    rel_y = float(position[1] - board_rect.top) / float(board_rect.height)
    screen_x = int(np.clip(rel_x * env.screen_size, 0, env.screen_size - 1))
    screen_y = int(np.clip(rel_y * env.screen_size, 0, env.screen_size - 1))
    if button == 1:
        return SkirmishAction.left_click(screen_x, screen_y)
    if button == 3:
        return SkirmishAction.right_click(screen_x, screen_y)
    return None


def draw_frame(
    surface: pygame.Surface,
    session: LiveSession,
    view: LiveView,
    *,
    fonts: dict[str, pygame.font.Font],
) -> None:
    surface.fill(BACKGROUND)
    _draw_top_bar(surface, session, view, fonts)
    draw_board(surface, session.env, session.observation, fonts=fonts)
    draw_channels(surface, session.observation, view, fonts=fonts)
    draw_hud(surface, session, view, fonts=fonts)


def draw_board(
    surface: pygame.Surface,
    env: TinySkirmishEnv,
    observation: ObservationBatch,
    *,
    fonts: dict[str, pygame.font.Font],
) -> None:
    state = env._require_state()
    pygame.draw.rect(surface, PANEL, BOARD_RECT)
    cell = BOARD_RECT.width // env.grid_size

    for y in range(env.grid_size):
        for x in range(env.grid_size):
            rect = pygame.Rect(
                BOARD_RECT.left + x * cell,
                BOARD_RECT.top + y * cell,
                cell,
                cell,
            )
            color = (17, 25, 31)
            if _grid_plane_value(observation, 11, env, x, y) > 0.0:
                color = _blend_color(color, (35, 78, 130), 0.28)
            if _grid_plane_value(observation, 12, env, x, y) > 0.0:
                color = _blend_color(color, (78, 50, 126), 0.22)
            if _grid_plane_value(observation, 10, env, x, y) > 0.0:
                color = _blend_color(color, CHANNEL_COLORS[10], 0.30)
            if _grid_plane_value(observation, 7, env, x, y) > 0.0:
                color = _blend_color(color, CHANNEL_COLORS[7], 0.42)
            if (x, y) in state.walls:
                color = CHANNEL_COLORS[0]
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRID_LINE, rect, 1)

    if state.last_target_grid is not None:
        tx, ty = state.last_target_grid
        if 0 <= tx < env.grid_size and 0 <= ty < env.grid_size:
            rect = pygame.Rect(
                BOARD_RECT.left + tx * cell,
                BOARD_RECT.top + ty * cell,
                cell,
                cell,
            )
            pygame.draw.rect(surface, CHANNEL_COLORS[8], rect, 4)
            pygame.draw.line(surface, CHANNEL_COLORS[8], rect.topleft, rect.bottomright, 2)
            pygame.draw.line(surface, CHANNEL_COLORS[8], rect.topright, rect.bottomleft, 2)

    for entity in state.entities:
        if not entity.alive:
            continue
        center = (
            BOARD_RECT.left + entity.x * cell + cell // 2,
            BOARD_RECT.top + entity.y * cell + cell // 2,
        )
        radius = max(10, cell // 3)
        color = CHANNEL_COLORS[1] if entity.kind == "friendly" else CHANNEL_COLORS[2]
        pygame.draw.circle(surface, (8, 12, 16), center, radius + 4)
        pygame.draw.circle(surface, color, center, radius)
        if entity.selected:
            pygame.draw.circle(surface, CHANNEL_COLORS[3], center, radius + 8, 4)
        _draw_health_bar(surface, entity.health, entity.max_health, center, cell)

    pygame.draw.rect(surface, PANEL_BORDER, BOARD_RECT, 2)
    label = fonts["small"].render("live skirmish state", True, MUTED)
    surface.blit(label, (BOARD_RECT.left, BOARD_RECT.top - 24))


def draw_channels(
    surface: pygame.Surface,
    observation: ObservationBatch,
    view: LiveView,
    *,
    fonts: dict[str, pygame.font.Font],
) -> None:
    pygame.draw.rect(surface, PANEL, CHANNEL_RECT)
    channel_ids = view.channel_ids()
    columns = 5 if view.channels == "core" else 4
    gap = 10
    label_height = 34
    tile_size = 96 if view.channels == "core" else 116
    x0 = CHANNEL_RECT.left + 14
    y0 = CHANNEL_RECT.top + 38

    header = f"channels={view.channels}"
    if view.channels == "all":
        header += f" page={view.channel_page + 1}/{view.page_count()}"
    surface.blit(fonts["small"].render(header, True, MUTED), (CHANNEL_RECT.left + 14, CHANNEL_RECT.top + 12))

    for index, channel_id in enumerate(channel_ids):
        row = index // columns
        col = index % columns
        x = x0 + col * (tile_size + gap)
        y = y0 + row * (tile_size + label_height + gap)
        plane = observation.spatial_obs[channel_id]
        tile = pygame_surface_from_channel(plane, channel_id, tile_size)
        surface.blit(tile, (x, y + label_height))
        min_value = float(np.min(plane))
        max_value = float(np.max(plane))
        label = f"{channel_id:02d} {channel_name(channel_id)}"
        stats = f"{min_value:.2f}..{max_value:.2f}"
        surface.blit(fonts["tiny"].render(label, True, TEXT), (x, y))
        surface.blit(fonts["tiny"].render(stats, True, MUTED), (x, y + 15))

    pygame.draw.rect(surface, PANEL_BORDER, CHANNEL_RECT, 2)


def draw_hud(
    surface: pygame.Surface,
    session: LiveSession,
    view: LiveView,
    *,
    fonts: dict[str, pygame.font.Font],
) -> None:
    pygame.draw.rect(surface, PANEL, HUD_RECT)
    pygame.draw.rect(surface, PANEL_BORDER, HUD_RECT, 2)
    state = session.env._require_state()
    snapshot = session.snapshot
    status = "paused" if view.paused else "running"
    if snapshot.termination:
        status = f"done:{snapshot.termination}"
    left = (
        f"step={state.step_count}/{session.steps} mode={session.mode} "
        f"status={status} fps={view.fps:.1f}"
    )
    action = (
        f"action={snapshot.action_name} target={snapshot.target} "
        f"reward={snapshot.reward:+.3f} total={snapshot.total_reward:+.3f}"
    )
    parts = f"parts={snapshot.reward_parts or {}}"
    events = f"events={','.join(snapshot.events) if snapshot.events else '-'}"
    controls = (
        "Space pause | N step | R reset | S/D/M mode | C channels | Tab page | "
        "+/- fps | mouse in manual | Esc quit"
    )
    surface.blit(fonts["body"].render(left, True, TEXT), (HUD_RECT.left + 16, HUD_RECT.top + 12))
    surface.blit(fonts["body"].render(action, True, TEXT), (HUD_RECT.left + 16, HUD_RECT.top + 36))
    surface.blit(fonts["small"].render(parts, True, MUTED), (HUD_RECT.left + 16, HUD_RECT.top + 62))
    surface.blit(fonts["small"].render(events, True, MUTED), (HUD_RECT.left + 16, HUD_RECT.top + 82))
    surface.blit(fonts["small"].render(controls, True, (130, 144, 156)), (HUD_RECT.left + 610, HUD_RECT.top + 82))


def pygame_surface_from_channel(
    plane: np.ndarray,
    channel_id: int,
    tile_size: int,
) -> pygame.Surface:
    image = colorize_channel(plane, channel_id=channel_id)
    raw = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    return pygame.transform.scale(raw, (tile_size, tile_size))


def build_fonts() -> dict[str, pygame.font.Font]:
    return {
        "title": pygame.font.SysFont("consolas", 24, bold=True),
        "body": pygame.font.SysFont("consolas", 18),
        "small": pygame.font.SysFont("consolas", 15),
        "tiny": pygame.font.SysFont("consolas", 12),
    }


def run_live(
    *,
    mode: str,
    seed: int,
    steps: int,
    fps: float,
    channels: str,
    real_snn: bool = False,
    device: str = "cpu",
    small: bool = False,
    deterministic: bool = False,
) -> None:
    pygame.init()
    pygame.display.set_caption("TinySkirmish Live")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = build_fonts()
    session = LiveSession(
        mode=mode,
        seed=seed,
        steps=steps,
        real_snn=real_snn,
        device=device,
        small=small,
        deterministic=deterministic,
    )
    view = LiveView(channels=channels, fps=max(0.5, float(fps)), paused=(mode == "manual"))
    accumulator = 0.0
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = _handle_key(event, session, view)
            elif event.type == pygame.MOUSEBUTTONDOWN and session.mode == "manual":
                action = manual_action_from_board(event.pos, event.button, env=session.env)
                if action is not None:
                    session.step(action)
                    view.paused = True

        if not view.paused:
            accumulator += dt
            interval = 1.0 / max(view.fps, 0.5)
            while accumulator >= interval and not view.paused:
                session.step()
                accumulator -= interval
                if session.snapshot.termination is not None:
                    view.paused = True

        draw_frame(screen, session, view, fonts=fonts)
        pygame.display.flip()

    pygame.quit()


def _handle_key(
    event: pygame.event.Event,
    session: LiveSession,
    view: LiveView,
) -> bool:
    if event.key == pygame.K_ESCAPE:
        return False
    if event.key == pygame.K_SPACE:
        view.paused = not view.paused
    elif event.key == pygame.K_n:
        session.step()
        view.paused = True
    elif event.key == pygame.K_r:
        session.reset()
        view.paused = session.mode == "manual"
    elif event.key == pygame.K_s:
        session.mode = "scripted"
        view.paused = False
    elif event.key == pygame.K_d:
        session.mode = "random"
        view.paused = False
    elif event.key == pygame.K_m:
        session.mode = "manual"
        view.paused = True
    elif event.key == pygame.K_c:
        view.toggle_channels()
    elif event.key == pygame.K_TAB:
        view.next_page()
    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        view.fps = min(30.0, view.fps + 1.0)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        view.fps = max(0.5, view.fps - 1.0)
    return True


def _draw_top_bar(
    surface: pygame.Surface,
    session: LiveSession,
    view: LiveView,
    fonts: dict[str, pygame.font.Font],
) -> None:
    title = "TinySkirmish Live"
    if session.real_snn:
        title += f" | real_snn device={session.device} small={session.small}"
    surface.blit(fonts["title"].render(title, True, TEXT), (24, 22))
    subtitle = "Pygame dashboard over the same 27x84x84 observation protocol"
    surface.blit(fonts["small"].render(subtitle, True, MUTED), (24, 52))
    channel = f"channel_page={view.channel_page + 1}/{view.page_count()}"
    surface.blit(fonts["small"].render(channel, True, MUTED), (1048, 52))


def _grid_plane_value(
    observation: ObservationBatch,
    channel_id: int,
    env: TinySkirmishEnv,
    grid_x: int,
    grid_y: int,
) -> float:
    sx, sy = env.grid_to_screen(grid_x, grid_y)
    return float(observation.spatial_obs[channel_id, sy, sx])


def _blend_color(
    base: tuple[int, int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(base[index] * (1.0 - alpha) + color[index] * alpha))
        for index in range(3)
    )


def _draw_health_bar(
    surface: pygame.Surface,
    health: int,
    max_health: int,
    center: tuple[int, int],
    cell_size: int,
) -> None:
    width = int(cell_size * 0.72)
    height = 6
    left = center[0] - width // 2
    top = center[1] + cell_size // 3
    ratio = 0.0 if max_health <= 0 else max(0.0, min(1.0, health / max_health))
    background = pygame.Rect(left, top, width, height)
    fill = pygame.Rect(left, top, int(width * ratio), height)
    pygame.draw.rect(surface, (35, 20, 20), background)
    pygame.draw.rect(surface, (80, 230, 120), fill)
    pygame.draw.rect(surface, (8, 12, 16), background, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Pygame renderer for TinySkirmish.")
    parser.add_argument("--mode", choices=("scripted", "random", "manual"), default="scripted")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--channels", choices=("core", "all"), default="core")
    parser.add_argument("--real-snn", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_live(
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        fps=args.fps,
        channels=args.channels,
        real_snn=args.real_snn,
        device=args.device,
        small=args.small,
        deterministic=args.deterministic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
