from __future__ import annotations

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from .live import (
    BOARD_RECT,
    WINDOW_SIZE,
    LiveSession,
    LiveView,
    build_fonts,
    draw_board,
    draw_channels,
    draw_frame,
    manual_action_from_board,
)
from .protocol import ACTION_LEFT_CLICK, ACTION_RIGHT_CLICK


def _surface_is_nonempty(surface: pygame.Surface) -> bool:
    pixels = pygame.surfarray.array3d(surface)
    return bool(pixels.max() > 0 and pixels.sum() > 1000)


def _make_session() -> LiveSession:
    return LiveSession(
        mode="scripted",
        seed=9,
        steps=12,
        real_snn=False,
        device="cpu",
        small=False,
        deterministic=False,
    )


def _assert_frame_renders() -> None:
    pygame.display.set_mode((1, 1))
    fonts = build_fonts()
    session = _make_session()
    view = LiveView(channels="core", paused=True, fps=4.0)
    surface = pygame.Surface(WINDOW_SIZE)

    draw_frame(surface, session, view, fonts=fonts)
    if not _surface_is_nonempty(surface):
        raise AssertionError("reset frame rendered as an empty surface")

    session.step()
    surface.fill((0, 0, 0))
    draw_frame(surface, session, view, fonts=fonts)
    if not _surface_is_nonempty(surface):
        raise AssertionError("stepped frame rendered as an empty surface")


def _assert_channel_modes_render() -> None:
    pygame.display.set_mode((1, 1))
    fonts = build_fonts()
    session = _make_session()
    surface = pygame.Surface(WINDOW_SIZE)

    draw_channels(surface, session.observation, LiveView(channels="core"), fonts=fonts)
    if not _surface_is_nonempty(surface):
        raise AssertionError("core channel view rendered as an empty surface")

    surface.fill((0, 0, 0))
    draw_channels(surface, session.observation, LiveView(channels="all"), fonts=fonts)
    if not _surface_is_nonempty(surface):
        raise AssertionError("all channel view rendered as an empty surface")


def _assert_board_renders() -> None:
    pygame.display.set_mode((1, 1))
    fonts = build_fonts()
    session = _make_session()
    surface = pygame.Surface(WINDOW_SIZE)

    draw_board(surface, session.env, session.observation, fonts=fonts)
    if not _surface_is_nonempty(surface):
        raise AssertionError("board rendered as an empty surface")


def _assert_manual_actions() -> None:
    session = _make_session()
    center = BOARD_RECT.center
    right = manual_action_from_board(center, 3, env=session.env)
    left = manual_action_from_board(center, 1, env=session.env)
    outside = manual_action_from_board((BOARD_RECT.right + 20, BOARD_RECT.bottom + 20), 3, env=session.env)
    if right is None or right.action_id != ACTION_RIGHT_CLICK:
        raise AssertionError("right mouse click did not map to RIGHT_CLICK")
    if left is None or left.action_id != ACTION_LEFT_CLICK:
        raise AssertionError("left mouse click did not map to LEFT_CLICK")
    if outside is not None:
        raise AssertionError("outside-board click produced an action")
    gx, gy = session.env.screen_to_grid(right.x, right.y)
    if not (0 <= gx < session.env.grid_size and 0 <= gy < session.env.grid_size):
        raise AssertionError("manual action target is outside the grid")


def main() -> int:
    pygame.init()
    checks = [
        _assert_frame_renders,
        _assert_channel_modes_render,
        _assert_board_renders,
        _assert_manual_actions,
    ]
    try:
        for check in checks:
            check()
            print(f"ok {check.__name__}")
    finally:
        pygame.quit()
    print("TinySkirmish live renderer self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
