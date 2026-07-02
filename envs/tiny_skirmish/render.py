from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import TinySkirmishEnv
from .protocol import (
    ACTION_NAMES,
    SPATIAL_CHANNELS,
    ObservationBatch,
    SkirmishAction,
)
from .real_snn_bridge import (
    build_real_policy,
    import_real_snn_modules,
    tiny_observation_to_real_batch,
)
from .rollout import scripted_action


try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - older Pillow fallback
    RESAMPLE_NEAREST = Image.NEAREST


CORE_CHANNEL_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13)

CHANNEL_SCHEMA: dict[int, str] = {
    0: "walls",
    1: "friendly",
    2: "enemy",
    3: "selected",
    4: "friendly_hp",
    5: "enemy_hp",
    6: "passable",
    7: "friendly_attack_range",
    8: "last_target",
    10: "enemy_attack_range",
    11: "inverse_enemy_distance",
    12: "inverse_selected_distance",
    13: "bias",
}

CHANNEL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (150, 154, 160),
    1: (60, 220, 150),
    2: (235, 80, 90),
    3: (255, 225, 80),
    4: (80, 255, 170),
    5: (255, 105, 120),
    6: (75, 90, 100),
    7: (255, 160, 60),
    8: (245, 70, 230),
    10: (255, 100, 35),
    11: (80, 160, 255),
    12: (170, 100, 255),
    13: (220, 220, 220),
}


@dataclass(slots=True)
class RenderRecord:
    step: int
    observation: ObservationBatch
    state_text: str
    action_name: str = "RESET"
    action_id: int | None = None
    target: tuple[int, int] | None = None
    reward: float = 0.0
    reward_parts: dict[str, float] | None = None
    events: tuple[str, ...] = ()
    termination: str | None = None


def channel_name(channel_id: int) -> str:
    return CHANNEL_SCHEMA.get(int(channel_id), f"unused_{channel_id}")


def render_overview(record: RenderRecord, *, scale: int = 4) -> Image.Image:
    spatial = record.observation.spatial_obs
    rgb = np.zeros((*spatial.shape[1:], 3), dtype=np.float32)
    rgb[:, :] = (14, 21, 26)

    _blend(rgb, spatial[6], (24, 34, 40), 0.95)
    _blend(rgb, spatial[11], (35, 70, 130), 0.32)
    _blend(rgb, spatial[12], (75, 50, 120), 0.28)
    _blend(rgb, spatial[7], CHANNEL_COLORS[7], 0.38)
    _blend(rgb, spatial[10], CHANNEL_COLORS[10], 0.28)
    _blend(rgb, spatial[0], CHANNEL_COLORS[0], 0.95)
    _blend(rgb, spatial[1], CHANNEL_COLORS[1], 1.0)
    _blend(rgb, spatial[2], CHANNEL_COLORS[2], 1.0)
    _blend(rgb, spatial[3], CHANNEL_COLORS[3], 0.82)
    _blend(rgb, spatial[8], CHANNEL_COLORS[8], 1.0)

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((image.width * scale, image.height * scale), RESAMPLE_NEAREST)
    _draw_grid(image, cell_size=7 * scale)

    header_height = 70
    canvas = Image.new("RGB", (image.width, image.height + header_height), (18, 24, 30))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 8), _record_title(record), fill=(235, 240, 245), font=font)
    draw.text((10, 28), _record_reward(record), fill=(190, 205, 215), font=font)
    draw.text((10, 48), "gray=walls green=friendly red=enemy yellow=selected magenta=target", fill=(150, 165, 175), font=font)
    return canvas


def render_channel_sheet(
    observation: ObservationBatch,
    *,
    mode: str = "core",
    tile_scale: int = 2,
    columns: int | None = None,
) -> Image.Image:
    observation.validate()
    if mode == "core":
        channel_ids = CORE_CHANNEL_IDS
        columns = columns or 4
    elif mode == "all":
        channel_ids = tuple(range(SPATIAL_CHANNELS))
        columns = columns or 6
    else:
        raise ValueError("channel mode must be 'core' or 'all'")

    tile_image_size = observation.spatial_obs.shape[-1] * tile_scale
    label_height = 42
    gap = 10
    margin = 12
    tile_w = tile_image_size
    tile_h = tile_image_size + label_height
    rows = int(np.ceil(len(channel_ids) / columns))
    width = margin * 2 + columns * tile_w + (columns - 1) * gap
    height = margin * 2 + rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (16, 22, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, channel_id in enumerate(channel_ids):
        row = index // columns
        col = index % columns
        x = margin + col * (tile_w + gap)
        y = margin + row * (tile_h + gap)
        plane = observation.spatial_obs[channel_id]
        tile = colorize_channel(plane, channel_id=channel_id)
        tile = tile.resize((tile_image_size, tile_image_size), RESAMPLE_NEAREST)
        canvas.paste(tile, (x, y + label_height))
        min_value = float(np.min(plane))
        max_value = float(np.max(plane))
        label = f"{channel_id:02d} {channel_name(channel_id)}"
        stats = f"min={min_value:.2f} max={max_value:.2f}"
        draw.text((x, y), label, fill=(235, 240, 245), font=font)
        draw.text((x, y + 17), stats, fill=(160, 176, 186), font=font)
    return canvas


def colorize_channel(plane: np.ndarray, *, channel_id: int) -> Image.Image:
    values = np.asarray(plane, dtype=np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        if max_value > 0.0:
            norm = np.ones_like(values, dtype=np.float32)
        else:
            norm = np.zeros_like(values, dtype=np.float32)
    else:
        norm = (values - min_value) / (max_value - min_value)
    color = np.asarray(CHANNEL_COLORS.get(channel_id, (95, 120, 145)), dtype=np.float32)
    rgb = norm[..., None] * color[None, None, :]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def export_record(
    record: RenderRecord,
    *,
    output_dir: Path,
    channel_mode: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / f"step_{record.step:03d}_overview.png"
    channels_path = output_dir / f"step_{record.step:03d}_channels.png"
    render_overview(record).save(overview_path)
    render_channel_sheet(record.observation, mode=channel_mode).save(channels_path)
    return overview_path, channels_path


def render_rollout(
    *,
    mode: str,
    seed: int,
    steps: int,
    output_dir: Path,
    channel_mode: str,
    render_every: int = 1,
    real_snn: bool = False,
    device: str = "cpu",
    small: bool = False,
    deterministic: bool = False,
) -> list[RenderRecord]:
    if render_every <= 0:
        raise ValueError("render_every must be positive")

    env = TinySkirmishEnv(seed=seed, max_steps=steps)
    observation = env.reset()
    records: list[RenderRecord] = [
        RenderRecord(step=0, observation=observation, state_text=env.render_text()),
    ]

    real_context = None
    if real_snn:
        real_context = _build_real_snn_context(
            device=device,
            small=small,
        )

    total_reward = 0.0
    events: dict[str, int] = {}
    summary_lines = [_summary_line(records[0])]

    for step_index in range(1, steps + 1):
        action = _choose_action(
            env=env,
            mode=mode,
            real_context=real_context,
            deterministic=deterministic,
        )
        result = env.step(action)
        total_reward += result.reward.total
        for event in result.reward.events:
            events[event] = events.get(event, 0) + 1
        record = RenderRecord(
            step=step_index,
            observation=result.observation,
            state_text=env.render_text(),
            action_name=ACTION_NAMES.get(action.action_id, str(action.action_id)),
            action_id=action.action_id,
            target=result.info.get("target_grid"),
            reward=result.reward.total,
            reward_parts=result.reward.compact_parts(),
            events=result.reward.events,
            termination=result.info.get("termination"),
        )
        records.append(record)
        summary_lines.append(_summary_line(record))
        if result.done or result.truncated:
            break

    rendered = []
    for record in records:
        if record.step == 0 or record.step % render_every == 0 or record.termination:
            rendered.append(record)
            export_record(record, output_dir=output_dir, channel_mode=channel_mode)

    summary_lines.append("")
    summary_lines.append(f"rendered_steps={[record.step for record in rendered]}")
    summary_lines.append(f"total_reward={total_reward:.6f}")
    summary_lines.append(f"events={events}")
    final = records[-1]
    summary_lines.append(f"termination={final.termination or 'none'}")
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return rendered


def _build_real_snn_context(*, device: str, small: bool) -> dict[str, object]:
    import torch

    real = import_real_snn_modules()
    PPO = real["PPO"]
    torch_device = torch.device(device)
    policy = build_real_policy(real, device=torch_device, small=small)
    ppo = PPO(
        policy,
        lr=1.0e-4,
        gamma=0.99,
        clip_epsilon=0.1,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        tbptt_window=8,
        rollout_cache_spatial_dtype="float32",
    )
    state = policy.init_concrete_state(batch_size=1, device=torch_device)
    return {
        "real": real,
        "ppo": ppo,
        "device": torch_device,
        "state": state,
    }


def _choose_action(
    *,
    env: TinySkirmishEnv,
    mode: str,
    real_context: dict[str, object] | None,
    deterministic: bool,
) -> SkirmishAction:
    if real_context is not None:
        batch = tiny_observation_to_real_batch(
            env.observe(),
            real_context["real"],
            device=real_context["device"],
        ).with_state(real_context["state"])
        sample = real_context["ppo"].select_action(batch, deterministic=deterministic)
        real_context["state"] = sample.next_state
        return SkirmishAction(sample.action_id, sample.x, sample.y)
    if mode == "scripted":
        return scripted_action(env)
    if mode == "random":
        return env.random_action()
    raise ValueError(f"unknown mode: {mode}")


def _blend(
    rgb: np.ndarray,
    plane: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    mask = np.clip(np.asarray(plane, dtype=np.float32), 0.0, 1.0)
    if not np.any(mask > 0.0):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    effective_alpha = (mask * float(alpha))[..., None]
    rgb[:] = rgb * (1.0 - effective_alpha) + color_arr * effective_alpha


def _draw_grid(image: Image.Image, *, cell_size: int) -> None:
    draw = ImageDraw.Draw(image)
    color = (50, 60, 68)
    for value in range(0, image.width + 1, cell_size):
        draw.line((value, 0, value, image.height), fill=color)
    for value in range(0, image.height + 1, cell_size):
        draw.line((0, value, image.width, value), fill=color)


def _record_title(record: RenderRecord) -> str:
    target = "-" if record.target is None else str(record.target)
    return f"step={record.step:03d} action={record.action_name} target={target}"


def _record_reward(record: RenderRecord) -> str:
    parts = record.reward_parts or {}
    events = ",".join(record.events) if record.events else "-"
    return f"reward={record.reward:+.3f} parts={parts} events={events}"


def _summary_line(record: RenderRecord) -> str:
    return (
        f"step={record.step:03d} action={record.action_name} target={record.target} "
        f"reward={record.reward:+.3f} parts={record.reward_parts or {}} "
        f"events={record.events} termination={record.termination}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render TinySkirmish spatial channels.")
    parser.add_argument("--mode", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("renders/scripted"))
    parser.add_argument("--channels", choices=("core", "all"), default="core")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--real-snn", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rendered = render_rollout(
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        output_dir=args.out,
        channel_mode=args.channels,
        render_every=args.render_every,
        real_snn=args.real_snn,
        device=args.device,
        small=args.small,
        deterministic=args.deterministic,
    )
    print(f"Rendered {len(rendered)} steps to {args.out}")
    print(f"Summary: {args.out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
