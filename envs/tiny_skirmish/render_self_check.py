from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from PIL import Image

from .render import CHANNEL_SCHEMA, CORE_CHANNEL_IDS, render_rollout
from .protocol import SPATIAL_CHANNELS


def _assert_rendered_png(path: Path, *, min_size: tuple[int, int]) -> None:
    if not path.exists():
        raise AssertionError(f"missing render file: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"empty render file: {path}")
    with Image.open(path) as image:
        if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
            raise AssertionError(f"unexpected image size for {path}: {image.size}")


def _assert_core_render() -> None:
    with TemporaryDirectory() as temp:
        out = Path(temp) / "core"
        rendered = render_rollout(
            mode="scripted",
            seed=9,
            steps=2,
            output_dir=out,
            channel_mode="core",
            render_every=1,
        )
        assert [record.step for record in rendered] == [0, 1, 2]
        _assert_rendered_png(out / "step_000_overview.png", min_size=(300, 380))
        _assert_rendered_png(out / "step_000_channels.png", min_size=(600, 600))
        summary = (out / "summary.txt").read_text(encoding="utf-8")
        assert "target_near_enemy" in summary


def _assert_all_channel_render() -> None:
    with TemporaryDirectory() as temp:
        out = Path(temp) / "all"
        rendered = render_rollout(
            mode="scripted",
            seed=9,
            steps=1,
            output_dir=out,
            channel_mode="all",
            render_every=1,
        )
        assert [record.step for record in rendered] == [0, 1]
        _assert_rendered_png(out / "step_000_channels.png", min_size=(900, 900))


def _assert_documented_channels() -> None:
    missing = [channel_id for channel_id in CORE_CHANNEL_IDS if channel_id not in CHANNEL_SCHEMA]
    if missing:
        raise AssertionError(f"core channel ids missing schema labels: {missing}")
    if any(channel_id < 0 or channel_id >= SPATIAL_CHANNELS for channel_id in CHANNEL_SCHEMA):
        raise AssertionError("channel schema contains out-of-range channel id")


def main() -> int:
    checks = [
        _assert_documented_channels,
        _assert_core_render,
        _assert_all_channel_render,
    ]
    for check in checks:
        check()
        print(f"ok {check.__name__}")
    print("TinySkirmish render self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
