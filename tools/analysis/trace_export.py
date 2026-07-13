"""Export an eval episode trace into JSON for the arch explorer.

eval.py --trace-episodes writes .pt sidecars (Utility/eval_trace.py,
format_version 1) under analysis_results/<run>/episode_traces/. This
module compacts one of them into trace_data.json for the explorer's
trace-replay view: per policy step the action / log-prob / value /
reward stream plus bit-packed friendly / enemy / selected screen masks
decoded from the player_relative and selected feature_screen channels.

Usage:
  python -m tools.analysis.trace_export <run> [--episode 1] [--mode det]
  python -m tools.analysis.trace_export <run> --list
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import torch

TRACE_JSON_SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES_ROOT = _REPO_ROOT / "analysis_results"
ARCH_EXPLORER_TRACE_JSON = (
    _REPO_ROOT / "tools" / "viz" / "arch_explorer" / "public" / "trace_data.json"
)
TRACE_FILE_RE = re.compile(r"^episode_(\d+)_(det|stoch)\.pt$")

# spatial_obs is the raw PySC2 feature_screen stack / 255
# (obs_space/obs_space_2.py:348-354); channel order is
# pysc2.lib.features.SCREEN_FEATURES.
PLAYER_RELATIVE_CHANNEL = 5
SELECTED_CHANNEL = 7
PR_FRIENDLY = 1
PR_ENEMY = 4
SCREEN_SIZE = 84


def _pack_mask(mask: np.ndarray) -> str:
    """Boolean [84, 84] -> base64 of np.packbits (big-endian bit order,
    row-major: bit index = y * 84 + x)."""
    return base64.b64encode(np.packbits(mask.reshape(-1))).decode("ascii")


def list_traces(
    run_name: str,
    traces_root: str | Path = DEFAULT_TRACES_ROOT,
) -> list[Path]:
    trace_dir = Path(traces_root) / run_name / "episode_traces"
    if not trace_dir.is_dir():
        return []
    found = []
    for path in trace_dir.iterdir():
        match = TRACE_FILE_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), match.group(2), path))
    return [path for _episode, _mode, path in sorted(found)]


def resolve_trace_path(
    run_name: str,
    episode: int,
    mode: str,
    traces_root: str | Path = DEFAULT_TRACES_ROOT,
) -> Path:
    path = (
        Path(traces_root)
        / run_name
        / "episode_traces"
        / f"episode_{int(episode):04d}_{mode}.pt"
    )
    if not path.exists():
        available = [p.name for p in list_traces(run_name, traces_root)]
        raise FileNotFoundError(
            f"trace not found: {path}"
            + (f" (available: {', '.join(available)})" if available else ""),
        )
    return path


def _step_entry(record: dict) -> dict:
    entry = {
        "t": int(record["step_index"]),
        "policy": bool(record["policy_step"]),
        "learnable": bool(record["learnable"]),
        "action": record["action"],
        "x": int(record["move_x"]),
        "y": int(record["move_y"]),
        "log_prob": round(float(record["log_prob"]), 4),
        "value": round(float(record["value"]), 4),
        "reward": round(float(record["reward"]), 4),
        "cum_reward": round(float(record["cumulative_reward"]), 4),
        "done": bool(record["done"]),
        "func": (record.get("dispatched_action") or {}).get("function_name"),
    }
    policy_input = record.get("policy_input")
    if policy_input is not None:
        # fp16-stored feature_screen/255: *255 and round recovers the
        # integer categorical values exactly (1/255 and 4/255 round-trip).
        spatial = policy_input["spatial_obs"].float() * 255.0
        player_relative = spatial[PLAYER_RELATIVE_CHANNEL].round().numpy()
        selected = spatial[SELECTED_CHANNEL].round().numpy()
        entry["friendly"] = _pack_mask(player_relative == PR_FRIENDLY)
        entry["enemy"] = _pack_mask(player_relative == PR_ENEMY)
        entry["selected"] = _pack_mask(selected > 0.5)
        entry["entities"] = int(policy_input["entity_mask"].sum())
        entry["selection"] = int(policy_input["selection_mask"].sum())
        entry["feedback"] = [
            round(float(value), 3)
            for value in policy_input["action_feedback_tokens"]
            .float()
            .reshape(-1)
            .tolist()
        ]
    return entry


def export_trace(trace_path: str | Path) -> dict:
    """One .pt trace -> JSON-safe dict (schema_version 1)."""
    trace_path = Path(trace_path)
    payload = torch.load(str(trace_path), map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != 1:
        raise ValueError(
            f"unsupported trace format_version {version!r} in {trace_path}",
        )
    return {
        "schema_version": TRACE_JSON_SCHEMA_VERSION,
        "kind": "arch-explorer-trace",
        "run": payload.get("run_name"),
        "episode_index": int(payload["episode_index"]),
        "mode": "det" if payload.get("deterministic") else "stoch",
        "total_reward": float(payload["total_reward"]),
        "steps_total": int(payload["steps"]),
        "checkpoint_episode": payload.get("checkpoint_episode"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "source_file": trace_path.name,
        "screen": {
            "size": SCREEN_SIZE,
            "masks": ["friendly", "enemy", "selected"],
            "encoding": "base64-packbits-rowmajor",
        },
        "steps": [_step_entry(record) for record in payload["records"]],
    }


def write_trace_json(
    run_name: str,
    episode: int = 1,
    mode: str = "det",
    traces_root: str | Path = DEFAULT_TRACES_ROOT,
    out_path: str | Path | None = None,
) -> Path:
    trace_path = resolve_trace_path(run_name, episode, mode, traces_root)
    data = export_trace(trace_path)
    out = Path(out_path) if out_path else ARCH_EXPLORER_TRACE_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.analysis.trace_export",
        description="Export an eval episode trace to the arch explorer.",
    )
    parser.add_argument("run", help="Run name under the traces root.")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("det", "stoch"), default="det",
    )
    parser.add_argument(
        "--traces-root",
        default=str(DEFAULT_TRACES_ROOT),
        help="Root holding <run>/episode_traces (default: analysis_results).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: tools/viz/arch_explorer/public/trace_data.json).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available traces for the run and exit.",
    )
    args = parser.parse_args()

    if args.list:
        traces = list_traces(args.run, args.traces_root)
        if not traces:
            print(f"No traces under {args.traces_root}/{args.run}/episode_traces")
        for path in traces:
            print(path.name)
        return

    out = write_trace_json(
        args.run,
        episode=args.episode,
        mode=args.mode,
        traces_root=args.traces_root,
        out_path=args.out,
    )
    print(f"Wrote {out}")
    print(
        "Explorer trace replay: `npm run dev` picks it up immediately; for "
        "`npm run preview` re-run `npm run build`.",
    )


if __name__ == "__main__":
    main()
