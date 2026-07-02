"""Analyze eval trace sidecar artifacts and optionally extract conv activations."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TOKEN_COUNT,
    ACTION_FEEDBACK_TOKEN_DIM,
    PolicyInputBatch,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_ACTION_LABELS = {
    0: "no-op",
    1: "move",
    2: "attack",
}
SMART_ACTION_LABELS = {
    0: "no-op",
    1: "smart",
}
SEMANTIC_CLICK_ACTION_LABELS = {
    0: "no-op",
    1: "left_click",
    2: "right_click",
}
SEMANTIC_CLICK_HEADS = {"token_pointer", "coarse_to_fine"}


def _cfg_defaults() -> dict:
    try:
        from Utility.config import cfg

        return {
            "run_name": getattr(cfg.environment, "run_name", "") or "",
            "analysis_dir": getattr(cfg.environment, "analysis_dir", "analysis_results"),
            "models_dir": getattr(cfg.environment, "models_dir", "models"),
            "checkpoint_filename": getattr(
                cfg.environment,
                "checkpoint_path",
                "checkpoint.pth",
            ),
            "best_checkpoint_filename": getattr(
                cfg.environment,
                "best_checkpoint_path",
                "best_checkpoint.pth",
            ),
        }
    except Exception:
        return {
            "run_name": "",
            "analysis_dir": "analysis_results",
            "models_dir": "models",
            "checkpoint_filename": "checkpoint.pth",
            "best_checkpoint_filename": "best_checkpoint.pth",
        }


def _resolve_trace_path(
    explicit_trace: str | None,
    run_name: str | None,
    mode: str,
    episode_index: int | None,
) -> Path:
    if explicit_trace:
        path = Path(explicit_trace)
        if not path.exists():
            raise FileNotFoundError(f"Trace not found: {path}")
        return path

    defaults = _cfg_defaults()
    name = run_name or defaults["run_name"]
    if not name:
        raise FileNotFoundError(
            "Pass --trace or --run-name so the trace file can be resolved.",
        )

    trace_root = Path(defaults["analysis_dir"]) / name / "episode_traces" / mode
    if not trace_root.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_root}")

    traces = sorted(trace_root.glob("episode_*.pt"))
    if not traces:
        raise FileNotFoundError(f"No trace files found under {trace_root}")

    if episode_index is not None:
        pattern = f"episode_{int(episode_index):04d}_*.pt"
        matches = sorted(trace_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No trace for episode {episode_index} found under {trace_root}",
            )
        return matches[-1]

    return max(traces, key=lambda path: path.stat().st_mtime)


def _default_output_dir(trace_path: Path) -> Path:
    return trace_path.parent / f"{trace_path.stem}_analysis"


def _ensure_numpy_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        fill = float(finite.mean()) if finite.size else 0.0
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


def _plot_grid(
    arrays: list[np.ndarray],
    *,
    title: str,
    output_path: Path,
    labels: list[str] | None = None,
    cols: int = 4,
    cmap: str = "viridis",
) -> None:
    if not arrays:
        return
    cols = max(1, int(cols))
    rows = math.ceil(len(arrays) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for axis in axes.flat:
        axis.axis("off")

    for index, arr in enumerate(arrays):
        axis = axes.flat[index]
        image = _ensure_numpy_image(arr)
        axis.imshow(image, cmap=cmap, origin="lower")
        if labels is not None and index < len(labels):
            axis.set_title(labels[index], fontsize=9)
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class EvalTraceAnalyzer:
    def __init__(self, trace_path: str | Path):
        self.trace_path = Path(trace_path)
        self.payload = torch.load(
            self.trace_path,
            map_location="cpu",
            weights_only=False,
        )
        self.records: list[dict[str, Any]] = list(self.payload.get("records", []))
        self.action_labels = self._resolve_action_labels()

    def _resolve_action_labels(self) -> dict[int, str]:
        checkpoint_path = self._resolve_checkpoint_path()
        config = (
            None
            if checkpoint_path is None
            else _load_json(checkpoint_path.with_name("effective_config.json"))
        )
        if isinstance(config, dict):
            model_cfg = config.get("model", {})
            if isinstance(model_cfg, dict):
                try:
                    if int(model_cfg.get("action_dim")) == 2:
                        return SMART_ACTION_LABELS.copy()
                    spatial_head_type = str(
                        model_cfg.get("spatial_head_type", ""),
                    ).lower()
                    if (
                        int(model_cfg.get("action_dim")) == 3
                        and spatial_head_type in SEMANTIC_CLICK_HEADS
                    ):
                        return SEMANTIC_CLICK_ACTION_LABELS.copy()
                except (TypeError, ValueError):
                    pass
        actions = {
            int(record["action"])
            for record in self.records
            if record.get("action") is not None
        }
        if actions and max(actions) <= 1:
            return SMART_ACTION_LABELS.copy()
        return DEFAULT_ACTION_LABELS.copy()

    @property
    def policy_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records
            if record.get("policy_input") is not None
        ]

    def get_step_record(self, step_index: int | None = None) -> dict[str, Any]:
        if not self.records:
            raise ValueError("Trace contains no records")
        if step_index is None:
            for record in self.records:
                if record.get("policy_input") is not None:
                    return record
            return self.records[0]
        for record in self.records:
            if int(record.get("step_index", -1)) == int(step_index):
                return record
        raise KeyError(f"Step index {step_index} not found in trace")

    def summarize(self) -> dict[str, Any]:
        action_counts = {label: 0 for label in self.action_labels.values()}
        dispatch_counts: dict[str, int] = {}
        learnable_steps = 0
        bootstrap_steps = 0
        cumulative_rewards = []
        step_rewards = []

        for record in self.records:
            action = record.get("action")
            if action in self.action_labels:
                action_counts[self.action_labels[int(action)]] += 1
            if record.get("learnable"):
                learnable_steps += 1
            if not record.get("policy_step", False):
                bootstrap_steps += 1
            dispatched = (
                record.get("dispatched_action", {}) or {}
            ).get("function_name") or "unknown"
            dispatch_counts[dispatched] = dispatch_counts.get(dispatched, 0) + 1
            cumulative_rewards.append(float(record.get("cumulative_reward", 0.0)))
            step_rewards.append(float(record.get("reward", 0.0)))

        summary = {
            "trace_path": str(self.trace_path),
            "run_name": self.payload.get("run_name"),
            "episode_index": int(self.payload.get("episode_index", 0)),
            "deterministic": bool(self.payload.get("deterministic", True)),
            "total_reward": float(self.payload.get("total_reward", 0.0)),
            "steps": int(self.payload.get("steps", len(self.records))),
            "records": len(self.records),
            "policy_steps": len(self.policy_records),
            "bootstrap_steps": int(bootstrap_steps),
            "learnable_steps": int(learnable_steps),
            "action_counts": action_counts,
            "dispatch_counts": dict(sorted(dispatch_counts.items())),
            "step_reward_mean": (
                float(np.mean(step_rewards)) if step_rewards else 0.0
            ),
            "step_reward_min": (
                float(np.min(step_rewards)) if step_rewards else 0.0
            ),
            "step_reward_max": (
                float(np.max(step_rewards)) if step_rewards else 0.0
            ),
            "final_cumulative_reward": (
                float(cumulative_rewards[-1]) if cumulative_rewards else 0.0
            ),
        }
        return summary

    def write_report(self, path: str | Path) -> Path:
        summary = self.summarize()
        lines = [
            "=" * 70,
            "Eval Trace Report",
            "=" * 70,
            "",
            f"trace_path:           {summary['trace_path']}",
            f"run_name:             {summary['run_name']}",
            f"episode_index:        {summary['episode_index']}",
            f"deterministic:        {summary['deterministic']}",
            f"total_reward:         {summary['total_reward']:.2f}",
            f"steps:                {summary['steps']}",
            f"policy_steps:         {summary['policy_steps']}",
            f"bootstrap_steps:      {summary['bootstrap_steps']}",
            f"learnable_steps:      {summary['learnable_steps']}",
            f"step_reward_mean:     {summary['step_reward_mean']:.3f}",
            f"step_reward_min/max:  {summary['step_reward_min']:.3f} / {summary['step_reward_max']:.3f}",
            "",
            "Action counts:",
        ]
        for name, count in summary["action_counts"].items():
            lines.append(f"  - {name}: {count}")
        lines.append("")
        lines.append("Dispatched action counts:")
        for name, count in summary["dispatch_counts"].items():
            lines.append(f"  - {name}: {count}")
        lines.append("")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _plot_reward_timeline(self, output_path: Path) -> None:
        steps = [int(record.get("step_index", 0)) for record in self.records]
        rewards = [float(record.get("reward", 0.0)) for record in self.records]
        cumulative = [
            float(record.get("cumulative_reward", 0.0)) for record in self.records
        ]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(steps, rewards, color="tab:blue")
        ax1.set_ylabel("step reward")
        ax1.set_title("Per-step reward")
        ax1.grid(True)
        ax2.plot(steps, cumulative, color="tab:green")
        ax2.set_xlabel("step")
        ax2.set_ylabel("cumulative reward")
        ax2.set_title("Cumulative reward")
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

    def _plot_action_timeline(self, output_path: Path) -> None:
        policy_records = self.policy_records
        if not policy_records:
            return
        steps = [int(record.get("step_index", 0)) for record in policy_records]
        actions = [int(record.get("action", 0)) for record in policy_records]
        action_ids = sorted(self.action_labels)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.scatter(steps, actions, c=actions, cmap="viridis", s=18)
        ax.set_yticks(action_ids, labels=[self.action_labels[i] for i in action_ids])
        ax.set_xlabel("step")
        ax.set_ylabel("policy action")
        ax.set_title("Policy action timeline")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

    def _plot_dispatched_action_counts(self, output_path: Path) -> None:
        summary = self.summarize()
        names = list(summary["dispatch_counts"].keys())
        counts = [summary["dispatch_counts"][name] for name in names]
        if not names:
            return
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(names, counts, color="tab:purple")
        ax.set_ylabel("count")
        ax.set_title("Dispatched action counts")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

    def _plot_target_scatter(self, output_path: Path) -> None:
        smart_x = []
        smart_y = []
        move_x = []
        move_y = []
        attack_x = []
        attack_y = []
        for record in self.policy_records:
            dispatched = (record.get("dispatched_action", {}) or {}).get("function_name")
            if dispatched == "Smart_screen":
                smart_x.append(int(record.get("move_x", 0)))
                smart_y.append(int(record.get("move_y", 0)))
            elif dispatched == "Move_screen":
                move_x.append(int(record.get("move_x", 0)))
                move_y.append(int(record.get("move_y", 0)))
            elif dispatched == "Attack_screen":
                attack_x.append(int(record.get("move_x", 0)))
                attack_y.append(int(record.get("move_y", 0)))
        fig, ax = plt.subplots(figsize=(7, 7))
        if smart_x:
            ax.scatter(smart_x, smart_y, label="smart", alpha=0.7, s=18)
        if move_x:
            ax.scatter(move_x, move_y, label="move", alpha=0.7, s=18)
        if attack_x:
            ax.scatter(attack_x, attack_y, label="attack", alpha=0.7, s=18)
        ax.set_xlim(0, 83)
        ax.set_ylim(0, 83)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Dispatched spatial targets")
        if smart_x or move_x or attack_x:
            ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

    def _plot_spatial_planes(self, output_path: Path, step_index: int | None = None) -> int:
        record = self.get_step_record(step_index=step_index)
        policy_input = record.get("policy_input")
        if policy_input is None:
            raise ValueError("Selected step has no policy_input")
        spatial = policy_input["spatial_obs"].detach().cpu().float().numpy()
        labels = [f"ch {idx:02d}" for idx in range(spatial.shape[0])]
        _plot_grid(
            [spatial[idx] for idx in range(spatial.shape[0])],
            title=f"Spatial planes at step {record['step_index']}",
            output_path=output_path,
            labels=labels,
            cols=3,
            cmap="viridis",
        )
        return int(record["step_index"])

    def _resolve_checkpoint_path(self, explicit_checkpoint: str | None = None) -> Path | None:
        if explicit_checkpoint:
            path = Path(explicit_checkpoint)
            return path if path.exists() else None
        raw = self.payload.get("checkpoint_path")
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path
        run_name = self.payload.get("run_name")
        if not run_name:
            return None
        defaults = _cfg_defaults()
        models_dir = Path(defaults["models_dir"]) / str(run_name)
        candidates = [
            models_dir / defaults["best_checkpoint_filename"],
            models_dir / defaults["checkpoint_filename"],
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _instantiate_policy(self, checkpoint_path: Path):
        from agent_core.spiking_policy import PolicyNetwork

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("agent_state")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint at {checkpoint_path} has no agent_state")

        sample_record = self.get_step_record(step_index=None)
        policy_input = sample_record.get("policy_input")
        if policy_input is None:
            raise ValueError("Trace has no policy_input record to derive model shape from")

        spatial_shape = tuple(policy_input["spatial_obs"].shape)
        vector_dim = int(policy_input["meta_vec"].shape[-1])

        effective_config = _load_json(checkpoint_path.with_name("effective_config.json")) or {}
        model_cfg = effective_config.get("model", {}) if isinstance(effective_config, dict) else {}
        action_dim = int(model_cfg.get("action_dim", max(self.action_labels) + 1))

        policy = PolicyNetwork(
            spatial_input_shape=spatial_shape,
            vector_input_dim=vector_dim,
            action_dim=action_dim,
            num_steps=int(model_cfg.get("num_steps", 1)),
            screen_size=int(model_cfg.get("screen_size", spatial_shape[-1])),
            fast_token_snn_alpha=float(
                model_cfg.get(
                    "fast_token_snn_alpha",
                    model_cfg.get("token_snn_alpha", 0.8),
                ),
            ),
            fast_token_snn_beta=float(
                model_cfg.get(
                    "fast_token_snn_beta",
                    model_cfg.get("token_snn_beta", 0.9),
                ),
            ),
            slow_token_snn_alpha=float(model_cfg.get("slow_token_snn_alpha", 0.92)),
            slow_token_snn_beta=float(model_cfg.get("slow_token_snn_beta", 0.97)),
            temporal_combine_mode=str(model_cfg.get("temporal_combine_mode", "mean")),
            attention_embed_dim=int(model_cfg.get("attention_embed_dim", 64)),
            attention_pool_size=int(model_cfg.get("attention_pool_size", 7)),
            attention_beta=float(model_cfg.get("attention_beta", 0.5)),
            spatial_head_type=str(model_cfg.get("spatial_head_type", "token_pointer")),
            coarse_grid_size=_optional_int(model_cfg.get("coarse_grid_size")),
            local_grid_size=_optional_int(model_cfg.get("local_grid_size")),
            target_decode_mode=str(model_cfg.get("target_decode_mode", "center")),
            fine_skip_connection=_config_bool(
                model_cfg.get("fine_skip_connection"),
                default=False,
            ),
            fine_skip_dim=_optional_int(model_cfg.get("fine_skip_dim")) or 32,
        )
        policy.load_state_dict(state_dict)
        policy = policy.to(device="cpu", dtype=torch.float32)
        policy.device = torch.device("cpu")
        policy.eval()
        return policy

    def _build_batch(self, record: dict[str, Any]) -> PolicyInputBatch:
        policy_input = record.get("policy_input")
        if policy_input is None:
            raise ValueError("Selected record has no policy_input")
        return PolicyInputBatch(
            spatial_obs=policy_input["spatial_obs"].unsqueeze(0).float(),
            entity_features=policy_input["entity_features"].unsqueeze(0).float(),
            entity_mask=policy_input["entity_mask"].unsqueeze(0),
            selection_features=policy_input["selection_features"].unsqueeze(0).float(),
            selection_mask=policy_input["selection_mask"].unsqueeze(0),
            action_feedback_tokens=policy_input.get(
                "action_feedback_tokens",
                torch.zeros(ACTION_FEEDBACK_TOKEN_COUNT, ACTION_FEEDBACK_TOKEN_DIM),
            ).unsqueeze(0).float(),
            meta_vec=policy_input["meta_vec"].unsqueeze(0).float(),
            state_in=None,
        )

    def _extract_conv_activations(
        self,
        *,
        step_index: int | None,
        checkpoint_path: Path | None,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint path available for activation extraction")
        record = self.get_step_record(step_index=step_index)
        batch = self._build_batch(record)
        policy = self._instantiate_policy(checkpoint_path)

        activations: dict[str, torch.Tensor] = {}
        handles = []
        for layer_name in ("conv1", "conv2", "conv3"):
            layer = getattr(policy, layer_name)

            def _capture(_module, _inputs, output, *, name=layer_name):
                activations[name] = output.detach().cpu()

            handles.append(layer.register_forward_hook(_capture))

        try:
            with torch.no_grad():
                policy.encode_step_tensors(
                    spatial_obs=batch.spatial_obs,
                    entity_features=batch.entity_features,
                    entity_mask=batch.entity_mask,
                    selection_features=batch.selection_features,
                    selection_mask=batch.selection_mask,
                    action_feedback_tokens=batch.action_feedback_tokens,
                    meta_vec=batch.meta_vec,
                    state_in=None,
                )
        finally:
            for handle in handles:
                handle.remove()

        return int(record["step_index"]), activations

    def _plot_conv_activations(
        self,
        output_dir: Path,
        *,
        step_index: int | None,
        checkpoint_path: Path | None,
        max_channels: int,
    ) -> list[str]:
        actual_step, activations = self._extract_conv_activations(
            step_index=step_index,
            checkpoint_path=checkpoint_path,
        )
        exported = []
        for layer_name in ("conv1", "conv2", "conv3"):
            tensor = activations.get(layer_name)
            if tensor is None:
                continue
            maps = tensor[0].float().numpy()
            scores = np.mean(np.abs(maps), axis=(1, 2))
            top_indices = list(np.argsort(scores)[::-1][: max(1, int(max_channels))])
            arrays = [maps[index] for index in top_indices]
            labels = [f"ch {index} | mean|a|={scores[index]:.3f}" for index in top_indices]
            filename = f"{layer_name}_activations_step_{actual_step:04d}.png"
            _plot_grid(
                arrays,
                title=f"{layer_name} activations at step {actual_step}",
                output_path=output_dir / filename,
                labels=labels,
                cols=4,
                cmap="magma",
            )
            exported.append(filename)
        return exported

    def export_panels(
        self,
        out_dir: str | Path,
        *,
        step_index: int | None = None,
        include_activations: bool = False,
        checkpoint_path: str | None = None,
        max_channels: int = 16,
    ) -> list[str]:
        output_root = Path(out_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []

        def _record(filename: str):
            exported.append(filename)

        report_path = self.write_report(output_root / "trace_report.txt")
        _record(report_path.name)

        self._plot_reward_timeline(output_root / "01_reward_timeline.png")
        _record("01_reward_timeline.png")

        self._plot_action_timeline(output_root / "02_action_timeline.png")
        _record("02_action_timeline.png")

        self._plot_dispatched_action_counts(output_root / "03_dispatched_action_counts.png")
        _record("03_dispatched_action_counts.png")

        self._plot_target_scatter(output_root / "04_spatial_targets.png")
        _record("04_spatial_targets.png")

        actual_step = self._plot_spatial_planes(
            output_root / "05_spatial_planes.png",
            step_index=step_index,
        )
        _record("05_spatial_planes.png")

        resolved_checkpoint = self._resolve_checkpoint_path(
            explicit_checkpoint=checkpoint_path,
        )
        if include_activations:
            exported.extend(
                self._plot_conv_activations(
                    output_root,
                    step_index=actual_step,
                    checkpoint_path=resolved_checkpoint,
                    max_channels=max_channels,
                ),
            )

        manifest_lines = [
            "Eval trace analysis bundle",
            f"trace={self.trace_path}",
            f"step_index={actual_step}",
            f"include_activations={include_activations}",
            f"checkpoint={resolved_checkpoint}",
            "",
            "files:",
        ] + [f"- {name}" for name in exported]
        (output_root / "manifest.txt").write_text(
            "\n".join(manifest_lines),
            encoding="utf-8",
        )
        _record("manifest.txt")
        return exported


def main():
    parser = argparse.ArgumentParser(description="Analyze eval trace sidecar artifacts.")
    parser.add_argument("--trace", default=None, help="Explicit .pt trace path.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name under analysis_results/<run_name>/episode_traces/",
    )
    parser.add_argument(
        "--mode",
        default="det",
        choices=["det", "stoch"],
        help="Trace mode when resolving from --run-name.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode index to resolve within the trace directory.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for the analysis bundle. Defaults to a sibling folder next to the trace.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="Specific step index to visualize. Defaults to the first policy-controlled step.",
    )
    parser.add_argument(
        "--activations",
        action="store_true",
        help="Also load the checkpoint and export conv1/conv2/conv3 activation maps for the selected step.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path override for --activations.",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=16,
        help="Max activation channels per conv layer to show in the exported grids.",
    )
    args = parser.parse_args()

    trace_path = _resolve_trace_path(
        explicit_trace=args.trace,
        run_name=args.run_name,
        mode=args.mode,
        episode_index=args.episode_index,
    )
    out_dir = Path(args.out) if args.out else _default_output_dir(trace_path)

    logger.info("Trace: %s", trace_path)
    logger.info("Output: %s", out_dir)

    analyzer = EvalTraceAnalyzer(trace_path)
    exported = analyzer.export_panels(
        out_dir,
        step_index=args.step_index,
        include_activations=args.activations,
        checkpoint_path=args.checkpoint,
        max_channels=args.max_channels,
    )
    logger.info("Exported %s file(s)", len(exported))


if __name__ == "__main__":
    main()
