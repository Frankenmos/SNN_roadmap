from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

try:
    import torchvision
except ImportError:
    # Kernel-grid views in the checkpoint tab degrade gracefully without it.
    torchvision = None

from tools.analysis import mission_control
from tools.analysis.analyze_pth import (
    collect_checkpoint_metadata,
    collect_extractor_state_rows,
    collect_time_constant_rows,
)
from tools.analysis.results import (
    GRAD_NORM_DECOMP_FIELDS,
    POLICY_MIX_FIELDS,
    SIL_FIELDS,
    TrainingAnalyzer,
)
from tools.registry.core import diff_entries, list_run_entries

# Anchor run discovery to the repo root so the dashboard works regardless
# of the CWD it is launched from (tools/analysis/dashboard.py -> repo root).
# SNN_DASHBOARD_MODELS_DIR overrides it (also used by the smoke test).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = Path(
    os.environ.get("SNN_DASHBOARD_MODELS_DIR", str(_REPO_ROOT / "models"))
)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None


def _db_fingerprint(db_path: str) -> str:
    """Staleness key for st.cache_data: mtime/size of the DB and its WAL
    sidecar, so a training run appending rows busts the cache."""
    parts = []
    for suffix in ("", "-wal"):
        try:
            stat = Path(str(db_path) + suffix).stat()
            parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append("absent")
    return "|".join(parts)


def _metric_text(value, fmt: str = ".2f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return format(number, fmt)


def _list_local_runs(models_dir: str | Path = _MODELS_DIR) -> list[str]:
    root = Path(models_dir)
    if not root.exists():
        return []
    runs = [
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "training_logs.db").exists()
    ]
    return sorted(runs, reverse=True)


def _local_checkpoint_candidates(run_name: str) -> list[str]:
    run_dir = _MODELS_DIR / run_name
    preferred = [
        run_dir / "best_checkpoint.pth",
        run_dir / "checkpoint.pth",
    ]
    extras = sorted(path for path in run_dir.glob("*.pth") if path not in preferred)
    all_paths = [path for path in preferred if path.exists()] + extras
    return [str(path) for path in all_paths]


@st.cache_data
def _persist_uploaded_file(file_bytes: bytes, filename: str, suffix: str) -> str:
    cache_dir = Path(tempfile.gettempdir()) / "snn_ppo_dashboard"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(file_bytes).hexdigest()
    safe_name = Path(filename).name or f"upload{suffix}"
    path = cache_dir / f"{digest}_{safe_name}"
    if not path.exists():
        path.write_bytes(file_bytes)
    return str(path)


@st.cache_data
def load_analysis_bundle(
    db_path: str,
    window: int,
    num_bins: int,
    win_threshold: float,
    db_fingerprint: str = "",
) -> dict:
    # db_fingerprint is unused in the body: it participates in the
    # st.cache_data key so the cache invalidates when the DB file changes.
    analyzer = TrainingAnalyzer(db_path)
    try:
        try:
            reward_components = analyzer.get_reward_components()
        except Exception:
            reward_components = pd.DataFrame()
        bundle = {
            "episodes": analyzer.get_episode_metrics(),
            "steps": analyzer.get_step_metrics(),
            "phase_mix": analyzer.action_mix_by_episode_phase(num_bins=num_bins),
            "reward_components": reward_components,
            "updates": analyzer.get_update_metrics(),
            "evals": analyzer.get_eval_metrics(),
            "eval_episodes": analyzer.get_eval_episode_metrics(),
            "action_labels": analyzer.action_labels.copy(),
            "action_semantics": analyzer.action_semantics,
            "run_config": analyzer.get_run_config(),
            "diagnosis": analyzer.diagnose(
                window=window,
                num_bins=num_bins,
                win_threshold=win_threshold,
            ),
        }
        return bundle
    finally:
        analyzer.close()


@st.cache_resource
def load_model_ckpt(ckpt_path: str):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "agent_state",
            "policy_net_state_dict",
            "model_state_dict",
            "state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def _severity_box(severity: str, message: str, knob: str) -> None:
    body = f"{message}\n\nSuggested knob: `{knob}`"
    if severity == "HIGH":
        st.error(body)
    elif severity == "MED":
        st.warning(body)
    else:
        st.info(body)


def _reward_figure(episodes_df: pd.DataFrame, rolling_stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes_df["episode_id"],
            y=episodes_df["total_reward"],
            mode="lines",
            name="Raw reward",
            line={"color": "rgba(31,119,180,0.25)"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_stats["episode_id"],
            y=rolling_stats["mean"],
            mode="lines",
            name="Rolling mean",
            line={"color": "#d62728"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_stats["episode_id"],
            y=rolling_stats["mean"] + rolling_stats["std"],
            mode="lines",
            name="+1 std",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_stats["episode_id"],
            y=rolling_stats["mean"] - rolling_stats["std"],
            mode="lines",
            name="-1 std",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.12)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Reward trajectory",
        xaxis_title="Episode",
        yaxis_title="Total reward",
    )
    return fig


def _episode_length_figure(episodes_df: pd.DataFrame, window: int) -> go.Figure:
    rolling = episodes_df["steps"].rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes_df["episode_id"],
            y=episodes_df["steps"],
            mode="lines",
            name="Raw length",
            line={"color": "rgba(44,160,44,0.25)"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=episodes_df["episode_id"],
            y=rolling,
            mode="lines",
            name=f"Rolling mean ({window})",
            line={"color": "#2ca02c"},
        )
    )
    fig.update_layout(
        title="Episode length",
        xaxis_title="Episode",
        yaxis_title="Steps",
    )
    return fig


def _reward_efficiency_figure(episodes_df: pd.DataFrame, window: int) -> go.Figure:
    efficiency = (
        episodes_df["total_reward"] / episodes_df["steps"].clip(lower=1)
    ).replace([np.inf, -np.inf], np.nan)
    rolling = efficiency.rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes_df["episode_id"],
            y=efficiency,
            mode="lines",
            name="Reward / step",
            line={"color": "rgba(148,103,189,0.25)"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=episodes_df["episode_id"],
            y=rolling,
            mode="lines",
            name=f"Rolling mean ({window})",
            line={"color": "#9467bd"},
        )
    )
    fig.update_layout(
        title="Reward efficiency",
        xaxis_title="Episode",
        yaxis_title="Reward / step",
    )
    return fig


def _oscillation_figure(cov: pd.Series) -> go.Figure:
    cov_df = cov.reset_index()
    cov_df.columns = ["episode_id", "cov"]
    fig = px.line(
        cov_df,
        x="episode_id",
        y="cov",
        title="Rolling oscillation score (CoV)",
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
    return fig


def _action_mix_figure(
    mix_df: pd.DataFrame, action_labels: dict[int, str]
) -> go.Figure:
    action_ids = sorted(action_labels)
    pivot = mix_df.pivot(index="bin", columns="action", values="prob").fillna(0.0)
    pivot = pivot.reindex(columns=action_ids, fill_value=0.0)
    fig = go.Figure()
    for action_id in action_ids:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[action_id],
                mode="lines",
                stackgroup="one",
                name=action_labels.get(action_id, str(action_id)),
            )
        )
    fig.update_layout(
        title="Action mix over time",
        xaxis_title="Episode bin",
        yaxis_title="Share",
        yaxis_range=[0, 1],
    )
    return fig


def _phase_action_mix_figure(
    phase_mix_df: pd.DataFrame,
    phase: str,
    action_labels: dict[int, str],
) -> go.Figure:
    phase_df = phase_mix_df[phase_mix_df["phase"] == phase]
    title = f"{phase.title()}-phase action mix"
    if phase_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    action_ids = sorted(action_labels)
    pivot = phase_df.pivot(index="bin", columns="action", values="prob").fillna(0.0)
    pivot = pivot.reindex(columns=action_ids, fill_value=0.0)
    fig = go.Figure()
    for action_id in action_ids:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[action_id],
                mode="lines",
                stackgroup="one",
                name=action_labels.get(action_id, str(action_id)),
                showlegend=(phase == "early"),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Episode bin",
        yaxis_title="Share",
        yaxis_range=[0, 1],
    )
    return fig


def _action_heatmap_figure(
    steps_df: pd.DataFrame,
    num_bins: int,
    action_labels: dict[int, str],
) -> go.Figure:
    if steps_df.empty:
        return go.Figure()
    max_ep = int(steps_df["episode_id"].max())
    bin_size = max(1, max_ep // max(1, num_bins))
    steps = steps_df.copy()
    steps["episode_bin"] = (steps["episode_id"] // bin_size) * bin_size
    counts = steps.groupby(["episode_bin", "action"]).size().reset_index(name="count")
    totals = steps.groupby("episode_bin").size().reset_index(name="total")
    counts = counts.merge(totals, on="episode_bin")
    counts["probability"] = counts["count"] / counts["total"]
    counts["action_label"] = counts["action"].map(
        lambda action_id: action_labels.get(int(action_id), str(action_id))
    )
    fig = px.density_heatmap(
        counts,
        x="episode_bin",
        y="action_label",
        z="probability",
        nbinsx=num_bins,
        title="Action probability heatmap",
        color_continuous_scale="Viridis",
        labels={
            "episode_bin": "Episode bin",
            "action_label": "Action",
            "probability": "Probability",
        },
    )
    return fig


def _move_target_heatmap_figure(
    steps_df: pd.DataFrame,
    action_semantics: str,
    noop_action_id: int,
) -> go.Figure:
    required = {"action", "move_x", "move_y"}
    if steps_df.empty or not required.issubset(set(steps_df.columns)):
        return go.Figure()
    if action_semantics == "smart_screen_v2":
        spatial_steps = steps_df[
            (steps_df["action"] != noop_action_id)
            & steps_df["move_x"].notna()
            & steps_df["move_y"].notna()
        ]
        title = "Smart target heatmap"
    else:
        spatial_steps = steps_df[
            (steps_df["action"] == 1)
            & steps_df["move_x"].notna()
            & steps_df["move_y"].notna()
        ]
        title = "Move target heatmap"
    if spatial_steps.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    fig = px.density_heatmap(
        spatial_steps,
        x="move_x",
        y="move_y",
        nbinsx=32,
        nbinsy=32,
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_title="move_x", yaxis_title="move_y")
    return fig


def _entropy_figure(entropy_df: pd.DataFrame) -> go.Figure:
    action_count = 0
    if not entropy_df.empty and "action_count" in entropy_df.columns:
        action_count = int(entropy_df["action_count"].iloc[0])
    action_count = max(2, action_count)
    fig = px.line(
        entropy_df,
        x="bin",
        y="entropy",
        title="Empirical action entropy",
    )
    fig.add_hline(y=float(np.log(action_count)), line_dash="dot", line_color="gray")
    fig.add_hline(y=0.1, line_dash="dot", line_color="red")
    fig.update_layout(xaxis_title="Episode bin", yaxis_title="Entropy")
    return fig


def _reward_components_figure(
    reward_components_df: pd.DataFrame, window: int
) -> go.Figure | None:
    if reward_components_df.empty:
        return None
    reward_cols = [
        column
        for column in reward_components_df.columns
        if "reward" in column and column not in {"total_reward", "episode_id"}
    ]
    if not reward_cols:
        return None
    per_ep = (
        reward_components_df.groupby("episode_id")[reward_cols].mean().reset_index()
    )
    for column in reward_cols:
        per_ep[column] = per_ep[column].rolling(window, min_periods=1).mean()
    melted = per_ep.melt(id_vars="episode_id", value_vars=reward_cols)
    fig = px.line(
        melted,
        x="episode_id",
        y="value",
        color="variable",
        title="Reward component trends",
    )
    fig.update_layout(xaxis_title="Episode", yaxis_title="Rolling mean value")
    return fig


def _ppo_metric_figure(ppo_updates_df: pd.DataFrame, metric: str) -> go.Figure:
    x_axis = (
        "update_id" if "update_id" in ppo_updates_df.columns else ppo_updates_df.index
    )
    fig = px.line(
        ppo_updates_df,
        x=x_axis,
        y=metric,
        title=f"{metric} over PPO updates",
    )
    fig.update_layout(xaxis_title="PPO update", yaxis_title=metric)
    return fig


def _speed_scatter_figure(
    ppo_updates_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    title: str,
) -> go.Figure:
    fig = px.scatter(
        ppo_updates_df,
        x=x_metric,
        y=y_metric,
        title=title,
    )
    fig.update_layout(xaxis_title=x_metric, yaxis_title=y_metric)
    return fig


def _timing_breakdown_figure(ppo_updates_df: pd.DataFrame) -> go.Figure:
    x_axis = (
        "update_id" if "update_id" in ppo_updates_df.columns else ppo_updates_df.index
    )
    timing_cols = [
        column
        for column in [
            "rollout_wall_seconds",
            "ray_get_wall_seconds",
            "update_wall_seconds",
            "cpu_to_gpu_transfer_wall_seconds",
            "chunk_pack_wall_seconds",
            "replay_forward_wall_seconds",
            "backward_optimizer_wall_seconds",
            "checkpoint_wall_seconds",
        ]
        if column in ppo_updates_df.columns and ppo_updates_df[column].notna().any()
    ]
    if not timing_cols:
        return go.Figure()
    plot_df = ppo_updates_df[
        [*([x_axis] if isinstance(x_axis, str) else []), *timing_cols]
    ].copy()
    if not isinstance(x_axis, str):
        plot_df = plot_df.assign(update_index=np.asarray(x_axis))
        x_axis = "update_index"
    melted = plot_df.melt(id_vars=x_axis, value_vars=timing_cols)
    fig = px.line(
        melted,
        x=x_axis,
        y="value",
        color="variable",
        title="Ray / learner timing breakdown",
    )
    fig.update_layout(xaxis_title="PPO update", yaxis_title="seconds")
    return fig


def _eval_figure(eval_runs_df: pd.DataFrame) -> go.Figure:
    if eval_runs_df.empty:
        return go.Figure()

    fig = go.Figure()
    has_det = "deterministic" in eval_runs_df.columns
    groups = (
        [("all", eval_runs_df)]
        if not has_det
        else [
            (
                "stochastic",
                eval_runs_df[eval_runs_df["deterministic"].fillna(0).astype(int) == 0],
            ),
            (
                "deterministic",
                eval_runs_df[eval_runs_df["deterministic"].fillna(0).astype(int) == 1],
            ),
        ]
    )
    colors = {
        "all": "#1f77b4",
        "stochastic": "#1f77b4",
        "deterministic": "#d62728",
    }
    fills = {
        "all": "rgba(31,119,180,0.10)",
        "stochastic": "rgba(31,119,180,0.10)",
        "deterministic": "rgba(214,39,40,0.10)",
    }
    for label, subset in groups:
        if subset.empty:
            continue
        if {"min_reward", "max_reward"}.issubset(set(subset.columns)):
            fig.add_trace(
                go.Scatter(
                    x=subset["episode_index"],
                    y=subset["max_reward"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=subset["episode_index"],
                    y=subset["min_reward"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=fills[label],
                    hoverinfo="skip",
                    name=f"{label.title()} min/max",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=subset["episode_index"],
                y=subset["mean_reward"],
                mode="lines+markers",
                name=f"{label.title()} mean",
                line={"color": colors[label]},
                error_y={
                    "type": "data",
                    "array": subset["std_reward"],
                    "visible": True,
                }
                if "std_reward" in subset.columns
                else None,
            )
        )
    fig.update_layout(xaxis_title="Episode", yaxis_title="Mean reward")
    return fig


def _eval_gap_figure(eval_runs_df: pd.DataFrame) -> go.Figure | None:
    required = {"episode_index", "mean_reward", "deterministic"}
    if eval_runs_df.empty or not required.issubset(set(eval_runs_df.columns)):
        return None
    pivot = eval_runs_df.copy()
    pivot["mode"] = np.where(
        pivot["deterministic"].fillna(0).astype(int) == 1,
        "deterministic",
        "stochastic",
    )
    pivot = pivot.pivot_table(
        index="episode_index",
        columns="mode",
        values="mean_reward",
        aggfunc="last",
    )
    if not {"deterministic", "stochastic"}.issubset(set(pivot.columns)):
        return None
    gap = (pivot["stochastic"] - pivot["deterministic"]).reset_index(name="gap")
    fig = px.line(
        gap,
        x="episode_index",
        y="gap",
        title="Eval reward gap (stochastic - deterministic)",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Episode", yaxis_title="Reward gap")
    return fig


_POLICY_COUNT_LABELS = {
    "rollout_policy_no_op_count": "no_op",
    "rollout_policy_left_click_count": "left_click",
    "rollout_policy_right_click_count": "right_click",
}
_FEEDBACK_COUNT_LABELS = {
    "rollout_feedback_near_enemy_smart_count": "near_enemy",
    "rollout_feedback_moved_toward_target_count": "moved_toward_target",
    "rollout_feedback_enemy_health_drop_after_smart_count": "enemy_health_drop",
    "rollout_feedback_null_unclear_smart_count": "null_unclear",
}


def _update_axis(ppo_updates_df: pd.DataFrame) -> pd.Series:
    if "update_id" in ppo_updates_df.columns:
        return ppo_updates_df["update_id"]
    return pd.Series(np.arange(len(ppo_updates_df)), index=ppo_updates_df.index)


def _update_action_mix_figure(ppo_updates_df: pd.DataFrame) -> go.Figure | None:
    """Action mix per PPO update from rollout_policy_*_count aggregates.

    This is the Ray-path replacement for the step-based action-mix panels:
    the Ray trainer logs per-update action counts instead of STEP rows.
    """
    cols = [c for c in _POLICY_COUNT_LABELS if c in ppo_updates_df.columns]
    if ppo_updates_df.empty or not cols:
        return None
    counts = ppo_updates_df[cols].fillna(0.0)
    totals = counts.sum(axis=1)
    if float(totals.sum()) <= 0.0:
        return None
    shares = counts.div(totals.clip(lower=1.0), axis=0)
    x = _update_axis(ppo_updates_df)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=shares[col],
                mode="lines",
                stackgroup="one",
                name=_POLICY_COUNT_LABELS[col],
            )
        )
    fig.update_layout(
        title="Rollout action mix per PPO update",
        xaxis_title="PPO update",
        yaxis_title="Share of rollout steps",
        yaxis_range=[0, 1],
    )
    return fig


def _feedback_quality_figure(ppo_updates_df: pd.DataFrame) -> go.Figure | None:
    """Click-outcome feedback per update, as a share of executed Smart
    clicks (rollout_feedback_smart_executed_count)."""
    executed_col = "rollout_feedback_smart_executed_count"
    cols = [c for c in _FEEDBACK_COUNT_LABELS if c in ppo_updates_df.columns]
    if ppo_updates_df.empty or executed_col not in ppo_updates_df.columns or not cols:
        return None
    executed = ppo_updates_df[executed_col].fillna(0.0)
    if float(executed.sum()) <= 0.0:
        return None
    x = _update_axis(ppo_updates_df)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ppo_updates_df[col].fillna(0.0) / executed.clip(lower=1.0),
                mode="lines",
                name=_FEEDBACK_COUNT_LABELS[col],
            )
        )
    fig.update_layout(
        title="Click feedback quality (share of executed Smart clicks)",
        xaxis_title="PPO update",
        yaxis_title="Share of executed clicks",
    )
    return fig


def _grad_norm_decomposition_figure(
    ppo_updates_df: pd.DataFrame,
) -> go.Figure | None:
    """Per-module gradient norms (trunk / actor / critic / target heads)
    next to the total grad_norm, log scale."""
    decomp_cols = [
        c
        for c in GRAD_NORM_DECOMP_FIELDS
        if c in ppo_updates_df.columns and ppo_updates_df[c].notna().any()
    ]
    if ppo_updates_df.empty or not decomp_cols:
        return None
    cols = decomp_cols
    if "grad_norm" in ppo_updates_df.columns:
        cols = ["grad_norm", *decomp_cols]
    x = _update_axis(ppo_updates_df)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ppo_updates_df[col],
                mode="lines",
                name=col.replace("grad_norm_", "") if col != "grad_norm" else "total",
            )
        )
    fig.update_layout(
        title="Gradient norm decomposition",
        xaxis_title="PPO update",
        yaxis_title="Gradient norm (log)",
        yaxis_type="log",
    )
    return fig


def _render_sil_health(ppo_updates_df: pd.DataFrame) -> None:
    """SIL trophy-buffer health from ppo_updates sil_* columns. Rendered
    only when the run actually logged SIL passes."""
    cols = [
        c
        for c in SIL_FIELDS
        if c in ppo_updates_df.columns and ppo_updates_df[c].notna().any()
    ]
    if not cols:
        return
    sil = ppo_updates_df[ppo_updates_df[cols[0]].notna()]
    if sil.empty:
        return
    st.markdown("#### SIL health")
    latest = sil.iloc[-1]
    metric_cols = st.columns(4)
    for slot, (column, label, fmt) in enumerate(
        [
            ("sil_buffer_size", "Trophy buffer size", ".0f"),
            ("sil_gate_open_fraction", "Gate open fraction", ".3f"),
            ("sil_steps_replayed", "Steps replayed (last)", ".0f"),
            ("sil_grad_norm", "SIL grad norm (last)", ".2f"),
        ]
    ):
        if column in sil.columns:
            metric_cols[slot].metric(label, _metric_text(latest.get(column), fmt))
    chart_specs = [
        ("sil_loss", "SIL loss"),
        ("sil_gate_open_fraction", "SIL gate open fraction"),
        ("sil_buffer_size", "SIL buffer size"),
        ("sil_steps_replayed", "SIL steps replayed"),
    ]
    chart_specs = [
        (column, title)
        for column, title in chart_specs
        if column in sil.columns and sil[column].notna().any()
    ]
    chart_columns = st.columns(2)
    x = _update_axis(sil)
    for index, (column, title) in enumerate(chart_specs):
        fig = px.line(x=x, y=sil[column], title=title)
        fig.update_layout(xaxis_title="PPO update", yaxis_title=column)
        with chart_columns[index % 2]:
            st.plotly_chart(fig, width="stretch")


def _registry_fingerprint(run_name: str) -> str:
    """Cache key over the run's .pth artifacts (names + mtime + size)."""
    run_dir = _MODELS_DIR / run_name
    paths = [run_dir / "checkpoint.pth", run_dir / "best_checkpoint.pth"]
    snapshot_dir = run_dir / "snapshots"
    if snapshot_dir.is_dir():
        paths.extend(sorted(snapshot_dir.glob("policy_u*.pth")))
    parts = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            continue
    return "|".join(parts)


@st.cache_data
def _load_registry_rows(run_name: str, fingerprint: str = "") -> list[dict]:
    # fingerprint only keys the cache (files may be appended by training).
    entries = list_run_entries(run_name, models_dir=_MODELS_DIR)
    rows = []
    for entry in entries:
        metadata = entry.metadata
        git_commit = metadata.get("git_commit")
        rows.append(
            {
                "file": entry.name,
                "kind": entry.kind,
                "policy_version": entry.policy_version,
                "episode": metadata.get("episode"),
                "size_mib": round(entry.size_mib, 2),
                "eval_mean": entry.eval_mean,
                "eval_at_version": entry.eval_policy_version,
                "git_commit": str(git_commit)[:8] if git_commit else None,
                "config_hash": metadata.get("config_hash"),
                "saved": metadata.get("wall_time_iso"),
                "path": str(entry.path),
            },
        )
    return rows


@st.cache_data
def _load_registry_diff(path_a: str, path_b: str, fingerprint: str = "") -> dict:
    return diff_entries(path_a, path_b)


def _render_snapshots_tab(selected_run: str | None) -> None:
    st.subheader("Snapshot registry")
    if not selected_run:
        st.info(
            "The snapshot registry needs a local run (upload mode has no run directory)."
        )
        return
    try:
        rows = _load_registry_rows(
            selected_run,
            _registry_fingerprint(selected_run),
        )
    except Exception as exc:
        st.error(f"Failed to read registry for `{selected_run}`: {exc}")
        return
    if not rows:
        st.info(
            "No .pth artifacts found for this run. To record snapshot "
            "lineage, enable the `snapshot_*` keys in config.yaml "
            "(distributed section). CLI: `python -m tools.registry`."
        )
        return

    st.dataframe(
        pd.DataFrame(rows).drop(columns=["path"]),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "CLI: `python -m tools.registry list <run>` | `show <ref>` | "
        "`diff <a> <b>` (refs accept `<run>:u<N>`, `:checkpoint`, `:best`)."
    )

    if len(rows) < 2:
        st.info("Need at least two artifacts to diff.")
        return
    st.markdown("#### Diff two artifacts")
    labels = [row["file"] for row in rows]
    path_by_label = {row["file"]: row["path"] for row in rows}
    col_a, col_b = st.columns(2)
    with col_a:
        label_a = st.selectbox("A (baseline)", labels, index=0)
    with col_b:
        label_b = st.selectbox("B (comparison)", labels, index=len(labels) - 1)
    if label_a == label_b:
        st.info("Pick two different artifacts.")
        return
    try:
        diff = _load_registry_diff(
            path_by_label[label_a],
            path_by_label[label_b],
            _registry_fingerprint(selected_run),
        )
    except Exception as exc:
        st.error(f"Diff failed: {exc}")
        return

    unchanged = sum(1 for row in diff["layers"] if row["l2"] == 0.0)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Global L2 delta", _metric_text(diff["total_l2"], ".4f"))
    metric_cols[1].metric("Layers compared", f"{len(diff['layers'])}")
    metric_cols[2].metric("Bit-identical layers", f"{unchanged}")

    if diff["metadata_diff"]:
        st.markdown("**Metadata differences (A -> B)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"key": key, "A": str(value_a), "B": str(value_b)}
                    for key, (value_a, value_b) in diff["metadata_diff"].items()
                ],
            ),
            width="stretch",
            hide_index=True,
        )
    config_diff = diff.get("config_diff") or {}
    if config_diff.get("changed"):
        st.markdown("**Config differences (A -> B)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"key": key, "A": str(value_a), "B": str(value_b)}
                    for key, (value_a, value_b) in config_diff["changed"].items()
                ],
            ),
            width="stretch",
            hide_index=True,
        )
    if diff["layers"]:
        st.markdown("**Top layer deltas (by L2)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "layer": row["name"],
                        "shape": str(row["shape"]),
                        "l2": row["l2"],
                        "rel_l2": row["rel_l2"],
                        "cosine": row["cosine"],
                    }
                    for row in diff["layers"][:40]
                ],
            ),
            width="stretch",
            hide_index=True,
        )
    for label, key in (("Only in A", "only_in_a"), ("Only in B", "only_in_b")):
        names = diff.get(key) or []
        if names:
            suffix = " ..." if len(names) > 10 else ""
            st.caption(f"{label}: {', '.join(names[:10])}{suffix}")


def _render_checkpoint_panel(ckpt_path: str | None) -> None:
    st.subheader("Checkpoint Introspection")
    if not ckpt_path:
        st.info("No checkpoint selected.")
        return

    try:
        ckpt = load_model_ckpt(ckpt_path)
    except Exception as exc:
        st.error(f"Failed to load checkpoint `{ckpt_path}`: {exc}")
        return
    state_dict = get_state_dict(ckpt)
    st.caption(f"Checkpoint: `{ckpt_path}`")

    if state_dict is None:
        st.error("Could not identify a state_dict in this checkpoint.")
        return

    metadata = collect_checkpoint_metadata(ckpt)
    if metadata:
        st.markdown("#### Checkpoint metadata")
        meta_df = pd.DataFrame(
            [{"key": key, "value": value} for key, value in sorted(metadata.items())],
        )
        st.dataframe(meta_df, width="stretch", hide_index=True)

    time_constant_rows = collect_time_constant_rows(state_dict)
    if time_constant_rows:
        st.markdown("#### Learned alpha / beta")
        st.dataframe(
            pd.DataFrame(time_constant_rows),
            width="stretch",
            hide_index=True,
        )

    extractor_rows = collect_extractor_state_rows(ckpt)
    if any(summary["rows"] for summary in extractor_rows.values()):
        st.markdown("#### Extractor normalizers")
        normalizer_cols = st.columns(2)
        for index, (normalizer_name, summary) in enumerate(extractor_rows.items()):
            with normalizer_cols[index % 2]:
                st.caption(
                    f"{normalizer_name}: count={summary['count']:.1f}, warm={summary['warm']}",
                )
                if summary["rows"]:
                    st.dataframe(
                        pd.DataFrame(summary["rows"]),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No saved stats.")

    layer_names = list(state_dict.keys())
    selected_layer = st.selectbox("Layer / parameter", layer_names)
    tensor = state_dict[selected_layer]
    tensor_float = tensor.float()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Shape: {tuple(tensor.shape)}")
        st.write(f"Dtype: {tensor.dtype}")
        std_value = (
            tensor_float.std(unbiased=False).item() if tensor_float.numel() > 1 else 0.0
        )
        st.write(f"Mean: {tensor_float.mean().item():.4f}")
        st.write(f"Std: {std_value:.4f}")
        st.write(f"Min: {tensor_float.min().item():.4f}")
        st.write(f"Max: {tensor_float.max().item():.4f}")

    with col2:
        flat_tensor = tensor_float.cpu().numpy().flatten()
        fig_hist = px.histogram(
            x=flat_tensor,
            nbins=50,
            title=f"Weight distribution: {selected_layer}",
            log_y=True,
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("#### Tensor visualization")
    if tensor.ndim == 4:
        grid_modes = (
            []
            if torchvision is None
            else ["Grid (First Input Channel)", "Grid (First Output Channel)"]
        )
        if torchvision is None:
            st.info(
                "torchvision is not installed - kernel grid views are "
                "unavailable. Showing the slice explorer only."
            )
        view_mode = st.radio(
            "View mode",
            [*grid_modes, "Slice Explorer"],
        )
        if view_mode == "Grid (First Input Channel)":
            kernels = tensor[:, 0:1, :, :].float()
            min_v, max_v = kernels.min(), kernels.max()
            if max_v > min_v:
                kernels = (kernels - min_v) / (max_v - min_v)
            grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            st.image(grid_np, caption="Kernels for input channel 0", width=600)
        elif view_mode == "Grid (First Output Channel)":
            kernels = tensor[0:1, :, :, :].permute(1, 0, 2, 3).float()
            min_v, max_v = kernels.min(), kernels.max()
            if max_v > min_v:
                kernels = (kernels - min_v) / (max_v - min_v)
            grid = torchvision.utils.make_grid(kernels, nrow=8, padding=1)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            st.image(grid_np, caption="Kernels for output channel 0", width=600)
        else:
            out_channels = tensor.shape[0]
            in_channels = tensor.shape[1]
            c1, c2 = st.columns(2)
            sel_out = c1.slider("Output channel", 0, out_channels - 1, 0)
            sel_in = c2.slider("Input channel", 0, in_channels - 1, 0)
            kernel = tensor[sel_out, sel_in].float().cpu().numpy()
            fig_kernel = px.imshow(
                kernel,
                color_continuous_scale="RdBu",
                title=f"Kernel [{sel_out}, {sel_in}]",
            )
            st.plotly_chart(fig_kernel, width="content")
    elif tensor.ndim == 2:
        matrix_slice = tensor_float.cpu().numpy()
        if tensor.numel() > 10000:
            st.warning("Matrix is large; showing the top-left 100x100 slice.")
            matrix_slice = matrix_slice[:100, :100]
        fig_matrix = px.imshow(
            matrix_slice,
            color_continuous_scale="RdBu",
            title="Weight heatmap",
        )
        st.plotly_chart(fig_matrix, width="stretch")
    else:
        st.info("No spatial visualization for this tensor rank.")


def _models_dir_fingerprint() -> str:
    """Cache key over every run DB's mtime/size, so the overview table
    refreshes when any run advances."""
    parts = []
    try:
        for run_dir in sorted(_MODELS_DIR.iterdir()):
            db_path = run_dir / "training_logs.db"
            if db_path.exists():
                stat = db_path.stat()
                parts.append(f"{run_dir.name}:{stat.st_mtime_ns}:{stat.st_size}")
    except OSError:
        pass
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _load_runs_overview(fingerprint: str) -> pd.DataFrame:
    del fingerprint  # cache key only
    rows = []
    for overview in mission_control.list_runs_overview(_MODELS_DIR):
        share = overview.right_click_share_last
        rows.append(
            {
                "run": overview.name,
                "updates": overview.updates_total,
                "last update": overview.last_update_index,
                "right-click share (last)": (
                    None if share is None else round(share, 3)
                ),
                "last eval": overview.last_eval_mean,
                "eval @ version": overview.last_eval_policy_version,
                "snapshots": overview.snapshot_count,
                "checkpoint": overview.has_checkpoint,
                "best": overview.has_best,
                "DB modified": overview.db_modified_iso,
            }
        )
    return pd.DataFrame(rows)


def _render_mission_control_tab() -> None:
    st.caption(
        "Read-only mission control: this dashboard never launches or stops "
        "processes. Generated commands are for your own terminal."
    )

    st.subheader("All runs")
    overview_df = _load_runs_overview(_models_dir_fingerprint())
    if overview_df.empty:
        st.info(f"No runs found under {_MODELS_DIR}.")
    else:
        st.dataframe(overview_df, width="stretch", hide_index=True)

    st.subheader("Config diff")
    config_runs = mission_control.runs_with_config(_MODELS_DIR)
    if len(config_runs) < 2:
        st.caption("Need at least two runs with effective_config.json to diff.")
    else:
        col_a, col_b = st.columns(2)
        run_a = col_a.selectbox("Run A", config_runs, index=0, key="mc_diff_a")
        run_b = col_b.selectbox(
            "Run B",
            config_runs,
            index=len(config_runs) - 1,
            key="mc_diff_b",
        )
        diff = mission_control.diff_run_configs(run_a, run_b, _MODELS_DIR)
        if diff.get("error"):
            st.warning(diff["error"])
        else:
            if diff["changed"]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"key": key, "A": repr(a), "B": repr(b)}
                            for key, (a, b) in diff["changed"].items()
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No changed keys between these runs.")
            for label, keys in (
                ("Only in A", diff["only_a"]),
                ("Only in B", diff["only_b"]),
            ):
                if keys:
                    st.caption(f"{label}: {', '.join(keys)}")

    st.subheader("Launch command generator")
    st.caption(
        "Flags mirror distributed/ray_train.py; anything left at its "
        "default falls back to config.yaml exactly as the trainer would. "
        "Snapshot recording is enabled by uncommenting the snapshot_* keys "
        "under distributed: in config.yaml."
    )
    col1, col2, col3 = st.columns(3)
    run_name = col1.text_input(
        "--run-name",
        value="",
        placeholder="e.g. banana_glasses_v8_...",
        key="mc_run_name",
    )
    num_actors = col2.number_input(
        "--num-actors",
        min_value=1,
        max_value=64,
        value=10,
        key="mc_actors",
    )
    max_updates = col3.number_input(
        "--max-updates (0 = config default)",
        min_value=0,
        value=0,
        key="mc_max_updates",
    )
    config_path = st.text_input(
        "--config (blank = repo config.yaml / SNN_CONFIG_PATH)",
        value="",
        key="mc_config",
    )
    command = mission_control.build_launch_command(
        run_name=run_name.strip(),
        num_actors=int(num_actors),
        max_updates=int(max_updates) or None,
        config_path=config_path.strip() or None,
    )
    st.code(command, language="powershell")
    if not run_name.strip():
        st.caption(
            "Without --run-name the trainer uses environment.run_name or a timestamp."
        )


def render_dashboard() -> None:
    st.set_page_config(layout="wide", page_title="SNN-PPO Analysis Dashboard")
    st.title("SNN-PPO Analysis Dashboard")
    st.sidebar.header("Data Sources")

    source_mode = st.sidebar.radio(
        "Training log source",
        ["Local run", "Upload DB"],
    )

    selected_run = None
    db_path = None

    if source_mode == "Local run":
        runs = _list_local_runs()
        if not runs:
            st.info("No local runs with `training_logs.db` found under `models/`.")
            return
        selected_run = st.sidebar.selectbox("Run", runs, index=0)
        db_path = str(_MODELS_DIR / selected_run / "training_logs.db")
        st.sidebar.caption(f"DB: `{db_path}`")
    else:
        uploaded_db = st.sidebar.file_uploader(
            "Upload training log (.db)",
            type="db",
        )
        if uploaded_db is not None:
            db_path = _persist_uploaded_file(
                uploaded_db.getvalue(),
                uploaded_db.name,
                ".db",
            )

    window = st.sidebar.slider("Rolling window", min_value=5, max_value=200, value=100)
    num_bins = st.sidebar.slider("Action bins", min_value=10, max_value=100, value=50)
    win_threshold = st.sidebar.number_input("Win threshold", value=25.0, step=25.0)

    uploaded_ckpt = st.sidebar.file_uploader(
        "Optional checkpoint (.pth)",
        type="pth",
    )
    ckpt_path = None
    if uploaded_ckpt is not None:
        ckpt_path = _persist_uploaded_file(
            uploaded_ckpt.getvalue(),
            uploaded_ckpt.name,
            ".pth",
        )
    elif selected_run is not None:
        candidates = _local_checkpoint_candidates(selected_run)
        if candidates:
            ckpt_path = st.sidebar.selectbox("Checkpoint", candidates, index=0)

    if not db_path:
        st.info("Choose a local run or upload a `training_logs.db` file.")
        return

    with st.spinner("Loading analysis..."):
        try:
            bundle = load_analysis_bundle(
                db_path,
                window,
                num_bins,
                win_threshold,
                db_fingerprint=_db_fingerprint(db_path),
            )
        except Exception as exc:
            st.error(f"Failed to load `{db_path}`: {exc}")
            return

    episodes_df = bundle["episodes"]
    steps_df = bundle["steps"]
    phase_mix_df = bundle["phase_mix"]
    reward_components_df = bundle["reward_components"]
    ppo_updates_df = bundle["updates"]
    eval_runs_df = bundle["evals"]
    eval_episodes_df = bundle.get("eval_episodes", pd.DataFrame())
    action_labels = bundle["action_labels"]
    action_semantics = bundle["action_semantics"]
    diagnosis = bundle["diagnosis"]

    summary = diagnosis["summary"]
    rolling_stats = diagnosis["rolling_stats"]
    entropy_df = diagnosis["entropy_series"]
    mix_df = diagnosis["action_mix"]
    action_shift = diagnosis["action_shift"]
    plateau_ep = diagnosis["plateau_episode"]
    late_cov = diagnosis["late_cov"]
    if episodes_df.empty:
        cov_series = pd.Series(dtype=float)
    else:
        cov_series = episodes_df["total_reward"].rolling(
            50, min_periods=1
        ).std().fillna(0) / (
            episodes_df["total_reward"].rolling(50, min_periods=1).mean().abs() + 1e-6
        )
        cov_series.index = episodes_df["episode_id"]

    if len(steps_df) > 1_000_000:
        st.warning(
            f"Large dataset detected ({len(steps_df):,} step rows). Some plots may take a moment."
        )

    metrics = st.columns(5)
    metrics[0].metric("Episodes", f"{summary['total_episodes']}")
    metrics[1].metric("Final 100 avg", _metric_text(summary["final_100_avg_reward"]))
    metrics[2].metric("Max reward", _metric_text(summary["max_total_reward"]))
    metrics[3].metric(
        "Plateau",
        f"ep {plateau_ep}" if plateau_ep is not None else "none",
    )
    if not eval_runs_df.empty:
        latest_eval = eval_runs_df.iloc[-1]
        metrics[4].metric(
            "Latest eval mean",
            _metric_text(latest_eval["mean_reward"]),
        )
    elif not ppo_updates_df.empty and "clip_fraction" in ppo_updates_df.columns:
        metrics[4].metric(
            "Late clip frac",
            _metric_text(ppo_updates_df["clip_fraction"].tail(20).mean(), ".3f"),
        )
    else:
        metrics[4].metric(
            "Avg ep length", _metric_text(summary["avg_episode_length"], ".1f")
        )

    st.caption(
        f"Source DB: `{db_path}` | action semantics: `{action_semantics}` "
        f"("
        + ", ".join(
            f"{action_id}={action_labels[action_id]}"
            for action_id in sorted(action_labels)
        )
        + ")"
    )

    with st.expander("Results-style diagnosis", expanded=True):
        if diagnosis["flags"]:
            for severity, message, knob in diagnosis["flags"]:
                _severity_box(severity, message, knob)
        else:
            st.success("No rule-based instability flag fired for this run.")

        info_cols = st.columns(3)
        info_cols[0].metric(
            "Late CoV",
            f"{late_cov:.3f}" if late_cov is not None else "n/a",
        )
        info_cols[1].metric(
            "Action-shift max",
            f"{float(action_shift.max()):.3f}" if not action_shift.empty else "n/a",
        )
        if not ppo_updates_df.empty and "mean_kl" in ppo_updates_df.columns:
            info_cols[2].metric(
                "Late KL",
                f"{float(ppo_updates_df['mean_kl'].tail(20).mean()):.4f}",
            )
        else:
            info_cols[2].metric("PPO updates", f"{len(ppo_updates_df)}")

    (
        tab_overview,
        tab_policy,
        tab_ppo,
        tab_rewards,
        tab_snapshots,
        tab_checkpoint,
        tab_mission,
    ) = st.tabs(
        [
            "Overview",
            "Policy",
            "PPO / Eval",
            "Reward Shaping",
            "Snapshots",
            "Checkpoint",
            "Mission Control",
        ]
    )

    with tab_overview:
        if episodes_df.empty:
            st.info(
                "No completed episodes in this DB yet. Episode-level panels "
                "will populate once episodes finish; PPO / Eval may already "
                "have data."
            )
        else:
            st.plotly_chart(
                _reward_figure(episodes_df, rolling_stats),
                width="stretch",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    _episode_length_figure(episodes_df, window),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    _oscillation_figure(cov_series),
                    width="stretch",
                )

            scatter = px.scatter(
                episodes_df,
                x="steps",
                y="total_reward",
                color="episode_id",
                title="Episode length vs reward",
                hover_data=["episode_id"],
            )
            st.plotly_chart(scatter, width="stretch")
            st.plotly_chart(
                _reward_efficiency_figure(episodes_df, window),
                width="stretch",
            )

    with tab_policy:
        update_mix_fig = _update_action_mix_figure(ppo_updates_df)
        feedback_fig = _feedback_quality_figure(ppo_updates_df)
        if update_mix_fig is not None or feedback_fig is not None:
            st.markdown("#### Rollout action mix (from `ppo_updates`)")
            if update_mix_fig is not None:
                st.plotly_chart(update_mix_fig, width="stretch")
            if feedback_fig is not None:
                st.plotly_chart(feedback_fig, width="stretch")

        if mix_df.empty:
            st.info(
                "Ray runs don't log per-step data (the Ray training path "
                "emits no STEP rows), so step-based action-mix, entropy, and "
                "target-heatmap panels are unavailable for this run. The "
                "per-update rollout aggregates above are the real signal."
                if not ppo_updates_df.empty
                else "No step data available for action-mix analysis."
            )
        else:
            st.plotly_chart(
                _action_mix_figure(mix_df, action_labels),
                width="stretch",
            )
            phase_cols = st.columns(3)
            for idx, phase in enumerate(["early", "mid", "late"]):
                with phase_cols[idx]:
                    st.plotly_chart(
                        _phase_action_mix_figure(phase_mix_df, phase, action_labels),
                        width="stretch",
                    )

            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    _action_heatmap_figure(steps_df, num_bins, action_labels),
                    width="stretch",
                )
            with right:
                st.plotly_chart(
                    _entropy_figure(entropy_df),
                    width="stretch",
                )
            st.plotly_chart(
                _move_target_heatmap_figure(
                    steps_df,
                    action_semantics=action_semantics,
                    noop_action_id=diagnosis["noop_action_id"],
                ),
                width="stretch",
            )

    with tab_ppo:
        if ppo_updates_df.empty:
            st.info("No `ppo_updates` table found in this DB.")
        else:
            metrics_to_plot = [
                "mean_kl",
                "clip_fraction",
                "explained_variance",
                "mean_entropy",
                "grad_norm",
                "lr",
                "mean_policy_loss",
                "mean_value_loss",
                "nonfinite_grad_steps",
                "skipped_optimizer_steps",
                "transitions_in_update",
                "return_mean",
                "return_std",
                "return_p10",
                "return_p50",
                "return_p90",
                "update_wall_seconds",
                "tbptt_chunks",
                "tbptt_chunk_groups",
                "tbptt_window",
                "tbptt_group_max_steps",
                "tbptt_group_mean_active_chunks",
                "tbptt_forward_calls",
                "rollout_wall_seconds",
                "ray_get_wall_seconds",
                "ray_submit_wall_seconds",
                "rollout_collect_overhead_wall_seconds",
                "rollout_steps_collected",
                "rollout_actor_count",
                "rollout_fragments_collected",
                "fragment_validation_wall_seconds",
                "learner_update_from_fragments_wall_seconds",
                "fragment_tensor_build_wall_seconds",
                "cpu_to_gpu_transfer_wall_seconds",
                "bootstrap_value_wall_seconds",
                "gae_wall_seconds",
                "tbptt_chunk_build_wall_seconds",
                "chunk_pack_wall_seconds",
                "replay_forward_wall_seconds",
                "loss_eval_wall_seconds",
                "backward_optimizer_wall_seconds",
                "ppo_epoch_wall_seconds",
                "payload_total_mib",
                "cuda_peak_allocated_gib",
                "cuda_peak_reserved_gib",
                "learner_transitions_per_second",
                "rollout_steps_per_second",
                "forward_calls_per_second",
                *POLICY_MIX_FIELDS,
                *GRAD_NORM_DECOMP_FIELDS,
                *SIL_FIELDS,
            ]
            present_metrics = [
                metric for metric in metrics_to_plot if metric in ppo_updates_df.columns
            ]
            default_metrics = [
                metric
                for metric in [
                    "mean_kl",
                    "clip_fraction",
                    "explained_variance",
                    "grad_norm",
                    "update_wall_seconds",
                ]
                if metric in present_metrics
            ]
            selected_metrics = st.multiselect(
                "PPO metrics",
                present_metrics,
                default=default_metrics or present_metrics[:2],
            )
            if selected_metrics:
                cols = st.columns(2)
                for index, metric in enumerate(selected_metrics):
                    with cols[index % 2]:
                        st.plotly_chart(
                            _ppo_metric_figure(ppo_updates_df, metric),
                            width="stretch",
                        )

            grad_decomp_fig = _grad_norm_decomposition_figure(ppo_updates_df)
            if grad_decomp_fig is not None:
                st.markdown("#### Gradient norm decomposition")
                st.plotly_chart(grad_decomp_fig, width="stretch")

            _render_sil_health(ppo_updates_df)

            speed_fields = {
                "update_wall_seconds",
                "tbptt_forward_calls",
                "transitions_in_update",
                "tbptt_chunks",
                "tbptt_chunk_groups",
                "tbptt_group_mean_active_chunks",
                "rollout_wall_seconds",
                "ray_get_wall_seconds",
                "replay_forward_wall_seconds",
                "backward_optimizer_wall_seconds",
                "learner_transitions_per_second",
                "rollout_steps_per_second",
                "forward_calls_per_second",
                "payload_total_mib",
                "cuda_peak_allocated_gib",
            }
            if speed_fields.intersection(set(ppo_updates_df.columns)):
                st.markdown("#### Ray / learner throughput")
                tail = ppo_updates_df.tail(max(1, len(ppo_updates_df) // 4)).copy()
                if {
                    "transitions_in_update",
                    "update_wall_seconds",
                }.issubset(set(tail.columns)):
                    tail["transitions_per_second"] = tail[
                        "transitions_in_update"
                    ] / tail["update_wall_seconds"].clip(lower=1.0e-6)
                if {
                    "tbptt_forward_calls",
                    "update_wall_seconds",
                }.issubset(set(tail.columns)):
                    tail["forward_calls_per_second"] = tail[
                        "tbptt_forward_calls"
                    ] / tail["update_wall_seconds"].clip(lower=1.0e-6)
                if {
                    "rollout_steps_collected",
                    "rollout_wall_seconds",
                }.issubset(set(tail.columns)):
                    tail["rollout_steps_per_second"] = tail[
                        "rollout_steps_collected"
                    ] / tail["rollout_wall_seconds"].clip(lower=1.0e-6)
                summary_cols = st.columns(4)
                if "update_wall_seconds" in tail.columns:
                    summary_cols[0].metric(
                        "Late update sec",
                        f"{float(tail['update_wall_seconds'].mean()):.2f}",
                    )
                if "transitions_per_second" in tail.columns:
                    summary_cols[1].metric(
                        "Learner steps / s",
                        f"{float(tail['transitions_per_second'].mean()):.1f}",
                    )
                elif "learner_transitions_per_second" in tail.columns:
                    summary_cols[1].metric(
                        "Learner steps / s",
                        f"{float(tail['learner_transitions_per_second'].mean()):.1f}",
                    )
                if "rollout_steps_per_second" in tail.columns:
                    summary_cols[2].metric(
                        "Rollout steps / s",
                        f"{float(tail['rollout_steps_per_second'].mean()):.1f}",
                    )
                elif "tbptt_forward_calls" in tail.columns:
                    summary_cols[2].metric(
                        "Late forward calls",
                        f"{float(tail['tbptt_forward_calls'].mean()):.1f}",
                    )
                if "tbptt_group_mean_active_chunks" in tail.columns:
                    summary_cols[3].metric(
                        "Active chunks",
                        f"{float(tail['tbptt_group_mean_active_chunks'].mean()):.2f}",
                    )

                timing_fig = _timing_breakdown_figure(ppo_updates_df)
                if timing_fig.data:
                    st.plotly_chart(timing_fig, width="stretch")

                speed_left, speed_right = st.columns(2)
                if {
                    "update_wall_seconds",
                    "transitions_in_update",
                }.issubset(set(ppo_updates_df.columns)):
                    with speed_left:
                        st.plotly_chart(
                            _speed_scatter_figure(
                                ppo_updates_df,
                                "update_wall_seconds",
                                "transitions_in_update",
                                "Transitions per update vs wall time",
                            ),
                            width="stretch",
                        )
                if {
                    "update_wall_seconds",
                    "tbptt_forward_calls",
                }.issubset(set(ppo_updates_df.columns)):
                    with speed_right:
                        st.plotly_chart(
                            _speed_scatter_figure(
                                ppo_updates_df,
                                "update_wall_seconds",
                                "tbptt_forward_calls",
                                "Forward calls vs wall time",
                            ),
                            width="stretch",
                        )

            if "nonfinite_grad_steps" in ppo_updates_df.columns:
                total_nonfinite = int(
                    ppo_updates_df["nonfinite_grad_steps"].fillna(0).sum()
                )
                total_skipped = int(
                    ppo_updates_df.get(
                        "skipped_optimizer_steps",
                        pd.Series(dtype=float),
                    )
                    .fillna(0)
                    .sum()
                )
                st.caption(
                    f"Non-finite grad steps: {total_nonfinite} | "
                    f"Skipped optimizer steps: {total_skipped}"
                )

        if eval_runs_df.empty:
            st.info("No `eval_runs` table found in this DB.")
        else:
            st.plotly_chart(_eval_figure(eval_runs_df), width="stretch")
            eval_gap_fig = _eval_gap_figure(eval_runs_df)
            if eval_gap_fig is not None:
                st.plotly_chart(eval_gap_fig, width="stretch")
            if not eval_episodes_df.empty:
                with st.expander("Individual evaluation episodes"):
                    st.dataframe(eval_episodes_df, width="stretch", hide_index=True)

    with tab_rewards:
        run_config = bundle.get("run_config")
        reward_config = (run_config or {}).get("reward")

        reward_fig = _reward_components_figure(reward_components_df, window)
        if reward_fig is None:
            st.info(
                "No per-step reward components in this DB. The Ray training "
                "path does not emit REWARD_COMP rows, so per-step shaping "
                "breakdowns are only available for local `train.py` runs."
            )
        else:
            st.caption(
                "Schema note: components use the legacy 6-column layout "
                "(v3 names: health / engagement / positioning / score / "
                "bonus / end-of-episode). For `defeat_roaches_v4` runs, the "
                "v4 action-guidance and smart-outcome terms are folded into "
                "`bonus_reward`."
            )
            st.plotly_chart(reward_fig, width="stretch")

        if not reward_components_df.empty:
            reward_cols = [
                column
                for column in reward_components_df.columns
                if "reward" in column and column not in {"total_reward", "episode_id"}
            ]
            if reward_cols:
                dist_fig = px.box(
                    reward_components_df,
                    y=reward_cols,
                    title="Reward component distribution",
                )
                st.plotly_chart(dist_fig, width="stretch")

        if isinstance(reward_config, dict) and reward_config:
            reward_name = reward_config.get("name", "unknown")
            st.markdown(
                f"#### Reward config for this run: `{reward_name}` (from `effective_config.json`)"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"parameter": key, "value": value}
                        for key, value in sorted(reward_config.items())
                        if key != "name"
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_snapshots:
        _render_snapshots_tab(selected_run)

    with tab_checkpoint:
        _render_checkpoint_panel(ckpt_path)

    with tab_mission:
        _render_mission_control_tab()


if __name__ == "__main__":
    render_dashboard()
