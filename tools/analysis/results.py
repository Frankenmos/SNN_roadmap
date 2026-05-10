"""
Training run analyzer for the SNN-PPO project.

Reads training_logs.db (SQLite), produces diagnostic plots and a
rule-based instability report. Extends the older TrainingAnalyzer with:

- plateau detection
- rolling oscillation score (coefficient of variation)
- empirical action-entropy (proxy for true policy entropy, which is not
  logged)
- action-mix drift
- win-rate proxy
- diagnose() that maps observed signals to concrete hyperparameter knobs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LEGACY_ACTION_LABELS = {0: "attack", 1: "move", 2: "no-op"}
CONDITIONED_ACTION_LABELS = {0: "no-op", 1: "move", 2: "attack"}
SMART_SCREEN_ACTION_LABELS = {0: "no-op", 1: "smart"}
SEMANTIC_CLICK_ACTION_LABELS = {0: "no-op", 1: "left_click", 2: "right_click"}
STREAM_ACTION_FEEDBACK_LABELS = SEMANTIC_CLICK_ACTION_LABELS.copy()
ACTION_LABELS = CONDITIONED_ACTION_LABELS.copy()
PHASE_LABELS = ("early", "mid", "late")
SEMANTIC_CLICK_SEMANTICS = {
    "semantic_pointer_v1",
    "stream_action_feedback_v1",
    "stream_action_effect_feedback_v2",
}
RAY_THROUGHPUT_FIELDS = [
    "rollout_wall_seconds",
    "ray_get_wall_seconds",
    "ray_submit_wall_seconds",
    "rollout_collect_overhead_wall_seconds",
    "rollout_collect_waves",
    "rollout_empty_waves",
    "rollout_steps_collected",
    "rollout_actor_count",
    "rollout_fragments_collected",
    "fragment_validation_wall_seconds",
    "learner_update_from_fragments_wall_seconds",
]
LEARNER_TIMING_FIELDS = [
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
    "checkpoint_wall_seconds",
    "episode_log_enqueue_wall_seconds",
]
PAYLOAD_FIELDS = [
    "learnable_transitions_in_update",
    "fragments_in_update",
    "payload_spatial_bytes",
    "payload_state_bytes",
    "payload_total_bytes",
    "payload_total_mib",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "rollout_cache_spatial_dtype",
    "episodes_logged_in_update",
]


class TrainingAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self.action_semantics = self._infer_action_semantics()
        if self.action_semantics == "smart_screen_v2":
            self.action_labels = SMART_SCREEN_ACTION_LABELS.copy()
            self.noop_action_id = 0
        elif self.action_semantics in SEMANTIC_CLICK_SEMANTICS:
            self.action_labels = SEMANTIC_CLICK_ACTION_LABELS.copy()
            self.noop_action_id = 0
        elif self.action_semantics == "conditioned_spatial_v1":
            self.action_labels = CONDITIONED_ACTION_LABELS.copy()
            self.noop_action_id = 0
        else:
            self.action_labels = LEGACY_ACTION_LABELS.copy()
            self.noop_action_id = 2
        self.action_ids = sorted(self.action_labels)
        # Cached DataFrames — loaded lazily.
        self._episodes: Optional[pd.DataFrame] = None
        self._steps: Optional[pd.DataFrame] = None
        self._reward_components: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Connection + raw loaders
    # ------------------------------------------------------------------
    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
            self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
            ).fetchall()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            if self.conn:
                self.conn.close()
                self.conn = None
            try:
                uri = f"file:{Path(self.db_path).resolve()}?mode=ro&immutable=1"
                self.conn = sqlite3.connect(uri, uri=True)
                self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
                ).fetchall()
                logger.info(f"Connected to immutable database snapshot: {self.db_path}")
            except sqlite3.Error:
                logger.error(f"Database connection failed: {e}")
                raise e

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def _load_effective_config(self) -> Optional[dict]:
        config_path = Path(self.db_path).with_name("effective_config.json")
        if not config_path.exists():
            return None
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _infer_action_semantics(self) -> str:
        config = self._load_effective_config()
        if isinstance(config, dict):
            model_cfg = config.get("model", {})
            distributed_cfg = config.get("distributed", {})
            configured_schema = str(
                config.get("policy_input_schema")
                or distributed_cfg.get("required_policy_input_schema", "")
            ).lower()
            if isinstance(model_cfg, dict):
                # Check action_dim first (preferred signal)
                value = model_cfg.get("action_dim")
                try:
                    if int(value) == 2:
                        return "smart_screen_v2"
                    if int(value) == 3:
                        spatial_head = str(
                            model_cfg.get("spatial_head_type", ""),
                        ).lower()
                        if (
                            configured_schema
                            in {
                                "stream_action_feedback_v1",
                                "stream_action_effect_feedback_v2",
                            }
                            or spatial_head == "coarse_to_fine"
                            or int(model_cfg.get("vector_input_dim", 0) or 0) == 15
                        ):
                            return configured_schema or "stream_action_feedback_v1"
                        if spatial_head == "token_pointer":
                            return "semantic_pointer_v1"
                except (TypeError, ValueError):
                    pass

                # Fallback: check spatial_head_type + meta_input_dim combo
                # This handles runs where action_dim wasn't written to config
                spatial_head = str(model_cfg.get("spatial_head_type", "")).lower()
                meta_dim = model_cfg.get("meta_input_dim") or model_cfg.get("vector_input_dim")
                try:
                    if spatial_head == "token_pointer" and int(meta_dim) == 19:
                        return "semantic_pointer_v1"
                except (TypeError, ValueError):
                    pass

                # Older conditioned-spatial runs had larger meta_vec
                for field_name in ("vector_input_dim", "meta_input_dim"):
                    value = model_cfg.get(field_name)
                    try:
                        if int(value) >= 32:
                            return "conditioned_spatial_v1"
                    except (TypeError, ValueError):
                        continue
        return "legacy_v0"

    def _table_columns(self, table_name: str) -> set[str]:
        try:
            rows = self.conn.execute(
                f"PRAGMA table_info({table_name})",
            ).fetchall()
        except sqlite3.Error:
            return set()
        return {row[1] for row in rows}

    def get_episode_metrics(self) -> pd.DataFrame:
        if self._episodes is None:
            self._episodes = pd.read_sql_query(
                "SELECT episode_id, total_reward, average_reward, steps, timestamp "
                "FROM episodes ORDER BY episode_id",
                self.conn,
            )
        return self._episodes

    def get_step_metrics(self) -> pd.DataFrame:
        if self._steps is None:
            cols = self._table_columns("steps")
            wanted = [
                "episode_id",
                "step_number",
                "action",
                "move_x",
                "move_y",
                "reward",
                "cumulative_reward",
            ]
            present = [name for name in wanted if name in cols]
            self._steps = pd.read_sql_query(
                f"SELECT {', '.join(present)} FROM steps "
                "ORDER BY episode_id, step_number",
                self.conn,
            )
        return self._steps

    def get_reward_components(self) -> pd.DataFrame:
        # Fix from prior version: the column is episode_id, not episode.
        if self._reward_components is None:
            self._reward_components = pd.read_sql_query(
                "SELECT episode_id, step, health_reward, engagement_reward, "
                "positioning_reward, score_reward, bonus_reward, "
                "end_of_episode_reward, total_reward "
                "FROM reward_components ORDER BY episode_id, step",
                self.conn,
            )
        return self._reward_components

    def get_update_metrics(self) -> pd.DataFrame:
        """Per-PPO-update metrics. Returns an empty DataFrame when the
        table is missing (legacy DBs) or empty (new DB, no update yet)."""
        cols = self._table_columns("ppo_updates")
        if not cols:
            return pd.DataFrame()
        wanted = [
            "update_id",
            "episode_id",
            "episode_index",
            "global_update_index",
            "policy_version",
            "policy_protocol_version",
            "policy_input_schema",
            "mean_policy_loss",
            "mean_value_loss",
            "mean_entropy",
            "mean_kl",
            "clip_fraction",
            "explained_variance",
            "grad_norm",
            "lr",
            "nonfinite_grad_steps",
            "skipped_optimizer_steps",
            "transitions_in_update",
            "learnable_transitions_in_update",
            "fragments_in_update",
            "return_mean",
            "return_std",
            "return_p10",
            "return_p50",
            "return_p90",
            "entity_mask_utilization",
            "entity_count_p50",
            "entity_count_p99",
            "selection_mask_utilization",
            "update_wall_seconds",
            "tbptt_chunks",
            "tbptt_chunk_groups",
            "tbptt_window",
            "tbptt_group_max_steps",
            "tbptt_group_mean_active_chunks",
            "tbptt_forward_calls",
        ]
        wanted.extend(RAY_THROUGHPUT_FIELDS)
        wanted.extend(LEARNER_TIMING_FIELDS)
        wanted.extend(PAYLOAD_FIELDS)
        present = [name for name in wanted if name in cols]
        if not present:
            return pd.DataFrame()
        try:
            updates = pd.read_sql_query(
                f"SELECT {', '.join(present)} FROM ppo_updates ORDER BY update_id",
                self.conn,
            )
            return self._with_update_derived_metrics(updates)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _with_update_derived_metrics(updates: pd.DataFrame) -> pd.DataFrame:
        if updates.empty:
            return updates
        updates = updates.copy()
        if {
            "transitions_in_update",
            "update_wall_seconds",
        }.issubset(set(updates.columns)):
            updates["learner_transitions_per_second"] = (
                updates["transitions_in_update"]
                / updates["update_wall_seconds"].clip(lower=1.0e-6)
            )
        if {
            "rollout_steps_collected",
            "rollout_wall_seconds",
        }.issubset(set(updates.columns)):
            updates["rollout_steps_per_second"] = (
                updates["rollout_steps_collected"]
                / updates["rollout_wall_seconds"].clip(lower=1.0e-6)
            )
        elif {
            "transitions_in_update",
            "rollout_wall_seconds",
        }.issubset(set(updates.columns)):
            updates["rollout_steps_per_second"] = (
                updates["transitions_in_update"]
                / updates["rollout_wall_seconds"].clip(lower=1.0e-6)
            )
        if {
            "tbptt_forward_calls",
            "update_wall_seconds",
        }.issubset(set(updates.columns)):
            updates["forward_calls_per_second"] = (
                updates["tbptt_forward_calls"]
                / updates["update_wall_seconds"].clip(lower=1.0e-6)
            )
        for source, dest in [
            ("cuda_peak_allocated_bytes", "cuda_peak_allocated_gib"),
            ("cuda_peak_reserved_bytes", "cuda_peak_reserved_gib"),
            ("payload_total_bytes", "payload_total_gib"),
        ]:
            if source in updates.columns:
                updates[dest] = updates[source] / float(1024**3)
        return updates

    def get_eval_metrics(self) -> pd.DataFrame:
        cols = self._table_columns("eval_runs")
        if not cols:
            return pd.DataFrame()
        wanted = [
            "eval_id",
            "episode_index",
            "num_episodes",
            "mean_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "deterministic",
            "timestamp",
        ]
        present = [name for name in wanted if name in cols]
        if not present:
            return pd.DataFrame()
        try:
            return pd.read_sql_query(
                f"SELECT {', '.join(present)} FROM eval_runs ORDER BY eval_id",
                self.conn,
            )
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Analysis primitives
    # ------------------------------------------------------------------
    def rolling_reward_stats(self, window: int = 100) -> pd.DataFrame:
        ep = self.get_episode_metrics()
        out = pd.DataFrame({"episode_id": ep["episode_id"]})
        out["mean"] = ep["total_reward"].rolling(window, min_periods=1).mean()
        out["std"] = ep["total_reward"].rolling(window, min_periods=1).std().fillna(0)
        out["min"] = ep["total_reward"].rolling(window, min_periods=1).min()
        out["max"] = ep["total_reward"].rolling(window, min_periods=1).max()
        return out

    def detect_plateau(
        self,
        window: int = 100,
        slope_threshold: float = 0.01,
        peak_frac: float = 0.75,
    ) -> Optional[int]:
        """Return episode_id where reward curve flattens, or None.

        Method: rolling-mean reward, then a sliding linear-regression slope
        over `window` episodes. First window whose |slope| < threshold AND
        whose center reward is at least peak_frac of the whole-run rolling
        max is flagged as the plateau point.
        """
        stats = self.rolling_reward_stats(window)
        mean = stats["mean"].values
        if len(mean) < 2 * window:
            return None

        peak_reached = peak_frac * mean.max()
        # Slope via normal equations on a simple linear fit.
        x = np.arange(window, dtype=np.float64)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()

        for start in range(0, len(mean) - window):
            seg = mean[start : start + window]
            slope = (x_centered * (seg - seg.mean())).sum() / denom
            center_idx = start + window // 2
            if abs(slope) < slope_threshold and mean[center_idx] >= peak_reached:
                return int(stats["episode_id"].iloc[center_idx])
        return None

    def oscillation_score(self, window: int = 50) -> pd.Series:
        """Rolling coefficient of variation of total_reward. Higher = more
        unstable. Indexed by episode_id."""
        ep = self.get_episode_metrics()
        r = ep["total_reward"]
        mean = r.rolling(window, min_periods=1).mean()
        std = r.rolling(window, min_periods=1).std().fillna(0)
        # Guard against division by zero early in training.
        cov = std / (mean.abs() + 1e-6)
        cov.index = ep["episode_id"]
        return cov

    def empirical_action_entropy(self, num_bins: int = 50) -> pd.DataFrame:
        """Per-bin empirical entropy H = -sum(p log p) over the high-level
        action id (0/1/2). This is the best proxy available given the
        current logging schema (no true policy entropy stored)."""
        steps = self.get_step_metrics()
        if steps.empty:
            return pd.DataFrame(columns=["bin", "entropy", "action_count"])

        max_ep = steps["episode_id"].max()
        bin_size = max(1, max_ep // num_bins)
        steps = steps.copy()
        steps["bin"] = (steps["episode_id"] // bin_size) * bin_size
        counts = steps.groupby(["bin", "action"]).size().reset_index(name="count")
        totals = steps.groupby("bin").size().reset_index(name="total")
        counts = counts.merge(totals, on="bin")
        counts["prob"] = counts["count"] / counts["total"]
        counts["term"] = -counts["prob"] * np.log(counts["prob"] + 1e-12)
        entropy = counts.groupby("bin")["term"].sum().reset_index(name="entropy")
        entropy["action_count"] = len(self.action_ids)
        return entropy

    def action_mix_over_time(self, num_bins: int = 50) -> pd.DataFrame:
        """Probability of each action id per episode-bin. Returns long-form
        DataFrame with columns [bin, action, prob]."""
        steps = self.get_step_metrics()
        if steps.empty:
            return pd.DataFrame(columns=["bin", "action", "prob"])

        max_ep = steps["episode_id"].max()
        bin_size = max(1, max_ep // num_bins)
        steps = steps.copy()
        steps["bin"] = (steps["episode_id"] // bin_size) * bin_size
        counts = steps.groupby(["bin", "action"]).size().reset_index(name="count")
        totals = steps.groupby("bin").size().reset_index(name="total")
        mix = counts.merge(totals, on="bin")
        mix["prob"] = mix["count"] / mix["total"]
        return mix[["bin", "action", "prob"]]

    def action_mix_by_episode_phase(self, num_bins: int = 50) -> pd.DataFrame:
        """Probability of each action by phase-of-episode and episode-bin.

        Phase is derived from normalized step progress:
        - early: first third
        - mid: middle third
        - late: final third
        """
        steps = self.get_step_metrics()
        episodes = self.get_episode_metrics()[["episode_id", "steps"]].rename(
            columns={"steps": "episode_steps"},
        )
        if steps.empty or episodes.empty:
            return pd.DataFrame(columns=["phase", "bin", "action", "prob"])

        merged = steps.merge(episodes, on="episode_id", how="left")
        merged["episode_steps"] = merged["episode_steps"].fillna(1).clip(lower=1)
        rel_pos = (merged["step_number"].astype(float) + 1.0) / merged[
            "episode_steps"
        ].astype(float)
        merged["phase"] = np.select(
            [
                rel_pos <= (1.0 / 3.0),
                rel_pos <= (2.0 / 3.0),
            ],
            [
                "early",
                "mid",
            ],
            default="late",
        )

        max_ep = int(merged["episode_id"].max())
        bin_size = max(1, max_ep // max(1, num_bins))
        merged["bin"] = (merged["episode_id"] // bin_size) * bin_size

        counts = (
            merged.groupby(["phase", "bin", "action"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            merged.groupby(["phase", "bin"])
            .size()
            .reset_index(name="total")
        )
        mix = counts.merge(totals, on=["phase", "bin"])
        mix["prob"] = mix["count"] / mix["total"]
        mix["phase"] = pd.Categorical(
            mix["phase"],
            categories=list(PHASE_LABELS),
            ordered=True,
        )
        return mix[["phase", "bin", "action", "prob"]].sort_values(
            ["phase", "bin", "action"],
        )

    def win_rate(self, threshold: float, window: int = 100) -> pd.Series:
        """Proxy win rate: fraction of episodes in rolling window with
        total_reward >= threshold. DefeatRoaches is a score-based minigame,
        so 'win' is a user-chosen threshold."""
        ep = self.get_episode_metrics()
        wins = (ep["total_reward"] >= threshold).astype(float)
        return wins.rolling(window, min_periods=1).mean()

    def action_mix_shift(self, num_bins: int = 50) -> pd.Series:
        """L1 distance between adjacent action-distribution bins. Large
        spikes = sudden policy shift (possible catastrophic forgetting)."""
        mix = self.action_mix_over_time(num_bins=num_bins)
        if mix.empty:
            return pd.Series(dtype=float)
        pivot = mix.pivot(index="bin", columns="action", values="prob").fillna(0)
        diff = pivot.diff().abs().sum(axis=1)
        return diff.fillna(0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def calculate_summary_statistics(self) -> dict:
        ep = self.get_episode_metrics()
        summary = {
            "total_episodes": int(len(ep)),
            "avg_episode_length": float(ep["steps"].mean()),
            "max_total_reward": float(ep["total_reward"].max()),
            "avg_total_reward": float(ep["total_reward"].mean()),
            "reward_std": float(ep["total_reward"].std()),
            "final_100_avg_reward": float(
                ep["total_reward"].tail(100).mean()
                if len(ep) >= 100
                else ep["total_reward"].mean()
            ),
        }
        try:
            rc = self.get_reward_components()
            for col in [
                "health_reward",
                "engagement_reward",
                "positioning_reward",
                "score_reward",
                "bonus_reward",
                "end_of_episode_reward",
            ]:
                summary[f"avg_{col}"] = float(rc[col].mean())
        except Exception as e:
            logger.warning(f"Could not calculate reward component statistics: {e}")
        return summary

    def _spatial_target_steps(self, steps: pd.DataFrame) -> pd.DataFrame:
        required = {"action", "move_x", "move_y"}
        if steps.empty or not required.issubset(set(steps.columns)):
            return pd.DataFrame()
        if self.action_semantics in {"smart_screen_v2", *SEMANTIC_CLICK_SEMANTICS}:
            action_mask = steps["action"] != self.noop_action_id
        elif self.action_semantics == "conditioned_spatial_v1":
            action_mask = steps["action"] != self.noop_action_id
        else:
            action_mask = steps["action"] == 1
        return steps[
            action_mask
            & steps["move_x"].notna()
            & steps["move_y"].notna()
        ]

    # ------------------------------------------------------------------
    # Rule-based diagnosis
    # ------------------------------------------------------------------
    def diagnose(
        self,
        window: int = 100,
        num_bins: int = 50,
        win_threshold: float = 25.0,
    ) -> dict:
        """Run all primitives and flag instability patterns. Returns a dict
        with raw numbers + a 'flags' list of (severity, message, knob)."""
        ep = self.get_episode_metrics()
        stats = self.rolling_reward_stats(window)
        cov = self.oscillation_score(window=50)
        entropy = self.empirical_action_entropy(num_bins=num_bins)
        mix = self.action_mix_over_time(num_bins=num_bins)
        shift = self.action_mix_shift(num_bins=num_bins)
        plateau_ep = self.detect_plateau(window=window)
        updates = self.get_update_metrics()
        evals = self.get_eval_metrics()

        flags = []

        # ---- PPO-side flags (require new logger instrumentation) ----
        if not updates.empty:
            # Use the last quarter of updates as "sustained" late-stage.
            tail = updates.tail(max(1, len(updates) // 4))
            mean_clip = tail["clip_fraction"].mean() if "clip_fraction" in tail else None
            mean_kl = tail["mean_kl"].mean() if "mean_kl" in tail else None
            mean_ev = (
                tail["explained_variance"].mean()
                if "explained_variance" in tail
                else None
            )
            inf_grad_updates = 0
            if "grad_norm" in tail.columns:
                inf_grad_updates = int(sum(
                    1
                    for value in tail["grad_norm"].dropna()
                    if not math.isfinite(float(value))
                ))
            nonfinite_grad_steps = 0
            skipped_optimizer_steps = 0
            if "nonfinite_grad_steps" in tail.columns:
                nonfinite_grad_steps = int(tail["nonfinite_grad_steps"].fillna(0).sum())
            if "skipped_optimizer_steps" in tail.columns:
                skipped_optimizer_steps = int(tail["skipped_optimizer_steps"].fillna(0).sum())

            if mean_clip is not None and mean_clip > 0.3 and len(tail) >= 10:
                flags.append((
                    "HIGH",
                    f"PPO clip fraction sustained high: {mean_clip:.2%} of "
                    f"samples clipped in last {len(tail)} updates.",
                    "clip_eps: 0.18 -> 0.10; lr: 1e-4 -> 5e-5",
                ))
            if mean_kl is not None and mean_kl > 0.05:
                flags.append((
                    "HIGH",
                    f"Approx KL(old||new) sustained high: {mean_kl:.3f} over "
                    f"last {len(tail)} updates. Policy moving too fast.",
                    "epochs: 20 -> 8; lower lr",
                ))
            if mean_ev is not None and mean_ev < 0.0:
                flags.append((
                    "HIGH",
                    f"Critic worse than mean: explained_variance = "
                    f"{mean_ev:.2f} (< 0) in last {len(tail)} updates.",
                    "critic_loss_coef up; investigate reward scale",
                ))
            if inf_grad_updates > 0 or nonfinite_grad_steps > 0:
                flags.append((
                    "HIGH",
                    f"Non-finite gradients detected late: {inf_grad_updates} update(s) "
                    f"with non-finite grad_norm, {nonfinite_grad_steps} skipped "
                    f"optimizer step(s) in the late-update tail.",
                    "inspect surrogate gradient stability and reward/value scale",
                ))
            if skipped_optimizer_steps > 0:
                flags.append((
                    "MED",
                    f"Optimizer steps were skipped {skipped_optimizer_steps} time(s) "
                    f"in the late-update tail due to invalid gradients.",
                    "check long-rollout batches and log per-batch failure context",
                ))
            if {
                "tbptt_group_mean_active_chunks",
                "tbptt_forward_calls",
                "update_wall_seconds",
            }.issubset(set(tail.columns)):
                mean_active_chunks = tail["tbptt_group_mean_active_chunks"].mean()
                mean_forward_calls = tail["tbptt_forward_calls"].mean()
                mean_update_seconds = tail["update_wall_seconds"].mean()
                if (
                    mean_active_chunks <= 1.25
                    and mean_forward_calls >= 1000
                    and mean_update_seconds >= 60.0
                ):
                    flags.append((
                        "MED",
                        "Learner is replaying mostly one TBPTT chunk at a time "
                        f"(active_chunks={mean_active_chunks:.2f}, "
                        f"forward_calls/update={mean_forward_calls:.0f}).",
                        "increase hyperparameters.batch_size above tbptt_window",
                    ))
            if {
                "replay_forward_wall_seconds",
                "update_wall_seconds",
            }.issubset(set(tail.columns)):
                replay_share = (
                    tail["replay_forward_wall_seconds"]
                    / tail["update_wall_seconds"].clip(lower=1.0e-6)
                ).mean()
                if replay_share > 0.65:
                    flags.append((
                        "MED",
                        f"Learner update is replay-forward dominated "
                        f"({replay_share:.0%} of update wall time).",
                        "pack more TBPTT chunks together; then consider model/kernel optimization",
                    ))

        # Late-stage instability
        if plateau_ep is not None and len(cov) > 200:
            late_cov = cov.tail(200).mean()
            if late_cov > 0.5:
                flags.append((
                    "HIGH",
                    f"Late-stage instability: mean CoV over last 200 eps = "
                    f"{late_cov:.2f} (> 0.5) after plateau at ep {plateau_ep}.",
                    "lr: 1e-4 -> 5e-5; epochs: 20 -> 10",
                ))

        # Policy collapse (entropy drop)
        if not entropy.empty and len(entropy) > 5:
            peak_idx = entropy["entropy"].idxmax()
            peak_ent = entropy["entropy"].iloc[peak_idx]
            post_peak = entropy.iloc[peak_idx:]
            collapse = post_peak[post_peak["entropy"] < 0.1]
            if peak_ent > 0.3 and not collapse.empty:
                flags.append((
                    "HIGH",
                    f"Policy collapse: empirical action entropy dropped from "
                    f"{peak_ent:.2f} (peak) to {collapse.iloc[0]['entropy']:.2f} "
                    f"at bin ep~{int(collapse.iloc[0]['bin'])}.",
                    "entropy_coef: 0.01 -> 0.03; add entropy annealing",
                ))

        # Exploration never commits
        if not entropy.empty:
            tail = entropy.tail(max(1, len(entropy) // 4))
            if (tail["entropy"] > 1.0).all() and len(ep) > 500:
                flags.append((
                    "MED",
                    f"Exploration never commits: entropy stays > 1.0 through "
                    f"end of run (tail mean {tail['entropy'].mean():.2f}).",
                    "entropy_coef: 0.01 -> 0.003; or inspect reward shaping",
                ))

        # No-op spam
        if not mix.empty:
            noop = mix[mix["action"] == self.noop_action_id].groupby("bin")["prob"].mean()
            late_noop = noop.tail(max(1, len(noop) // 4)).mean() if not noop.empty else None
            if late_noop is not None and late_noop > 0.5:
                flags.append((
                    "MED",
                    f"{self.action_labels[self.noop_action_id].title()} dominates: "
                    f"{late_noop:.1%} of late-training steps "
                    f"are {self.action_labels[self.noop_action_id]}.",
                    "add step-level penalty; verify reward function",
                ))

        # Plateau with variance blow-up
        if plateau_ep is not None:
            plateau_mask = ep["episode_id"] >= plateau_ep
            post = stats.loc[plateau_mask.values, "std"]
            if len(post) > 200:
                early_std = post.head(100).mean()
                late_std = post.tail(100).mean()
                if late_std > 2.0 * max(early_std, 1.0):
                    flags.append((
                        "HIGH",
                        f"Plateau-with-variance: reward std grew from "
                        f"{early_std:.1f} to {late_std:.1f} after plateau "
                        f"(ep {plateau_ep}).",
                        "epochs: 20 -> 8; update_frequency: 10 -> 20",
                    ))

        # Catastrophic-forgetting-like action shifts. Suppressed when bins
        # collapse to single episodes (max_ep < 2*num_bins) — L1 between
        # per-episode action histograms is noise, not drift.
        ep_count = len(ep)
        bins_are_meaningful = ep_count >= 2 * num_bins
        if (
            not shift.empty
            and shift.max() > 0.3
            and bins_are_meaningful
            and ep_count >= 500
        ):
            big = shift[shift > 0.3]
            flags.append((
                "MED",
                f"Sharp action-mix shift (L1 > 0.3) at bin(s) "
                f"{list(big.index[:3])} — possible catastrophic forgetting.",
                "clip_eps: 0.18 -> 0.10; lower lr",
            ))

        return {
            "plateau_episode": plateau_ep,
            "late_cov": float(cov.tail(200).mean()) if len(cov) > 200 else None,
            "entropy_series": entropy,
            "action_mix": mix,
            "rolling_stats": stats,
            "action_shift": shift,
            "updates": updates,
            "evals": evals,
            "flags": flags,
            "summary": self.calculate_summary_statistics(),
            "action_semantics": self.action_semantics,
            "action_labels": self.action_labels.copy(),
            "noop_action_id": self.noop_action_id,
            "config": {"window": window, "num_bins": num_bins, "win_threshold": win_threshold},
        }

    def write_report(self, diagnosis: dict, path: str):
        """Write a plain-text instability report from a diagnosis dict."""
        lines = []
        lines.append("=" * 70)
        lines.append("SNN-PPO Training Instability Report")
        lines.append("=" * 70)
        lines.append("")

        s = diagnosis["summary"]
        lines.append("Summary:")
        lines.append(f"  episodes:              {s['total_episodes']}")
        lines.append(f"  avg episode length:    {s['avg_episode_length']:.1f}")
        lines.append(f"  avg total reward:      {s['avg_total_reward']:.2f}")
        lines.append(f"  max total reward:      {s['max_total_reward']:.2f}")
        lines.append(f"  reward std (full run): {s['reward_std']:.2f}")
        lines.append(f"  final 100-ep avg:      {s['final_100_avg_reward']:.2f}")
        lines.append(f"  action semantics:      {diagnosis.get('action_semantics', self.action_semantics)}")
        action_labels = diagnosis.get("action_labels", self.action_labels)
        lines.append(
            "  action labels:         "
            f"0={action_labels.get(0, '0')}, "
            f"1={action_labels.get(1, '1')}, "
            f"2={action_labels.get(2, '2')}"
        )
        lines.append("")

        plateau = diagnosis["plateau_episode"]
        lines.append(
            f"Plateau detection: {'ep ' + str(plateau) if plateau else 'none detected'}"
        )
        if diagnosis["late_cov"] is not None:
            lines.append(f"Late-stage CoV (last 200 eps): {diagnosis['late_cov']:.3f}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("Data-driven flags:")
        lines.append("-" * 70)
        if not diagnosis["flags"]:
            lines.append("  (no rule triggered — training looks healthy by log signals)")
        else:
            for sev, msg, knob in diagnosis["flags"]:
                lines.append(f"  [{sev}] {msg}")
                lines.append(f"        suggested knob: {knob}")
                lines.append("")

        lines.append("-" * 70)
        lines.append("Interpretation Notes:")
        lines.append("-" * 70)
        lines.append(
            "  1. Current code has already fixed two older structural suspects:"
        )
        lines.append(
            "     - state-replay PPO training now uses stored SNN state"
        )
        lines.append(
            "     - entropy bonus is normalized per head instead of favoring move samples"
        )
        lines.append(
            "  2. If you are analyzing a run produced before those fixes landed,"
        )
        lines.append(
            "     older collapse patterns may reflect historical bugs rather than the"
        )
        lines.append(
            "     current trainer."
        )
        lines.append("")

        updates = diagnosis.get("updates", pd.DataFrame())
        evals = diagnosis.get("evals", pd.DataFrame())
        lines.append("-" * 70)
        lines.append("Instrumentation status:")
        lines.append("-" * 70)
        if updates is None or updates.empty:
            lines.append("  ppo_updates table empty - using empirical proxies.")
            lines.append("  - True policy entropy / KL / clip-fraction / value-loss"
                         " not logged in this run.")
            lines.append("  - Action entropy below is an EMPIRICAL proxy from"
                         " the 3-way action id.")
        else:
            tail = updates.tail(max(1, len(updates) // 4))
            lines.append(f"  ppo_updates: {len(updates)} rows logged.")
            lines.append(f"  Late-stage (last {len(tail)} updates) means:")
            for column, label, fmt in [
                ("mean_entropy", "mean_entropy", ".3f"),
                ("mean_kl", "mean_kl (approx)", ".4f"),
                ("clip_fraction", "clip_fraction", ".3f"),
                ("explained_variance", "explained_variance", ".3f"),
                ("update_wall_seconds", "update_wall_sec", ".2f"),
                ("rollout_wall_seconds", "rollout_wall_sec", ".2f"),
                ("learner_transitions_per_second", "learner steps/sec", ".2f"),
                ("rollout_steps_per_second", "rollout steps/sec", ".2f"),
                ("tbptt_forward_calls", "tbptt fwd calls", ".1f"),
                ("tbptt_group_mean_active_chunks", "active chunks", ".2f"),
                ("replay_forward_wall_seconds", "replay_forward_s", ".2f"),
                ("backward_optimizer_wall_seconds", "backward_opt_s", ".2f"),
                ("chunk_pack_wall_seconds", "chunk_pack_s", ".2f"),
                ("cpu_to_gpu_transfer_wall_seconds", "cpu_to_gpu_s", ".2f"),
                ("payload_total_mib", "payload MiB", ".1f"),
                ("cuda_peak_allocated_gib", "cuda peak GiB", ".2f"),
            ]:
                if column in tail.columns and tail[column].notna().any():
                    lines.append(
                        f"    {label:20s}= {tail[column].mean():{fmt}}"
                    )
            finite_grad_norms = [
                float(value)
                for value in tail["grad_norm"].dropna()
                if math.isfinite(float(value))
            ] if "grad_norm" in tail.columns else []
            if finite_grad_norms:
                lines.append(f"    {'grad_norm (finite)':20s}= {np.mean(finite_grad_norms):.3f}")
            if "grad_norm" in tail.columns:
                inf_grad_updates = sum(
                    1
                    for value in tail["grad_norm"].dropna()
                    if not math.isfinite(float(value))
                )
                lines.append(f"    {'grad_norm inf cnt':20s}= {inf_grad_updates}")
            if "lr" in tail.columns and tail["lr"].notna().any():
                lines.append(f"    {'lr':20s}= {tail['lr'].mean():.2e}")
            if "nonfinite_grad_steps" in tail.columns:
                lines.append(
                    f"    {'nonfinite_grad_steps':20s}= {int(tail['nonfinite_grad_steps'].fillna(0).sum())}"
                )
            if "skipped_optimizer_steps" in tail.columns:
                lines.append(
                    f"    {'skipped_opt_steps':20s}= {int(tail['skipped_optimizer_steps'].fillna(0).sum())}"
                )
            if "transitions_in_update" in tail.columns:
                lines.append(
                    f"    {'transitions/update':20s}= {tail['transitions_in_update'].mean():.1f}"
                )
            if "entity_mask_utilization" in tail.columns:
                lines.append(
                    f"    {'entity_mask_util':20s}= {tail['entity_mask_utilization'].mean():.3f}"
                )
            if "entity_count_p50" in tail.columns:
                lines.append(
                    f"    {'entity_count_p50':20s}= {tail['entity_count_p50'].mean():.2f}"
                )
            if "entity_count_p99" in tail.columns:
                lines.append(
                    f"    {'entity_count_p99':20s}= {tail['entity_count_p99'].mean():.2f}"
                )
            if "selection_mask_utilization" in tail.columns:
                lines.append(
                    f"    {'selection_mask_util':20s}= {tail['selection_mask_utilization'].mean():.3f}"
                )
        if evals is None or evals.empty:
            lines.append("  eval_runs: none logged")
        else:
            latest_eval = evals.iloc[-1]
            lines.append(f"  eval_runs: {len(evals)} rows logged.")
            extras = []
            if "num_episodes" in evals.columns:
                extras.append(f"n={int(latest_eval['num_episodes'])}")
            if "deterministic" in evals.columns:
                extras.append(f"det={bool(latest_eval['deterministic'])}")
            tag = f" ({', '.join(extras)})" if extras else ""
            lines.append(
                f"  latest eval @ ep {int(latest_eval['episode_index'])}{tag}: "
                f"mean={float(latest_eval['mean_reward']):.2f}, "
                f"std={float(latest_eval['std_reward']):.2f}"
            )
        lines.append("")

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Report written to {path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_training_progress(
        self,
        window: int = 100,
        num_bins: int = 50,
        plateau_ep: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        ep = self.get_episode_metrics()
        stats = self.rolling_reward_stats(window=window)
        cov = self.oscillation_score(window=50)
        entropy = self.empirical_action_entropy(num_bins=num_bins)
        mix = self.action_mix_over_time(num_bins=num_bins)

        fig, axes = plt.subplots(5, 1, figsize=(13, 20))

        # 1. Reward + rolling mean + std band
        ax = axes[0]
        ax.plot(ep["episode_id"], ep["total_reward"], color="tab:blue",
                alpha=0.25, label="raw reward")
        ax.plot(stats["episode_id"], stats["mean"], color="tab:red",
                label=f"rolling mean (w={window})")
        ax.fill_between(
            stats["episode_id"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="tab:red", alpha=0.15, label="+/- 1 std",
        )
        if plateau_ep is not None:
            ax.axvline(plateau_ep, ls="--", color="k", alpha=0.5,
                       label=f"plateau (ep {plateau_ep})")
        ax.set_title("Total reward per episode")
        ax.set_xlabel("episode"); ax.set_ylabel("reward")
        ax.legend(); ax.grid(True)

        # 2. Episode length
        ax = axes[1]
        ep_rolling = ep["steps"].rolling(window, min_periods=1).mean()
        ax.plot(ep["episode_id"], ep["steps"], color="tab:green", alpha=0.25,
                label="raw")
        ax.plot(ep["episode_id"], ep_rolling, color="tab:red",
                label=f"rolling mean (w={window})")
        ax.set_title("Episode length over time")
        ax.set_xlabel("episode"); ax.set_ylabel("steps")
        ax.legend(); ax.grid(True)

        # 3. Oscillation score
        ax = axes[2]
        ax.plot(cov.index, cov.values, color="tab:purple")
        ax.axhline(0.5, ls=":", color="k", alpha=0.4, label="0.5 (instability threshold)")
        if plateau_ep is not None:
            ax.axvline(plateau_ep, ls="--", color="k", alpha=0.5)
        ax.set_title("Rolling oscillation score (coefficient of variation, w=50)")
        ax.set_xlabel("episode"); ax.set_ylabel("CoV")
        ax.legend(); ax.grid(True)

        # 4. Empirical action entropy
        ax = axes[3]
        if not entropy.empty:
            ax.plot(entropy["bin"], entropy["entropy"], color="tab:orange",
                    marker="o", markersize=3)
        ax.axhline(
            np.log(len(self.action_ids)),
            ls=":",
            color="k",
            alpha=0.4,
            label=f"log({len(self.action_ids)}) (uniform)",
        )
        ax.axhline(0.1, ls=":", color="r", alpha=0.4, label="0.1 (collapse threshold)")
        if plateau_ep is not None:
            ax.axvline(plateau_ep, ls="--", color="k", alpha=0.5)
        ax.set_title("Empirical action entropy over time (proxy for policy entropy)")
        ax.set_xlabel("episode (binned)"); ax.set_ylabel("H(action)")
        ax.legend(); ax.grid(True)

        # 5. Action mix stacked area
        ax = axes[4]
        if not mix.empty:
            pivot = mix.pivot(index="bin", columns="action", values="prob").fillna(0)
            pivot = pivot.reindex(columns=self.action_ids, fill_value=0)
            ax.stackplot(
                pivot.index,
                [pivot[action_id].values for action_id in self.action_ids],
                labels=[self.action_labels[action_id] for action_id in self.action_ids],
                alpha=0.7,
            )
        ax.set_title("Action mix over time")
        ax.set_xlabel("episode (binned)"); ax.set_ylabel("share")
        ax.set_ylim(0, 1)
        if not mix.empty:
            ax.legend(loc="upper right")
        ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=110)
            logger.info(f"Training-progress plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_reward_components(self, save_path: Optional[str] = None):
        try:
            rc = self.get_reward_components()
            component_cols = [
                "health_reward", "engagement_reward", "positioning_reward",
                "score_reward", "bonus_reward", "end_of_episode_reward",
            ]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            if sns is not None:
                sns.boxplot(data=rc[component_cols], ax=ax1)
            else:
                ax1.boxplot(
                    [rc[col].dropna().values for col in component_cols],
                    tick_labels=component_cols,
                )
            ax1.set_title("Distribution of reward components")
            ax1.set_xlabel("component"); ax1.set_ylabel("value")
            for label in ax1.get_xticklabels():
                label.set_rotation(30)

            for col in component_cols:
                per_ep = rc.groupby("episode_id")[col].mean()
                rolling = per_ep.rolling(window=50, min_periods=1).mean()
                ax2.plot(rolling.index, rolling.values, label=col)
            ax2.set_title("Reward components evolution (rolling mean, w=50)")
            ax2.set_xlabel("episode"); ax2.set_ylabel("avg value")
            ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            ax2.grid(True)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=110)
                logger.info(f"Reward-component plot saved to {save_path}")
            else:
                plt.show()
            plt.close(fig)
        except Exception as e:
            logger.error(f"Could not plot reward components: {e}")

    def plot_win_rate(
        self,
        threshold: float,
        window: int = 100,
        save_path: Optional[str] = None,
    ):
        wr = self.win_rate(threshold, window=window)
        ep = self.get_episode_metrics()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ep["episode_id"], wr.values, color="tab:cyan")
        ax.set_title(f"Win rate (reward >= {threshold}, rolling w={window})")
        ax.set_xlabel("episode"); ax.set_ylabel("fraction above threshold")
        ax.set_ylim(0, 1); ax.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=110)
            logger.info(f"Win-rate plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def export_ai_friendly_panels(
        self,
        out_dir: str,
        window: int = 100,
        num_bins: int = 50,
        plateau_ep: Optional[int] = None,
    ) -> list[str]:
        """Export a small bundle of high-signal static panels intended to
        be easy to share back into text-only workflows.

        The output is a set of focused PNGs rather than one giant dashboard
        screenshot. Missing-data panels are skipped instead of failing.
        """
        output_root = Path(out_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        exported: list[str] = []

        def _save(fig, filename: str):
            path = output_root / filename
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            exported.append(filename)
            logger.info(f"AI-friendly panel saved to {path}")

        ep = self.get_episode_metrics()
        stats = self.rolling_reward_stats(window=window)
        cov = self.oscillation_score(window=50)
        entropy = self.empirical_action_entropy(num_bins=num_bins)
        mix = self.action_mix_over_time(num_bins=num_bins)
        phase_mix = self.action_mix_by_episode_phase(num_bins=num_bins)
        steps = self.get_step_metrics()
        updates = self.get_update_metrics()
        evals = self.get_eval_metrics()
        reward_components = self.get_reward_components()

        # 1. Reward trajectory
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(
            ep["episode_id"], ep["total_reward"],
            color="tab:blue", alpha=0.25, label="raw reward",
        )
        ax.plot(
            stats["episode_id"], stats["mean"],
            color="tab:red", label=f"rolling mean (w={window})",
        )
        ax.fill_between(
            stats["episode_id"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color="tab:red", alpha=0.15, label="+/- 1 std",
        )
        if plateau_ep is not None:
            ax.axvline(
                plateau_ep,
                ls="--",
                color="k",
                alpha=0.5,
                label=f"plateau (ep {plateau_ep})",
            )
        ax.set_title("Reward trajectory")
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.legend()
        ax.grid(True)
        _save(fig, "01_reward_trajectory.png")

        # 2. Episode length
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ep_rolling = ep["steps"].rolling(window, min_periods=1).mean()
        ax.plot(ep["episode_id"], ep["steps"], color="tab:green", alpha=0.25, label="raw")
        ax.plot(
            ep["episode_id"], ep_rolling,
            color="tab:red", label=f"rolling mean (w={window})",
        )
        ax.set_title("Episode length")
        ax.set_xlabel("episode")
        ax.set_ylabel("steps")
        ax.legend()
        ax.grid(True)
        _save(fig, "02_episode_length.png")

        # 3. Reward efficiency
        fig, ax = plt.subplots(figsize=(11, 4.5))
        efficiency = (
            ep["total_reward"] / ep["steps"].clip(lower=1)
        ).replace([np.inf, -np.inf], np.nan)
        ax.plot(
            ep["episode_id"], efficiency,
            color="tab:purple", alpha=0.25, label="reward / step",
        )
        ax.plot(
            ep["episode_id"],
            efficiency.rolling(window, min_periods=1).mean(),
            color="tab:red",
            label=f"rolling mean (w={window})",
        )
        ax.set_title("Reward efficiency")
        ax.set_xlabel("episode")
        ax.set_ylabel("reward / step")
        ax.legend()
        ax.grid(True)
        _save(fig, "03_reward_efficiency.png")

        # 4. Oscillation score
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(cov.index, cov.values, color="tab:purple")
        ax.axhline(0.5, ls=":", color="k", alpha=0.4, label="0.5 threshold")
        if plateau_ep is not None:
            ax.axvline(plateau_ep, ls="--", color="k", alpha=0.5)
        ax.set_title("Rolling oscillation score")
        ax.set_xlabel("episode")
        ax.set_ylabel("CoV")
        ax.legend()
        ax.grid(True)
        _save(fig, "04_oscillation_score.png")

        # 5. Action entropy
        if not entropy.empty:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(
                entropy["bin"], entropy["entropy"],
                color="tab:orange", marker="o", markersize=3,
            )
            ax.axhline(
                np.log(len(self.action_ids)),
                ls=":",
                color="k",
                alpha=0.4,
                label=f"log({len(self.action_ids)})",
            )
            ax.axhline(0.1, ls=":", color="r", alpha=0.4, label="0.1 threshold")
            if plateau_ep is not None:
                ax.axvline(plateau_ep, ls="--", color="k", alpha=0.5)
            ax.set_title("Empirical action entropy")
            ax.set_xlabel("episode (binned)")
            ax.set_ylabel("H(action)")
            ax.legend()
            ax.grid(True)
            _save(fig, "05_action_entropy.png")

        # 6. Whole-run action mix
        if not mix.empty:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            pivot = mix.pivot(index="bin", columns="action", values="prob").fillna(0)
            pivot = pivot.reindex(columns=self.action_ids, fill_value=0)
            ax.stackplot(
                pivot.index,
                [pivot[action_id].values for action_id in self.action_ids],
                labels=[self.action_labels[action_id] for action_id in self.action_ids],
                alpha=0.75,
            )
            ax.set_title("Action mix over time")
            ax.set_xlabel("episode (binned)")
            ax.set_ylabel("share")
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right")
            ax.grid(True)
            _save(fig, "06_action_mix.png")

        # 7. Early / mid / late action mix
        if not phase_mix.empty:
            fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
            for ax, phase in zip(axes, PHASE_LABELS):
                phase_df = phase_mix[phase_mix["phase"] == phase]
                if phase_df.empty:
                    continue
                pivot = phase_df.pivot(index="bin", columns="action", values="prob").fillna(0)
                pivot = pivot.reindex(columns=self.action_ids, fill_value=0)
                ax.stackplot(
                    pivot.index,
                    [pivot[action_id].values for action_id in self.action_ids],
                    labels=[self.action_labels[action_id] for action_id in self.action_ids],
                    alpha=0.75,
                )
                ax.set_title(f"{phase.title()}-phase action mix")
                ax.set_ylabel("share")
                ax.set_ylim(0, 1)
                ax.grid(True)
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("episode (binned)")
            _save(fig, "07_phase_action_mix.png")

        # 8. Spatial target heatmap
        spatial_steps = self._spatial_target_steps(steps)
        if not spatial_steps.empty:
                fig, ax = plt.subplots(figsize=(7, 6))
                heatmap, xedges, yedges = np.histogram2d(
                    spatial_steps["move_x"],
                    spatial_steps["move_y"],
                    bins=32,
                )
                image = ax.imshow(
                    heatmap.T,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
                ax.set_title("Spatial target heatmap")
                ax.set_xlabel("move_x bin")
                ax.set_ylabel("move_y bin")
                fig.colorbar(image, ax=ax, label="count")
                _save(fig, "08_move_target_heatmap.png")

        # 9. TBPTT / speed
        speed_fields = {
            "update_wall_seconds",
            "tbptt_forward_calls",
            "tbptt_chunk_groups",
            "tbptt_group_mean_active_chunks",
            "transitions_in_update",
        }
        if not updates.empty and speed_fields.intersection(set(updates.columns)):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            x = updates["update_id"] if "update_id" in updates.columns else np.arange(len(updates))

            if "update_wall_seconds" in updates.columns:
                axes[0, 0].plot(x, updates["update_wall_seconds"], color="tab:blue")
                axes[0, 0].set_title("Update wall seconds")
                axes[0, 0].set_ylabel("seconds")
                axes[0, 0].grid(True)

            if "tbptt_forward_calls" in updates.columns:
                axes[0, 1].plot(x, updates["tbptt_forward_calls"], color="tab:orange")
                axes[0, 1].set_title("TBPTT forward calls")
                axes[0, 1].grid(True)

            if "tbptt_chunk_groups" in updates.columns:
                axes[1, 0].plot(x, updates["tbptt_chunk_groups"], color="tab:green")
                axes[1, 0].set_title("TBPTT chunk groups")
                axes[1, 0].set_xlabel("ppo update")
                axes[1, 0].grid(True)

            if {
                "transitions_in_update",
                "update_wall_seconds",
            }.issubset(set(updates.columns)):
                throughput = (
                    updates["transitions_in_update"] /
                    updates["update_wall_seconds"].clip(lower=1.0e-6)
                )
                axes[1, 1].plot(x, throughput, color="tab:red")
                axes[1, 1].set_title("Transitions per second")
                axes[1, 1].set_xlabel("ppo update")
                axes[1, 1].grid(True)
            elif "tbptt_group_mean_active_chunks" in updates.columns:
                axes[1, 1].plot(
                    x,
                    updates["tbptt_group_mean_active_chunks"],
                    color="tab:red",
                )
                axes[1, 1].set_title("Mean active chunks")
                axes[1, 1].set_xlabel("ppo update")
                axes[1, 1].grid(True)

            _save(fig, "09_tbptt_speed.png")

        # 10. Learner/Ray timing breakdown
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
            if column in updates.columns and updates[column].notna().any()
        ]
        if not updates.empty and timing_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = updates["update_id"] if "update_id" in updates.columns else np.arange(len(updates))
            for column in timing_cols:
                ax.plot(x, updates[column], marker="o", markersize=3, label=column)
            ax.set_title("Ray / learner timing breakdown")
            ax.set_xlabel("ppo update")
            ax.set_ylabel("seconds")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(True)
            _save(fig, "13_learner_timing_breakdown.png")

        # 11. Payload / CUDA footprint
        memory_cols = [
            column
            for column in [
                "payload_total_mib",
                "cuda_peak_allocated_gib",
                "cuda_peak_reserved_gib",
            ]
            if column in updates.columns and updates[column].notna().any()
        ]
        if not updates.empty and memory_cols:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            x = updates["update_id"] if "update_id" in updates.columns else np.arange(len(updates))
            for column in memory_cols:
                ax.plot(x, updates[column], marker="o", markersize=3, label=column)
            ax.set_title("Rollout payload and CUDA peak footprint")
            ax.set_xlabel("ppo update")
            ax.set_ylabel("MiB / GiB")
            ax.legend()
            ax.grid(True)
            _save(fig, "14_payload_cuda_footprint.png")

        # 12. Eval split
        if not evals.empty and {"episode_index", "mean_reward"}.issubset(set(evals.columns)):
            fig, ax = plt.subplots(figsize=(11, 4.5))
            if "deterministic" in evals.columns:
                for label, det_value, color in [
                    ("stochastic", 0, "tab:blue"),
                    ("deterministic", 1, "tab:red"),
                ]:
                    subset = evals[evals["deterministic"].fillna(0).astype(int) == det_value]
                    if subset.empty:
                        continue
                    ax.plot(
                        subset["episode_index"],
                        subset["mean_reward"],
                        marker="o",
                        color=color,
                        label=label,
                    )
                    if {"min_reward", "max_reward"}.issubset(set(subset.columns)):
                        ax.fill_between(
                            subset["episode_index"],
                            subset["min_reward"],
                            subset["max_reward"],
                            color=color,
                            alpha=0.10,
                        )
            else:
                ax.plot(
                    evals["episode_index"],
                    evals["mean_reward"],
                    marker="o",
                    color="tab:blue",
                    label="eval mean",
                )
            ax.set_title("Evaluation reward")
            ax.set_xlabel("episode")
            ax.set_ylabel("mean reward")
            ax.legend()
            ax.grid(True)
            _save(fig, "10_eval_split.png")

        # 13. Eval gap
        if (
            not evals.empty
            and {"episode_index", "mean_reward", "deterministic"}.issubset(set(evals.columns))
        ):
            pivot = evals.copy()
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
            if {"deterministic", "stochastic"}.issubset(set(pivot.columns)):
                gap = pivot["stochastic"] - pivot["deterministic"]
                fig, ax = plt.subplots(figsize=(11, 4.5))
                ax.plot(gap.index, gap.values, color="tab:brown", marker="o")
                ax.axhline(0.0, ls=":", color="k", alpha=0.4)
                ax.set_title("Eval reward gap (stochastic - deterministic)")
                ax.set_xlabel("episode")
                ax.set_ylabel("reward gap")
                ax.grid(True)
                _save(fig, "11_eval_gap.png")

        # 14. Reward component trends
        if not reward_components.empty:
            reward_cols = [
                column
                for column in reward_components.columns
                if "reward" in column and column not in {"total_reward", "episode_id"}
            ]
            if reward_cols:
                fig, ax = plt.subplots(figsize=(12, 5))
                for col in reward_cols:
                    per_ep = reward_components.groupby("episode_id")[col].mean()
                    rolling = per_ep.rolling(window=50, min_periods=1).mean()
                    ax.plot(rolling.index, rolling.values, label=col)
                ax.set_title("Reward component trends")
                ax.set_xlabel("episode")
                ax.set_ylabel("rolling mean value")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.grid(True)
                _save(fig, "12_reward_component_trends.png")

        manifest_lines = [
            "AI-friendly export bundle",
            f"window={window}",
            f"num_bins={num_bins}",
            f"action_semantics={self.action_semantics}",
            "action_labels="
            + ",".join(
                f"{action_id}:{self.action_labels[action_id]}"
                for action_id in self.action_ids
            ),
            "",
            "files:",
        ] + [f"- {name}" for name in exported]
        (output_root / "manifest.txt").write_text(
            "\n".join(manifest_lines),
            encoding="utf-8",
        )
        logger.info(f"AI-friendly manifest saved to {output_root / 'manifest.txt'}")

        return exported

    def export_metrics(self, path: str):
        summary = self.calculate_summary_statistics()
        pd.DataFrame(list(summary.items()),
                     columns=["metric", "value"]).to_csv(path, index=False)
        logger.info(f"Summary metrics exported to {path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cfg_defaults():
    """Pull per-run layout defaults from config if available; safe to call
    from a pure-analysis context where the training deps aren't installed."""
    try:
        from Utility.config import cfg
        return {
            "run_name": getattr(cfg.environment, "run_name", "") or "",
            "models_dir": getattr(cfg.environment, "models_dir", "models"),
            "analysis_dir": getattr(cfg.environment, "analysis_dir",
                                    "analysis_results"),
            "db_filename": getattr(cfg.environment, "db_path",
                                   "training_logs.db"),
        }
    except Exception:
        return {
            "run_name": "",
            "models_dir": "models",
            "analysis_dir": "analysis_results",
            "db_filename": "training_logs.db",
        }


def _latest_run_name(models_dir: str, db_filename: str) -> Optional[str]:
    root = Path(models_dir)
    if not root.exists():
        return None
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and (child / db_filename).exists()
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name


def _infer_run_name_from_db_path(
    db_path: str,
    models_dir: str,
    db_filename: str,
) -> Optional[str]:
    path = Path(db_path)
    try:
        if (
            path.name == db_filename
            and path.parent.parent.name == Path(models_dir).name
        ):
            return path.parent.name
    except IndexError:
        return None
    return None


def main():
    defaults = _cfg_defaults()
    parser = argparse.ArgumentParser(description="SNN-PPO training analyzer.")
    parser.add_argument("--run-name", default=defaults["run_name"] or None,
                        help="Per-run subfolder name under "
                             "{models_dir}/ and {analysis_dir}/. If omitted "
                             "and config has no run_name, resolves to the "
                             "latest local run when possible.")
    parser.add_argument("--db", default=None,
                        help="Explicit path to training_logs.db. "
                             "Overrides the per-run join.")
    parser.add_argument("--out", default=None,
                        help="Explicit output directory. "
                             "Overrides the per-run join.")
    parser.add_argument("--window", type=int, default=100,
                        help="Rolling window in episodes")
    parser.add_argument("--num-bins", type=int, default=50,
                        help="Number of episode bins for action/entropy plots")
    parser.add_argument("--win-threshold", type=float, default=25.0,
                        help="Reward threshold for win-rate proxy")
    parser.add_argument("--report", action="store_true",
                        help="Write instability_report.txt in --out")
    parser.add_argument(
        "--aismart",
        action="store_true",
        help="Export focused dashboard-style PNG panels into "
             "<out>/ai_friendly_results/ for easy sharing back into text workflows.",
    )
    args = parser.parse_args()

    # Resolve paths per the per-run layout. Explicit --db/--out win.
    run_name = args.run_name
    if args.db is not None:
        db_path = args.db
        if not run_name:
            run_name = _infer_run_name_from_db_path(
                db_path,
                defaults["models_dir"],
                defaults["db_filename"],
            )
        if not run_name:
            run_name = _latest_run_name(
                defaults["models_dir"],
                defaults["db_filename"],
            )
    elif run_name:
        db_path = str(Path(defaults["models_dir"]) / run_name
                      / defaults["db_filename"])
    else:
        run_name = _latest_run_name(
            defaults["models_dir"],
            defaults["db_filename"],
        )
        if run_name:
            db_path = str(Path(defaults["models_dir"]) / run_name
                          / defaults["db_filename"])
        else:
            db_path = defaults["db_filename"]  # flat-layout fallback

    if args.db is None and args.out is None and run_name is None:
        db_path = defaults["db_filename"]  # flat-layout fallback

    if args.out is not None:
        out_dir = Path(args.out)
    elif run_name:
        out_dir = Path(defaults["analysis_dir"]) / run_name
    else:
        out_dir = Path(defaults["analysis_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"DB: {db_path}")
    logger.info(f"Output: {out_dir}")

    analyzer = TrainingAnalyzer(db_path)
    try:
        diagnosis = analyzer.diagnose(
            window=args.window,
            num_bins=args.num_bins,
            win_threshold=args.win_threshold,
        )

        analyzer.plot_training_progress(
            window=args.window,
            num_bins=args.num_bins,
            plateau_ep=diagnosis["plateau_episode"],
            save_path=str(out_dir / "training_progress.png"),
        )
        analyzer.plot_reward_components(
            save_path=str(out_dir / "reward_components.png"),
        )
        analyzer.plot_win_rate(
            threshold=args.win_threshold,
            window=args.window,
            save_path=str(out_dir / "win_rate.png"),
        )
        analyzer.export_metrics(str(out_dir / "training_metrics.csv"))

        if args.report:
            analyzer.write_report(
                diagnosis,
                str(out_dir / "instability_report.txt"),
            )

        if args.aismart:
            analyzer.export_ai_friendly_panels(
                out_dir=str(out_dir / "ai_friendly_results"),
                window=args.window,
                num_bins=args.num_bins,
                plateau_ep=diagnosis["plateau_episode"],
            )

        # Console summary
        print("\n=== Diagnosis ===")
        print(f"plateau episode: {diagnosis['plateau_episode']}")
        print(f"flags: {len(diagnosis['flags'])}")
        for sev, msg, knob in diagnosis["flags"]:
            print(f"  [{sev}] {msg}")
            print(f"        -> {knob}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
