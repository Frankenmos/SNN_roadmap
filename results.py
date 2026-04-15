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
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ACTION_LABELS = {0: "attack", 1: "move", 2: "no-op"}


class TrainingAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        # Cached DataFrames — loaded lazily.
        self._episodes: Optional[pd.DataFrame] = None
        self._steps: Optional[pd.DataFrame] = None
        self._reward_components: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Connection + raw loaders
    # ------------------------------------------------------------------
    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

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
            self._steps = pd.read_sql_query(
                "SELECT episode_id, step_number, action, reward, cumulative_reward "
                "FROM steps ORDER BY episode_id, step_number",
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
        try:
            return pd.read_sql_query(
                "SELECT update_id, episode_id, mean_policy_loss, "
                "mean_value_loss, mean_entropy, mean_kl, clip_fraction, "
                "explained_variance, grad_norm, lr "
                "FROM ppo_updates ORDER BY update_id",
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
            return pd.DataFrame(columns=["bin", "entropy"])

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

        flags = []

        # ---- PPO-side flags (require new logger instrumentation) ----
        if not updates.empty:
            # Use the last quarter of updates as "sustained" late-stage.
            tail = updates.tail(max(1, len(updates) // 4))
            mean_clip = tail["clip_fraction"].mean()
            mean_kl = tail["mean_kl"].mean()
            mean_ev = tail["explained_variance"].mean()

            if mean_clip > 0.3:
                flags.append((
                    "HIGH",
                    f"PPO clip fraction sustained high: {mean_clip:.2%} of "
                    f"samples clipped in last {len(tail)} updates.",
                    "clip_eps: 0.18 -> 0.10; lr: 1e-4 -> 5e-5",
                ))
            if mean_kl > 0.05:
                flags.append((
                    "HIGH",
                    f"Approx KL(old||new) sustained high: {mean_kl:.3f} over "
                    f"last {len(tail)} updates. Policy moving too fast.",
                    "epochs: 20 -> 8; lower lr",
                ))
            if mean_ev < 0.0:
                flags.append((
                    "HIGH",
                    f"Critic worse than mean: explained_variance = "
                    f"{mean_ev:.2f} (< 0) in last {len(tail)} updates.",
                    "critic_loss_coef up; investigate reward scale",
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
            noop = mix[mix["action"] == 2].groupby("bin")["prob"].mean()
            if not noop.empty and noop.tail(max(1, len(noop) // 4)).mean() > 0.5:
                flags.append((
                    "MED",
                    f"No-op dominates: {noop.tail(len(noop)//4).mean():.1%} "
                    f"of late-training steps are no-op.",
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

        # Catastrophic-forgetting-like action shifts
        if not shift.empty and shift.max() > 0.3:
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
            "flags": flags,
            "summary": self.calculate_summary_statistics(),
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
        lines.append("Architecture-specific suspects (cannot verify from logs):")
        lines.append("-" * 70)
        lines.append(
            "  1. Stateful/stateless SNN mismatch. Rollout carries self.snn_state "
            "across env steps (PPO_CNN_agent.py:83-85), but training evaluates "
            "with state=None (PPO.py:179-181). The PPO importance ratio is "
            "built on inconsistent policies; bias grows with episode length."
        )
        lines.append(
            "  2. Move-head entropy asymmetry. Joint log-prob masking means a "
            "move-action sample gets H ~ log(3) + 2*log(84) ~ 9.9, while an "
            "attack sample gets H ~ log(3) ~ 1.1. With entropy_coef=0.01 this "
            "is a ~10x bias favoring movement, which can suppress commitment "
            "to attacks."
        )
        lines.append("")

        updates = diagnosis.get("updates", pd.DataFrame())
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
            lines.append(f"    mean_entropy        = {tail['mean_entropy'].mean():.3f}")
            lines.append(f"    mean_kl (approx)    = {tail['mean_kl'].mean():.4f}")
            lines.append(f"    clip_fraction       = {tail['clip_fraction'].mean():.3f}")
            lines.append(f"    explained_variance  = {tail['explained_variance'].mean():.3f}")
            lines.append(f"    grad_norm           = {tail['grad_norm'].mean():.3f}")
            lines.append(f"    lr                  = {tail['lr'].mean():.2e}")
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
        ax.axhline(np.log(3), ls=":", color="k", alpha=0.4, label="log(3) (uniform)")
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
            pivot = pivot.reindex(columns=[0, 1, 2], fill_value=0)
            ax.stackplot(
                pivot.index,
                [pivot[0].values, pivot[1].values, pivot[2].values],
                labels=[ACTION_LABELS[0], ACTION_LABELS[1], ACTION_LABELS[2]],
                alpha=0.7,
            )
        ax.set_title("Action mix over time")
        ax.set_xlabel("episode (binned)"); ax.set_ylabel("share")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right"); ax.grid(True)

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

            sns.boxplot(data=rc[component_cols], ax=ax1)
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


def main():
    defaults = _cfg_defaults()
    parser = argparse.ArgumentParser(description="SNN-PPO training analyzer.")
    parser.add_argument("--run-name", default=defaults["run_name"] or None,
                        help="Per-run subfolder name under "
                             "{models_dir}/ and {analysis_dir}/. If omitted "
                             "and config has no run_name, falls back to the "
                             "flat layout (legacy).")
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
    args = parser.parse_args()

    # Resolve paths per the per-run layout. Explicit --db/--out win.
    run_name = args.run_name
    if args.db is not None:
        db_path = args.db
    elif run_name:
        db_path = str(Path(defaults["models_dir"]) / run_name
                      / defaults["db_filename"])
    else:
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
