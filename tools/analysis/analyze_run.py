#!/usr/bin/env python
"""
Lightweight CLI helper for quick DB and checkpoint inspection.

This intentionally stays simpler than results.py:
- quick summaries in the terminal
- tolerant of older and newer DB schemas
- convenient per-run path resolution
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def _cfg_defaults() -> dict:
    try:
        from Utility.config import cfg

        return {
            "run_name": getattr(cfg.environment, "run_name", "") or "",
            "models_dir": getattr(cfg.environment, "models_dir", "models"),
            "db_filename": getattr(cfg.environment, "db_path", "training_logs.db"),
            "checkpoint_filename": getattr(
                cfg.environment, "checkpoint_path", "checkpoint.pth"
            ),
            "best_checkpoint_filename": getattr(
                cfg.environment, "best_checkpoint_path", "best_checkpoint.pth"
            ),
        }
    except Exception:
        return {
            "run_name": "",
            "models_dir": "models",
            "db_filename": "training_logs.db",
            "checkpoint_filename": "checkpoint.pth",
            "best_checkpoint_filename": "best_checkpoint.pth",
        }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.Error:
        return set()
    return {row[1] for row in rows}


def _connect_db(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1").fetchall()
        return conn
    except sqlite3.Error as exc:
        try:
            conn.close()
        except Exception:
            pass
        uri = f"file:{Path(db_path).resolve()}?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
            ).fetchall()
            print("Opened DB as immutable read-only snapshot.")
            return conn
        except sqlite3.Error:
            raise exc


def _resolve_db_path(args: argparse.Namespace) -> str:
    defaults = _cfg_defaults()
    if args.db:
        return args.db
    run_name = args.run_name or defaults["run_name"]
    if not run_name:
        raise SystemExit("Pass --db PATH or --run-name NAME for mode='db'.")
    return str(Path(defaults["models_dir"]) / run_name / defaults["db_filename"])


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    defaults = _cfg_defaults()
    if args.ckpt:
        return args.ckpt
    run_name = args.run_name or defaults["run_name"]
    if not run_name:
        raise SystemExit("Pass --ckpt PATH or --run-name NAME for mode='pth'.")

    run_dir = Path(defaults["models_dir"]) / run_name
    preferred = []
    if args.which in {"auto", "best"}:
        preferred.append(run_dir / defaults["best_checkpoint_filename"])
    if args.which in {"auto", "checkpoint"}:
        preferred.append(run_dir / defaults["checkpoint_filename"])

    for path in preferred:
        if path.exists():
            return str(path)

    raise SystemExit(
        f"Could not resolve a checkpoint in {run_dir}. "
        "Pass --ckpt explicitly."
    )


def analyze_db(db_path: str) -> None:
    print(f"=== DB analysis for {db_path} ===")
    conn = _connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*), AVG(total_reward), MAX(total_reward), AVG(steps) FROM episodes"
    )
    row = cursor.fetchone()
    if row is None:
        print("No episodes found.")
        conn.close()
        return

    num_episodes, avg_reward, max_reward, avg_steps = row
    print(f"Number of episodes      : {num_episodes}")
    print(
        f"Average total reward    : {avg_reward:.3f}"
        if avg_reward is not None
        else "Average total reward    : N/A"
    )
    print(
        f"Max total reward        : {max_reward:.3f}"
        if max_reward is not None
        else "Max total reward        : N/A"
    )
    print(
        f"Average episode length  : {avg_steps:.3f}"
        if avg_steps is not None
        else "Average episode length  : N/A"
    )
    print()

    print("Action distribution (steps table):")
    cursor.execute("SELECT action, COUNT(*) FROM steps GROUP BY action ORDER BY action")
    action_counts = cursor.fetchall()
    if not action_counts:
        print("  No steps recorded.")
    else:
        total_steps = sum(count for _, count in action_counts)
        for action, count in action_counts:
            frac = count / total_steps if total_steps > 0 else 0.0
            print(f"  action={action}: count={count}, {frac:.2%} of all steps")
    print()

    if _table_exists(conn, "reward_components"):
        print("Mean reward components:")
        cursor.execute(
            """
            SELECT
                AVG(health_reward),
                AVG(engagement_reward),
                AVG(positioning_reward),
                AVG(score_reward),
                AVG(bonus_reward),
                AVG(end_of_episode_reward),
                AVG(total_reward)
            FROM reward_components
            """
        )
        rc = cursor.fetchone()
        if rc is None or all(value is None for value in rc):
            print("  No reward_components recorded.")
        else:
            labels = [
                "health_reward",
                "engagement_reward",
                "positioning_reward",
                "score_reward",
                "bonus_reward",
                "end_of_episode_reward",
                "total_reward (per step)",
            ]
            for label, value in zip(labels, rc):
                print(
                    f"  {label:24s}: {value:.6f}"
                    if value is not None
                    else f"  {label:24s}: N/A"
                )
        print()

    if _table_exists(conn, "ppo_updates"):
        cols = _table_columns(conn, "ppo_updates")
        cursor.execute("SELECT COUNT(*) FROM ppo_updates")
        num_updates = cursor.fetchone()[0]
        print(f"PPO updates logged       : {num_updates}")
        if num_updates > 0:
            fields = [
                "mean_kl",
                "clip_fraction",
                "explained_variance",
                "mean_entropy",
                "grad_norm",
                "lr",
                "nonfinite_grad_steps",
                "skipped_optimizer_steps",
                "transitions_in_update",
                "learnable_transitions_in_update",
                "fragments_in_update",
                "update_wall_seconds",
                "rollout_wall_seconds",
                "ray_get_wall_seconds",
                "tbptt_forward_calls",
                "tbptt_group_mean_active_chunks",
                "cpu_to_gpu_transfer_wall_seconds",
                "chunk_pack_wall_seconds",
                "replay_forward_wall_seconds",
                "backward_optimizer_wall_seconds",
                "payload_total_mib",
                "cuda_peak_allocated_bytes",
            ]
            present = [field for field in fields if field in cols]
            if present:
                order_col = "update_id" if "update_id" in cols else "rowid"
                tail_query = (
                    f"SELECT {', '.join(present)} FROM ppo_updates "
                    f"ORDER BY {order_col} DESC LIMIT 25"
                )
                tail_rows = conn.execute(tail_query).fetchall()
                print("Late PPO summary (last 25 updates):")
                column_to_values = {
                    name: [row[idx] for row in tail_rows]
                    for idx, name in enumerate(present)
                }
                for name, values in column_to_values.items():
                    clean = []
                    for value in values:
                        if value is None:
                            continue
                        numeric = float(value)
                        if name == "grad_norm" and not math.isfinite(numeric):
                            continue
                        clean.append(numeric)
                    if clean:
                        print(f"  {name:24s}: {sum(clean) / len(clean):.6f}")
                    elif name == "grad_norm":
                        inf_count = sum(
                            1
                            for value in values
                            if value is not None and not math.isfinite(float(value))
                        )
                        print(f"  {name:24s}: no finite values ({inf_count} inf/nan)")
                if {
                    "transitions_in_update",
                    "update_wall_seconds",
                }.issubset(column_to_values):
                    pairs = [
                        (steps, seconds)
                        for steps, seconds in zip(
                            column_to_values["transitions_in_update"],
                            column_to_values["update_wall_seconds"],
                        )
                        if steps is not None and seconds not in (None, 0)
                    ]
                    if pairs:
                        values = [float(steps) / float(seconds) for steps, seconds in pairs]
                        print(f"  {'learner steps/sec':24s}: {sum(values) / len(values):.6f}")
                if "cuda_peak_allocated_bytes" in column_to_values:
                    clean = [
                        float(value) / float(1024**3)
                        for value in column_to_values["cuda_peak_allocated_bytes"]
                        if value is not None
                    ]
                    if clean:
                        print(f"  {'cuda peak alloc GiB':24s}: {sum(clean) / len(clean):.6f}")
                if "grad_norm" in column_to_values:
                    inf_count = sum(
                        1
                        for value in column_to_values["grad_norm"]
                        if value is not None and not math.isfinite(float(value))
                    )
                    print(f"  {'grad_norm inf/nan':24s}: {inf_count}")
        print()

    if _table_exists(conn, "eval_runs"):
        cursor.execute("SELECT COUNT(*) FROM eval_runs")
        num_evals = cursor.fetchone()[0]
        print(f"Evaluation runs logged   : {num_evals}")
        if num_evals > 0:
            cols = _table_columns(conn, "eval_runs")
            present = [
                field
                for field in [
                    "eval_id",
                    "episode_index",
                    "mean_reward",
                    "std_reward",
                    "deterministic",
                ]
                if field in cols
            ]
            row = conn.execute(
                f"SELECT {', '.join(present)} FROM eval_runs "
                "ORDER BY eval_id DESC LIMIT 1"
            ).fetchone()
            if row:
                latest = dict(zip(present, row))
                episode_text = latest.get("episode_index", "n/a")
                mean_text = latest.get("mean_reward", "n/a")
                std_text = latest.get("std_reward", "n/a")
                det_text = latest.get("deterministic", "n/a")
                print(
                    "Latest eval             : "
                    f"ep={episode_text}, mean={mean_text}, std={std_text}, "
                    f"deterministic={det_text}"
                )
        print()

    conn.close()
    print("=== End of DB analysis ===")


def _pick_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "agent_state",
            "policy_net_state_dict",
            "model_state_dict",
            "state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if torch is not None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def inspect_checkpoint(ckpt_path: str) -> None:
    if torch is None:
        raise RuntimeError("torch is not installed in this environment.")

    print(f"=== Checkpoint analysis for {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("Top-level checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    print()

    state_dict = _pick_state_dict(ckpt)
    if state_dict is None:
        print("Could not automatically identify a model state_dict.")
        return

    print("Interpreting part of checkpoint as model state_dict.")
    print(f"Number of tensors in state_dict: {len(state_dict)}")
    print()

    total_params = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        numel = tensor.numel()
        total_params += numel
        stats_tensor = tensor.float().view(-1)
        std = (
            stats_tensor.std(unbiased=False).item()
            if stats_tensor.numel() > 1
            else 0.0
        )
        shape_str = "x".join(str(dimension) for dimension in tensor.shape)
        print(
            f"{name:40s} | shape={shape_str:15s} | "
            f"numel={numel:8d} | "
            f"mean={stats_tensor.mean().item():+.4f} | "
            f"std={std:.4f} | "
            f"min={stats_tensor.min().item():+.4f} | "
            f"max={stats_tensor.max().item():+.4f}"
        )

    print()
    print(f"Total number of parameters: {total_params:,}")
    print("=== End of checkpoint analysis ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick helper for training DBs and checkpoints.",
    )
    parser.add_argument(
        "--mode",
        choices=["db", "pth"],
        default="db",
        help="What to analyze.",
    )
    parser.add_argument("--run-name", default=None, help="Run name under models/.")
    parser.add_argument("--db", type=str, help="Path to training_logs.db.")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint.")
    parser.add_argument(
        "--which",
        choices=["auto", "best", "checkpoint"],
        default="auto",
        help="Checkpoint choice when resolving from --run-name.",
    )
    args = parser.parse_args()

    if args.mode == "db":
        analyze_db(_resolve_db_path(args))
    else:
        inspect_checkpoint(_resolve_checkpoint_path(args))


if __name__ == "__main__":
    main()
