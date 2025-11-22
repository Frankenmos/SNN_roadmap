#!/usr/bin/env python
"""
Small analysis helper for the SNN-PPO project.

Two main entrypoints:

1) DB-centric analysis (training_logs.db):
   - prints number of episodes
   - avg / max total_reward
   - avg episode length
   - action histogram
   - mean reward components

2) Checkpoint-centric analysis (.pth):
   - prints top-level keys of checkpoint
   - guesses the model state_dict
   - prints per-parameter: name, shape, number of elements
   - prints total number of parameters and basic stats
"""

import argparse
import sqlite3
from collections import Counter

import math

try:
    import torch
except ImportError:
    torch = None


# ----------------------
# DB-CENTRIC ANALYSIS
# ----------------------

def analyze_db(db_path: str) -> None:
    print(f"=== DB analysis for {db_path} ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Episodes summary
    cursor.execute("SELECT COUNT(*), AVG(total_reward), MAX(total_reward), AVG(steps) FROM episodes")
    row = cursor.fetchone()
    if row is None:
        print("No episodes found.")
        return

    num_episodes, avg_reward, max_reward, avg_steps = row
    print(f"Number of episodes      : {num_episodes}")
    print(f"Average total reward    : {avg_reward:.3f}" if avg_reward is not None else "Average total reward    : N/A")
    print(f"Max total reward        : {max_reward:.3f}" if max_reward is not None else "Max total reward        : N/A")
    print(f"Average episode length  : {avg_steps:.3f}" if avg_steps is not None else "Average episode length  : N/A")
    print()

    # Action distribution from steps table
    print("Action distribution (steps table):")
    cursor.execute("SELECT action, COUNT(*) FROM steps GROUP BY action ORDER BY action")
    action_counts = cursor.fetchall()
    if not action_counts:
        print("  No steps recorded.")
    else:
        total_steps = sum(c for _, c in action_counts)
        for action, count in action_counts:
            frac = count / total_steps if total_steps > 0 else 0.0
            print(f"  action={action}: count={count}, {frac:.2%} of all steps")
    print()

    # Reward components means
    print("Mean reward components (reward_components table):")
    cursor.execute("""
        SELECT
            AVG(health_reward),
            AVG(engagement_reward),
            AVG(positioning_reward),
            AVG(score_reward),
            AVG(bonus_reward),
            AVG(end_of_episode_reward),
            AVG(total_reward)
        FROM reward_components
    """)
    rc = cursor.fetchone()
    if rc is None or all(v is None for v in rc):
        print("  No reward_components recorded.")
    else:
        (h, e, p, s, b, end, tot) = rc
        print(f"  health_reward           : {h:.6f}" if h is not None else "  health_reward           : N/A")
        print(f"  engagement_reward       : {e:.6f}" if e is not None else "  engagement_reward       : N/A")
        print(f"  positioning_reward      : {p:.6f}" if p is not None else "  positioning_reward      : N/A")
        print(f"  score_reward            : {s:.6f}" if s is not None else "  score_reward            : N/A")
        print(f"  bonus_reward            : {b:.6f}" if b is not None else "  bonus_reward            : N/A")
        print(f"  end_of_episode_reward   : {end:.6f}" if end is not None else "  end_of_episode_reward   : N/A")
        print(f"  total_reward (per step) : {tot:.6f}" if tot is not None else "  total_reward (per step) : N/A")

    conn.close()
    print("=== End of DB analysis ===")


# ----------------------
# PTH-CENTRIC ANALYSIS
# ----------------------

def _pick_state_dict(ckpt):
    """
    Try to guess what part of the checkpoint is the model state_dict.
    Returns a dict of tensors if successful, otherwise None.
    """
    if isinstance(ckpt, dict):
        # Add "agent_state" because your PPO_CNN_run.py saves that
        for key in ["agent_state", "policy_net_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # Fallback: raw state_dict (all values are tensors)
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    return None



def inspect_checkpoint(ckpt_path: str) -> None:
    if torch is None:
        raise RuntimeError("torch is not installed in this environment, cannot inspect .pth")

    print(f"=== Checkpoint analysis for {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("Top-level checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    print()

    state_dict = _pick_state_dict(ckpt)
    if state_dict is None:
        print("Could not automatically identify a model state_dict inside the checkpoint.")
        print("If this checkpoint contains something else (e.g. full trainer object), "
              "you'll need to adapt this script.")
        return

    print("Interpreting part of checkpoint as model state_dict.")
    print(f"Number of tensors in state_dict: {len(state_dict)}")
    print()

    total_params = 0
    param_stats = {}

    for name, tensor in state_dict.items():
        numel = tensor.numel()
        shape = tuple(tensor.shape)
        total_params += numel

        # collect basic stats
        with torch.no_grad():
            mean = tensor.float().mean().item()
            std = tensor.float().std().item()
            min_val = tensor.float().min().item()
            max_val = tensor.float().max().item()

        param_stats[name] = {
            "shape": shape,
            "numel": numel,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

    print(f"Total number of parameters: {total_params:,}")
    print()

    # Print a short summary per parameter
    for name, stats in param_stats.items():
        shape_str = "x".join(str(d) for d in stats["shape"])
        print(f"{name:40s} | shape={shape_str:15s} | "
              f"numel={stats['numel']:8d} | "
              f"mean={stats['mean']:+.4f} | std={stats['std']:.4f} | "
              f"min={stats['min']:+.4f} | max={stats['max']:+.4f}")

    print("=== End of checkpoint analysis ===")


# ----------------------
# CLI
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Small helper to analyze training DB and model checkpoints.")
    parser.add_argument("--mode", choices=["db", "pth"], required=True,
                        help="What to analyze: 'db' for SQLite logs, 'pth' for a checkpoint.")
    parser.add_argument("--db", type=str, help="Path to SQLite training log (when mode=='db').")
    parser.add_argument("--ckpt", type=str, help="Path to .pth checkpoint (when mode=='pth').")

    args = parser.parse_args()

    if args.mode == "db":
        if not args.db:
            raise SystemExit("You must pass --db PATH when mode=='db'.")
        analyze_db(args.db)

    elif args.mode == "pth":
        if not args.ckpt:
            raise SystemExit("You must pass --ckpt PATH when mode=='pth'.")
        inspect_checkpoint(args.ckpt)


if __name__ == "__main__":
    main()
