#!/usr/bin/env python
"""
Simple .pth checkpoint inspector for the SNN-PPO policy.

Usage:
    python analyze_pth.py checkpoint.pth
"""

import sys
import math
import torch


def pick_state_dict(ckpt):
    """
    Try to guess which part of the checkpoint is the model state_dict.
    Adapts to your current format: ckpt["agent_state"].
    """
    if isinstance(ckpt, dict):
        # Your current saving convention
        for key in ["agent_state", "policy_net_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # Fallback: raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    return None


def inspect_checkpoint(path: str) -> None:
    print(f"=== Checkpoint analysis for {path} ===")
    ckpt = torch.load(path, map_location="cpu")

    print("Top-level checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    print()

    state_dict = pick_state_dict(ckpt)
    if state_dict is None:
        print("Could not find a model state_dict inside this checkpoint.")
        print("Top-level object might be something else (e.g., full trainer).")
        return

    print("Interpreting part of checkpoint as model state_dict.")
    print(f"Number of tensors in state_dict: {len(state_dict)}")
    print()

    total_params = 0
    param_stats = {}

    for name, tensor in state_dict.items():
        t = tensor.float().view(-1)  # flatten for stats
        numel = t.numel()
        total_params += numel

        mean = t.mean().item()
        if numel > 1:
            # use unbiased=False to avoid degrees-of-freedom warnings
            std = t.std(unbiased=False).item()
        else:
            std = 0.0  # scalar: no meaningful std

        min_val = t.min().item()
        max_val = t.max().item()

        param_stats[name] = {
            "shape": tuple(tensor.shape),
            "numel": numel,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

    print(f"Total number of parameters: {total_params:,}")
    print()

    # Pretty-print per-parameter summary
    for name, stats in param_stats.items():
        shape_str = "x".join(str(d) for d in stats["shape"]) if stats["shape"] else ""
        std_str = f"{stats['std']:.4f}" if stats["numel"] > 1 else "0.0000"

        print(f"{name:35s} | shape={shape_str:15s} | "
              f"numel={stats['numel']:8d} | "
              f"mean={stats['mean']:+.4f} | "
              f"std={std_str} | "
              f"min={stats['min']:+.4f} | "
              f"max={stats['max']:+.4f}")

    print("=== End of checkpoint analysis ===")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_pth.py checkpoint.pth")
        raise SystemExit(1)

    ckpt_path = sys.argv[1]
    inspect_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
