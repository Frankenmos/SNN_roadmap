#!/usr/bin/env python
"""
Checkpoint inspector plus optional 2D weight-map visualizer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _cfg_defaults() -> dict:
    try:
        from Utility.config import cfg

        return {
            "run_name": getattr(cfg.environment, "run_name", "") or "",
            "models_dir": getattr(cfg.environment, "models_dir", "models"),
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
            "checkpoint_filename": "checkpoint.pth",
            "best_checkpoint_filename": "best_checkpoint.pth",
        }


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    defaults = _cfg_defaults()
    if args.ckpt:
        return args.ckpt

    run_name = args.run_name or defaults["run_name"]
    if not run_name:
        raise SystemExit("Pass a checkpoint path or use --run-name NAME.")

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
        f"Could not resolve a checkpoint in {run_dir}. Pass --ckpt explicitly."
    )


def pick_state_dict(ckpt):
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


ENTITY_NORMALIZED_FIELDS = (
    "health",
    "health_ratio",
    "shield",
    "shield_ratio",
    "energy",
    "energy_ratio",
    "weapon_cooldown",
    "x",
    "y",
    "radius",
    "build_progress",
    "order_id_0",
    "order_id_1",
    "assigned_harvesters",
    "ideal_harvesters",
)
SELECTION_NORMALIZED_FIELDS = (
    "health",
    "shields",
    "energy",
    "transport_slots_taken",
    "build_progress",
)
DEFAULT_NORMALIZER_WARMUP_COUNT = 32.0


def collect_checkpoint_metadata(ckpt) -> dict:
    metadata = {}
    if not isinstance(ckpt, dict):
        return metadata

    for key, value in ckpt.items():
        if key in {
            "agent_state",
            "policy_net_state_dict",
            "model_state_dict",
            "state_dict",
            "optimizer_state",
            "scheduler_state",
            "extractor_state",
        }:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            metadata[key] = float(value.detach().cpu().item())
        elif isinstance(value, (list, tuple)) and len(value) <= 5:
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                metadata[key] = list(value)

    metadata["has_optimizer_state"] = isinstance(ckpt.get("optimizer_state"), dict)
    metadata["has_scheduler_state"] = isinstance(ckpt.get("scheduler_state"), dict)
    metadata["has_extractor_state"] = isinstance(ckpt.get("extractor_state"), dict)
    state_dict = pick_state_dict(ckpt)
    if isinstance(state_dict, dict):
        metadata["state_tensor_count"] = len(state_dict)
        metadata["parameter_count"] = int(
            sum(
                int(tensor.numel())
                for tensor in state_dict.values()
                if isinstance(tensor, torch.Tensor)
            )
        )
    return metadata


def collect_time_constant_rows(state_dict) -> list[dict]:
    rows = []
    if not isinstance(state_dict, dict):
        return rows
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "alpha" not in name and "beta" not in name:
            continue
        flat = tensor.detach().cpu().float().reshape(-1)
        rows.append(
            {
                "name": name,
                "kind": "alpha" if "alpha" in name else "beta",
                "module": name.rsplit(".", 1)[0] if "." in name else name,
                "numel": int(flat.numel()),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
            }
        )
    return rows


def _normalizer_rows(state: dict | None, field_names: tuple[str, ...]) -> dict:
    if not isinstance(state, dict):
        return {
            "count": 0.0,
            "warm": False,
            "rows": [],
        }

    count = float(state.get("count", 0.0))
    mean = np.asarray(state.get("mean", []), dtype=np.float64)
    m2 = np.asarray(state.get("m2", []), dtype=np.float64)
    size = min(len(field_names), len(mean), len(m2))
    denom = max(count - 1.0, 1.0)
    rows = []
    for index in range(size):
        variance = max(float(m2[index]) / denom, 0.0)
        rows.append(
            {
                "field": field_names[index],
                "mean": float(mean[index]),
                "std": float(math.sqrt(variance)),
                "m2": float(m2[index]),
            }
        )
    return {
        "count": count,
        "warm": count >= DEFAULT_NORMALIZER_WARMUP_COUNT,
        "rows": rows,
    }


def collect_extractor_state_rows(ckpt) -> dict:
    extractor_state = {}
    if isinstance(ckpt, dict):
        extractor_state = ckpt.get("extractor_state", {}) or {}
    return {
        "entity_normalizer": _normalizer_rows(
            extractor_state.get("entity_normalizer"),
            ENTITY_NORMALIZED_FIELDS,
        ),
        "selection_normalizer": _normalizer_rows(
            extractor_state.get("selection_normalizer"),
            SELECTION_NORMALIZED_FIELDS,
        ),
    }


def extract_weight_vectors(state_dict, max_points: int = 5000):
    vectors = []
    metadata = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not name.endswith("weight"):
            continue

        arr = tensor.detach().cpu().float().numpy()
        shape = arr.shape
        layer = name.rsplit(".", 1)[0] if "." in name else name

        if arr.ndim == 2:
            for index in range(shape[0]):
                vectors.append(arr[index].ravel().copy())
                metadata.append(
                    {
                        "layer": layer,
                        "name": name,
                        "unit_idx": index,
                        "shape": shape,
                    }
                )
        elif arr.ndim >= 3:
            for index in range(shape[0]):
                vectors.append(arr[index].ravel().copy())
                metadata.append(
                    {
                        "layer": layer,
                        "name": name,
                        "unit_idx": index,
                        "shape": shape,
                    }
                )
        else:
            vectors.append(arr.ravel().copy())
            metadata.append(
                {
                    "layer": layer,
                    "name": name,
                    "unit_idx": None,
                    "shape": shape,
                }
            )

        if len(vectors) >= max_points:
            print(f"[extract_weight_vectors] Reached max_points={max_points}, stopping.")
            break

    if not vectors:
        print("[extract_weight_vectors] No weight vectors found.")
        return None, None

    return vectors, metadata


def build_2d_embedding(vectors, method: str = "tsne") -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the 2D weight map. "
            "Install it or run with --no-map."
        ) from exc

    lengths = [vector.size for vector in vectors]
    max_len = max(lengths)
    num_vectors = len(vectors)

    matrix = np.zeros((num_vectors, max_len), dtype=np.float32)
    for index, vector in enumerate(vectors):
        matrix[index, : vector.size] = vector

    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-8
    matrix = (matrix - mean) / std

    n_components = min(50, matrix.shape[0], matrix.shape[1])
    if n_components < 2:
        print("[build_2d_embedding] Not enough data for PCA; using trivial projection.")
        return matrix[:, :2]

    print(f"[build_2d_embedding] Running PCA -> {n_components}D...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)

    if method != "tsne":
        raise ValueError(f"Unknown method: {method}")

    perplexity = min(30.0, max(5.0, (num_vectors - 1) / 3))
    print(
        f"[build_2d_embedding] Running t-SNE -> 2D "
        f"(N={num_vectors}, perplexity={perplexity:.1f})..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        verbose=1,
    )
    return tsne.fit_transform(reduced)


def plot_embedding(embedding: np.ndarray, metadata, title: str = "Weight-space map") -> None:
    if embedding is None or metadata is None:
        print("[plot_embedding] Nothing to plot.")
        return

    layers = [item["layer"] for item in metadata]
    unique_layers = sorted(set(layers))
    layer_to_idx = {layer: index for index, layer in enumerate(unique_layers)}
    color_indices = np.array([layer_to_idx[layer] for layer in layers])

    plt.figure(figsize=(9, 7))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_indices,
        s=8,
        alpha=0.8,
        cmap="tab20",
    )

    handles = []
    labels = []
    for layer in unique_layers[:12]:
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", markeredgewidth=0.0)
        )
        labels.append(layer)

    if len(unique_layers) > 12:
        print(
            f"[plot_embedding] Too many layers ({len(unique_layers)}), "
            "truncating legend to 12."
        )

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(handles, labels, fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def inspect_checkpoint(path: str, make_map: bool = True, max_points: int = 5000) -> None:
    print(f"=== Checkpoint analysis for {path} ===")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print("Top-level checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    print()

    state_dict = pick_state_dict(ckpt)
    if state_dict is None:
        print("Could not find a model state_dict inside this checkpoint.")
        return

    metadata = collect_checkpoint_metadata(ckpt)
    if metadata:
        print("Checkpoint metadata:")
        for key in sorted(metadata):
            print(f"  {key}: {metadata[key]}")
        print()

    time_constant_rows = collect_time_constant_rows(state_dict)
    if time_constant_rows:
        print("Learned alpha/beta parameters:")
        for row in time_constant_rows:
            print(
                f"  {row['name']:35s} | numel={row['numel']:4d} | "
                f"mean={row['mean']:+.4f} | std={row['std']:.4f} | "
                f"min={row['min']:+.4f} | max={row['max']:+.4f}"
            )
        print()

    extractor_rows = collect_extractor_state_rows(ckpt)
    for normalizer_name, summary in extractor_rows.items():
        if not summary["rows"]:
            continue
        print(
            f"{normalizer_name}: count={summary['count']:.1f} | "
            f"warm={summary['warm']}"
        )
        for row in summary["rows"]:
            print(
                f"  {row['field']:24s} | mean={row['mean']:+.4f} | "
                f"std={row['std']:.4f}"
            )
        print()

    print("Interpreting part of checkpoint as model state_dict.")
    print(f"Number of tensors in state_dict: {len(state_dict)}")
    print()

    total_params = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        flat = tensor.float().view(-1)
        numel = flat.numel()
        total_params += numel
        std = flat.std(unbiased=False).item() if numel > 1 else 0.0
        shape_str = "x".join(str(dimension) for dimension in tensor.shape)
        print(
            f"{name:35s} | shape={shape_str:15s} | "
            f"numel={numel:8d} | "
            f"mean={flat.mean().item():+.4f} | "
            f"std={std:.4f} | "
            f"min={flat.min().item():+.4f} | "
            f"max={flat.max().item():+.4f}"
        )

    print()
    print(f"Total number of parameters: {total_params:,}")
    print("=== End of checkpoint analysis ===")
    print()

    if make_map:
        print("[inspect_checkpoint] Extracting weight vectors for 2D map...")
        vectors, metadata = extract_weight_vectors(state_dict, max_points=max_points)
        if vectors is None:
            return

        print("[inspect_checkpoint] Building 2D embedding...")
        embedding = build_2d_embedding(vectors, method="tsne")

        print("[inspect_checkpoint] Plotting embedding...")
        plot_embedding(embedding, metadata, title=f"Weight-space map: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint inspector and optional weight-space visualizer.",
    )
    parser.add_argument("ckpt", nargs="?", help="Explicit checkpoint path.")
    parser.add_argument("--run-name", default=None, help="Run name under models/.")
    parser.add_argument(
        "--which",
        choices=["auto", "best", "checkpoint"],
        default="auto",
        help="Checkpoint choice when resolving from --run-name.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip the 2D weight-space map.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum number of unit/filter vectors to embed.",
    )
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint_path(args)
    inspect_checkpoint(
        ckpt_path,
        make_map=not args.no_map,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
