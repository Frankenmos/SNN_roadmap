#!/usr/bin/env python
"""
Checkpoint inspector plus optional 2D weight-map visualizer.
"""

from __future__ import annotations

import argparse
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
