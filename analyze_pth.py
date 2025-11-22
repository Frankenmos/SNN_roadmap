#!/usr/bin/env python
"""
Simple .pth checkpoint inspector + 2D weight-map visualizer.

Usage:
    python analyze_pth.py checkpoint.pth

Requires:
    numpy
    matplotlib
    scikit-learn
"""

import sys
import math
import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def extract_weight_vectors(state_dict, max_points=5000):
    """
    Turn the state_dict into a list of weight vectors (one per neuron/filter)
    plus metadata for plotting.

    Returns:
        vectors:  list of 1D numpy arrays
        metadata: list of dicts with keys:
                  - 'layer' (str)
                  - 'name'  (full param name)
                  - 'unit_idx' (int) or None
                  - 'shape' (tuple)
    """
    vectors = []
    metadata = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # We mostly care about weights; you can relax this if you want.
        if not name.endswith("weight"):
            continue

        arr = tensor.detach().cpu().float().numpy()
        shape = arr.shape
        ndim = arr.ndim

        # Infer a "layer name" (everything before the last dot)
        if "." in name:
            layer = name.rsplit(".", 1)[0]
        else:
            layer = name

        # Linear: [out_features, in_features] -> row per neuron
        if ndim == 2:
            out_features = shape[0]
            for i in range(out_features):
                v = arr[i].ravel()
                vectors.append(v.copy())
                metadata.append({
                    "layer": layer,
                    "name": name,
                    "unit_idx": i,
                    "shape": shape,
                })

        # Conv or similar: [out_channels, ...] -> filter per out_channel
        elif ndim >= 3:
            out_channels = shape[0]
            for i in range(out_channels):
                v = arr[i].ravel()
                vectors.append(v.copy())
                metadata.append({
                    "layer": layer,
                    "name": name,
                    "unit_idx": i,
                    "shape": shape,
                })

        # 1D weights (e.g. BatchNorm, embedding row, etc.) -> treat as single vector
        elif ndim == 1:
            v = arr.ravel()
            vectors.append(v.copy())
            metadata.append({
                "layer": layer,
                "name": name,
                "unit_idx": None,
                "shape": shape,
            })

        # Scalar or weird shapes: still include, but as a single vector
        else:
            v = arr.ravel()
            vectors.append(v.copy())
            metadata.append({
                "layer": layer,
                "name": name,
                "unit_idx": None,
                "shape": shape,
            })

        if len(vectors) >= max_points:
            print(f"[extract_weight_vectors] Reached max_points={max_points}, stopping.")
            break

    if not vectors:
        print("[extract_weight_vectors] No weight vectors found.")
        return None, None

    return vectors, metadata


def build_2d_embedding(vectors, method="tsne"):
    """
    Build a 2D embedding (PCA + t-SNE) from a list of 1D numpy arrays.

    Args:
        vectors: list of 1D np arrays, possibly different lengths
        method:  currently only "tsne" implemented

    Returns:
        embedding: np.ndarray of shape [N, 2]
    """
    # Pad all vectors to the same length
    lengths = [v.size for v in vectors]
    max_len = max(lengths)
    N = len(vectors)

    X = np.zeros((N, max_len), dtype=np.float32)
    for i, v in enumerate(vectors):
        L = v.size
        X[i, :L] = v

    # Optional: standardize each dimension (helps PCA / t-SNE)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_std = (X - mean) / std

    # PCA to reduce dimension first
    n_components_pca = min(50, X_std.shape[0], X_std.shape[1])
    if n_components_pca < 2:
        # Not enough data, just take first two coords as "embedding"
        print("[build_2d_embedding] Not enough data for PCA; returning trivial 2D projection.")
        return X_std[:, :2]

    print(f"[build_2d_embedding] Running PCA -> {n_components_pca}D...")
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_std)

    if method == "tsne":
        # Perplexity must be < N; keep it sane.
        perplexity = min(30.0, max(5.0, (N - 1) / 3))
        print(f"[build_2d_embedding] Running t-SNE -> 2D (N={N}, perplexity={perplexity:.1f})...")
        tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    init="pca",   # 'pca' is supported in old & new versions
    verbose=1,
)

        X_2d = tsne.fit_transform(X_pca)
        return X_2d

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_embedding(embedding, metadata, title="Weight-space map"):
    """
    Plot the 2D embedding, colored by layer.
    """
    if embedding is None or metadata is None:
        print("[plot_embedding] Nothing to plot.")
        return

    embedding = np.asarray(embedding)
    assert embedding.shape[1] == 2

    layers = [m["layer"] for m in metadata]
    unique_layers = sorted(set(layers))
    layer_to_idx = {layer: i for i, layer in enumerate(unique_layers)}

    # Map each point to a color index
    color_indices = np.array([layer_to_idx[l] for l in layers])

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_indices,
        s=8,
        alpha=0.8,
        cmap="tab20",
    )

    # Build legend manually: one entry per layer (might want to truncate if too many)
    handles = []
    labels = []
    for layer, idx in layer_to_idx.items():
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="",
                markeredgewidth=0.0
            )
        )
        labels.append(layer)

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")

    # Limit legend if there are too many layers
    max_legend_items = 12
    if len(handles) > max_legend_items:
        print(f"[plot_embedding] Too many layers ({len(handles)}), truncating legend to {max_legend_items}.")
        handles = handles[:max_legend_items]
        labels = labels[:max_legend_items]

    plt.legend(handles, labels, fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def inspect_checkpoint(path: str, make_map: bool = True, max_points: int = 5000) -> None:
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
        if not isinstance(tensor, torch.Tensor):
            continue

        t = tensor.float().view(-1)  # flatten for stats
        numel = t.numel()
        total_params += numel

        mean = t.mean().item()
        if numel > 1:
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
    print()

    # Build and show 2D map
    if make_map:
        print("[inspect_checkpoint] Extracting weight vectors for 2D map...")
        vectors, metadata = extract_weight_vectors(state_dict, max_points=max_points)
        if vectors is None:
            return

        print("[inspect_checkpoint] Building 2D embedding...")
        embedding = build_2d_embedding(vectors, method="tsne")

        print("[inspect_checkpoint] Plotting embedding...")
        plot_embedding(embedding, metadata, title=f"Weight-space map: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_pth.py checkpoint.pth [--no-map] [--max-points N]")
        raise SystemExit(1)

    ckpt_path = sys.argv[1]
    make_map = "--no-map" not in sys.argv

    # Optional max_points argument
    max_points = 5000
    if "--max-points" in sys.argv:
        idx = sys.argv.index("--max-points")
        if idx + 1 < len(sys.argv):
            try:
                max_points = int(sys.argv[idx + 1])
            except ValueError:
                print("Invalid value for --max-points, using default 5000")

    inspect_checkpoint(ckpt_path, make_map=make_map, max_points=max_points)


if __name__ == "__main__":
    main()
    