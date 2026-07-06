"""Registry core: enumerate, inspect, and diff policy .pth artifacts.

Reads the same checkpoint layout train.py/ray_train.py write
(agent_state / extractor_state / policy_version metadata keys) and the
snapshots written by Utility.checkpoint_snapshots. Eval scores are
joined from the run's training_logs.db (eval_runs.policy_version) when
available.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tools.analysis.analyze_pth import (
    collect_checkpoint_metadata,
    collect_extractor_state_rows,
    collect_time_constant_rows,
    pick_state_dict,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
SNAPSHOT_FILE_RE = re.compile(r"^policy_u(\d+)\.pth$")


@dataclass
class RegistryEntry:
    path: Path
    kind: str  # "snapshot" | "checkpoint" | "best"
    policy_version: int | None
    size_bytes: int
    metadata: dict = field(default_factory=dict)
    eval_mean: float | None = None
    eval_policy_version: int | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_mib(self) -> float:
        return self.size_bytes / float(1024**2)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _entry_from_path(path: Path, kind: str) -> RegistryEntry:
    ckpt = _load_checkpoint(path)
    metadata = collect_checkpoint_metadata(ckpt)
    version = metadata.get("policy_version", metadata.get("global_update_index"))
    return RegistryEntry(
        path=path,
        kind=kind,
        policy_version=int(version) if version is not None else None,
        size_bytes=path.stat().st_size,
        metadata=metadata,
    )


def _eval_rows(run_dir: Path) -> list[tuple[int, float]]:
    """(policy_version, mean_reward) rows from the run DB, version-sorted."""
    db_path = run_dir / "training_logs.db"
    if not db_path.exists():
        return []
    try:
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT policy_version, mean_reward FROM eval_runs "
                "WHERE policy_version IS NOT NULL ORDER BY policy_version",
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [(int(version), float(mean)) for version, mean in rows]


def _attach_eval_scores(entries: list[RegistryEntry], run_dir: Path) -> None:
    """Join each entry to the latest eval at or before its version."""
    rows = _eval_rows(run_dir)
    if not rows:
        return
    for entry in entries:
        if entry.policy_version is None:
            continue
        best = None
        for version, mean in rows:
            if version <= entry.policy_version:
                best = (version, mean)
            else:
                break
        if best is not None:
            entry.eval_policy_version, entry.eval_mean = best[0], best[1]


def list_run_entries(
    run_name: str,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> list[RegistryEntry]:
    """All lineage artifacts of a run: snapshots (version-sorted), then
    checkpoint.pth and best_checkpoint.pth."""
    run_dir = Path(models_dir) / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    entries: list[RegistryEntry] = []
    snapshot_dir = run_dir / "snapshots"
    if snapshot_dir.is_dir():
        snapshots = []
        for path in snapshot_dir.iterdir():
            match = SNAPSHOT_FILE_RE.match(path.name)
            if match:
                snapshots.append((int(match.group(1)), path))
        for _version, path in sorted(snapshots):
            entries.append(_entry_from_path(path, "snapshot"))
    for filename, kind in (
        ("checkpoint.pth", "checkpoint"),
        ("best_checkpoint.pth", "best"),
    ):
        path = run_dir / filename
        if path.exists():
            entries.append(_entry_from_path(path, kind))

    _attach_eval_scores(entries, run_dir)
    return entries


def resolve_ref(
    ref: str,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Resolve a CLI reference to a .pth path.

    Accepted forms:
    - an existing file path;
    - "<run>:u<N>"       -> models/<run>/snapshots/policy_u<N>.pth
    - "<run>:checkpoint" -> models/<run>/checkpoint.pth
    - "<run>:best"       -> models/<run>/best_checkpoint.pth
    - "<run>:latest" or bare "<run>" -> highest-version snapshot,
      falling back to checkpoint.pth.
    """
    direct = Path(ref)
    if direct.is_file():
        return direct

    run_name, _, selector = ref.partition(":")
    selector = selector or "latest"
    run_dir = Path(models_dir) / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"'{ref}' is neither a file nor a run under {models_dir}",
        )

    if selector == "checkpoint":
        path = run_dir / "checkpoint.pth"
    elif selector == "best":
        path = run_dir / "best_checkpoint.pth"
    elif selector.startswith("u") and selector[1:].isdigit():
        path = run_dir / "snapshots" / f"policy_{selector}.pth"
    elif selector == "latest":
        snapshot_dir = run_dir / "snapshots"
        candidates = []
        if snapshot_dir.is_dir():
            for child in snapshot_dir.iterdir():
                match = SNAPSHOT_FILE_RE.match(child.name)
                if match:
                    candidates.append((int(match.group(1)), child))
        if candidates:
            return max(candidates)[1]
        path = run_dir / "checkpoint.pth"
    else:
        raise ValueError(
            f"unknown selector '{selector}' in '{ref}' "
            "(use u<N>, checkpoint, best, or latest)",
        )
    if not path.exists():
        raise FileNotFoundError(f"resolved '{ref}' to missing file: {path}")
    return path


def _module_param_counts(state_dict: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        module = name.split(".", 1)[0]
        counts[module] = counts.get(module, 0) + int(tensor.numel())
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def show_entry(path: str | Path) -> dict:
    """Metadata, per-module parameter counts, learned alpha/beta summary,
    and extractor-normalizer summary for one artifact."""
    path = Path(path)
    ckpt = _load_checkpoint(path)
    state_dict = pick_state_dict(ckpt) or {}
    return {
        "path": str(path),
        "metadata": collect_checkpoint_metadata(ckpt),
        "module_param_counts": _module_param_counts(state_dict),
        "time_constants": collect_time_constant_rows(state_dict),
        "extractor": collect_extractor_state_rows(ckpt),
    }


def _run_dir_for(path: Path) -> Path:
    """models/<run>/snapshots/x.pth -> models/<run>; models/<run>/x.pth
    -> models/<run>."""
    parent = path.resolve().parent
    if parent.name == "snapshots":
        return parent.parent
    return parent


def _flatten_config(value, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            sub_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_config(sub_value, sub_prefix))
    else:
        flat[prefix] = value
    return flat


def _config_diff(path_a: Path, path_b: Path) -> dict:
    """Key-level diff of the two runs' effective_config.json (empty when
    either side has none)."""
    configs = []
    for path in (path_a, path_b):
        config_path = _run_dir_for(path) / "effective_config.json"
        if not config_path.exists():
            return {}
        try:
            configs.append(
                _flatten_config(json.loads(config_path.read_text(encoding="utf-8"))),
            )
        except (OSError, json.JSONDecodeError):
            return {}
    flat_a, flat_b = configs
    changed = {
        key: (flat_a[key], flat_b[key])
        for key in sorted(set(flat_a) & set(flat_b))
        if flat_a[key] != flat_b[key]
    }
    only_a = sorted(set(flat_a) - set(flat_b))
    only_b = sorted(set(flat_b) - set(flat_a))
    return {"changed": changed, "only_a": only_a, "only_b": only_b}


def diff_entries(path_a: str | Path, path_b: str | Path) -> dict:
    """Layer-wise weight deltas (L2 / relative L2 / cosine, sorted by L2
    descending), plus metadata and config diffs. Works across runs."""
    path_a, path_b = Path(path_a), Path(path_b)
    ckpt_a, ckpt_b = _load_checkpoint(path_a), _load_checkpoint(path_b)
    state_a = pick_state_dict(ckpt_a) or {}
    state_b = pick_state_dict(ckpt_b) or {}

    keys_a = {k for k, v in state_a.items() if isinstance(v, torch.Tensor)}
    keys_b = {k for k, v in state_b.items() if isinstance(v, torch.Tensor)}
    layers = []
    shape_mismatches = []
    for name in sorted(keys_a & keys_b):
        tensor_a = state_a[name].detach().float()
        tensor_b = state_b[name].detach().float()
        if tensor_a.shape != tensor_b.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "shape_a": tuple(tensor_a.shape),
                    "shape_b": tuple(tensor_b.shape),
                },
            )
            continue
        flat_a = tensor_a.reshape(-1)
        flat_b = tensor_b.reshape(-1)
        norm_a = float(flat_a.norm())
        norm_b = float(flat_b.norm())
        l2 = float((flat_b - flat_a).norm())
        cosine = float(
            torch.dot(flat_a, flat_b) / (max(norm_a * norm_b, 1e-12)),
        )
        layers.append(
            {
                "name": name,
                "shape": tuple(tensor_a.shape),
                "numel": int(flat_a.numel()),
                "l2": l2,
                "rel_l2": l2 / max(norm_a, 1e-12),
                "cosine": cosine,
            },
        )
    layers.sort(key=lambda row: -row["l2"])

    meta_a = collect_checkpoint_metadata(ckpt_a)
    meta_b = collect_checkpoint_metadata(ckpt_b)
    metadata_diff = {
        key: (meta_a.get(key), meta_b.get(key))
        for key in sorted(set(meta_a) | set(meta_b))
        if meta_a.get(key) != meta_b.get(key)
    }

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "layers": layers,
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "shape_mismatches": shape_mismatches,
        "metadata_diff": metadata_diff,
        "config_diff": _config_diff(path_a, path_b),
        "total_l2": float(
            sum(row["l2"] ** 2 for row in layers) ** 0.5,
        ),
    }
