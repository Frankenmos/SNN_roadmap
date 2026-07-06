"""Registry core: enumerate, inspect, and diff policy .pth artifacts.

Reads the same checkpoint layout train.py/ray_train.py write
(agent_state / extractor_state / policy_version metadata keys) and the
snapshots written by Utility.checkpoint_snapshots. Eval scores are
joined from the run's training_logs.db (eval_runs.policy_version) when
available.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


# ----------------------------------------------------------------------
# Export: run bundle for the 3D architecture explorer (live mode)
# ----------------------------------------------------------------------
ARCH_EXPLORER_PUBLIC_JSON = (
    _REPO_ROOT / "tools" / "viz" / "arch_explorer" / "public" / "run_data.json"
)

# Column selections are intersected with the actual table schema (PRAGMA)
# so legacy DBs missing newer columns still export cleanly.
_HISTORY_COLUMNS = (
    "global_update_index",
    "mean_entropy",
    "mean_kl",
    "grad_norm",
    "grad_norm_trunk",
    "grad_norm_actor_head",
    "grad_norm_critic_head",
    "grad_norm_target_head",
    "rollout_policy_no_op_count",
    "rollout_policy_left_click_count",
    "rollout_policy_right_click_count",
    "sil_gate_open_fraction",
    "sil_buffer_size",
)
_ENTRY_ROW_COLUMNS = _HISTORY_COLUMNS + (
    "clip_fraction",
    "rollout_feedback_smart_executed_count",
    "rollout_feedback_near_enemy_smart_count",
    "rollout_feedback_enemy_health_drop_after_smart_count",
    "sil_loss",
    "sil_steps_replayed",
)
_EVAL_EXPORT_COLUMNS = (
    "episode_index",
    "policy_version",
    "num_episodes",
    "mean_reward",
    "std_reward",
    "deterministic",
)


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    try:
        return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def _time_constant_rows_with_effective(state_dict: dict) -> list[dict]:
    """collect_time_constant_rows plus effective_* stats.

    snnTorch's Synaptic/Leaky forward uses alpha.clamp(0, 1) /
    beta.clamp(0, 1), so the clamped value is what the network actually
    runs with (raw values can drift outside [0, 1] during training)."""
    rows = []
    for row in collect_time_constant_rows(state_dict):
        tensor = state_dict[row["name"]].detach().cpu().float().reshape(-1)
        effective = tensor.clamp(0.0, 1.0)
        row = dict(row)
        row["effective_mean"] = float(effective.mean().item())
        row["effective_min"] = float(effective.min().item())
        row["effective_max"] = float(effective.max().item())
        rows.append(row)
    return rows


def _update_row_at_version(
    conn: sqlite3.Connection,
    columns: list[str],
    version: int | None,
) -> dict | None:
    """The ppo_updates row at the greatest global_update_index <= version
    (policy_version == global_update_index by construction)."""
    if version is None or "global_update_index" not in columns:
        return None
    wanted = [name for name in _ENTRY_ROW_COLUMNS if name in columns]
    try:
        row = conn.execute(
            f"SELECT {', '.join(wanted)} FROM ppo_updates "
            "WHERE global_update_index <= ? "
            "ORDER BY global_update_index DESC LIMIT 1",
            (version,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return dict(zip(wanted, row))


def _update_history(
    conn: sqlite3.Connection,
    columns: list[str],
    max_points: int,
) -> dict | None:
    """Strided per-update series (always keeps the final row)."""
    wanted = [name for name in _HISTORY_COLUMNS if name in columns]
    if "global_update_index" not in wanted:
        return None
    try:
        rows = conn.execute(
            f"SELECT {', '.join(wanted)} FROM ppo_updates "
            "ORDER BY global_update_index",
        ).fetchall()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    stride = max(1, math.ceil(len(rows) / max(max_points, 2)))
    sampled = list(rows[::stride])
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return {
        "total_updates": len(rows),
        "stride": stride,
        "series": {
            name: [row[index] for row in sampled]
            for index, name in enumerate(wanted)
        },
    }


def _eval_export_rows(conn: sqlite3.Connection) -> list[dict]:
    columns = _table_columns(conn, "eval_runs")
    wanted = [name for name in _EVAL_EXPORT_COLUMNS if name in columns]
    if not wanted:
        return []
    try:
        rows = conn.execute(
            f"SELECT {', '.join(wanted)} FROM eval_runs ORDER BY eval_id",
        ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(zip(wanted, row)) for row in rows]


def _export_config(run_dir: Path) -> dict:
    """Headline settings from effective_config.json, including the SNN
    alpha/beta init values the learned constants are compared against."""
    config_path = run_dir / "effective_config.json"
    if not config_path.exists():
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    model = raw.get("model") or {}
    ppo = raw.get("ppo") or {}
    reward = raw.get("reward") or {}
    return {
        "reward_name": reward.get("name"),
        "spatial_head_type": model.get("spatial_head_type"),
        "lr": ppo.get("lr"),
        "sil_enabled": ppo.get("sil_enabled"),
        "snn_init": {
            "fast_alpha": model.get("fast_token_snn_alpha"),
            "fast_beta": model.get("fast_token_snn_beta"),
            "slow_alpha": model.get("slow_token_snn_alpha"),
            "slow_beta": model.get("slow_token_snn_beta"),
        },
    }


def _json_sanitize(value):
    """Replace non-finite floats with None: sqlite/torch can hold NaN/inf,
    and JSON.parse in the explorer rejects bare NaN tokens."""
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_run_data(
    run_name: str,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    max_points: int = 400,
) -> dict:
    """JSON-safe bundle of a run's artifact lineage (with learned time
    constants per artifact), per-update training history, and eval scores.
    Consumed by tools/viz/arch_explorer's live mode."""
    run_dir = Path(models_dir) / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    # Same artifact set as list_run_entries, but one torch.load per file
    # (metadata and time constants come from the same checkpoint dict).
    paths: list[tuple[str, Path]] = []
    snapshot_dir = run_dir / "snapshots"
    if snapshot_dir.is_dir():
        snapshots = []
        for path in snapshot_dir.iterdir():
            match = SNAPSHOT_FILE_RE.match(path.name)
            if match:
                snapshots.append((int(match.group(1)), path))
        for _version, path in sorted(snapshots):
            paths.append(("snapshot", path))
    for filename, kind in (
        ("checkpoint.pth", "checkpoint"),
        ("best_checkpoint.pth", "best"),
    ):
        path = run_dir / filename
        if path.exists():
            paths.append((kind, path))

    conn = _connect_ro(run_dir / "training_logs.db")
    update_columns = _table_columns(conn, "ppo_updates") if conn else []
    eval_rows = _eval_rows(run_dir)

    entries = []
    module_param_counts: dict[str, int] = {}
    for kind, path in paths:
        ckpt = _load_checkpoint(path)
        metadata = collect_checkpoint_metadata(ckpt)
        state_dict = pick_state_dict(ckpt) or {}
        version = metadata.get(
            "policy_version", metadata.get("global_update_index"),
        )
        version = int(version) if version is not None else None
        module_param_counts = (
            _module_param_counts(state_dict) or module_param_counts
        )

        eval_mean = eval_version = None
        if version is not None:
            for row_version, mean in eval_rows:
                if row_version <= version:
                    eval_version, eval_mean = row_version, mean
                else:
                    break

        selector = f"u{version}" if kind == "snapshot" else kind
        entries.append(
            {
                "ref": f"{run_name}:{selector}",
                "file": path.name,
                "kind": kind,
                "policy_version": version,
                "episode": metadata.get("episode"),
                "wall_time_iso": metadata.get("wall_time_iso"),
                "git_commit": metadata.get("git_commit"),
                "size_mib": round(path.stat().st_size / float(1024**2), 2),
                "eval_mean": eval_mean,
                "eval_policy_version": eval_version,
                "time_constants": _time_constant_rows_with_effective(
                    state_dict,
                ),
                "update_row": (
                    _update_row_at_version(conn, update_columns, version)
                    if conn
                    else None
                ),
            },
        )

    history = _update_history(conn, update_columns, max_points) if conn else None
    evals = _eval_export_rows(conn) if conn else []
    if conn:
        conn.close()

    return _json_sanitize(
        {
            "schema_version": 1,
            "kind": "arch-explorer-run-data",
            "run": run_name,
            "generated_iso": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            "config": _export_config(run_dir),
            "module_param_counts": module_param_counts,
            "entries": entries,
            "history": history,
            "evals": evals,
        },
    )


def write_run_data_json(
    run_name: str,
    out_path: str | Path | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    max_points: int = 400,
) -> Path:
    """Export and write run_data.json (default: the explorer's public/
    directory, so `npm run dev` serves it immediately)."""
    data = export_run_data(
        run_name, models_dir=models_dir, max_points=max_points,
    )
    out = Path(out_path) if out_path else ARCH_EXPLORER_PUBLIC_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out
