"""Two-tier policy snapshot writer for the Ray training path.

Writes models/<run>/snapshots/policy_u{N}.pth on a dense-early /
sparse-later schedule, where N is the durable policy_version
(agent.ppo.update_count, survives resume). Snapshots contain policy
weights + extractor/normalizer state + metadata and deliberately NO
optimizer or scheduler state, so they stay small (~a few MiB) and are
lineage artifacts, not resume points. checkpoint.pth/best_checkpoint.pth
semantics are untouched.

Extractor state is mandatory in every snapshot: a snapshot whose
normalizers report count=0 would silently feed the policy un-normalized
features at eval time (the count=0 lesson). The Ray caller must sync
actor normalizer stats into the learner extractor before saving.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_DIR_NAME = "snapshots"
SNAPSHOT_FILE_PREFIX = "policy_u"


@dataclass(frozen=True)
class SnapshotSchedule:
    """Dense-early / sparse-later cadence, in policy_version units.

    A version N is due when:
    - N <= dense_until and N % dense_every == 0, or
    - N >  dense_until and N % sparse_every == 0.
    An interval of 0 disables that tier; both 0 disables snapshots.
    """

    dense_every: int = 0
    dense_until: int = 0
    sparse_every: int = 0

    @classmethod
    def from_distributed_config(cls, get_value) -> "SnapshotSchedule":
        """Build from a `(name, fallback) -> value` config getter, e.g.
        ray_train's _distributed_value. Absent keys leave snapshots off."""
        return cls(
            dense_every=int(get_value("snapshot_dense_every_updates", 0) or 0),
            dense_until=int(get_value("snapshot_dense_until_update", 0) or 0),
            sparse_every=int(get_value("snapshot_sparse_every_updates", 0) or 0),
        )

    @property
    def enabled(self) -> bool:
        return self.dense_every > 0 or self.sparse_every > 0

    def is_due(self, policy_version: int) -> bool:
        version = int(policy_version)
        if version <= 0:
            return False
        if self.dense_every > 0 and version <= self.dense_until:
            return version % self.dense_every == 0
        if self.sparse_every > 0 and version > self.dense_until:
            return version % self.sparse_every == 0
        return False


def snapshot_path(run_dir: str | os.PathLike, policy_version: int) -> Path:
    return (
        Path(run_dir)
        / SNAPSHOT_DIR_NAME
        / f"{SNAPSHOT_FILE_PREFIX}{int(policy_version)}.pth"
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _config_hash(run_dir: str | os.PathLike) -> str | None:
    """Short digest of the run's effective_config.json (fallback: the
    config.yaml copy saved at run start)."""
    for name in ("effective_config.json", "config.yaml"):
        path = Path(run_dir) / name
        if path.exists():
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            except OSError:
                continue
    return None


def build_snapshot_payload(
    agent,
    *,
    episode: int,
    run_dir: str | os.PathLike,
    run_name: str = "",
) -> dict:
    """Snapshot dict. Key names match save_checkpoint's so existing
    tooling (pick_state_dict, collect_checkpoint_metadata, the dashboard
    checkpoint tab) reads snapshots without changes."""
    extractor_state = agent.extractor.state_dict()
    if not isinstance(extractor_state, dict) or not extractor_state:
        raise ValueError(
            "Refusing to snapshot without extractor state: normalizer "
            "stats are mandatory in every snapshot (count=0 lesson).",
        )
    policy_version = int(getattr(agent.ppo, "update_count", 0))
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "agent_state": agent.policy.state_dict(),
        "extractor_state": extractor_state,
        "episode": int(episode),
        "policy_version": policy_version,
        "global_update_index": policy_version,
        "policy_protocol_version": POLICY_PROTOCOL_VERSION,
        "policy_input_schema": POLICY_INPUT_SCHEMA,
        "run_name": str(run_name),
        "wall_time_unix": float(time.time()),
        "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(run_dir),
    }


def save_policy_snapshot(
    agent,
    *,
    run_dir: str | os.PathLike,
    episode: int,
    run_name: str = "",
) -> Path | None:
    """Atomically write the snapshot for the agent's current
    policy_version. Never raises: snapshotting is an observability
    feature and must not take down a training run."""
    try:
        payload = build_snapshot_payload(
            agent,
            episode=episode,
            run_dir=run_dir,
            run_name=run_name,
        )
        path = snapshot_path(run_dir, payload["policy_version"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".pth.tmp")
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
        print(f"Policy snapshot saved to {path}.")
        return path
    except Exception as exc:  # noqa: BLE001 - deliberate: never kill training
        print(f"Warning: policy snapshot failed (training continues): {exc}")
        return None
