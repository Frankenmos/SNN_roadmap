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
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from agent_core.policy_protocol import POLICY_INPUT_SCHEMA, POLICY_PROTOCOL_VERSION
from Utility.model_git import register_snapshot, sha256_file

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
    def from_distributed_config(cls, get_value) -> SnapshotSchedule:
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
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
    return None


def _phase_event(run_dir: Path, phase_id: int) -> dict:
    path = run_dir / "resume_events.jsonl"
    if not path.exists():
        return {}
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read phase identity from {path}: {exc}") from exc
    matching = [row for row in rows if int(row.get("phase_id", -1)) == int(phase_id)]
    if not matching:
        raise ValueError(f"No resume event for phase_id={phase_id} in {path}")
    return matching[-1]


def build_snapshot_payload(
    agent,
    *,
    episode: int,
    run_dir: str | os.PathLike,
    run_name: str = "",
    phase_id: int = 0,
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
    run_dir = Path(run_dir)
    run_name = str(run_name or run_dir.name)
    manifest_path = run_dir / "run_manifest.json"
    manifest_sha256 = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.exists()
        else None
    )
    phase_event = _phase_event(run_dir, phase_id)
    source = dict(phase_event.get("source", {}))
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
        "phase_id": int(phase_id),
        "wall_time_unix": float(time.time()),
        "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(run_dir),
        "run_manifest_sha256": manifest_sha256,
        "source_identity": source,
        "phase_effective_config_sha256": phase_event.get(
            "effective_config_sha256",
        ),
        "policy_config": (
            agent.policy.resolved_config()
            if hasattr(agent.policy, "resolved_config")
            else None
        ),
    }


def save_policy_snapshot(
    agent,
    *,
    run_dir: str | os.PathLike,
    episode: int,
    run_name: str = "",
    phase_id: int = 0,
) -> Path | None:
    """Atomically write the snapshot for the agent's current
    policy_version. Never raises: snapshotting is an observability
    feature and must not take down a training run."""
    temp_path = None
    try:
        run_name = str(run_name or Path(run_dir).name)
        payload = build_snapshot_payload(
            agent,
            episode=episode,
            run_dir=run_dir,
            run_name=run_name,
            phase_id=phase_id,
        )
        path = snapshot_path(run_dir, payload["policy_version"])
        staging_dir = Path(run_dir) / "model_git" / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        temp_path = staging_dir / f"policy_u{payload['policy_version']}.{os.getpid()}.tmp"
        torch.save(payload, temp_path)
        object_path, event = register_snapshot(
            temp_path,
            run_dir=run_dir,
            metadata={
                "run_name": str(run_name),
                "phase_id": int(phase_id),
                "policy_version": int(payload["policy_version"]),
                "episode": int(episode),
                "config_sha256": payload.get("config_hash"),
                "run_manifest_sha256": payload.get("run_manifest_sha256"),
                "source_identity": payload.get("source_identity", {}),
                "phase_effective_config_sha256": payload.get(
                    "phase_effective_config_sha256",
                ),
            },
        )

        # Compatibility ref for existing analysis tools. It is write-once;
        # divergent resumes at the same update remain separately addressable
        # through their content hashes in model_git/index.jsonl.
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with object_path.open("rb") as source_handle:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(descriptor, "wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, 1024 * 1024)
        elif sha256_file(path) != event["artifact_sha256"]:
            print(
                f"Policy compatibility ref {path} already names different content; "
                "preserved it and recorded this fork only in model_git.",
            )
        print(
            f"Policy snapshot {event['artifact_sha256'][:12]} saved to {object_path}; "
            f"compatibility ref: {path}.",
        )
        return path
    except Exception as exc:  # noqa: BLE001 - deliberate: never kill training
        print(f"Warning: policy snapshot failed (training continues): {exc}")
        return None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
