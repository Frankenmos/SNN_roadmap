"""Content-addressed, append-only lineage storage for policy snapshots.

The mutable training checkpoint remains the resume mechanism. This module owns
small immutable policy objects and their auditable ancestry/index metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_GIT_DIR = "model_git"
OBJECTS_DIR = "objects"
INDEX_FILE = "index.jsonl"
TAGS_FILE = "tags.jsonl"
FORK_FILE = "fork_parent.json"
SCHEMA_VERSION = 1


class ModelGitError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_git_dir(run_dir: str | os.PathLike) -> Path:
    return Path(run_dir) / MODEL_GIT_DIR


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelGitError(f"Malformed {path} line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ModelGitError(f"Expected an object in {path} line {line_number}")
        rows.append(row)
    return rows


def _append_chained_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(path)
    event = dict(event)
    event["schema_version"] = SCHEMA_VERSION
    event["sequence"] = len(rows)
    event["previous_event_sha256"] = rows[-1].get("event_sha256") if rows else None
    event["event_sha256"] = hashlib.sha256(_canonical_bytes(event)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _copy_object_exclusive(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        if sha256_file(destination) != expected_sha256:
            raise ModelGitError(
                f"Hash collision or corrupted object: {destination}",
            ) from exc
        return
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _fork_parent(run_dir: Path) -> str | None:
    path = model_git_dir(run_dir) / FORK_FILE
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["parent_sha256"])
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ModelGitError(f"Unreadable fork ancestry at {path}: {exc}") from exc


def register_snapshot(
    staged_path: str | os.PathLike,
    *,
    run_dir: str | os.PathLike,
    metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Store a staged .pth once and append its lineage event."""
    staged_path = Path(staged_path)
    run_dir = Path(run_dir)
    root = model_git_dir(run_dir)
    artifact_sha256 = sha256_file(staged_path)
    object_path = root / OBJECTS_DIR / f"{artifact_sha256}.pth"
    _copy_object_exclusive(staged_path, object_path, artifact_sha256)

    prior = [row for row in read_jsonl(root / INDEX_FILE) if row.get("event") == "snapshot"]
    existing = next(
        (row for row in prior if row.get("artifact_sha256") == artifact_sha256),
        None,
    )
    if existing is not None:
        return object_path, existing
    parent_sha256 = (
        str(prior[-1]["artifact_sha256"])
        if prior
        else _fork_parent(run_dir)
    )
    event = _append_chained_event(
        root / INDEX_FILE,
        {
            **dict(metadata),
            "event": "snapshot",
            "timestamp_utc": _utc_now(),
            "artifact_sha256": artifact_sha256,
            "parent_sha256": parent_sha256,
            "object": f"{OBJECTS_DIR}/{artifact_sha256}.pth",
        },
    )
    return object_path, event


def initialize_fork(run_dir: str | os.PathLike, *, parent_sha256: str, parent_ref: str) -> Path:
    """Set ancestry once, before the child run has any registered objects."""
    root = model_git_dir(run_dir)
    if read_jsonl(root / INDEX_FILE):
        raise ModelGitError("Cannot set fork ancestry after snapshots exist")
    path = root / FORK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "model-git-fork-parent",
        "created_at_utc": _utc_now(),
        "parent_sha256": str(parent_sha256),
        "parent_ref": str(parent_ref),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ModelGitError(f"Fork ancestry already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def create_tag(run_dir: str | os.PathLike, *, tag: str, artifact_sha256: str) -> dict[str, Any]:
    if not tag or any(char.isspace() for char in tag):
        raise ModelGitError("Tag must be non-empty and contain no whitespace")
    path = model_git_dir(run_dir) / TAGS_FILE
    if any(row.get("tag") == tag for row in read_jsonl(path)):
        raise ModelGitError(f"Immutable tag already exists: {tag}")
    return _append_chained_event(
        path,
        {
            "event": "tag",
            "timestamp_utc": _utc_now(),
            "tag": tag,
            "artifact_sha256": artifact_sha256,
        },
    )


def verify_run(run_dir: str | os.PathLike) -> dict[str, Any]:
    import torch

    run_dir = Path(run_dir)
    root = model_git_dir(run_dir)
    errors = []
    rows = read_jsonl(root / INDEX_FILE)
    fork_path = root / FORK_FILE
    fork_parent = None
    if fork_path.exists():
        try:
            fork_payload = json.loads(fork_path.read_text(encoding="utf-8"))
            if fork_payload.get("schema_version") != SCHEMA_VERSION:
                errors.append("fork ancestry schema mismatch")
            if fork_payload.get("kind") != "model-git-fork-parent":
                errors.append("fork ancestry kind mismatch")
            fork_parent = str(fork_payload["parent_sha256"])
            parent_ref = str(fork_payload.get("parent_ref", ""))
            parent_run_name, separator, _selector = parent_ref.partition(":")
            if not separator or not parent_run_name:
                errors.append("fork ancestry parent ref mismatch")
            else:
                parent_rows = read_jsonl(
                    run_dir.parent
                    / parent_run_name
                    / MODEL_GIT_DIR
                    / INDEX_FILE,
                )
                if not any(
                    row.get("artifact_sha256") == fork_parent
                    for row in parent_rows
                ):
                    errors.append("fork ancestry parent object is not indexed")
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            errors.append(f"unreadable fork ancestry: {exc}")
    previous = None
    previous_artifact = fork_parent
    artifacts = set()
    for sequence, row in enumerate(rows):
        claimed = row.get("event_sha256")
        unsigned = dict(row)
        unsigned.pop("event_sha256", None)
        actual_event = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        if row.get("sequence") != sequence:
            errors.append(f"index sequence mismatch at row {sequence}")
        if row.get("previous_event_sha256") != previous:
            errors.append(f"index chain mismatch at row {sequence}")
        if claimed != actual_event:
            errors.append(f"index event hash mismatch at row {sequence}")
        previous = claimed
        if row.get("event") != "snapshot":
            errors.append(f"unknown index event at row {sequence}")
        artifact = str(row.get("artifact_sha256", ""))
        if len(artifact) != 64 or any(char not in "0123456789abcdef" for char in artifact):
            errors.append(f"invalid artifact hash at row {sequence}")
        if artifact in artifacts:
            errors.append(f"duplicate artifact event {artifact}")
        if row.get("parent_sha256") != previous_artifact:
            errors.append(f"parent lineage mismatch at row {sequence}")
        previous_artifact = artifact
        required_identity = {
            "run_name",
            "phase_id",
            "policy_version",
            "config_sha256",
            "run_manifest_sha256",
            "source_identity",
            "phase_effective_config_sha256",
        }
        missing_identity = sorted(required_identity - set(row))
        if missing_identity:
            errors.append(
                f"missing snapshot identity at row {sequence}: {', '.join(missing_identity)}",
            )
        if row.get("run_name") != run_dir.name:
            errors.append(f"run identity mismatch at row {sequence}")
        if not isinstance(row.get("phase_id"), int):
            errors.append(f"phase identity mismatch at row {sequence}")
        if not isinstance(row.get("source_identity"), dict):
            errors.append(f"source identity mismatch at row {sequence}")
        if artifact:
            artifacts.add(artifact)
            expected_object = f"{OBJECTS_DIR}/{artifact}.pth"
            if row.get("object") != expected_object:
                errors.append(f"object path mismatch at row {sequence}")
            object_path = root / expected_object
            if not object_path.exists():
                errors.append(f"missing object {artifact}")
            elif sha256_file(object_path) != artifact:
                errors.append(f"object hash mismatch {artifact}")
            else:
                try:
                    payload = torch.load(
                        object_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    for payload_key, event_key in (
                        ("run_name", "run_name"),
                        ("phase_id", "phase_id"),
                        ("policy_version", "policy_version"),
                        ("config_hash", "config_sha256"),
                        ("run_manifest_sha256", "run_manifest_sha256"),
                        ("source_identity", "source_identity"),
                        (
                            "phase_effective_config_sha256",
                            "phase_effective_config_sha256",
                        ),
                    ):
                        if payload.get(payload_key) != row.get(event_key):
                            errors.append(
                                f"object/index {event_key} mismatch at row {sequence}",
                            )
                    if not isinstance(payload.get("agent_state"), dict):
                        errors.append(f"object missing policy state at row {sequence}")
                    if not isinstance(payload.get("extractor_state"), dict):
                        errors.append(f"object missing extractor state at row {sequence}")
                except Exception as exc:  # noqa: BLE001 - verifier reports damage
                    errors.append(f"unreadable object {artifact}: {type(exc).__name__}: {exc}")
    object_dir = root / OBJECTS_DIR
    if object_dir.is_dir():
        indexed_names = {f"{artifact}.pth" for artifact in artifacts}
        for object_path in object_dir.glob("*.pth"):
            if object_path.name not in indexed_names:
                errors.append(f"orphan object {object_path.name}")
    previous_tag = None
    for sequence, tag in enumerate(read_jsonl(root / TAGS_FILE)):
        claimed = tag.get("event_sha256")
        unsigned = dict(tag)
        unsigned.pop("event_sha256", None)
        actual_event = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        if tag.get("sequence") != sequence:
            errors.append(f"tag sequence mismatch at row {sequence}")
        if tag.get("previous_event_sha256") != previous_tag:
            errors.append(f"tag chain mismatch at row {sequence}")
        if claimed != actual_event:
            errors.append(f"tag event hash mismatch at row {sequence}")
        previous_tag = claimed
        if str(tag.get("artifact_sha256")) not in artifacts:
            errors.append(f"tag {tag.get('tag')} points outside this run")
    return {
        "ok": not errors,
        "objects": len(artifacts),
        "index_events": len(rows),
        "errors": errors,
    }
