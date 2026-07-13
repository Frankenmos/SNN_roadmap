"""Immutable run identity and append-only launch provenance.

The SQLite database is the measurement stream.  These files answer a
different question: which exact experiment produced those measurements, and
what changed whenever the run was resumed?
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"
RESUME_EVENTS_FILENAME = "resume_events.jsonl"


class RunManifestError(RuntimeError):
    """Raised when an existing immutable manifest is malformed or incompatible."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", *args],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def collect_source_state(repo_root: str | os.PathLike) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    status = _run_git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    diff = _run_git(repo_root, ["diff", "--binary", "HEAD", "--"])
    status_lines = (
        [] if status is None else [line for line in status.splitlines() if line]
    )
    return {
        "git_commit": None if commit is None else commit.strip() or None,
        "git_dirty": None if status is None else bool(status_lines),
        "git_status": status_lines,
        "git_status_sha256": (
            None
            if status is None
            else hashlib.sha256(status.encode("utf-8")).hexdigest()
        ),
        "git_diff_sha256": (
            None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest()
        ),
    }


def collect_runtime_state() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        runtime.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            },
        )
    except Exception as exc:  # pragma: no cover - torch is present in production
        runtime["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return runtime


def _redact_argv(argv: Iterable[str]) -> list[str]:
    """Keep launch commands useful without persisting obvious CLI secrets."""
    redacted = []
    redact_next = False
    sensitive_fragments = ("token", "secret", "password", "api-key", "api_key")
    for raw in argv:
        value = str(raw)
        lowered = value.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-") and any(
            part in lowered for part in sensitive_fragments
        ):
            if "=" in value:
                redacted.append(value.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(value)
                redact_next = True
            continue
        redacted.append(value)
    return redacted


def build_manifest_payload(
    *,
    run_name: str,
    effective_config: dict[str, Any],
    launch_mode: str,
    repo_root: str | os.PathLike,
    config_path: str | os.PathLike | None,
    argv: Iterable[str] | None = None,
    resolved_launch: dict[str, Any] | None = None,
    adopted_existing_run: bool = False,
) -> dict[str, Any]:
    config_path = None if config_path is None else str(Path(config_path).resolve())
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "kind": "snn-run-manifest",
        "run_name": str(run_name),
        "created_at_utc": utc_now_iso(),
        "adopted_existing_run": bool(adopted_existing_run),
        "launch": {
            "mode": str(launch_mode),
            "argv": _redact_argv(sys.argv if argv is None else argv),
            "working_directory": str(Path.cwd().resolve()),
            "resolved": dict(resolved_launch or {}),
        },
        "config_source": {
            "path": config_path,
            "sha256": file_sha256(config_path),
        },
        "effective_config": effective_config,
        "effective_config_sha256": json_sha256(effective_config),
        "source": collect_source_state(repo_root),
        "runtime": collect_runtime_state(),
    }


def ensure_run_manifest(
    run_dir: str | os.PathLike,
    payload: dict[str, Any],
) -> tuple[Path, bool]:
    """Write the manifest once. Existing manifests are validated, never changed."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_MANIFEST_FILENAME
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunManifestError(
                f"Existing immutable run manifest is unreadable: {path}: {exc}",
            ) from exc
        if existing.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            raise RunManifestError(
                f"Unsupported run manifest schema in {path}: {existing.get('schema_version')!r}",
            ) from None
        if str(existing.get("run_name")) != str(payload.get("run_name")):
            raise RunManifestError(
                f"Run manifest name mismatch in {path}: "
                f"{existing.get('run_name')!r} != {payload.get('run_name')!r}",
            ) from None
        return path, False

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path, True


def next_phase_id(run_dir: str | os.PathLike) -> int:
    """Return the next monotonically increasing launch phase for this run."""
    path = Path(run_dir) / RESUME_EVENTS_FILENAME
    if not path.exists():
        return 0
    highest = -1
    _line_number = 0
    try:
        for _line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            phase_id = row.get("phase_id")
            if phase_id is None:
                phase_id = highest + 1
            highest = max(highest, int(phase_id))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RunManifestError(
            f"Cannot allocate the next run phase from {path} at line "
            f"{_line_number or '?'}: {exc}",
        ) from exc
    return highest + 1


def append_resume_event(
    run_dir: str | os.PathLike,
    *,
    event_type: str,
    effective_config: dict[str, Any],
    launch_mode: str,
    repo_root: str | os.PathLike,
    config_path: str | os.PathLike | None,
    checkpoint_present: bool,
    phase_id: int,
    argv: Iterable[str] | None = None,
    resolved_launch: dict[str, Any] | None = None,
) -> Path:
    """Append one self-contained launch record; previous records stay untouched."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RESUME_EVENTS_FILENAME
    config_path = None if config_path is None else str(Path(config_path).resolve())
    event = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "event": str(event_type),
        "phase_id": int(phase_id),
        "timestamp_utc": utc_now_iso(),
        "launch": {
            "mode": str(launch_mode),
            "argv": _redact_argv(sys.argv if argv is None else argv),
            "working_directory": str(Path.cwd().resolve()),
            "resolved": dict(resolved_launch or {}),
        },
        "checkpoint_present": bool(checkpoint_present),
        "config_source": {
            "path": config_path,
            "sha256": file_sha256(config_path),
        },
        "effective_config": effective_config,
        "effective_config_sha256": json_sha256(effective_config),
        "source": collect_source_state(repo_root),
        "runtime": collect_runtime_state(),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
