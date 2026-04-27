from __future__ import annotations

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class DirectoryLock:
    """Small cross-process lock based on atomic directory creation."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        timeout_seconds: float = 180.0,
        poll_seconds: float = 0.1,
        stale_seconds: float = 600.0,
    ) -> None:
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.stale_seconds = float(stale_seconds)
        self._acquired = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                os.mkdir(self.path)
                self._acquired = True
                self._write_owner_file()
                return
            except FileExistsError:
                self._remove_if_stale()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for SC2 runtime lock: {self.path}",
                    )
                time.sleep(self.poll_seconds)

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "DirectoryLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _write_owner_file(self) -> None:
        try:
            owner = self.path / "owner.txt"
            owner.write_text(
                f"pid={os.getpid()}\ncreated={time.time():.6f}\n",
                encoding="utf-8",
            )
        except OSError:
            return

    def _remove_if_stale(self) -> None:
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return
        if age < self.stale_seconds:
            return
        shutil.rmtree(self.path, ignore_errors=True)


def default_sc2_create_game_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "snn_sc2_create_game.lock"


@contextmanager
def sc2_create_game_lock(
    *,
    enabled: bool = True,
    path: str | os.PathLike | None = None,
):
    if not enabled:
        yield
        return
    lock_path = default_sc2_create_game_lock_path() if path is None else Path(path)
    with DirectoryLock(lock_path):
        yield
