"""Checkpoint/snapshot registry: list, show, and diff .pth lineage
artifacts written by training (checkpoint.pth, best_checkpoint.pth,
snapshots/policy_u{N}.pth).

CLI: python -m tools.registry {list,show,diff} ...
"""

from tools.registry.core import (
    RegistryEntry,
    diff_entries,
    list_run_entries,
    resolve_ref,
    show_entry,
)

__all__ = [
    "RegistryEntry",
    "diff_entries",
    "list_run_entries",
    "resolve_ref",
    "show_entry",
]
