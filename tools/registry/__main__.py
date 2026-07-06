"""CLI for the checkpoint/snapshot registry.

Usage:
  python -m tools.registry list <run> [--models-dir DIR]
  python -m tools.registry show <ref> [--models-dir DIR]
  python -m tools.registry diff <ref-a> <ref-b> [--top N] [--models-dir DIR]

A <ref> is a .pth path or "<run>[:selector]" with selector one of
u<N> / checkpoint / best / latest (default latest).
"""

from __future__ import annotations

import argparse

from tools.registry.core import (
    DEFAULT_MODELS_DIR,
    diff_entries,
    list_run_entries,
    resolve_ref,
    show_entry,
)


def _cmd_list(args: argparse.Namespace) -> None:
    entries = list_run_entries(args.run, models_dir=args.models_dir)
    if not entries:
        print(f"No .pth artifacts found for run '{args.run}'.")
        return
    header = (
        f"{'file':32s} {'kind':10s} {'version':>8s} {'episode':>9s} "
        f"{'MiB':>7s} {'eval_mean':>10s} {'git':8s} {'config':12s} {'saved':20s}"
    )
    print(header)
    print("-" * len(header))
    for entry in entries:
        meta = entry.metadata
        eval_text = "-"
        if entry.eval_mean is not None:
            marker = (
                "" if entry.eval_policy_version == entry.policy_version else "~"
            )
            eval_text = f"{marker}{entry.eval_mean:.1f}"
        print(
            f"{entry.name:32s} "
            f"{entry.kind:10s} "
            f"{str(entry.policy_version if entry.policy_version is not None else '-'):>8s} "
            f"{str(meta.get('episode', '-')):>9s} "
            f"{entry.size_mib:>7.1f} "
            f"{eval_text:>10s} "
            f"{str(meta.get('git_commit', '-'))[:8]:8s} "
            f"{str(meta.get('config_hash', '-')):12s} "
            f"{str(meta.get('wall_time_iso', '-')):20s}"
        )
    print(
        "\n(eval_mean joined from training_logs.db eval_runs; '~' = latest "
        "eval at an earlier policy_version, not this exact one)",
    )


def _cmd_show(args: argparse.Namespace) -> None:
    path = resolve_ref(args.ref, models_dir=args.models_dir)
    info = show_entry(path)
    print(f"=== {info['path']} ===\n")

    print("Metadata:")
    for key in sorted(info["metadata"]):
        print(f"  {key}: {info['metadata'][key]}")

    print("\nParameters by module:")
    total = sum(info["module_param_counts"].values())
    for module, count in info["module_param_counts"].items():
        share = 100.0 * count / max(total, 1)
        print(f"  {module:32s} {count:>12,d}  ({share:.1f}%)")
    print(f"  {'TOTAL':32s} {total:>12,d}")

    if info["time_constants"]:
        print("\nLearned alpha/beta:")
        for row in info["time_constants"]:
            print(
                f"  {row['name']:44s} {row['kind']:5s} "
                f"mean={row['mean']:+.4f} std={row['std']:.4f} "
                f"min={row['min']:+.4f} max={row['max']:+.4f}",
            )

    print("\nExtractor normalizers:")
    for name, summary in info["extractor"].items():
        print(
            f"  {name}: count={summary['count']:.1f} warm={summary['warm']} "
            f"fields={len(summary['rows'])}",
        )


def _cmd_diff(args: argparse.Namespace) -> None:
    path_a = resolve_ref(args.ref_a, models_dir=args.models_dir)
    path_b = resolve_ref(args.ref_b, models_dir=args.models_dir)
    diff = diff_entries(path_a, path_b)

    print(f"A: {diff['path_a']}")
    print(f"B: {diff['path_b']}")
    print(f"\nTotal weight delta (global L2): {diff['total_l2']:.4f}")

    if diff["metadata_diff"]:
        print("\nMetadata differences (A -> B):")
        for key, (value_a, value_b) in diff["metadata_diff"].items():
            print(f"  {key}: {value_a!r} -> {value_b!r}")

    config_diff = diff["config_diff"]
    if config_diff.get("changed") or config_diff.get("only_a") or config_diff.get("only_b"):
        print("\nConfig differences (effective_config.json, A -> B):")
        for key, (value_a, value_b) in config_diff.get("changed", {}).items():
            print(f"  {key}: {value_a!r} -> {value_b!r}")
        for key in config_diff.get("only_a", []):
            print(f"  {key}: only in A")
        for key in config_diff.get("only_b", []):
            print(f"  {key}: only in B")

    for label, names in (
        ("Only in A", diff["only_in_a"]),
        ("Only in B", diff["only_in_b"]),
    ):
        if names:
            print(f"\n{label} ({len(names)}):")
            for name in names[:20]:
                print(f"  {name}")
            if len(names) > 20:
                print(f"  ... and {len(names) - 20} more")

    if diff["shape_mismatches"]:
        print(f"\nShape mismatches ({len(diff['shape_mismatches'])}):")
        for row in diff["shape_mismatches"]:
            print(f"  {row['name']}: {row['shape_a']} vs {row['shape_b']}")

    top = args.top
    print(f"\nTop {min(top, len(diff['layers']))} layers by L2 delta:")
    print(f"  {'layer':44s} {'l2':>10s} {'rel_l2':>8s} {'cosine':>8s}")
    for row in diff["layers"][:top]:
        print(
            f"  {row['name']:44s} {row['l2']:>10.4f} "
            f"{row['rel_l2']:>8.4f} {row['cosine']:>8.4f}",
        )
    unchanged = sum(1 for row in diff["layers"] if row["l2"] == 0.0)
    print(
        f"\n{len(diff['layers'])} shared layers compared, "
        f"{unchanged} bit-identical.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.registry",
        description="List, inspect, and diff policy .pth artifacts.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Models root (default: <repo>/models).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="Table of a run's snapshots and checkpoints.",
    )
    list_parser.add_argument("run", help="Run name under the models dir.")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser(
        "show", help="Metadata / param counts / alpha-beta for one artifact.",
    )
    show_parser.add_argument(
        "ref", help="Path or <run>[:u<N>|:checkpoint|:best|:latest].",
    )
    show_parser.set_defaults(func=_cmd_show)

    diff_parser = subparsers.add_parser(
        "diff", help="Layer-wise weight/metadata/config diff of two artifacts.",
    )
    diff_parser.add_argument("ref_a")
    diff_parser.add_argument("ref_b")
    diff_parser.add_argument(
        "--top", type=int, default=25, help="Layers to print (default 25).",
    )
    diff_parser.set_defaults(func=_cmd_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
