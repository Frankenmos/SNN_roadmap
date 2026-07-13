"""CLI for the checkpoint/snapshot registry.

Usage:
  python -m tools.registry list <run> [--models-dir DIR]
  python -m tools.registry show <ref> [--models-dir DIR]
  python -m tools.registry diff <ref-a> <ref-b> [--top N] [--models-dir DIR]
  python -m tools.registry export <run> [--out PATH] [--max-points N]

A <ref> is a .pth path or "<run>[:selector]" with selector one of
u<N> / checkpoint / best / latest (default latest).

`export` writes run_data.json for the 3D architecture explorer's live
mode (default target: tools/viz/arch_explorer/public/run_data.json).
"""

from __future__ import annotations

import argparse

from tools.registry.core import (
    DEFAULT_MODELS_DIR,
    diff_entries,
    fork_run_lineage,
    list_run_entries,
    resolve_ref,
    show_entry,
    tag_ref,
    verify_run_registry,
    write_run_data_json,
)


def _cmd_list(args: argparse.Namespace) -> None:
    entries = list_run_entries(args.run, models_dir=args.models_dir)
    if not entries:
        print(f"No .pth artifacts found for run '{args.run}'.")
        return
    header = (
        f"{'file':32s} {'kind':10s} {'version':>8s} {'episode':>9s} "
        f"{'MiB':>7s} {'eval_mean':>10s} {'artifact':12s} {'parent':12s} "
        f"{'phase':>5s} {'saved':20s}"
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
            f"{str(entry.artifact_sha256 or '-')[:12]:12s} "
            f"{str(entry.parent_sha256 or '-')[:12]:12s} "
            f"{str(meta.get('phase_id', '-')):>5s} "
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
    print(f"Artifact SHA-256: {info['artifact_sha256']}")
    if info["lineage"]:
        print(f"Parent SHA-256: {info['lineage'].get('parent_sha256')}")
        print("Lineage identity:")
        for key in (
            "run_name",
            "phase_id",
            "policy_version",
            "config_sha256",
            "phase_effective_config_sha256",
            "run_manifest_sha256",
        ):
            print(f"  {key}: {info['lineage'].get(key)}")
        source = info["lineage"].get("source_identity") or {}
        print(f"  source.git_commit: {source.get('git_commit')}")
        print(f"  source.git_dirty: {source.get('git_dirty')}")
        print(f"  source.git_diff_sha256: {source.get('git_diff_sha256')}")

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
        print(
            f"\nConfig differences ({diff.get('config_diff_source', 'unknown')}, A -> B):",
        )
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


def _cmd_export(args: argparse.Namespace) -> None:
    out = write_run_data_json(
        args.run,
        out_path=args.out,
        models_dir=args.models_dir,
        max_points=args.max_points,
    )
    print(f"Wrote {out}")
    print(
        "Explorer live mode: `npm run dev` picks it up immediately; "
        "for `npm run preview` re-run `npm run build` (or export with "
        "--out <explorer>/dist/run_data.json).",
    )


def _cmd_tag(args: argparse.Namespace) -> None:
    event = tag_ref(
        args.run,
        args.tag,
        args.ref,
        models_dir=args.models_dir,
    )
    print(f"Tagged {event['artifact_sha256']} as {args.run}:tag/{args.tag}")


def _cmd_fork(args: argparse.Namespace) -> None:
    path = fork_run_lineage(
        args.child_run,
        args.parent_ref,
        models_dir=args.models_dir,
    )
    print(f"Initialized immutable fork ancestry at {path}")


def _cmd_verify(args: argparse.Namespace) -> None:
    result = verify_run_registry(args.run, models_dir=args.models_dir)
    print(
        f"{'OK' if result['ok'] else 'FAILED'}: {result['objects']} objects, "
        f"{result['index_events']} index events",
    )
    for error in result["errors"]:
        print(f"  {error}")
    if not result["ok"]:
        raise SystemExit(1)


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

    export_parser = subparsers.add_parser(
        "export",
        help="Write run_data.json for the 3D explorer's live mode.",
    )
    export_parser.add_argument("run", help="Run name under the models dir.")
    export_parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: tools/viz/arch_explorer/public/run_data.json).",
    )
    export_parser.add_argument(
        "--max-points",
        type=int,
        default=400,
        help="Max points per history series after downsampling (default 400).",
    )
    export_parser.set_defaults(func=_cmd_export)

    tag_parser = subparsers.add_parser(
        "tag", help="Create an immutable human-readable tag for an object.",
    )
    tag_parser.add_argument("run")
    tag_parser.add_argument("tag")
    tag_parser.add_argument("ref")
    tag_parser.set_defaults(func=_cmd_tag)

    fork_parser = subparsers.add_parser(
        "fork", help="Set a child run's parent object before its first snapshot.",
    )
    fork_parser.add_argument("child_run")
    fork_parser.add_argument("parent_ref")
    fork_parser.set_defaults(func=_cmd_fork)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify append-only index hashes and object contents.",
    )
    verify_parser.add_argument("run")
    verify_parser.set_defaults(func=_cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
