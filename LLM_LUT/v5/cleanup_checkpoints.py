"""
Safely clean up intermediate fine-tune checkpoints, keeping only the best epoch.

Usage:
    cd LLM_LUT/v5
    python cleanup_checkpoints.py --root /data/mingyu/LLM_LUT/v5/results --dry-run
    python cleanup_checkpoints.py --root /data/mingyu/LLM_LUT/v5/results
    python cleanup_checkpoints.py --root /data/mingyu/LLM_LUT/v4/results --dry-run

It reads summary.json in each result directory, finds the best epoch by PPL,
and deletes other epoch checkpoints.
"""

import os
import json
import argparse
from pathlib import Path


def find_summary_dirs(root: str):
    """Yield directories that contain a summary.json."""
    for dirpath, dirnames, filenames in os.walk(root):
        if "summary.json" in filenames:
            yield dirpath


def best_epoch_from_summary(summary_path: str):
    with open(summary_path, "r") as f:
        summary = json.load(f)
    after = summary.get("after", [])
    if not after:
        return None, None
    # Choose best by lowest PPL; tie-break by lowest KL
    best = min(
        after,
        key=lambda x: (x.get("ppl", float("inf")), x.get("kl", float("inf"))),
    )
    return best.get("epoch"), best


def collect_paths_to_keep(checkpoints: dict):
    """Given a checkpoints dict from summary, return absolute paths to keep."""
    keep = set()
    for rel_path in checkpoints.values():
        abs_path = os.path.abspath(rel_path)
        keep.add(abs_path)
        # If it's a LUT directory, keep everything inside
        if os.path.isdir(abs_path):
            for root, _, files in os.walk(abs_path):
                for name in files:
                    keep.add(os.path.join(root, name))
    return keep


def cleanup_dir(result_dir: str, dry_run: bool):
    summary_path = os.path.join(result_dir, "summary.json")
    if not os.path.exists(summary_path):
        return 0, 0

    best_epoch, best_entry = best_epoch_from_summary(summary_path)
    if best_epoch is None:
        print(f"[SKIP] {result_dir}: no 'after' epochs in summary.json")
        return 0, 0

    keep_paths = collect_paths_to_keep(best_entry.get("checkpoints", {}))
    keep_paths.add(os.path.abspath(summary_path))

    deleted = []
    skipped = []
    for name in os.listdir(result_dir):
        abs_path = os.path.abspath(os.path.join(result_dir, name))
        if abs_path in keep_paths:
            skipped.append(name)
            continue
        # Don't delete non-checkpoint files like logs
        if not (
            name.endswith(".pt")
            or name.endswith("_lut")
            or name.endswith("_down_proj")
            or name.endswith("_o_proj")
        ):
            skipped.append(name)
            continue
        deleted.append(abs_path)

    total_bytes = 0
    for p in deleted:
        if os.path.isfile(p):
            total_bytes += os.path.getsize(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    total_bytes += os.path.getsize(os.path.join(root, f))

    print(f"\n[DIR] {result_dir}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Keep: {len(skipped)} items")
    print(f"  Delete: {len(deleted)} items, ~{total_bytes / 1024**3:.2f} GiB")

    if not dry_run:
        for p in deleted:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                import shutil
                shutil.rmtree(p)
        print(f"  -> deleted")

    return len(deleted), total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing result subdirs")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted")
    args = parser.parse_args()

    total_deleted = 0
    total_bytes = 0
    for result_dir in find_summary_dirs(args.root):
        n, b = cleanup_dir(result_dir, args.dry_run)
        total_deleted += n
        total_bytes += b

    print(f"\n{'='*60}")
    print(f"Total items to delete: {total_deleted}")
    print(f"Total space to free: {total_bytes / 1024**3:.2f} GiB")
    if args.dry_run:
        print("This was a dry-run. Rerun without --dry-run to actually delete.")


if __name__ == "__main__":
    main()
