"""
清理微调结果中的冗余 down_proj checkpoint，只保留每个实验 PPL 最优的 epoch。

用法:
    cd LLM_LUT/v4
    python cleanup_checkpoints.py --results-dir results --dry-run
    python cleanup_checkpoints.py --results-dir results

可选:
    --keep-last-n N   额外保留最后 N 个 epoch（默认 0）
    --dry-run         只打印会删除哪些文件，不真正删除
"""

import os
import json
import glob
import argparse
from pathlib import Path


def find_summary_dirs(results_dir: str):
    """Yield subdirectories that contain a summary.json file."""
    for entry in os.scandir(results_dir):
        if not entry.is_dir():
            continue
        summary_path = os.path.join(entry.path, "summary.json")
        if os.path.exists(summary_path):
            yield entry.path, summary_path


def best_epoch_from_summary(summary_path: str):
    """Return (best_epoch, best_ppl, total_epochs) based on lowest PPL."""
    with open(summary_path, "r") as f:
        summary = json.load(f)
    after = summary.get("after", [])
    if not after:
        return None, None, 0

    best = min(after, key=lambda x: x.get("ppl", float("inf")))
    return best.get("epoch"), best.get("ppl"), len(after)


def collect_ckpt_files(exp_dir: str):
    """Collect all l*_epoch*_down_proj.pt files in an experiment directory."""
    pattern = os.path.join(exp_dir, "l*_epoch*_down_proj.pt")
    return sorted(glob.glob(pattern))


def parse_epoch_from_filename(name: str):
    """Parse epoch number from 'l15_epoch10_down_proj.pt'."""
    # name format: l{layer}_epoch{epoch}_down_proj.pt
    prefix = "_epoch"
    suffix = "_down_proj.pt"
    if prefix not in name or not name.endswith(suffix):
        return None
    start = name.index(prefix) + len(prefix)
    end = name.index(suffix)
    try:
        return int(name[start:end])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--keep-last-n", type=int, default=0,
                        help="Also keep the last N epochs in addition to the best epoch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    total_deleted = 0
    total_bytes_saved = 0

    for exp_dir, summary_path in find_summary_dirs(results_dir):
        best_epoch, best_ppl, total_epochs = best_epoch_from_summary(summary_path)
        if best_epoch is None:
            print(f"[SKIP] {exp_dir}: no valid summary")
            continue

        keep_epochs = {best_epoch}
        if args.keep_last_n > 0:
            for e in range(max(1, total_epochs - args.keep_last_n + 1), total_epochs + 1):
                keep_epochs.add(e)

        files = collect_ckpt_files(exp_dir)
        to_delete = []
        for path in files:
            name = os.path.basename(path)
            epoch = parse_epoch_from_filename(name)
            if epoch is None or epoch not in keep_epochs:
                to_delete.append(path)

        if not to_delete:
            print(f"[KEEP] {exp_dir}: best epoch {best_epoch} (PPL {best_ppl:.2f}), nothing to delete")
            continue

        bytes_saved = 0
        for path in to_delete:
            bytes_saved += os.path.getsize(path)

        action = "Would delete" if args.dry_run else "Deleted"
        print(f"[{action}] {exp_dir}: best epoch {best_epoch} (PPL {best_ppl:.2f}), "
              f"keeping epochs {sorted(keep_epochs)}, deleting {len(to_delete)} files "
              f"({bytes_saved / (1024**2):.1f} MiB)")

        if not args.dry_run:
            for path in to_delete:
                os.remove(path)

        total_deleted += len(to_delete)
        total_bytes_saved += bytes_saved

    print("=" * 70)
    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} {total_deleted} checkpoint files, "
          f"saved {total_bytes_saved / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
