#!/usr/bin/env python3
"""
cleanup_v6_outputs.py

清理 LLM_LUT/v6 实验过程中产生的旧输出目录和临时文件。

默认只扫描当前目录下匹配以下模式的目录：
  - outputs_ffn_lut_*
  - onpolicy_data_*
  - worstcase_*
  - v3_shared.log / v4_tail.log / collect_onpolicy*.log 等日志文件

用法：
  # 先 dry-run 看会删什么
  python cleanup_v6_outputs.py --dry-run

  # 确认删除
  python cleanup_v6_outputs.py --yes

  # 只删特定模式
  python cleanup_v6_outputs.py --yes --patterns "outputs_ffn_lut_layer1_*" "outputs_ffn_lut_layer39_*"

安全：
  - 默认不删除任何文件
  - 必须加 --yes 才会真正执行
  - 不会删除 .py、README、checkpoints 以外的 .pt 文件
  - 不会递归删除当前目录外的任何路径
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List


DEFAULT_DIR_PATTERNS = [
    "outputs_ffn_lut_*",
    "onpolicy_data_*",
    "onpolicy_layer39_*",
    "onpolicy_layer*_",
    "worstcase_*",
]

DEFAULT_FILE_PATTERNS = [
    "*.log",
    "candidate_prompts.jsonl",
]

PROTECTED_NAMES = {
    "build_candidate_pool.py",
    "build_lut_ffn_output.py",
    "build_lut_ffn_output_v3_independent.py",
    "build_lut_ffn_output_v3_shared_coarse.py",
    "build_tail_aware_hard_correction.py",
    "collect_onpolicy_data.py",
    "cleanup_v6_outputs.py",
    "diagnose_lut.py",
    "run_model_eval.py",
    "v6_replacement_engine.py",
    "requirements.txt",
    "README_build_lut_ffn_output.md",
    "README_run_model_eval.md",
    "README_build_candidate_pool.md",
    "example_custom_prompts.jsonl",
    ".git",
    ".gitignore",
    "AGENTS.md",
    "docs",
}


def match_any(name: str, patterns: List[str]) -> bool:
    import fnmatch
    for p in patterns:
        if fnmatch.fnmatch(name, p):
            return True
    return False


def scan_targets(root: Path, dir_patterns: List[str], file_patterns: List[str]):
    dirs_to_delete = []
    files_to_delete = []

    for item in root.iterdir():
        if item.name in PROTECTED_NAMES:
            continue
        if item.is_dir() and match_any(item.name, dir_patterns):
            dirs_to_delete.append(item)
        elif item.is_file() and match_any(item.name, file_patterns):
            files_to_delete.append(item)

    return dirs_to_delete, files_to_delete


def format_size(path: Path) -> str:
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def main():
    parser = argparse.ArgumentParser(description="Clean up v6 experimental outputs")
    parser.add_argument("--yes", action="store_true", help="Actually delete files (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--patterns", nargs="+", default=None, help="Override directory patterns to delete")
    parser.add_argument("--file-patterns", nargs="+", default=None, help="Override file patterns to delete")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}")
        sys.exit(1)

    dir_patterns = args.patterns if args.patterns else DEFAULT_DIR_PATTERNS
    file_patterns = args.file_patterns if args.file_patterns else DEFAULT_FILE_PATTERNS

    dirs, files = scan_targets(root, dir_patterns, file_patterns)

    if not dirs and not files:
        print(f"No matching files/directories found in {root}")
        sys.exit(0)

    print(f"Scanning: {root}")
    print(f"Directory patterns: {dir_patterns}")
    print(f"File patterns: {file_patterns}")
    print()

    if dirs:
        print(f"Directories to delete ({len(dirs)}):")
        for d in sorted(dirs):
            print(f"  [DIR]  {d.name}  ({format_size(d)})")
        print()

    if files:
        print(f"Files to delete ({len(files)}):")
        for f in sorted(files):
            print(f"  [FILE] {f.name}  ({format_size(f)})")
        print()

    if not args.yes:
        print("This is a dry-run. Add --yes to actually delete.")
        sys.exit(0)

    deleted_dirs = 0
    deleted_files = 0
    for d in dirs:
        shutil.rmtree(d)
        deleted_dirs += 1
    for f in files:
        f.unlink()
        deleted_files += 1

    print(f"Deleted {deleted_dirs} directories and {deleted_files} files.")


if __name__ == "__main__":
    main()
