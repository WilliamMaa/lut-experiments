#!/usr/bin/env python3
"""
download_model.py

从 HuggingFace 下载模型，支持断点续传，存储到指定目录（不是 cache 目录，避免重复占用）。

用法：
  python -u download_model.py \
    --model_id Qwen/Qwen3.6-35B-A3B \
    --local_dir /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --resume

需要 huggingface-cli 或 huggingface_hub。
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_storage(path: str, min_gb: float = 100.0):
    """检查目标目录所在分区的可用空间。"""
    abs_path = Path(path).expanduser().resolve()
    abs_path.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(abs_path)
    free_gb = stat.free / (1024 ** 3)
    print(f"Target: {abs_path}")
    print(f"Free space: {free_gb:.1f} GB")
    print(f"Required: {min_gb:.1f} GB")
    if free_gb < min_gb:
        print(f"ERROR: Not enough free space. Need at least {min_gb:.1f} GB.")
        sys.exit(1)
    return free_gb


def download_with_cli(model_id: str, local_dir: str, resume: bool, token: str = None):
    """使用 huggingface-cli download。"""
    cmd = [
        "huggingface-cli", "download",
        model_id,
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False",
    ]
    if resume:
        cmd.append("--resume-download")
    if token:
        cmd.extend(["--token", token])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_with_hub(model_id: str, local_dir: str, resume: bool, token: str = None):
    """使用 huggingface_hub snapshot_download。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    print(f"Using huggingface_hub.snapshot_download for {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=resume,
        token=token,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token（如果需要）")
    parser.add_argument("--min_gb", type=float, default=100.0, help="所需最小空间（GB）")
    parser.add_argument("--method", choices=["cli", "hub", "auto"], default="auto",
                        help="下载方式：cli 用 huggingface-cli，hub 用 snapshot_download")
    args = parser.parse_args()

    # 检查空间
    check_storage(args.local_dir, args.min_gb)

    # 选择下载方式
    method = args.method
    if method == "auto":
        if shutil.which("huggingface-cli"):
            method = "cli"
        else:
            method = "hub"

    print(f"Download method: {method}")

    if method == "cli":
        ok = download_with_cli(args.model_id, args.local_dir, args.resume, args.token)
    else:
        ok = download_with_hub(args.model_id, args.local_dir, args.resume, args.token)

    if ok:
        print(f"\nDownloaded to {args.local_dir}")
        print("You can now run extract_shared_expert.py")
    else:
        print("\nDownload failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
