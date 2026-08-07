#!/usr/bin/env python3
"""
Reconstruct metadata.json from existing sample_*.pt files.

Use when collect_shared_expert_data.py was killed before writing metadata.json.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--hook_path", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_dir = output_dir / "input"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = sorted(input_dir.glob("sample_*.pt"))
    num_files = len(files)
    total_tokens = 0
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=True)
        total_tokens += x.shape[0]

    hook_path = args.hook_path or f"model.model.layers[{args.layer_idx}].mlp.shared_expert"

    metadata = {
        "layer_idx": args.layer_idx,
        "num_files": num_files,
        "total_tokens": total_tokens,
        "hook_path": hook_path,
        "max_new_tokens": args.max_new_tokens,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Reconstructed metadata.json: {metadata_path}")
    print(f"  num_files: {num_files}")
    print(f"  total_tokens: {total_tokens}")


if __name__ == "__main__":
    main()
