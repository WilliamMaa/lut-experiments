"""
Patch existing v5 checkpoints to add/store the correct `input_dim`.

This fixes checkpoints built before build_lut.py saved `input_dim` explicitly,
where finetune.py inferred it from max channel index and could get a value
smaller than the true hidden size (e.g. 3582 instead of 3584 for Qwen2.5-7B).

Usage:
    cd LLM_LUT/v5
    python patch_ckpt_input_dim.py \
        --checkpoint_root ../v5/outputs_tree_21_23 \
        --input_dim 3584
"""

import os
import glob
import argparse
import torch


def patch_checkpoint(path: str, input_dim: int, dry_run: bool = False):
    ckpt = torch.load(path, map_location="cpu")
    addr_type = ckpt.get("address_type")
    if addr_type not in ("tree", "high_order"):
        return False

    old_dim = ckpt.get("input_dim")
    if old_dim == input_dim:
        return False  # already correct

    if dry_run:
        print(f"[dry-run] {path}: would set input_dim {old_dim} -> {input_dim}")
        return True

    ckpt["input_dim"] = input_dim
    torch.save(ckpt, path)
    print(f"[patched] {path}: input_dim {old_dim} -> {input_dim}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True,
                        help="Root directory containing checkpoints/l*/g*/*.pt")
    parser.add_argument("--input_dim", type=int, required=True,
                        help="True input dimension (e.g. 3584 for Qwen2.5-7B)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be changed")
    args = parser.parse_args()

    pattern = os.path.join(args.checkpoint_root, "checkpoints", "l*", "g*", "*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No checkpoints found matching {pattern}")
        return

    patched = 0
    for p in paths:
        if patch_checkpoint(p, args.input_dim, dry_run=args.dry_run):
            patched += 1
    print(f"\nDone. Patched {patched} checkpoint(s) out of {len(paths)}.")


if __name__ == "__main__":
    main()
