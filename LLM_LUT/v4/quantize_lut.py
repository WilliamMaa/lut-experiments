"""
LUT 量化工具：把 v3 生成的 FP32 table checkpoint 量化成 FP16 或 INT8。

用法:
    cd LLM_LUT/v4
    python quantize_lut.py \
        --checkpoint_root ../v3/outputs \
        --output_root ../v3/outputs_quantized \
        --dtype int8

输出结构保持与输入一致:
    outputs_quantized/checkpoints/l{layer}/g{count}/replacement_l{layer}g{gid}.pt
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict

import torch

V3_DIR = os.path.join(os.path.dirname(__file__), "..", "v3")
sys.path.insert(0, V3_DIR)


def quantize_tensor_symmetric_int8(tensor: torch.Tensor) -> Dict:
    """Per-tensor symmetric INT8 quantization."""
    max_abs = tensor.abs().max()
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs.item() / 127.0
    int8_tensor = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return {
        "table": int8_tensor,
        "scale": scale,
        "zero_point": 0.0,
        "quantization": "symmetric_int8",
    }


def quantize_checkpoint(input_path: str, output_path: str, dtype: str):
    """Quantize a single per-group checkpoint and save it."""
    ckpt = torch.load(input_path, map_location="cpu")
    table = ckpt["table"].float()

    if dtype == "fp16":
        new_ckpt = {
            **ckpt,
            "table": table.half(),
            "quantization": "fp16",
        }
    elif dtype == "int8":
        q = quantize_tensor_symmetric_int8(table)
        new_ckpt = {
            **ckpt,
            "table": q["table"],
            "scale": q["scale"],
            "zero_point": q["zero_point"],
            "quantization": q["quantization"],
        }
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_ckpt, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="../v3/outputs",
                        help="Root directory containing v3 checkpoints/")
    parser.add_argument("--output_root", default="../v3/outputs_quantized",
                        help="Output root directory")
    parser.add_argument("--dtype", default="int8", choices=["fp16", "int8"],
                        help="Target quantization dtype")
    parser.add_argument("--layers", default="19,20,21,22,23",
                        help="Comma-separated layer IDs to quantize (default: L19-L23)")
    parser.add_argument("--group_counts", default="4,8,12,16",
                        help="Comma-separated group counts to quantize")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    group_counts = [int(x.strip()) for x in args.group_counts.split(",")]

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Quantize LUT checkpoints: {args.dtype}")
    print("=" * 70)

    total_original_bytes = 0
    total_quantized_bytes = 0
    num_checkpoints = 0

    for layer_id in layers:
        for count in group_counts:
            ckpt_dir = os.path.join(args.checkpoint_root, "checkpoints", f"l{layer_id}", f"g{count}")
            if not os.path.isdir(ckpt_dir):
                continue
            pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
            for input_path in sorted(glob.glob(pattern)):
                name = os.path.basename(input_path)
                output_path = os.path.join(
                    args.output_root, "checkpoints", f"l{layer_id}", f"g{count}", name
                )
                if os.path.exists(output_path):
                    print(f"  [skip] {output_path} already exists")
                    continue

                ckpt = torch.load(input_path, map_location="cpu")
                table = ckpt["table"]
                original_bytes = table.element_size() * table.numel()
                quantize_checkpoint(input_path, output_path, args.dtype)

                q_ckpt = torch.load(output_path, map_location="cpu")
                q_table = q_ckpt["table"]
                quantized_bytes = q_table.element_size() * q_table.numel()

                total_original_bytes += original_bytes
                total_quantized_bytes += quantized_bytes
                num_checkpoints += 1
                print(f"  {name}: {original_bytes} B -> {quantized_bytes} B")

    ratio = total_original_bytes / total_quantized_bytes if total_quantized_bytes > 0 else 0
    summary = {
        "dtype": args.dtype,
        "num_checkpoints": num_checkpoints,
        "original_bytes": total_original_bytes,
        "quantized_bytes": total_quantized_bytes,
        "compression_ratio": ratio,
    }
    summary_path = os.path.join(args.output_root, f"quantize_summary_{args.dtype}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("QUANTIZE COMPLETE")
    print("=" * 70)
    print(f"Checkpoints: {num_checkpoints}")
    print(f"Original:    {total_original_bytes} B")
    print(f"Quantized:   {total_quantized_bytes} B")
    print(f"Ratio:       {ratio:.2f}x")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
