#!/usr/bin/env python3
"""Export K=2 cluster assignment and per-block statistics for Clustered VQK.

Usage:
  cd LLM_LUT/v8
  python -u vqk/analyze_clusters.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --module_path self_attn.o_proj \
    --weight_bits 4 \
    --block_size 128 \
    --num_clusters 2 \
    --seed 42 \
    --output_json results/v8_cluster_analysis_k2.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import load_model_and_tokenizer
from vqk.clustered_vqk_linear import ClusteredVQKLinear, _block_features, _kmeans, _qmax, _optimal_threshold


def _kurtosis(w: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis along last dim (flattened block)."""
    w_f = w.float()
    mean = w_f.mean(dim=-1, keepdim=True)
    std = w_f.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
    z = (w_f - mean) / std
    return (z ** 4).mean(dim=-1) - 3.0


def analyze_clusters(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer, _ = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        args.device,
        args.device_map,
    )
    model.eval()

    layer = model.model.layers[args.layer_idx]
    target = layer
    for name in args.module_path.split(".")[:-1]:
        target = getattr(target, name)
    attr_name = args.module_path.split(".")[-1]
    base_module = getattr(target, attr_name)

    # Instantiate clustered layer to get assignments and scales.
    clustered = ClusteredVQKLinear(
        base_module,
        weight_bits=args.weight_bits,
        activation_bits=4,  # not used for weight-only analysis
        block_size=args.block_size,
        activation_mode="per-token-per-block",
        num_clusters=args.num_clusters,
    ).to(next(base_module.parameters()).device).eval()

    weight = base_module.weight.detach()
    out_features, in_features = weight.shape
    num_blocks = in_features // args.block_size
    weight_blocks = weight.view(out_features, num_blocks, args.block_size)

    # Raw per-block statistics.
    w_flat = weight_blocks.transpose(0, 1).reshape(num_blocks, -1).float()
    block_max_abs = w_flat.abs().max(dim=1).values
    block_std = w_flat.std(dim=1, unbiased=False)
    block_kurtosis = _kurtosis(w_flat)
    block_mean_abs = w_flat.abs().mean(dim=1)
    p999 = torch.quantile(w_flat.abs(), 0.999, dim=1)
    block_outlier_ratio = (w_flat.abs() > p999.unsqueeze(1)).float().mean(dim=1)

    assignments = clustered.cluster_assignments.cpu()
    cluster_scales = clustered.cluster_scales.cpu()

    # Per-block quantization MSE using its cluster's chosen threshold/scale.
    qmax = _qmax(args.weight_bits)
    block_mse = []
    for b in range(num_blocks):
        c = int(assignments[b].item())
        scale = cluster_scales[c].item()
        w_b = weight_blocks[:, b, :].reshape(-1).float()
        q = torch.clamp(torch.round(w_b / scale), -qmax, qmax)
        w_hat = q * scale
        mse = (w_b - w_hat).pow(2).mean().item()
        block_mse.append(mse)

    # Build per-block records.
    block_records = []
    for b in range(num_blocks):
        c = int(assignments[b].item())
        block_records.append({
            "block_id": b,
            "cluster_id": c,
            "max_abs": block_max_abs[b].item(),
            "std": block_std[b].item(),
            "kurtosis": block_kurtosis[b].item(),
            "mean_abs": block_mean_abs[b].item(),
            "outlier_ratio": block_outlier_ratio[b].item(),
            "cluster_scale": cluster_scales[c].item(),
            "quantization_mse": block_mse[b],
        })

    # Cluster summaries.
    cluster_summaries = []
    for c in range(args.num_clusters):
        mask = assignments == c
        n_blocks = int(mask.sum().item())
        cluster_summaries.append({
            "cluster_id": c,
            "num_blocks": n_blocks,
            "fraction_blocks": n_blocks / num_blocks,
            "threshold": cluster_scales[c].item() * qmax,
            "scale": cluster_scales[c].item(),
            "mean_max_abs": block_max_abs[mask].mean().item() if n_blocks else 0.0,
            "std_max_abs": block_max_abs[mask].std(unbiased=False).item() if n_blocks else 0.0,
            "mean_std": block_std[mask].mean().item() if n_blocks else 0.0,
            "mean_kurtosis": block_kurtosis[mask].mean().item() if n_blocks else 0.0,
            "mean_outlier_ratio": block_outlier_ratio[mask].mean().item() if n_blocks else 0.0,
            "mean_quantization_mse": sum(r["quantization_mse"] for r in block_records if r["cluster_id"] == c) / max(1, n_blocks),
            "block_ids": [i for i, r in enumerate(block_records) if r["cluster_id"] == c],
        })

    result = {
        "model_path": args.model_path,
        "layer_idx": args.layer_idx,
        "module_path": args.module_path,
        "weight_bits": args.weight_bits,
        "block_size": args.block_size,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "num_blocks": num_blocks,
        "out_features": out_features,
        "in_features": in_features,
        "cluster_summaries": cluster_summaries,
        "block_records": block_records,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {output_path}")
    for cs in cluster_summaries:
        print(
            f"Cluster {cs['cluster_id']}: {cs['num_blocks']} blocks "
            f"({cs['fraction_blocks']:.1%}), "
            f"threshold={cs['threshold']:.4f}, "
            f"mean_max_abs={cs['mean_max_abs']:.4f}, "
            f"mean_std={cs['mean_std']:.4f}, "
            f"mean_mse={cs['mean_quantization_mse']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze Clustered VQK block clusters")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--module_path", default="self_attn.o_proj")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_json", default="results/v8_cluster_analysis_k2.json")
    args = parser.parse_args()
    analyze_clusters(args)


if __name__ == "__main__":
    main()
