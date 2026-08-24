#!/usr/bin/env python3
"""Stage 3 distribution analysis for v8 VQK.

Collects real forward activation at layer39.self_attn.o_proj and analyzes
per-block activation / quantization / output-error statistics for a chosen
integer VQK config.

Usage:
  cd LLM_LUT/v8
  python -u vqk/run_stage3.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_file /data/v8_eval_texts.jsonl \
    --max_prompts 100 \
    --max_length 1024 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_json results/v8_stage3_w4a4_per_token_per_block.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.prompts import load_eval_texts
from common.utils import load_model_and_tokenizer
from vqk.integer_vqk_linear import IntegerVQKLinear


def analyze_activation(
    int_layer: IntegerVQKLinear,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-block statistics for one forward activation tensor.

    Args:
        int_layer: the integer VQK layer whose quantizer is used.
        x: input activation, shape (..., in_features).

    Returns:
        Dict mapping stat name -> tensor of shape (..., num_blocks).
    """
    mode = int_layer.activation_mode
    q_x, s_x = int_layer._quantize_activation(x)

    in_features = int_layer.in_features
    block_size = int_layer.block_size
    num_blocks = int_layer.num_blocks

    # Dequantize to compare with original x.
    if mode == "per-token":
        x_hat = q_x.float() * s_x.float()
    elif mode == "per-token-per-block":
        q_x_blocks = q_x.view(*q_x.shape[:-1], num_blocks, block_size).float()
        x_hat_blocks = q_x_blocks * s_x.float()  # s_x: (..., num_blocks, 1)
        x_hat = x_hat_blocks.view(q_x.shape)
    else:
        raise ValueError(f"Unknown activation mode: {mode}")

    x_f = x.float()
    x_blocks = x_f.view(*x_f.shape[:-1], num_blocks, block_size)
    abs_blocks = x_blocks.abs()

    # Per-block statistics along the block_size dimension.
    block_min = x_blocks.min(dim=-1).values
    block_max = x_blocks.max(dim=-1).values
    block_max_abs = abs_blocks.max(dim=-1).values
    block_mean = x_blocks.mean(dim=-1)
    block_std = x_blocks.std(dim=-1, unbiased=False)
    block_p99 = torch.quantile(abs_blocks, 0.99, dim=-1)
    block_p999 = torch.quantile(abs_blocks, 0.999, dim=-1)

    # Outlier ratio: fraction of values exceeding the 99.9 percentile.
    outlier_mask = abs_blocks > block_p999.unsqueeze(-1)
    outlier_ratio = outlier_mask.float().mean(dim=-1)

    # Quantization error per block.
    quant_err = (x_f - x_hat).abs()
    quant_err_blocks = quant_err.view(*quant_err.shape[:-1], num_blocks, block_size)
    block_quant_err_mean = quant_err_blocks.mean(dim=-1)
    block_quant_err_max = quant_err_blocks.max(dim=-1).values
    block_quant_err_rel = block_quant_err_mean / block_max_abs.clamp_min(1e-8)

    return {
        "min": block_min,
        "max": block_max,
        "max_abs": block_max_abs,
        "mean": block_mean,
        "std": block_std,
        "p99": block_p99,
        "p999": block_p999,
        "outlier_ratio": outlier_ratio,
        "quant_err_mean": block_quant_err_mean,
        "quant_err_max": block_quant_err_max,
        "quant_err_rel": block_quant_err_rel,
    }


def flatten_batch_and_seq(t: torch.Tensor) -> torch.Tensor:
    """Flatten all but the last dimension (num_blocks) into tokens."""
    return t.reshape(-1, t.shape[-1])


def aggregate_across_tokens(per_token_stats: List[torch.Tensor]) -> Dict[str, List[float]]:
    """Aggregate a list of (num_tokens, num_blocks) tensors across tokens.

    Returns per-block mean/std/min/max/median.
    """
    cat = torch.cat(per_token_stats, dim=0)  # (total_tokens, num_blocks)
    cat_f = cat.float()
    return {
        "mean": cat_f.mean(dim=0).tolist(),
        "std": cat_f.std(dim=0, unbiased=False).tolist(),
        "min": cat_f.min(dim=0).values.tolist(),
        "max": cat_f.max(dim=0).values.tolist(),
        "median": cat_f.median(dim=0).values.tolist(),
    }


def run_stage3(args):
    texts = load_eval_texts(args.eval_file, args.max_prompts)
    print(f"Loaded {len(texts)} texts for distribution analysis")

    model, tokenizer, _ = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        args.device,
        args.device_map,
    )
    model.eval()

    target_module = model.model.layers[args.layer_idx].self_attn.o_proj
    device = next(target_module.parameters()).device
    print(f"Target module: layer {args.layer_idx} {args.module_path}, device={device}")

    int_layer = IntegerVQKLinear(
        target_module,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        block_size=args.block_size,
        activation_mode=args.activation_mode,
    ).to(device).eval()

    # Containers for per-token statistics.
    block_stat_tensors: Dict[str, List[torch.Tensor]] = {
        "min": [],
        "max": [],
        "max_abs": [],
        "mean": [],
        "std": [],
        "p99": [],
        "p999": [],
        "outlier_ratio": [],
        "quant_err_mean": [],
        "quant_err_max": [],
        "quant_err_rel": [],
    }
    output_l2_list: List[torch.Tensor] = []
    output_rel_l2_list: List[torch.Tensor] = []
    output_max_abs_list: List[torch.Tensor] = []

    def hook_fn(module, inputs, output):
        x = inputs[0].detach()
        y_base = output.detach()
        with torch.no_grad():
            y_hat = int_layer(x)
            stats = analyze_activation(int_layer, x)

            # Flatten batch and sequence dimensions -> (num_tokens, num_blocks).
            for key in block_stat_tensors:
                block_stat_tensors[key].append(
                    flatten_batch_and_seq(stats[key]).cpu()
                )

            # Output error statistics per token.
            err = (y_hat.float() - y_base.float())
            err_l2 = err.norm(dim=-1, p=2)
            y_base_l2 = y_base.float().norm(dim=-1, p=2).clamp_min(1e-8)
            err_rel_l2 = err_l2 / y_base_l2
            err_max_abs = err.abs().max(dim=-1).values

            output_l2_list.append(err_l2.cpu())
            output_rel_l2_list.append(err_rel_l2.cpu())
            output_max_abs_list.append(err_max_abs.cpu())

    handle = target_module.register_forward_hook(hook_fn)

    total_tokens = 0
    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue
        with torch.no_grad():
            model(input_ids)
        total_tokens += input_ids.shape[1]
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(texts)} texts, {total_tokens} tokens")

    handle.remove()
    print(f"\nTotal tokens collected: {total_tokens}")

    # Aggregate per-block statistics across all tokens.
    aggregated_block_stats = {
        key: aggregate_across_tokens(tensors)
        for key, tensors in block_stat_tensors.items()
    }

    # Aggregate output error statistics across all tokens.
    output_l2_cat = torch.cat([t.flatten() for t in output_l2_list]).float()
    output_rel_l2_cat = torch.cat([t.flatten() for t in output_rel_l2_list]).float()
    output_max_abs_cat = torch.cat([t.flatten() for t in output_max_abs_list]).float()

    output_error_stats = {
        "l2": {
            "mean": output_l2_cat.mean().item(),
            "std": output_l2_cat.std(unbiased=False).item(),
            "min": output_l2_cat.min().item(),
            "max": output_l2_cat.max().item(),
            "median": output_l2_cat.median().item(),
            "p99": torch.quantile(output_l2_cat, 0.99).item(),
            "p999": torch.quantile(output_l2_cat, 0.999).item(),
        },
        "rel_l2": {
            "mean": output_rel_l2_cat.mean().item(),
            "std": output_rel_l2_cat.std(unbiased=False).item(),
            "min": output_rel_l2_cat.min().item(),
            "max": output_rel_l2_cat.max().item(),
            "median": output_rel_l2_cat.median().item(),
            "p99": torch.quantile(output_rel_l2_cat, 0.99).item(),
            "p999": torch.quantile(output_rel_l2_cat, 0.999).item(),
        },
        "max_abs": {
            "mean": output_max_abs_cat.mean().item(),
            "std": output_max_abs_cat.std(unbiased=False).item(),
            "min": output_max_abs_cat.min().item(),
            "max": output_max_abs_cat.max().item(),
            "median": output_max_abs_cat.median().item(),
            "p99": torch.quantile(output_max_abs_cat, 0.99).item(),
            "p999": torch.quantile(output_max_abs_cat, 0.999).item(),
        },
    }

    result = {
        "model_path": args.model_path,
        "layer_idx": args.layer_idx,
        "module_path": args.module_path,
        "config": {
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "block_size": args.block_size,
            "activation_mode": args.activation_mode,
        },
        "num_prompts": len(texts),
        "max_length": args.max_length,
        "total_tokens": total_tokens,
        "block_stats": aggregated_block_stats,
        "output_error_stats": output_error_stats,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {output_path}")
    print(f"  Output error L2 mean: {output_error_stats['l2']['mean']:.6f}")
    print(f"  Output error rel-L2 mean: {output_error_stats['rel_l2']['mean']:.6f}")
    print(f"  Output error max-abs mean: {output_error_stats['max_abs']['mean']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="v8 VQK Stage 3 distribution analysis")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--max_prompts", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--module_path", default="self_attn.o_proj")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--activation_mode", default="per-token-per-block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_json", default="results/v8_stage3_w4a4_per_token_per_block.json")
    args = parser.parse_args()
    run_stage3(args)


if __name__ == "__main__":
    main()
