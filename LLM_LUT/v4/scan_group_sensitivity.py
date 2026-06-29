"""
Per-layer group sensitivity scan.

Given a baseline multi-layer config, evaluate the PPL impact of increasing
exactly one layer's group count at a time. This identifies which layers are
tolerant to more LUT replacement and which are sensitive.

Usage:
    cd LLM_LUT/v4
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python scan_group_sensitivity.py \
        --device cuda:0 --isolate_gpu \
        --model Qwen/Qwen2.5-7B-Instruct \
        --baseline "15:10,16:10,17:8,18:8,19:8,20:8,21:12,22:16,23:16,24:10,25:8,26:8,27:8" \
        --checkpoint_root ../v3/outputs_int8 \
        --summary_root ../v3/outputs \
        --lut_dtype int8 \
        --output_path results/group_sensitivity_scan.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch

from search_layer_configs import (
    load_layer_summary,
    get_available_group_counts,
    evaluate_multi_layer,
    compute_mac_reduction,
    compute_lut_storage,
)
from trainable_engine import load_model_and_data
from metrics import compute_baseline_probs, compute_model_metrics, format_bytes


def parse_config(arg: str) -> List[Tuple[int, int]]:
    """Parse '15:10,16:10,...' -> [(15, 10), (16, 10), ...]."""
    configs = []
    for part in arg.split(","):
        lid, cnt = part.strip().split(":")
        configs.append((int(lid), int(cnt)))
    return configs


def format_config(configs: List[Tuple[int, int]]) -> str:
    return ",".join(f"{lid}:{cnt}" for lid, cnt in configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--baseline", required=True,
                        help="Baseline config, e.g. '15:10,16:10,...'")
    parser.add_argument("--mode", default="increase", choices=["increase", "decrease"],
                        help="Scan direction: increase (test +1 group count) or decrease (test -1 group count).")
    parser.add_argument("--checkpoint_root", default="../v3/outputs_int8")
    parser.add_argument("--summary_root", default="../v3/outputs")
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolate_gpu", action="store_true")
    parser.add_argument("--lut_dtype", default="int8", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--output_path", default="results/group_sensitivity_scan.json")
    args = parser.parse_args()

    baseline_configs = parse_config(args.baseline)
    layers = [lid for lid, _ in baseline_configs]
    baseline_counts = {lid: cnt for lid, cnt in baseline_configs}

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Per-Layer Group Sensitivity Scan")
    print("=" * 70)
    print(f"Baseline: {format_config(baseline_configs)}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Summary root: {args.summary_root}")
    print("=" * 70)

    # Load summaries and determine next group count for each layer.
    summaries = {lid: load_layer_summary(args.summary_root, lid) for lid in layers}
    variants = []  # List of (description, configs)
    for lid in layers:
        available = sorted(get_available_group_counts(summaries[lid]))
        current = baseline_counts[lid]
        if args.mode == "increase":
            candidates = [c for c in available if c > current]
            if not candidates:
                print(f"  L{lid}: already at max available group count ({current})")
                continue
            next_cnt = candidates[0]
        else:  # decrease
            candidates = [c for c in available if c < current]
            if not candidates:
                print(f"  L{lid}: already at min available group count ({current})")
                continue
            next_cnt = candidates[-1]  # closest lower count
        variant = [(l, next_cnt if l == lid else baseline_counts[l]) for l in layers]
        variants.append((f"L{lid}:{current}->{next_cnt}", variant))
        print(f"  L{lid}: will test {current} -> {next_cnt}")

    if not variants:
        print("No variants to test.")
        return

    # Load model and data once.
    print("\n[1/2] Loading model and data...")
    model, tokenizer, _calib_loader, eval_loader = load_model_and_data(
        args.model, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    print("\n[2/2] Computing baseline reference probabilities...")
    model.eval()
    with torch.no_grad():
        reference_probs = compute_baseline_probs(model, eval_loader)

    def eval_config(configs):
        with torch.no_grad():
            metrics = evaluate_multi_layer(
                model, eval_loader, reference_probs, configs, args.checkpoint_root,
                lut_dtype=args.lut_dtype, summaries=summaries,
            )
        mac_ratio = compute_mac_reduction(configs, hidden_size, intermediate_size, num_layers)
        storage_bytes = compute_lut_storage(configs, args.checkpoint_root, lut_dtype=args.lut_dtype)
        return {
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "mac_reduction_ratio": mac_ratio,
            "lut_storage_bytes": storage_bytes,
            "lut_storage_human": format_bytes(storage_bytes),
        }

    # Evaluate baseline.
    print(f"\n[Eval] Baseline: {format_config(baseline_configs)}")
    baseline_result = eval_config(baseline_configs)
    print(f"  KL={baseline_result['kl']:.4f}, PPL={baseline_result['ppl']:.2f}, "
          f"Acc={baseline_result['acc']:.4f}, MAC↓={baseline_result['mac_reduction_ratio']*100:.2f}%")

    # Evaluate each variant.
    print(f"\n[Eval] Testing one-layer group {('increments' if args.mode == 'increase' else 'decrements')}...")
    variant_results = []
    for desc, configs in variants:
        print(f"  {desc}: {format_config(configs)} ...", end=" ")
        try:
            result = eval_config(configs)
            delta_ppl = result["ppl"] - baseline_result["ppl"]
            delta_mac = result["mac_reduction_ratio"] - baseline_result["mac_reduction_ratio"]
            result["delta_ppl"] = delta_ppl
            result["delta_mac_reduction_ratio"] = delta_mac
            result["description"] = desc
            result["configs"] = configs
            variant_results.append(result)
            print(f"PPL={result['ppl']:.2f} ({delta_ppl:+.2f}), "
                  f"MAC↓={result['mac_reduction_ratio']*100:.2f}% ({delta_mac*100:+.2f}%)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Rank by PPL delta (smaller = more tolerant).
    variant_results.sort(key=lambda r: r["delta_ppl"])

    print("\n" + "=" * 70)
    if args.mode == "increase":
        print("RANKING: most tolerant -> most sensitive (smaller ΔPPL is better)")
    else:
        print("RANKING: most beneficial -> least beneficial (more negative ΔPPL is better)")
    print("=" * 70)
    print(f"{'Variant':>12} | {'PPL':>8} | {'ΔPPL':>8} | {'ΔMAC↓':>8} | {'Acc':>6}")
    print("-" * 70)
    for r in variant_results:
        print(f"{r['description']:>12} | {r['ppl']:>8.2f} | {r['delta_ppl']:>+8.2f} | "
              f"{r['delta_mac_reduction_ratio']*100:>7.2f}% | {r['acc']:>6.4f}")

    # Save results.
    output = {
        "model": args.model,
        "baseline": {
            "configs": baseline_configs,
            **baseline_result,
        },
        "variants": variant_results,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {args.output_path}")


if __name__ == "__main__":
    main()
