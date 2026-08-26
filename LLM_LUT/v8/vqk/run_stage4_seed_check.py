#!/usr/bin/env python3
"""Stage 4 seed-stability check for K=2 Clustered VQK.

Runs the same W4A4 K=2 config with several random seeds and reports whether
both the cluster assignment and the evaluation metrics are stable.

Usage:
  cd LLM_LUT/v8
  python -u vqk/run_stage4_seed_check.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_file /data/v8_eval_texts.jsonl \
    --prompt_file /data/1000_prompts.jsonl \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_json results/v8_stage4_seed_check_k2.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.evaluator import Evaluator
from common.prompts import load_eval_texts, load_prompts
from vqk.clustered_vqk_patch import ClusteredVQKPatch


SEEDS = [0, 1, 2, 42, 999]


def run_seed_check(args):
    texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    prompts = load_prompts(args.prompt_file, args.max_eval_samples)

    evaluator = Evaluator(
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        logit_metrics=args.logit_metrics,
    )

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed check: K=2, seed={seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)

        patch = ClusteredVQKPatch(
            layer_idx=39,
            module_path="self_attn.o_proj",
            weight_bits=4,
            activation_bits=4,
            block_size=args.block_size,
            activation_mode="per-token-per-block",
            num_clusters=2,
        )

        result = evaluator.evaluate(
            patch=patch,
            texts=texts,
            prompts=prompts,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_json=f"{args.output_dir}/stage4_k2_seed{seed}.json",
            verbose=True,
        )

        # Record cluster assignment if available after patch uninstall.
        # Since uninstall clears _replacement, we re-instantiate once more just
        # to read the assignment (cheap, no forward pass).
        torch.manual_seed(seed)
        patch2 = ClusteredVQKPatch(
            layer_idx=39,
            module_path="self_attn.o_proj",
            weight_bits=4,
            activation_bits=4,
            block_size=args.block_size,
            activation_mode="per-token-per-block",
            num_clusters=2,
        )
        patch2.install(evaluator.student)
        assignment = patch2._replacement.cluster_assignments.cpu().tolist()
        patch2.uninstall(evaluator.student)

        results.append({
            "seed": seed,
            "ppl": result["patched"]["ppl"],
            "ppl_delta": result["delta"]["ppl"],
            "ppl_relative": result["delta"]["ppl_relative"],
            "logit_kl": result["logit_metrics"].get("avg_kl"),
            "top1_agreement": result["logit_metrics"].get("top1_agreement"),
            "top5_agreement": result["logit_metrics"].get("top5_agreement"),
            "cluster_assignment": assignment,
            "storage_stats": result.get("storage_stats", {}),
        })

    # Stability summary.
    ppls = [r["ppl"] for r in results]
    kls = [r["logit_kl"] for r in results]
    assignments = [tuple(r["cluster_assignment"]) for r in results]
    assignment_stable = len(set(assignments)) == 1

    summary = {
        "model_path": args.model_path,
        "layer_idx": 39,
        "module_path": "self_attn.o_proj",
        "config": {
            "weight_bits": 4,
            "activation_bits": 4,
            "block_size": args.block_size,
            "activation_mode": "per-token-per-block",
            "num_clusters": 2,
        },
        "seeds": SEEDS,
        "results": results,
        "stability": {
            "assignment_stable": assignment_stable,
            "ppl_min": min(ppls),
            "ppl_max": max(ppls),
            "ppl_range": max(ppls) - min(ppls),
            "kl_min": min(kls),
            "kl_max": max(kls),
            "kl_range": max(kls) - min(kls),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Seed stability summary (K=2):")
    print(f"{'='*60}")
    print(f"  Assignment stable across seeds: {assignment_stable}")
    print(f"  PPL range: {summary['stability']['ppl_range']:.6f}")
    print(f"  KL range:  {summary['stability']['kl_range']:.6f}")
    for r in results:
        print(
            f"  seed={r['seed']:>4}  PPL={r['ppl']:.4f}  "
            f"ΔPPL={r['ppl_delta']:+.4f}  KL={r['logit_kl']:.4f}  "
            f"Top-1={r['top1_agreement']:.2%}"
        )
    print(f"\n[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 4 seed stability check")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--logit_metrics", action="store_true", default=True)
    parser.add_argument("--output_json", default="results/v8_stage4_seed_check_k2.json")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    run_seed_check(args)


if __name__ == "__main__":
    main()
