#!/usr/bin/env python3
"""Stage 4 runner for v8 VQK Plan C: Clustered VQK.

Fixes W4A4 per-token-per-block block=128 and varies the number of weight-block
clusters K = 2 / 4 / 8.  Each cluster gets its own clipping threshold / scale.

Usage:
  cd LLM_LUT/v8
  python -u vqk/run_stage4.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_file /data/v8_eval_texts.jsonl \
    --prompt_file /data/1000_prompts.jsonl \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_json results/v8_stage4_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.evaluator import Evaluator
from common.prompts import load_eval_texts, load_prompts
from vqk.clustered_vqk_patch import ClusteredVQKPatch


def run_stage4(args):
    texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    prompts = load_prompts(args.prompt_file, args.max_eval_samples)

    evaluator = Evaluator(
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        logit_metrics=args.logit_metrics,
    )

    configs = [
        {"num_clusters": 2},
        {"num_clusters": 4},
        {"num_clusters": 8},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Stage 4: W4A4 block={args.block_size} K={cfg['num_clusters']}")
        print(f"{'='*60}")

        patch = ClusteredVQKPatch(
            layer_idx=39,
            module_path="self_attn.o_proj",
            weight_bits=4,
            activation_bits=4,
            block_size=args.block_size,
            activation_mode="per-token-per-block",
            num_clusters=cfg["num_clusters"],
        )

        result = evaluator.evaluate(
            patch=patch,
            texts=texts,
            prompts=prompts,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_json=f"{args.output_dir}/stage4_w4a4_blk{args.block_size}_k{cfg['num_clusters']}.json",
            verbose=True,
        )

        results.append({
            "num_clusters": cfg["num_clusters"],
            "block_size": args.block_size,
            "weight_bits": 4,
            "activation_bits": 4,
            "activation_mode": "per-token-per-block",
            "ppl": result["patched"]["ppl"],
            "ppl_delta": result["delta"]["ppl"],
            "ppl_relative": result["delta"]["ppl_relative"],
            "logit_kl": result["logit_metrics"].get("avg_kl"),
            "top1_agreement": result["logit_metrics"].get("top1_agreement"),
            "top5_agreement": result["logit_metrics"].get("top5_agreement"),
            "eos_success_rate": result["patched"]["generation_metrics"]["eos_success_rate"],
            "repetition_rate": result["patched"]["generation_metrics"]["repetition_rate"],
            "storage_stats": result.get("storage_stats", {}),
        })

    summary = {
        "model_path": args.model_path,
        "layer_idx": 39,
        "module_path": "self_attn.o_proj",
        "block_size": args.block_size,
        "weight_bits": 4,
        "activation_bits": 4,
        "activation_mode": "per-token-per-block",
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Stage 4 summary:")
    print(f"{'='*60}")
    for r in results:
        eff_bits = r["storage_stats"].get("effective_bits_per_weight", 0)
        print(
            f"K={r['num_clusters']:<2}  "
            f"PPL={r['ppl']:.4f}  ΔPPL={r['ppl_delta']:+.4f}  "
            f"KL={r['logit_kl']:.4f}  Top-1={r['top1_agreement']:.2%}  "
            f"eff_bits={eff_bits:.4f}"
        )
    print(f"\n[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="v8 VQK Stage 4 runner")
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
    parser.add_argument("--output_json", default="results/v8_stage4_results.json")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    run_stage4(args)


if __name__ == "__main__":
    main()
