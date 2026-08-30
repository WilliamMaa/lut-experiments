#!/usr/bin/env python3
"""Run any GDN replacement patch end-to-end on Qwen3.6-35B-A3B.

Usage:
  cd LLM_LUT/v8
  python -u qwen_toolkit/run_gdn_replacement.py \
    --patch fixed_basis \
    --layer_idx 20 \
    --rank 32 \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_file /data/mamingyu/v6/candidate_prompts.jsonl \
    --prompt_file /data/mamingyu/v6/candidate_prompts.jsonl \
    --max_eval_samples 64 \
    --max_new_tokens 128 \
    --max_length 512 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --logit_metrics \
    --output_json results/fixed_basis_l20_r32.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import Evaluator
from common.prompts import load_eval_texts, load_prompts
from qwen_toolkit.patches.fixed_basis_state_patch import FixedBasisStatePatch


PATCH_REGISTRY = {
    "fixed_basis": FixedBasisStatePatch,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, choices=list(PATCH_REGISTRY.keys()))
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--logit_metrics", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    prompts = load_prompts(args.prompt_file, args.max_eval_samples)

    patch_cls = PATCH_REGISTRY[args.patch]
    if args.patch == "fixed_basis":
        patch = patch_cls(layer_idx=args.layer_idx, rank=args.rank)
    else:
        patch = patch_cls(layer_idx=args.layer_idx)

    print(f"Running patch: {patch.name()}")
    print(f"Config: {patch.config()}")
    print(f"Storage stats: {patch.storage_stats()}")

    evaluator = Evaluator(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        logit_metrics=args.logit_metrics,
    )

    result = evaluator.evaluate(
        patch=patch,
        texts=texts,
        prompts=prompts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_json=args.output_json,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
