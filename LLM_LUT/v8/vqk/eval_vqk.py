#!/usr/bin/env python3
"""VQK single-module model-level evaluation entry.

Usage:
  cd LLM_LUT/v8
  python -u vqk/eval_vqk.py \
    --model_path /data/models/Qwen3.6-35B-A3B \
    --eval_file /data/ppl_texts.jsonl \
    --prompt_file /data/1000_prompts.jsonl \
    --layer_idx 39 \
    --module_path self_attn.o_proj \
    --quant_method vqk \
    --bits 4 \
    --block_size 64 \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --logit_metrics \
    --output_json results/vqk_l39_o_proj_b4_blk64.json
"""

import argparse
import sys
from pathlib import Path

from common.evaluator import Evaluator
from common.prompts import DEFAULT_PROMPTS, load_eval_texts, load_prompts
from vqk.vqk_patch import VQKPatch


def main():
    parser = argparse.ArgumentParser(description="VQK single-module model-level evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", default=None, help="JSONL/text file with long texts for PPL evaluation")
    parser.add_argument("--prompt_file", default=None, help="JSONL/text file with short prompts for generation evaluation")
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--module_path", default="self_attn.o_proj")
    parser.add_argument("--quant_method", default="vqk", choices=["vqk", "int"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--logit_metrics", action="store_true")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--prompt", action="append", default=None)
    args = parser.parse_args()

    if args.prompt_file:
        prompts = load_prompts(args.prompt_file, args.max_eval_samples)
    elif args.prompt:
        prompts = args.prompt
    else:
        prompts = DEFAULT_PROMPTS

    if args.eval_file:
        texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    else:
        print("No --eval_file provided, using prompts for PPL (noisy on short prompts)")
        texts = prompts[: args.max_eval_samples]

    patch = VQKPatch(
        layer_idx=args.layer_idx,
        module_path=args.module_path,
        bits=args.bits,
        block_size=args.block_size,
        quant_method=args.quant_method,
    )

    evaluator = Evaluator(
        model_path=args.model_path,
        device=args.device,
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

    if args.output_json:
        print(f"\n[Done] result saved to {args.output_json}")
    return result


if __name__ == "__main__":
    main()
