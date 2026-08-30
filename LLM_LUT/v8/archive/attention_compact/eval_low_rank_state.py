#!/usr/bin/env python3
"""Evaluate low-rank GDN recurrent state propagation."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import Evaluator
from common.prompts import load_eval_texts, load_prompts
from attention_compact.low_rank_state_patch import LowRankStatePatch


def main():
    parser = argparse.ArgumentParser(description="Evaluate low-rank GDN recurrent state")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_json", default="results/gdn_low_rank.json")
    args = parser.parse_args()

    texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    prompts = load_prompts(args.prompt_file, args.max_eval_samples)

    evaluator = Evaluator(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        logit_metrics=False,
    )

    patch = LowRankStatePatch(rank=args.rank, layer_indices=args.layers)
    result = evaluator.evaluate(
        patch=patch,
        texts=texts,
        prompts=prompts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_json=args.output_json,
    )
    print("\nDelta PPL:", result["delta"]["ppl"])
    print("Delta EOS rate:", result["delta"]["eos_success_rate"])


if __name__ == "__main__":
    main()
