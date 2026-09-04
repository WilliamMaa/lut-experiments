#!/usr/bin/env python3
"""Run KV cache compression methods end-to-end on Qwen3.6-35B-A3B."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import Evaluator
from common.prompts import load_eval_texts, load_prompts, load_multi_turn_prompts
from kv_cache.kv_cache_patch import KIVICachePatch, RetentionCachePatch, HeavyHitterCachePatch, HeavyHitterAttnScorePatch


PATCH_REGISTRY = {
    "kivi": KIVICachePatch,
    "retention": RetentionCachePatch,
    "heavy_hitter": HeavyHitterCachePatch,
    "heavy_hitter_attn": HeavyHitterAttnScorePatch,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, choices=list(PATCH_REGISTRY.keys()))
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--max_cache_len", type=int, default=512)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_tokens", type=int, default=128)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--min_prompt_length", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--multi_turn", action="store_true", help="Use multi-turn conversation eval instead of single prompts")
    parser.add_argument("--multi_turn_file", default="data/multi_turn_prompts.jsonl", help="Path to multi-turn prompts JSONL")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--logit_metrics", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    texts = load_eval_texts(
        args.eval_file, args.max_eval_samples,
        min_length=args.min_prompt_length, sort_by_length=args.min_prompt_length > 0,
    )
    prompts = None
    multi_turn_samples = None
    if args.multi_turn:
        multi_turn_samples = load_multi_turn_prompts(args.multi_turn_file, args.max_eval_samples)
    else:
        prompts = load_prompts(
            args.prompt_file, args.max_eval_samples,
            min_length=args.min_prompt_length, sort_by_length=args.min_prompt_length > 0,
        )

    patch_cls = PATCH_REGISTRY[args.patch]
    if args.patch == "kivi":
        patch = patch_cls(k_bits=args.k_bits, v_bits=args.v_bits)
    elif args.patch == "retention":
        patch = patch_cls(max_cache_len=args.max_cache_len, sink_tokens=args.sink_tokens)
    elif args.patch == "heavy_hitter":
        patch = patch_cls(
            max_cache_len=args.max_cache_len,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
        )
    elif args.patch == "heavy_hitter_attn":
        patch = patch_cls(
            max_cache_len=args.max_cache_len,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
        )
    else:
        patch = patch_cls()

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
        multi_turn_samples=multi_turn_samples,
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
