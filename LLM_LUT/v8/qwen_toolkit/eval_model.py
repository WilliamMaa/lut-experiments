#!/usr/bin/env python3
"""Standard end-to-end evaluation for Qwen3.6-35B-A3B replacement experiments.

Usage:
  cd LLM_LUT/v8
  python -u qwen_toolkit/eval_model.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_texts data/eval_texts.jsonl \
    --device_map balanced_low_0 \
    --output_json results/baseline_eval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_texts", required=True, help="jsonl file with {'text': ...}")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_texts", type=int, default=0, help="0 means all")
    parser.add_argument("--output_json", default="results/baseline_eval.json")
    parser.add_argument("--run_baseline", action="store_true", help="Also evaluate baseline alongside patched model")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    evaluator = Evaluator(args.model_path, args.torch_dtype, device_map=args.device_map)

    with open(args.eval_texts, "r", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    if args.num_texts > 0:
        texts = texts[: args.num_texts]
    print(f"Evaluating on {len(texts)} texts")

    result = evaluator.evaluate(
        texts,
        max_length=args.max_length,
        run_baseline=args.run_baseline,
        compute_kl=True,
        compute_ppl=True,
        compute_top1=True,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
