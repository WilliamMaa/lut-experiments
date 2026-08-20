#!/usr/bin/env python3
"""Run a pure baseline model-level evaluation with the v8 framework.

This is a thin wrapper around v8.common.evaluator.Evaluator with the NullPatch.

Usage:
  cd LLM_LUT/v8
  python -u run_baseline_eval.py \
    --model_path /data/models/Qwen3.6-35B-A3B \
    --eval_file eval.jsonl \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --logit_metrics \
    --output_json results/v8_baseline.json
"""

from common.evaluator import main

if __name__ == "__main__":
    main()
