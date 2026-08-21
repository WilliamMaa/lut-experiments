#!/usr/bin/env python3
"""Generate evaluation texts from prompts using the baseline model.

The produced JSONL can be used as --eval_file for v8 PPL evaluation.
Each line contains a "text" field equal to `prompt + generated_continuation`.

Usage:
  cd LLM_LUT/v8
  python -u common/generate_eval_texts.py \
    --model_path /data/models/Qwen3.6-35B-A3B \
    --prompt_file /data/1000_prompts.jsonl \
    --max_samples 128 \
    --max_new_tokens 512 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_jsonl /data/v8_eval_texts.jsonl
"""

import argparse
import json
from pathlib import Path

import torch

from utils import load_model_and_tokenizer
from prompts import load_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate eval texts from prompts")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file, args.max_samples)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        args.device,
        args.device_map,
    )
    model.eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length - args.max_new_tokens,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            record = {
                "text": full_text,
                "prompt": prompt,
                "sample_idx": i,
                "model_path": args.model_path,
                "max_new_tokens": args.max_new_tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated_count += 1

            if generated_count % 10 == 0:
                print(f"  generated {generated_count}/{len(prompts)}")

    print(f"\n[Done] wrote {generated_count} eval texts to {output_path}")
    print(f"  each text = prompt + {args.max_new_tokens}-token baseline continuation")


if __name__ == "__main__":
    main()
