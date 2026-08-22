#!/usr/bin/env python3
"""
Run model-level evaluation with V6 LUT replacement.

- Loads a causal LM from transformers
- Loads V6 LUT checkpoints
- Computes baseline PPL and generation outputs
- Installs replacement hook and computes the same metrics

Usage:
  python run_model_eval.py \
    --model_path Qwen/Qwen3-3.5B \
    --checkpoint_dir ./outputs_ffn_lut_layer1_4groups_.../checkpoints \
    --layer_idx 1 \
    --eval_file eval.jsonl \
    --max_eval_samples 128 \
    --device cuda:0

python run_model_eval.py \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --checkpoint_dir ./worstcase_32g_full_ffn/checkpoints \
  --layer_idx 1 \
  --hook_path "model.model.layers[1].mlp" \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --prompt "诸葛亮第一次北伐为什么会失败？请分别分析街亭失守、用人问题、蜀魏国力差距和整体战略选择的影响，并说明把失败简单归因于马谡是否合理。" \
  --max_new_tokens 1024 \
  --output_json ./worstcase_32g_full_ffn_model_eval_2.json
"""

import os
import json
import math
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v6_replacement_engine import V6ReplacementEngine


DEFAULT_PROMPTS = [
    "What is the capital of Japan?",
    "Explain the concept of overfitting in machine learning.",
    "If a train travels at 60 km/h for 2 hours, how far does it go?",
    "Write a Python function to reverse a string.",
    "请介绍一下长城的历史。",
    "What are the main differences between TCP and UDP?",
]


def load_eval_texts(eval_file: str, max_samples: int):
    """Load evaluation texts from a JSONL or plain text file."""
    texts = []
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"eval_file not found: {eval_file}")
    if path.suffix in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", obj.get("sentence", "")))
                if text:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                if len(texts) >= max_samples:
                    break
    return texts


def compute_ppl(model, tokenizer, texts, device, max_length=512, batch_size=1):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.shape[1]
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def run_generation(model, tokenizer, prompts, device, max_new_tokens=128):
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        results.append({"prompt": prompt, "output": generated})
    return results


def main():
    parser = argparse.ArgumentParser(description="V6 LUT model-level evaluation")
    parser.add_argument("--model_path", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with replacement_g*.pt LUT checkpoints")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index where the FFN lives")
    parser.add_argument("--hook_path", default=None, help="Python expression to locate the hook module, e.g. 'model.model.layers[1].mlp'")
    parser.add_argument("--eval_file", default=None, help="JSONL or text file with evaluation sentences")
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", default=None, help="HuggingFace device_map for multi-GPU inference, e.g. balanced_low_0. Do NOT use 'auto'.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_json", default=None, help="Path to write the PPL/generation summary JSON")
    parser.add_argument("--prompt", action="append", default=None, help="Custom prompt for generation. Repeat for multiple prompts. If omitted, use built-in prompts.")
    parser.add_argument("--verify_replacement", action="store_true", default=True, help="Verify that the hook actually changes the MLP output (default: True)")
    parser.add_argument("--no_verify_replacement", action="store_true", default=False, help="Skip replacement verification")
    args = parser.parse_args()

    if args.device_map == "auto":
        raise ValueError("device_map='auto' is forbidden by project red line. Use an explicit map like 'balanced_low_0'.")

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading model: {args.model_path}")
    if args.device_map is not None:
        print(f"  device_map={args.device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=args.device_map,
        )
        device = next(model.parameters()).device
        print(f"  first-layer device is {device}")
        engine_device = None
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        engine_device = device
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Evaluation texts
    if args.eval_file:
        texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    else:
        print("No --eval_file provided, using prompts for PPL (not ideal)")
        texts = prompts[: args.max_eval_samples]
    print(f"Evaluating on {len(texts)} samples")

    # Baseline
    print("\n===== Baseline (no LUT) =====")
    baseline_ppl = compute_ppl(model, tokenizer, texts, device, max_length=args.max_length)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    baseline_gen = run_generation(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
    for item in baseline_gen:
        print(f"  Prompt: {item['prompt']}")
        print(f"  Output: {item['output']}")
        print()

    # With LUT
    print("\n===== With V6 LUT =====")
    engine = V6ReplacementEngine(
        model=model,
        layer_idx=args.layer_idx,
        checkpoint_dir=args.checkpoint_dir,
        device=engine_device,
        hook_path=args.hook_path,
    )
    engine.install()

    # Verify that the hook actually changes the MLP output
    if args.verify_replacement and not args.no_verify_replacement:
        ok = engine.verify_replacement(model.config.hidden_size)
        if not ok:
            print("\n[Warning] Replacement verification failed; results below may not reflect LUT behavior.")

    lut_ppl = compute_ppl(model, tokenizer, texts, device, max_length=args.max_length)
    print(f"LUT PPL: {lut_ppl:.4f}")
    lut_gen = run_generation(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
    for item in lut_gen:
        print(f"  Prompt: {item['prompt']}")
        print(f"  Output: {item['output']}")
        print()

    engine.uninstall()

    print("\n===== Summary =====")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"LUT PPL:      {lut_ppl:.4f}")
    print(f"PPL delta:    {lut_ppl - baseline_ppl:+.4f}")

    if args.output_json:
        summary = {
            "model_path": args.model_path,
            "checkpoint_dir": str(args.checkpoint_dir),
            "layer_idx": args.layer_idx,
            "hook_path": args.hook_path,
            "device": str(device),
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "max_new_tokens": args.max_new_tokens,
            "prompts": prompts,
            "baseline_ppl": baseline_ppl,
            "lut_ppl": lut_ppl,
            "ppl_delta": lut_ppl - baseline_ppl,
            "baseline_generations": baseline_gen,
            "lut_generations": lut_gen,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary written to {args.output_json}")


if __name__ == "__main__":
    main()
