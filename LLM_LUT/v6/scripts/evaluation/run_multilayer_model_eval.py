#!/usr/bin/env python3
"""
run_multilayer_model_eval.py

模型级评估：同时替换多层 shared_expert，评估 PPL 和生成质量。

用法：
  python -u run_multilayer_model_eval.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --layer_idx 37 --checkpoint_dir outputs_l37_as_v4/checkpoints \
    --layer_idx 38 --checkpoint_dir outputs_l38_as_v4/checkpoints \
    --layer_idx 39 --checkpoint_dir outputs_l39_as_v4/checkpoints \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --max_eval_samples 128 \
    --max_new_tokens 4096 \
    --output_json multilayer_l37_39_eval.json
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List

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
    parser = argparse.ArgumentParser(description="V6 multi-layer LUT model-level evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", action="append", type=int, required=True,
                        help="Layer index to replace. Repeat for multiple layers.")
    parser.add_argument("--checkpoint_dir", action="append", type=str, required=True,
                        help="LUT checkpoint directory for the corresponding --layer_idx.")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", default=None,
                        help="HuggingFace device_map for multi-GPU, e.g. balanced_low_0. Do NOT use 'auto'.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--verify_replacement", action="store_true", default=True)
    parser.add_argument("--no_verify_replacement", action="store_true", default=False)
    args = parser.parse_args()

    if len(args.layer_idx) != len(args.checkpoint_dir):
        raise ValueError("Number of --layer_idx and --checkpoint_dir must match")

    if args.device_map == "auto":
        raise ValueError("device_map='auto' is forbidden. Use an explicit map like 'balanced_low_0'.")

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
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

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

    # Install engines for all replacement layers
    print("\n===== With V6 Multi-Layer LUT =====")
    engines = []
    for idx, ckpt_dir in zip(args.layer_idx, args.checkpoint_dir):
        hook_path = f"model.model.layers[{idx}].mlp.shared_expert"
        hook_module = eval(hook_path, {"model": model})
        engine_device = next(hook_module.parameters()).device
        print(f"Installing engine for layer {idx} from {ckpt_dir} on {engine_device}")
        engine = V6ReplacementEngine(
            model=model,
            layer_idx=idx,
            checkpoint_dir=ckpt_dir,
            device=engine_device,
            hook_path=hook_path,
        )
        engine.install()
        engines.append(engine)

    if args.verify_replacement and not args.no_verify_replacement:
        for engine in engines:
            ok = engine.verify_replacement(model.config.hidden_size)
            if not ok:
                print(f"[Warning] Replacement verification failed for layer {engine.layer_idx}")

    lut_ppl = compute_ppl(model, tokenizer, texts, device, max_length=args.max_length)
    print(f"LUT PPL: {lut_ppl:.4f}")
    lut_gen = run_generation(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
    for item in lut_gen:
        print(f"  Prompt: {item['prompt']}")
        print(f"  Output: {item['output']}")
        print()

    for engine in engines:
        engine.uninstall()

    print("\n===== Summary =====")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"LUT PPL:      {lut_ppl:.4f}")
    print(f"PPL delta:    {lut_ppl - baseline_ppl:+.4f}")

    if args.output_json:
        summary = {
            "model_path": args.model_path,
            "layers": args.layer_idx,
            "checkpoint_dirs": args.checkpoint_dir,
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
