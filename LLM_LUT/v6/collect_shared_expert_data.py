#!/usr/bin/env python3
"""
collect_shared_expert_data.py

采集指定 layer 的 shared_expert 输入/输出数据，用于训练 LUT。

注意：采集的是 shared_expert 的输出，不是完整 MoE block 的输出。

用法：
  python -u collect_shared_expert_data.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --calib_file candidate_prompts.jsonl \
    --output_dir /data/ai2/datasets/lut_distill_dataset/layer39_shared_expert_v3 \
    --max_samples 200000 \
    --max_tokens_per_prompt 512 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16
"""

import os
import json
import argparse
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_replacement_engine import V6ReplacementEngine


def load_calibration_texts(calib_file: str, max_prompts: int) -> List[str]:
    """从 JSONL 加载 prompt 文本。支持多种字段名。"""
    texts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 优先用 prompt 字段
            text = obj.get("prompt", obj.get("text", obj.get("content", obj.get("sentence", obj.get("input", "")))))
            if text:
                texts.append(text)
            if len(texts) >= max_prompts:
                break
    print(f"Loaded {len(texts)} calibration prompts")
    return texts


class ExpertCapture:
    """Forward hook: capture input and output of a module."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def __call__(self, module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        y = output[0] if isinstance(output, tuple) else output
        self.inputs.append(x.detach().cpu())
        self.outputs.append(y.detach().cpu())

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def concat(self):
        if not self.inputs:
            return None, None
        x = torch.cat(self.inputs, dim=0)
        y = torch.cat(self.outputs, dim=0)
        return x, y


def collect_shared_expert_data(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    output_dir: Path,
    max_tokens_per_prompt: int,
    max_new_tokens: int,
    max_total_tokens: int,
    generation_kwargs: dict,
    resume: bool = False,
):
    output_dir = Path(output_dir)
    input_dir = output_dir / "input"
    output_moe_dir = output_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_moe_dir.mkdir(parents=True, exist_ok=True)

    # Hook on shared_expert
    hook_path = f"model.model.layers[{layer_idx}].mlp.shared_expert"
    try:
        module = eval(hook_path, {"model": model})
    except AttributeError:
        raise ValueError(f"Cannot find {hook_path}")

    capture = ExpertCapture()
    handle = module.register_forward_hook(capture)
    print(f"Registered hook on {hook_path}: {type(module).__name__}")

    model.eval()
    file_counter = 0
    total_tokens = 0
    start_text_idx = 0

    metadata_path = output_dir / "metadata.json"
    if resume and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        file_counter = metadata.get("num_files", 0)
        total_tokens = metadata.get("total_tokens", 0)
        start_text_idx = file_counter
        print(f"[Resume] Found {file_counter} files, {total_tokens} tokens; will continue from prompt {start_text_idx}/{len(texts)}")

    pbar = tqdm(texts[start_text_idx:], desc="Collecting shared_expert data", initial=start_text_idx, total=len(texts))
    for text in pbar:
        if total_tokens >= max_total_tokens:
            break

        capture.clear()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens_per_prompt,
            padding=False,
        )
        # Move inputs to the same device as the model to avoid the
        # "input_ids on cpu but model on cuda" warning and the implicit copy overhead.
        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)

        try:
            with torch.no_grad():
                if max_new_tokens > 0:
                    # Generate continuation so LUT sees rollout states, not just prompt states.
                    _ = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **generation_kwargs,
                    )
                else:
                    _ = model(input_ids, use_cache=False)
        except Exception as e:
            print(f"Warning: Failed to process prompt: {e}")
            continue

        x_tensor, y_tensor = capture.concat()
        if x_tensor is None or y_tensor is None:
            continue

        # Validate
        assert x_tensor.shape == y_tensor.shape, \
            f"Shape mismatch: {x_tensor.shape} vs {y_tensor.shape}"

        # Save per prompt
        input_path = input_dir / f"sample_{file_counter:06d}.pt"
        output_path = output_moe_dir / f"sample_{file_counter:06d}.pt"
        torch.save(x_tensor, input_path)
        torch.save(y_tensor, output_path)

        file_counter += 1
        total_tokens += x_tensor.shape[0]
        pbar.set_postfix({"files": file_counter, "tokens": total_tokens})

        capture.clear()

        # 每保存一个文件就写 metadata，这样 killed 后也能 resume
        metadata = {
            "layer_idx": layer_idx,
            "num_files": file_counter,
            "total_tokens": total_tokens,
            "hook_path": hook_path,
            "max_new_tokens": max_new_tokens,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if file_counter % 100 == 0:
            torch.cuda.empty_cache()

    handle.remove()

    print(f"\nCollected {file_counter} files, {total_tokens} total tokens")
    print(f"Input:  {input_dir}")
    print(f"Output:  {output_moe_dir}")

    return file_counter, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--calib_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_prompts", type=int, default=1000)
    parser.add_argument("--max_tokens_per_prompt", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="If > 0, generate this many new tokens per prompt instead of only forwarding the prompt.")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Sample during generation (default: True).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_total_tokens", type=int, default=500000)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--replace_layer_idx", action="append", type=int, default=None,
                        help="Layer indices to replace with V6 LUT engine during data collection. Repeat for multiple layers.")
    parser.add_argument("--replace_checkpoint_dir", action="append", type=str, default=None,
                        help="Checkpoint directories for each replaced layer, in the same order as --replace_layer_idx.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing metadata.json in output_dir, skip already collected prompts.")
    args = parser.parse_args()

    if (args.replace_layer_idx is None) != (args.replace_checkpoint_dir is None):
        raise ValueError("--replace_layer_idx and --replace_checkpoint_dir must both be provided or both omitted")
    if args.replace_layer_idx is not None and len(args.replace_layer_idx) != len(args.replace_checkpoint_dir):
        raise ValueError("Number of --replace_layer_idx and --replace_checkpoint_dir must match")

    if args.device_map == "auto":
        raise ValueError("device_map='auto' forbidden. Use 'balanced_low_0'.")

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    model.eval()

    # Install replacement engines for already-trained layers, so the target layer sees
    # the on-policy distribution.
    # NOTE: hook_path must point to shared_expert, NOT the full mlp, because the LUT
    # checkpoint only approximates shared_expert. Hooking the full mlp would overwrite
    # the routed experts' output with the shared_expert LUT output.
    replacement_engines = []
    if args.replace_layer_idx is not None:
        for idx, ckpt_dir in zip(args.replace_layer_idx, args.replace_checkpoint_dir):
            hook_path = f"model.model.layers[{idx}].mlp.shared_expert"
            print(f"[Replace] Installing V6 engine for layer {idx} from {ckpt_dir}")
            engine = V6ReplacementEngine(model, idx, ckpt_dir, device=None, hook_path=hook_path)
            engine.install()
            replacement_engines.append(engine)

    try:
        texts = load_calibration_texts(args.calib_file, args.max_prompts)

        generation_kwargs = {}
        if args.do_sample:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        else:
            generation_kwargs["do_sample"] = False

        num_files, num_tokens = collect_shared_expert_data(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layer_idx=args.layer_idx,
            output_dir=Path(args.output_dir),
            max_tokens_per_prompt=args.max_tokens_per_prompt,
            max_new_tokens=args.max_new_tokens,
            max_total_tokens=args.max_total_tokens,
            generation_kwargs=generation_kwargs,
            resume=args.resume,
        )

        print(f"\nDone: {num_files} files, {num_tokens} tokens")
        print(f"Output: {args.output_dir}")
    finally:
        for engine in replacement_engines:
            engine.uninstall()


if __name__ == "__main__":
    main()
