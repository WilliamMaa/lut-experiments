#!/usr/bin/env python3
"""Minimal hook recorder for Qwen3.6-35B-A3B.

Attach forward hooks to arbitrary submodules and record inputs/outputs as CPU tensors.

Usage:
  cd LLM_LUT/v8
  python -u qwen_toolkit/hook_recorder.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --layer_idx 20 \
    --submodules linear_attn,in_proj_qkv,out_proj \
    --prompt "The quick brown fox jumps over the lazy dog." \
    --output_path results/hook_records/layer20.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer


class HookRecorder:
    def __init__(self):
        self.records = []
        self.handles = []

    def _to_cpu(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().float()
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_cpu(t) for t in x)
        if isinstance(x, dict):
            return {k: self._to_cpu(v) for k, v in x.items()}
        return x

    def _hook(self, name):
        def fn(module, input, output):
            self.records.append(
                {
                    "name": name,
                    "input": self._to_cpu(input),
                    "output": self._to_cpu(output),
                }
            )
        return fn

    def attach(self, module, submodule_path):
        parts = submodule_path.split(".")
        target = module
        for p in parts:
            target = getattr(target, p)
        handle = target.register_forward_hook(self._hook(submodule_path))
        self.handles.append(handle)
        return self

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def save(self, path):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.records, out_path)
        print(f"Saved {len(self.records)} records to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--submodules", default="linear_attn", help="comma-separated submodule paths")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_path", default="results/hook_records/layer20.pt")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    recorder = HookRecorder()
    layer = model.model.layers[args.layer_idx]
    for name in args.submodules.split(","):
        recorder.attach(layer, name)
        print(f"Attached hook to layer {args.layer_idx}.{name}")

    enc = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        model(input_ids, use_cache=True)

    recorder.remove_all()
    recorder.save(args.output_path)


if __name__ == "__main__":
    main()
