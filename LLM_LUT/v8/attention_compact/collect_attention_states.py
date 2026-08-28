#!/usr/bin/env python3
"""Collect attention states for Attention Compact Memory / LUT experiments.

Stage 0: record Q, K, V, attention probabilities, attention output, and residual
inputs for selected layers during a single prefill pass on long texts.

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/collect_attention_states.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --eval_file /data/v8_eval_texts.jsonl \
    --max_prompts 100 \
    --max_length 2048 \
    --layers 8 24 39 \
    --output_dir data/attention_states \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import load_model_and_tokenizer
from common.prompts import load_eval_texts


class AttentionStateCollector:
    """Hook attention projections and attention output in selected layers."""

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self._buffers = {l: {} for l in layers}
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]

            def make_input_hook(layer_idx):
                def hook(module, inputs, output):
                    # inputs is a tuple; first element is hidden_states
                    x = inputs[0]
                    self._buffers[layer_idx]["residual_input"] = x.detach().cpu()
                return hook

            def make_q_hook(layer_idx):
                def hook(module, inputs, output):
                    self._buffers[layer_idx]["q"] = output.detach().cpu()
                return hook

            def make_k_hook(layer_idx):
                def hook(module, inputs, output):
                    self._buffers[layer_idx]["k"] = output.detach().cpu()
                return hook

            def make_v_hook(layer_idx):
                def hook(module, inputs, output):
                    self._buffers[layer_idx]["v"] = output.detach().cpu()
                return hook

            def make_attn_hook(layer_idx):
                def hook(module, inputs, output):
                    # For Qwen3-5-MoE, self_attn returns (attn_output, attn_weights)
                    attn_output = output[0] if isinstance(output, tuple) else output
                    self._buffers[layer_idx]["attention_output"] = attn_output.detach().cpu()
                return hook

            # Capture residual input at self_attn entry.
            self.hooks.append(layer.self_attn.register_forward_pre_hook(make_input_hook(layer_idx)))
            self.hooks.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
            self.hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
            self.hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))
            self.hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_and_clear(self):
        data = {}
        for layer_idx in self.layers:
            for key, tensor in self._buffers[layer_idx].items():
                # Convert to float32 for storage and downstream CPU analysis.
                data[f"{key}_{layer_idx}"] = tensor.float()
        self._buffers = {l: {} for l in self.layers}
        return data


def collect(args):
    texts = load_eval_texts(args.eval_file, args.max_prompts)
    print(f"Loaded {len(texts)} texts")

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        args.device,
        args.device_map,
    )
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = AttentionStateCollector(model, args.layers)

    manifest = {
        "model_path": args.model_path,
        "layers": args.layers,
        "max_length": args.max_length,
        "samples": [],
    }

    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        print(f"\n[{i+1}/{len(texts)}] length={input_ids.shape[1]}")
        with torch.no_grad():
            # output_attentions=True makes the model return attention probabilities.
            outputs = model(
                input_ids,
                output_attentions=True,
                use_cache=False,
            )

        data = collector.get_and_clear()

        # Attention probabilities can be recomputed from Q/K during evaluation,
        # so we do not store the full (heads, seq, seq) tensor to save disk space.
        # (For 100 samples of 2048 tokens this would be ~150 GB.)

        output_path = output_dir / f"sample_{i:04d}.pt"
        torch.save(data, output_path)

        manifest["samples"].append({
            "sample_idx": i,
            "text_preview": text[:200],
            "length": input_ids.shape[1],
            "output_path": str(output_path),
        })
        print(f"  saved {output_path}")

    collector.remove()

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[Manifest] {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect attention states for compact memory experiments")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--max_prompts", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 39])
    parser.add_argument("--output_dir", default="data/attention_states")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
