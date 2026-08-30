#!/usr/bin/env python3
"""Inspect Qwen3.6-35B-A3B architecture and produce a machine-readable layer map.

Usage:
  cd LLM_LUT/v8
  python -u qwen_toolkit/inspect_model.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --device_map balanced_low_0 \
    --output_json results/qwen_layer_map.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer


def _layer_type(layer):
    if hasattr(layer, "linear_attn"):
        return "linear_attention"
    if hasattr(layer, "self_attn"):
        return "full_attention"
    return "unknown"


def _module_name(cls):
    return f"{cls.__module__}.{cls.__name__}"


def inspect(args):
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    config = model.config

    summary = {
        "model_path": args.model_path,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None),
        "linear_num_key_heads": getattr(config, "linear_num_key_heads", None),
        "linear_num_value_heads": getattr(config, "linear_num_value_heads", None),
        "linear_key_head_dim": getattr(config, "linear_key_head_dim", None),
        "linear_value_head_dim": getattr(config, "linear_value_head_dim", None),
        "full_attention_interval": getattr(config, "full_attention_interval", None),
    }

    layers = []
    for idx, layer in enumerate(model.model.layers):
        lt = _layer_type(layer)
        attn_module = None
        if lt == "linear_attention":
            attn_module = layer.linear_attn
        elif lt == "full_attention":
            attn_module = layer.self_attn

        layer_info = {
            "layer_idx": idx,
            "layer_type": lt,
            "attn_module_class": _module_name(type(attn_module)) if attn_module else None,
            "layer_class": _module_name(type(layer)),
            "device": str(next(layer.parameters(), torch.tensor(0)).device),
        }

        if attn_module is not None:
            params = {
                n: {
                    "shape": list(p.shape),
                    "dtype": str(p.dtype),
                    "params": p.numel(),
                }
                for n, p in attn_module.named_parameters()
            }
            layer_info["attn_params"] = params
            layer_info["attn_param_count"] = sum(p["params"] for p in params.values())

        layers.append(layer_info)

    summary["layers"] = layers
    summary["gdn_layers"] = [l["layer_idx"] for l in layers if l["layer_type"] == "linear_attention"]
    summary["full_attention_layers"] = [l["layer_idx"] for l in layers if l["layer_type"] == "full_attention"]
    summary["total_attn_params"] = sum(l.get("attn_param_count", 0) for l in layers)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved layer map to {out_path}")

    # Human-readable summary
    print(f"\nnum_hidden_layers: {summary['num_hidden_layers']}")
    print(f"GDN layers: {summary['gdn_layers']}")
    print(f"Full-attention layers: {summary['full_attention_layers']}")
    print(f"Total attention params: {summary['total_attn_params'] / 1e6:.2f}M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--output_json", default="results/qwen_layer_map.json")
    args = parser.parse_args()
    inspect(args)


if __name__ == "__main__":
    main()
