#!/usr/bin/env python3
"""
extract_shared_expert.py

从完整 Qwen3.6-35B-A3B 模型中提取指定 layer 的 shared_expert 权重，
保存为单 expert .pt 文件，供后续 LUT 训练作为 teacher。

用法：
  python -u extract_shared_expert.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --output_path /root/data1/rce/OLMo-core/tmp/qwen_35b_shared_expert_l39.pt \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16
"""

import argparse
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, action="append", required=True)
    parser.add_argument("--output_path", type=str, action="append", required=True)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    args = parser.parse_args()

    if len(args.layer_idx) != len(args.output_path):
        raise ValueError("Number of --layer_idx and --output_path must match")

    if args.device_map == "auto":
        raise ValueError("device_map='auto' forbidden. Use 'balanced_low_0' or explicit map.")

    dtype = getattr(torch, args.torch_dtype)
    print(f"Loading model from {args.model_path}")
    print(f"  dtype: {args.torch_dtype}")
    print(f"  device_map: {args.device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    for layer_idx, output_path in zip(args.layer_idx, args.output_path):
        # Extract shared_expert weights
        layer = model.model.layers[layer_idx]
        expert = layer.mlp.shared_expert

        state_dict = expert.state_dict()
        # Add shared_expert_gate if exists (the weighting gate)
        gate_weight = getattr(layer.mlp, "shared_expert_gate", None)
        if gate_weight is not None:
            state_dict["shared_expert_gate.weight"] = gate_weight.weight.detach().cpu()

        torch.save(state_dict, output_path)
        print(f"\nSaved shared_expert layer {layer_idx} to {output_path}")
        print("Keys:")
        for k in state_dict.keys():
            print(f"  {k}: {tuple(state_dict[k].shape)}")



if __name__ == "__main__":
    main()
