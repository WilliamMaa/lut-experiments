#!/usr/bin/env python3
"""Probe Qwen3.6-35B-A3B Gated DeltaNet recurrent state structure.

This is a pure diagnostic script. It does not train or patch anything.
It loads the model, lists the layer types, picks a Gated DeltaNet (GDN)
layer, runs one prefill + one decode step, and prints the shapes/dtypes of
all intermediate tensors and of the recurrent state stored in the cache.

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/probe_gdn.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
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


def _find_attention_module(layer):
    """Return (attribute_name, module) for the attention-like submodule."""
    for name in ("linear_attn", "self_attn", "attn", "attention"):
        if hasattr(layer, name):
            return name, getattr(layer, name)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Probe GDN recurrent state structure")
    parser.add_argument("--model_path", required=True, help="Path or HF name of Qwen3.6-35B-A3B")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--target_layer", type=int, default=None, help="GDN layer to inspect")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    # ------------------------------------------------------------------
    # 1. Config summary
    # ------------------------------------------------------------------
    config = model.config
    text_cfg = config.text_config if hasattr(config, "text_config") else config

    print("=" * 60)
    print("Model config summary")
    print("=" * 60)
    keys = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "full_attention_interval",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    ]
    for k in keys:
        print(f"  {k}: {getattr(text_cfg, k, None)}")
    print(f"  layer_types: {getattr(text_cfg, 'layer_types', None)}")

    # ------------------------------------------------------------------
    # 2. Layer layout
    # ------------------------------------------------------------------
    layers = model.model.layers
    gdn_layers = []
    full_layers = []

    print("\n" + "=" * 60)
    print("Layer layout")
    print("=" * 60)
    for i, layer in enumerate(layers):
        attn_name, attn = _find_attention_module(layer)
        cls_name = type(attn).__name__ if attn is not None else type(layer).__name__
        layer_type = getattr(attn, "layer_type", None) if attn is not None else None
        marker = ""
        if "GatedDeltaNet" in cls_name:
            gdn_layers.append(i)
            marker = "  <- GDN"
        elif "Attention" in cls_name:
            full_layers.append(i)
            marker = "  <- full-attn"
        print(f"  Layer {i:2d}: attn_attr={attn_name or 'N/A':12s} class={cls_name:35s} layer_type={layer_type}{marker}")

    print(f"\nGDN layers: {gdn_layers}")
    print(f"Full-attention layers: {full_layers}")

    if not gdn_layers:
        raise RuntimeError("No GatedDeltaNet layers found; this script is only for hybrid GDN models.")

    # ------------------------------------------------------------------
    # 3. Pick a target layer
    # ------------------------------------------------------------------
    target = args.target_layer
    if target is None:
        # Avoid the very first and very last GDN layers; pick a middle one.
        target = gdn_layers[len(gdn_layers) // 2]
    if target not in gdn_layers:
        print(f"Warning: layer {target} is not a GDN layer; defaulting to closest GDN layer.")
        target = min(gdn_layers, key=lambda x: abs(x - target))
    print(f"\nTarget GDN layer: {target}")

    # ------------------------------------------------------------------
    # 4. Register hooks on the target GDN module
    # ------------------------------------------------------------------
    layer = layers[target]
    _, attn = _find_attention_module(layer)
    captured = {}

    def _make_output_hook(name):
        def hook(module, inputs, output):
            out0 = output[0] if isinstance(output, tuple) else output
            captured[name] = out0.detach().cpu()
        return hook

    def _make_input_hook(name):
        def hook(module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captured[name] = x.detach().cpu()
        return hook

    hooks = [
        # GDN is called with keyword-only arguments, so we capture its input
        # from the first projection instead of from the module pre-hook.
        attn.in_proj_qkv.register_forward_pre_hook(_make_input_hook("gdn_input")),
        attn.in_proj_qkv.register_forward_hook(_make_output_hook("in_proj_qkv")),
        attn.conv1d.register_forward_hook(_make_output_hook("conv1d_out")),
        attn.in_proj_a.register_forward_hook(_make_output_hook("gate_a_raw")),
        attn.in_proj_b.register_forward_hook(_make_output_hook("gate_b_raw")),
        attn.in_proj_z.register_forward_hook(_make_output_hook("gate_z_raw")),
        attn.register_forward_hook(_make_output_hook("gdn_output")),
    ]

    # ------------------------------------------------------------------
    # 5. Prefill with cache enabled
    # ------------------------------------------------------------------
    enc = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        input_ids = torch.cat([input_ids, input_ids], dim=-1)
    input_ids = input_ids.to(device)

    print("\n" + "=" * 60)
    print(f"Prefill: input_ids shape = {tuple(input_ids.shape)}")
    print("=" * 60)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    past = outputs.past_key_values
    print(f"past_key_values type: {type(past).__name__}")
    print(f"past_key_values length: {len(past) if past is not None else 'N/A'}")

    # ------------------------------------------------------------------
    # Helper: extract a layer cache and recurrent state from various cache formats
    # ------------------------------------------------------------------
    def _layer_cache(cache, layer_idx):
        if cache is None:
            return None
        if hasattr(cache, "layers"):
            return cache.layers[layer_idx]
        if isinstance(cache, (tuple, list)) and layer_idx < len(cache):
            return cache[layer_idx]
        return None

    def _recurrent_state_from_layer_cache(layer_cache):
        if layer_cache is None:
            return None
        # Custom Transformers cache layer object (e.g. LinearAttentionLayer)
        if hasattr(layer_cache, "recurrent_states") and isinstance(layer_cache.recurrent_states, dict):
            rs = layer_cache.recurrent_states.get(0)
            if torch.is_tensor(rs):
                return rs
        if isinstance(layer_cache, dict):
            rs = layer_cache.get("recurrent_states")
            if isinstance(rs, (list, tuple)) and len(rs) > 0 and torch.is_tensor(rs[0]):
                return rs[0]
            if torch.is_tensor(rs):
                return rs
            return None
        if isinstance(layer_cache, (list, tuple)):
            # Best guess: the 4D tensor is the recurrent state; 3D is conv state.
            for v in layer_cache:
                if torch.is_tensor(v) and v.dim() == 4:
                    return v
            return None
        return None

    def _inspect_cache(cache, label):
        info = {}
        lc = _layer_cache(cache, target)
        if isinstance(cache, (tuple, list)):
            len_str = str(len(cache))
        elif hasattr(cache, "__len__"):
            try:
                len_str = str(len(cache))
            except Exception:
                len_str = "callable"
        else:
            len_str = "N/A"
        print(f"\n[{label}] cache type={type(cache).__name__}, len={len_str}")
        if lc is None:
            print(f"  layer {target} cache not found")
            return info

        # Case 1: Transformers DynamicCache exposes custom layer objects (e.g. LinearAttentionLayer)
        if hasattr(lc, "conv_states") or hasattr(lc, "recurrent_states"):
            for attr_name in ("conv_states", "recurrent_states"):
                if not hasattr(lc, attr_name):
                    continue
                states = getattr(lc, attr_name)
                if not isinstance(states, dict):
                    continue
                init_flags = getattr(lc, f"is_{attr_name}_initialized", {})
                for idx, tensor in states.items():
                    key = f"{attr_name}[{idx}]"
                    if not torch.is_tensor(tensor):
                        print(f"    {key}: None (uninitialized)")
                        info[key] = None
                        continue
                    print(f"    {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                          f"device={tensor.device} initialized={init_flags.get(idx, '?')}")
                    info[key] = {"shapes": [tuple(tensor.shape)], "dtypes": [str(tensor.dtype)]}
            return info

        # Case 2: plain dict layer cache
        if isinstance(lc, dict):
            print(f"  layer {target} keys: {sorted(lc.keys())}")
            for k, v in lc.items():
                if isinstance(v, (list, tuple)):
                    shapes = [tuple(t.shape) if torch.is_tensor(t) else str(type(t).__name__) for t in v]
                    dtypes = [str(t.dtype) if torch.is_tensor(t) else None for t in v]
                elif torch.is_tensor(v):
                    shapes = [tuple(v.shape)]
                    dtypes = [str(v.dtype)]
                else:
                    shapes = str(type(v).__name__)
                    dtypes = None
                print(f"    {k}: shapes={shapes}, dtypes={dtypes}")
                info[k] = {"shapes": shapes, "dtypes": dtypes}
            return info

        # Case 3: tuple/list layer cache
        if isinstance(lc, (list, tuple)):
            for idx, v in enumerate(lc):
                if torch.is_tensor(v):
                    print(f"    [{idx}] tensor shape={tuple(v.shape)} dtype={v.dtype}")
                    info[f"entry_{idx}"] = {"shapes": [tuple(v.shape)], "dtypes": [str(v.dtype)]}
                else:
                    print(f"    [{idx}] {type(v).__name__}")
                    info[f"entry_{idx}"] = str(type(v).__name__)
            return info

        return info

    # ------------------------------------------------------------------
    # 6. Inspect the cache object returned after prefill
    # ------------------------------------------------------------------
    cache_info_prefill = _inspect_cache(past, "prefill")
    st_prefill = _recurrent_state_from_layer_cache(_layer_cache(past, target))
    if torch.is_tensor(st_prefill):
        print(f"\nPrefill recurrent state (layer {target}):")
        print(f"  shape      : {tuple(st_prefill.shape)}")
        print(f"  dtype      : {st_prefill.dtype}")
        print(f"  device     : {st_prefill.device}")
        print(f"  bytes      : {st_prefill.numel() * st_prefill.element_size()}")
        print(f"  min/max/mean: {st_prefill.float().min().item():.4f} / "
              f"{st_prefill.float().max().item():.4f} / {st_prefill.float().mean().item():.4f}")

    # Snapshot prefill intermediates before decode overwrites them.
    captured_prefill = {k: v for k, v in captured.items()}
    captured.clear()

    # ------------------------------------------------------------------
    # 7. Print intermediate shapes captured by hooks (prefill)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Captured intermediate tensors during PREFILL")
    print("=" * 60)
    for name in ["gdn_input", "in_proj_qkv", "conv1d_out", "gate_a_raw", "gate_b_raw", "gate_z_raw", "gdn_output"]:
        if name not in captured_prefill:
            print(f"  {name}: MISSING")
            continue
        t = captured_prefill[name]
        print(f"  {name}: shape={tuple(t.shape)}, dtype={t.dtype}")
        if t.numel() <= 1_000_000:
            print(f"          min={t.float().min().item():.4f}, max={t.float().max().item():.4f}, "
                  f"mean={t.float().mean().item():.4f}")

    # ------------------------------------------------------------------
    # 8. Single-token decode to confirm recurrent state update
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Single-token decode with cache")
    print("=" * 60)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 0
    next_token = torch.tensor([[eos_id]], device=device)

    with torch.no_grad():
        outputs2 = model(next_token, past_key_values=past, use_cache=True, return_dict=True)
    past2 = outputs2.past_key_values

    cache_info_decode = _inspect_cache(past2, "decode")
    st_decode = _recurrent_state_from_layer_cache(_layer_cache(past2, target))
    if torch.is_tensor(st_decode):
        print(f"\nDecode recurrent state (layer {target}):")
        print(f"  shape      : {tuple(st_decode.shape)}")
        print(f"  dtype      : {st_decode.dtype}")
        print(f"  device     : {st_decode.device}")
        print(f"  bytes      : {st_decode.numel() * st_decode.element_size()}")
        print(f"  min/max/mean: {st_decode.float().min().item():.4f} / "
              f"{st_decode.float().max().item():.4f} / {st_decode.float().mean().item():.4f}")
        delta = (st_decode - st_prefill).float()
        print(f"  delta w.r.t prefill: min={delta.min().item():.4f}, max={delta.max().item():.4f}, "
              f"mean_abs={delta.abs().mean().item():.4f}, norm={delta.pow(2).sum().sqrt().item():.4f}")

    # ------------------------------------------------------------------
    # 9. Save diagnostic artifacts
    # ------------------------------------------------------------------
    out_dir = Path("results/gdn_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(st_prefill):
        torch.save(st_prefill.cpu(), out_dir / f"layer{target}_recurrent_state_prefill.pt")
    if torch.is_tensor(st_decode):
        torch.save(st_decode.cpu(), out_dir / f"layer{target}_recurrent_state_decode.pt")

    diag = {
        "model_path": args.model_path,
        "target_layer": target,
        "config": {k: getattr(text_cfg, k, None) for k in keys + ["layer_types"]},
        "gdn_layers": gdn_layers,
        "full_layers": full_layers,
        "cache_info_prefill": cache_info_prefill,
        "cache_info_decode": cache_info_decode,
        "captured_prefill_shapes": {k: tuple(v.shape) for k, v in captured_prefill.items()},
        "captured_decode_shapes": {k: tuple(v.shape) for k, v in captured.items()},
    }
    out_path = out_dir / f"layer{target}_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"\nSaved diagnostic summary to {out_path}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
