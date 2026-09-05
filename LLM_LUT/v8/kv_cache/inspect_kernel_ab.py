#!/usr/bin/env python3
"""A/B test: original attention vs the stashing eager kernel.

Runs the same input through the model twice and compares logits. The patch
does not touch weights, so if the kernel is mathematically correct the two
logits must match closely. A large divergence localizes the quality loss to
the kernel (which corrupts EVERY forward pass, PPL included), not to the
cache eviction policy.

Also probes and dumps the exact attention_mask / kwargs the kernel receives
on the first calls, so mask-handling bugs are visible directly.

Usage (on remote):
    python kv_cache/inspect_kernel_ab.py \
        --model_path /home/u/downloads/models/Qwen3.6-35B-A3B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.utils import load_model_and_tokenizer
import kv_cache.attention_scores as attn_scores

PROBE = {"calls": []}


def _fp64_reference(query, key, value, attention_mask, scaling):
    """Ground-truth eager attention in float64 on CPU (same math, high precision)."""
    q = query.detach().double().cpu()
    k = key.detach().double().cpu()
    v = value.detach().double().cpu()
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    w = torch.matmul(q, k.transpose(-1, -2)) * scaling
    if torch.is_tensor(attention_mask):
        w = w + attention_mask.detach().double().cpu()[..., : k.shape[-2]]
    p = torch.softmax(w, dim=-1)
    return torch.matmul(p, v)


def probing_kernel(module, query, key, value, attention_mask, *args, **kwargs):
    out, attn_weights = attn_scores._ORIG_KERNEL(module, query, key, value,
                                                 attention_mask, *args, **kwargs)
    if len(PROBE["calls"]) < 3:
        mask = attention_mask
        info = {
            "layer_idx": getattr(module, "layer_idx", None),
            "q_shape": tuple(query.shape),
            "k_shape": tuple(key.shape),
            "v_shape": tuple(value.shape),
            "mask_type": type(mask).__name__,
            "kwargs": {
                k2: (tuple(v2.shape) if torch.is_tensor(v2) else v2)
                for k2, v2 in kwargs.items()
            },
        }
        if torch.is_tensor(mask):
            info["mask_shape"] = tuple(mask.shape)
            info["mask_dtype"] = str(mask.dtype)
            if mask.dtype.is_floating_point:
                flat = mask.float().flatten()
                info["mask_min"] = round(flat.min().item(), 4)
                info["mask_max"] = round(flat.max().item(), 4)
                info["mask_frac_min"] = round(
                    (flat < -1e6).float().mean().item(), 4)
            else:
                info["mask_true_frac"] = round(
                    mask.bool().float().mean().item(), 4)
        elif isinstance(mask, dict):
            info["mask_entries"] = {
                k2: (tuple(v2.shape) if torch.is_tensor(v2) else type(v2).__name__)
                for k2, v2 in mask.items()
            }
        # Per-layer ground truth check: is OUR output close to exact math?
        scale = kwargs.get("scaling") or query.shape[-1] ** -0.5
        ref = _fp64_reference(query, key, value, attention_mask, scale)
        d = (out.detach().double().cpu() - ref).abs()
        info["layer_vs_fp64_max"] = round(d.max().item(), 6)
        info["layer_vs_fp64_mean"] = round(d.mean().item(), 6)
        info["out_absmax"] = round(out.detach().abs().max().item(), 4)
        PROBE["calls"].append(info)
    return out, attn_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--seq_len", type=int, default=64)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, device_map=args.device_map)
    print("default attn implementation:", model.config._attn_implementation)

    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=args.seq_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"input shape: {tuple(input_ids.shape)}")

    with torch.no_grad():
        logits_ref = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits.float()

    # Install our kernel exactly like the eval patch does, then add the probe.
    bank = attn_scores.AttentionScoreBank()
    attn_scores.install_eager_score_stash(model, bank)
    attn_scores._registry_target()["eager"] = probing_kernel

    with torch.no_grad():
        logits_new = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits.float()

    diff = (logits_ref - logits_new).abs()
    print("\n=== A/B RESULT ===")
    print(f"max abs diff:    {diff.max().item():.6f}")
    print(f"mean abs diff:   {diff.mean().item():.6f}")
    agree = (logits_ref.argmax(-1) == logits_new.argmax(-1)).float().mean().item()
    print(f"top-1 agreement: {agree:.2%}")
    print(f"ref logit range: [{logits_ref.min().item():.2f}, {logits_ref.max().item():.2f}]")

    print(f"\nbank entries: {sorted(bank.scores.keys())}")
    print(f"bank score lengths: {[tuple(v.shape) for v in bank.scores.values()]}")
    print(f"\nkernel probe (first {len(PROBE['calls'])} calls):")
    for c in PROBE["calls"]:
        print(" ", c)


if __name__ == "__main__":
    main()
