#!/usr/bin/env python3
"""A/B test: baseline forward vs the score-stashing sdpa wrapper.

The patch must not change the model's forward output at all (the wrapper
delegates to the original sdpa). This script verifies that: same input run
before and after install, logits must match closely. It also verifies the
score bank gets populated, and probes the exact mask/kwargs the wrapper sees.

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


def probing_wrapper(module, query, key, value, attention_mask, *args, **kwargs):
    if len(PROBE["calls"]) < 3:
        mask = attention_mask
        info = {
            "layer_idx": getattr(module, "layer_idx", None),
            "q_shape": tuple(query.shape),
            "k_shape": tuple(key.shape),
            "v_shape": tuple(value.shape),
            "mask_type": type(mask).__name__,
            "is_causal": mask is None and query.shape[-2] > 1,
            "scaling": kwargs.get("scaling"),
        }
        if torch.is_tensor(mask):
            info["mask_shape"] = tuple(mask.shape)
            info["mask_dtype"] = str(mask.dtype)
        PROBE["calls"].append(info)
    return attn_scores._sdpa_stash(module, query, key, value, attention_mask,
                                   *args, **kwargs)


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

    # Install the stash wrapper exactly like the eval patch does, then add
    # the probe on top.
    bank = attn_scores.AttentionScoreBank()
    attn_scores.install_eager_score_stash(model, bank)
    attn_scores._registry_target()["sdpa"] = probing_wrapper

    with torch.no_grad():
        logits_new = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits.float()

    diff = (logits_ref - logits_new).abs()
    print("\n=== A/B RESULT ===")
    print(f"max abs diff:    {diff.max().item():.6f}")
    print(f"mean abs diff:   {diff.mean().item():.6f}")
    agree = (logits_ref.argmax(-1) == logits_new.argmax(-1)).float().mean().item()
    print(f"top-1 agreement: {agree:.2%}")

    print(f"\nbank entries: {sorted(bank.scores.keys())}")
    print(f"bank score lengths: {[tuple(v.shape) for v in bank.scores.values()]}")
    print(f"\nwrapper probe (first {len(PROBE['calls'])} calls):")
    for c in PROBE["calls"]:
        print(" ", c)


if __name__ == "__main__":
    main()
