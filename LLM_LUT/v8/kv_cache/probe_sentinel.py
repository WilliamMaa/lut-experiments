#!/usr/bin/env python3
"""Diagnose the sentinel-fact failure: is the answer span evicted, or kept but unread?

Runs ONE turn (document + question) through heavy_hitter_attn with a fresh
cache, then inspects every full-attn layer's compressed cache:

  - the ground-truth answer span's original prefill positions
  - whether each is in the kept set (sink/hh/recent, via _hh_orig_idx)
  - its prefill attention-mass score and rank among all positions
  - whether it falls inside the obs-window exclusion zone

This distinguishes the two failure modes, which demand opposite fixes:
  evicted          -> selection problem (scores don't value the fact)
  kept but unread  -> addressing problem (queries don't attend it)

Usage:
  python kv_cache/probe_sentinel.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --multi_turn_file data/multi_turn_prompts_v3.jsonl \
    --doc_index 0 --turn 4 --answer_text "178亿元至182亿元" \
    --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
    [--merge_evicted] [--shared_selection] [--k_bits 8 --v_bits 8]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer
from common.prompts import load_multi_turn_prompts
from kv_cache.kv_cache_patch import HeavyHitterAttnScorePatch


def find_span_positions(prompt_ids, doc_ids_window):
    """First occurrence of the doc-window token subsequence inside prompt ids."""
    n = len(doc_ids_window)
    p = prompt_ids
    for i in range(len(p) - n + 1):
        if p[i:i + n] == doc_ids_window:
            return list(range(i, i + n))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--multi_turn_file", required=True)
    ap.add_argument("--doc_index", type=int, default=0)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--answer_text", required=True)
    ap.add_argument("--max_cache_len", type=int, default=128)
    ap.add_argument("--sink_tokens", type=int, default=4)
    ap.add_argument("--recent_tokens", type=int, default=32)
    ap.add_argument("--obs_window", type=int, default=64)
    ap.add_argument("--merge_evicted", action="store_true")
    ap.add_argument("--shared_selection", action="store_true")
    ap.add_argument("--k_bits", type=int, default=16)
    ap.add_argument("--v_bits", type=int, default=16)
    ap.add_argument("--device_map", default="balanced_low_0")
    ap.add_argument("--torch_dtype", default="bfloat16")
    args = ap.parse_args()

    samples = load_multi_turn_prompts(args.multi_turn_file, max_samples=args.doc_index + 1)
    sample = samples[args.doc_index]
    document = sample["document"]
    question = sample["questions"][args.turn]

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map,
    )

    # Build the single-turn prompt exactly like run_multi_turn_generation turn 0.
    no_think_text = (
        "直接给出最终回复，不要输出思考过程、分析步骤、'Here's a thinking process' "
        "或任何元说明。"
    )
    messages = [
        {"role": "system", "content": no_think_text},
        {"role": "user", "content": f"{document}\n\n{question}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    )
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Locate the answer span: tokenize the doc, find the char span of the
    # answer text, take a token window around it, search in the prompt ids.
    char_start = document.find(args.answer_text)
    if char_start < 0:
        print(f"[probe] answer text not found in document: {args.answer_text}")
        sys.exit(1)
    enc = tokenizer(document, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    doc_ids = enc["input_ids"]
    span_tok_idx = [i for i, (a, b) in enumerate(offsets)
                    if a >= char_start and b <= char_start + len(args.answer_text)]
    if not span_tok_idx:
        # fall back: nearest token containing the span start
        span_tok_idx = [min(range(len(offsets)),
                            key=lambda i: abs(offsets[i][0] - char_start))]
    lo = max(0, span_tok_idx[0] - 3)
    hi = min(len(doc_ids), span_tok_idx[-1] + 4)
    window_ids = doc_ids[lo:hi]
    prompt_ids = input_ids[0].tolist()
    found = find_span_positions(prompt_ids, window_ids)
    if found is None:
        print("[probe] could not align doc token window to prompt ids")
        sys.exit(1)
    span_prompt_pos = found[span_tok_idx[0] - lo: span_tok_idx[-1] - lo + 1]
    span_orig = [p for p in span_prompt_pos]  # single prefill, pos == orig
    print(f"[probe] prompt_len={prompt_len} budget={args.max_cache_len}")
    print(f"[probe] answer span tokens: "
          f"{[tokenizer.decode([prompt_ids[p]]) for p in span_prompt_pos]}")
    print(f"[probe] answer orig positions: {span_orig}")
    print(f"[probe] obs_window exclusion zone: "
          f"[{prompt_len - args.obs_window}, {prompt_len})")

    patch = HeavyHitterAttnScorePatch(
        max_cache_len=args.max_cache_len,
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        obs_window=args.obs_window,
        merge_evicted=args.merge_evicted,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        shared_selection=args.shared_selection,
    )
    print(f"[probe] patch: {patch.name()}")
    patch.install(model)
    cache = patch.get_cache(device)

    gen_kwargs = dict(
        max_new_tokens=8, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        past_key_values=cache,
    )
    with torch.no_grad():
        out = model.generate(input_ids=input_ids,
                             attention_mask=torch.ones_like(input_ids), **gen_kwargs)
    print(f"[probe] generated: {tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)!r}")

    # Inspect each full-attn layer's compressed cache.
    for idx, layer in enumerate(cache.layers):
        if not (hasattr(layer, "keys") and torch.is_tensor(layer.keys)
                and layer.keys.dim() == 4 and getattr(layer, "_hh_orig_idx", None) is not None):
            continue
        orig_idx = layer._hh_orig_idx
        scores = getattr(layer, "_hh_prefill_scores", None)
        kept = set(orig_idx.tolist())
        plen = scores.shape[-1] if scores is not None else prompt_len
        line = f"[layer {idx:2d}] kept_len={orig_idx.numel():4d} | span:"
        for p in span_orig:
            s = scores[p].item() if scores is not None and p < plen else float("nan")
            rank = int((scores > scores[p]).sum().item()) if scores is not None and p < plen else -1
            in_win = p >= plen - args.obs_window
            line += (f" pos={p} kept={'Y' if p in kept else 'N'}"
                     f" score={s:.4g} rank={rank}/{plen}"
                     f"{' IN_OBS_WIN' if in_win else ''} |")
        print(line)

    patch.uninstall(model)


if __name__ == "__main__":
    main()
