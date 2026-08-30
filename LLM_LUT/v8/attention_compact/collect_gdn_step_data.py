#!/usr/bin/env python3
"""Collect per-token GDN inputs/states/outputs during decode.

This is a diagnostic/prototype data collector. It runs a prefill on one prompt,
then generates `num_decode_steps` tokens, and records for a target GDN layer at
 every decode step:
   q_t, k_t, v_t, g_t, beta_t, S_{t-1}, S_t, core_attn_out_t

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/collect_gdn_step_data.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --prompt "The quick brown fox jumps over the lazy dog." \
    --layer_idx 20 \
    --num_decode_steps 64 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_path data/gdn_step_data/layer20.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer


STEP_DATA = []
_TARGET_LAYER = None


def _make_hooked_recurrent_fn(orig_fn):
    def hooked_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False, **kwargs):
        # We only care about single-token (decode) calls.
        if query is None or query.shape[2] != 1:
            return orig_fn(query, key, value, g, beta, initial_state=initial_state, output_final_state=output_final_state, **kwargs)
        STEP_DATA.append({
            "q": query.detach().cpu().float(),
            "k": key.detach().cpu().float(),
            "v": value.detach().cpu().float(),
            "g": g.detach().cpu().float(),
            "beta": beta.detach().cpu().float(),
            "S_in": initial_state.detach().cpu().float() if torch.is_tensor(initial_state) else None,
        })
        out, final = orig_fn(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            **kwargs,
        )
        STEP_DATA[-1]["S_out"] = final.detach().cpu().float() if torch.is_tensor(final) else None
        STEP_DATA[-1]["out"] = out.detach().cpu().float() if torch.is_tensor(out) else None
        return out, final

    return hooked_recurrent_gated_delta_rule


def _make_hooked_chunk_fn(orig_fn):
    def hooked_chunk_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False, **kwargs):
        # If the model falls back to chunk path for seq_len=1 decode, capture it too.
        if query is None or query.shape[2] != 1:
            return orig_fn(query, key, value, g, beta, initial_state=initial_state, output_final_state=output_final_state, **kwargs)
        STEP_DATA.append({
            "q": query.detach().cpu().float(),
            "k": key.detach().cpu().float(),
            "v": value.detach().cpu().float(),
            "g": g.detach().cpu().float(),
            "beta": beta.detach().cpu().float(),
            "S_in": initial_state.detach().cpu().float() if torch.is_tensor(initial_state) else None,
        })
        out, final = orig_fn(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            **kwargs,
        )
        STEP_DATA[-1]["S_out"] = final.detach().cpu().float() if torch.is_tensor(final) else None
        STEP_DATA[-1]["out"] = out.detach().cpu().float() if torch.is_tensor(out) else None
        return out, final

    return hooked_chunk_gated_delta_rule


def _patch_layer_forward(layer, layer_idx):
    """Patch only the target layer to tag which step data belongs to it."""
    orig_forward = layer.linear_attn.forward

    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return orig_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

    layer.linear_attn.forward = forward


def collect(args):
    global _TARGET_LAYER, STEP_DATA
    _TARGET_LAYER = args.layer_idx
    STEP_DATA.clear()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    # Monkey-patch the recurrent/chunk functions in the modeling module.
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as gdn_module

    orig_recurrent = gdn_module.torch_recurrent_gated_delta_rule
    orig_chunk = gdn_module.torch_chunk_gated_delta_rule
    gdn_module.torch_recurrent_gated_delta_rule = _make_hooked_recurrent_fn(orig_recurrent)
    gdn_module.torch_chunk_gated_delta_rule = _make_hooked_chunk_fn(orig_chunk)

    try:
        # Prefill.
        enc = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_prefill_length)
        input_ids = enc["input_ids"].to(device)
        print(f"Prefill length: {input_ids.shape[1]}")
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True, return_dict=True)
        cache = outputs.past_key_values

        # Decode steps.
        next_id = input_ids[:, -1:]
        for step in range(args.num_decode_steps):
            with torch.no_grad():
                outputs = model(next_id, past_key_values=cache, use_cache=True, return_dict=True)
            logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            cache = outputs.past_key_values
            if step % 10 == 0:
                print(f"  decode step {step}/{args.num_decode_steps}")

        # Save.
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(STEP_DATA, out_path)
        meta = {
            "model_path": args.model_path,
            "layer_idx": args.layer_idx,
            "prompt": args.prompt,
            "num_decode_steps": args.num_decode_steps,
            "output_path": str(out_path),
            "num_steps_collected": len(STEP_DATA),
        }
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(STEP_DATA)} steps to {out_path}")
    finally:
        gdn_module.torch_recurrent_gated_delta_rule = orig_recurrent
        gdn_module.torch_chunk_gated_delta_rule = orig_chunk


def main():
    parser = argparse.ArgumentParser(description="Collect per-token GDN step data")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--num_decode_steps", type=int, default=64)
    parser.add_argument("--max_prefill_length", type=int, default=256)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_path", default="data/gdn_step_data/layer20.pt")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
