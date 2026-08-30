#!/usr/bin/env python3
"""Collect per-token GDN inputs/states/outputs during decode for ONE target layer.

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/collect_gdn_step_data.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --prompt "The quick brown fox jumps over the lazy dog." \
    --layer_idx 20 \
    --num_decode_steps 1 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --output_path data/gdn_step_data/layer20.pt
"""

import argparse
import functools
import importlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer


STEP_DATA = []
_CAPTURE_ACTIVE = False


def _make_hooked_recurrent_fn(orig_fn):
    def hooked(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        **kwargs,
    ):
        global _CAPTURE_ACTIVE

        # In the official GDN recurrent/chunk functions, tensors enter as
        # [B, seq_len, num_heads, head_dim] before the internal transpose.
        # Single-token decode => seq_len == 1, i.e. query.shape[1] == 1.
        is_decode = query is not None and query.shape[1] == 1
        print(
            "[KERNEL]",
            f"capture={_CAPTURE_ACTIVE}",
            f"is_decode={is_decode}",
            f"qshape={tuple(query.shape) if query is not None else None}",
            f"id={id(orig_fn)}",
            flush=True,
        )

        if not (_CAPTURE_ACTIVE and is_decode):
            return orig_fn(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                **kwargs,
            )

        print(
            "[HOOK HIT]",
            f"q={tuple(query.shape)}",
            f"k={tuple(key.shape)}",
            f"v={tuple(value.shape)}",
            f"S_in={tuple(initial_state.shape) if torch.is_tensor(initial_state) else None}",
            flush=True,
        )

        record = {
            "q": query.detach().cpu().float(),
            "k": key.detach().cpu().float(),
            "v": value.detach().cpu().float(),
            "g": g.detach().cpu().float(),
            "beta": beta.detach().cpu().float(),
            "S_in": (
                initial_state.detach().cpu().float()
                if torch.is_tensor(initial_state)
                else None
            ),
        }

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

        record["S_out"] = final.detach().cpu().float() if torch.is_tensor(final) else None
        record["out"] = out.detach().cpu().float() if torch.is_tensor(out) else None
        STEP_DATA.append(record)

        return out, final

    return hooked


def _patch_target_layer(layer, raw_forward):
    global _CAPTURE_ACTIVE

    if not hasattr(layer, "linear_attn"):
        raise ValueError(f"Layer does not have linear_attn: {type(layer).__name__}")

    attn = layer.linear_attn
    orig_forward = raw_forward

    def forward(*args, **kwargs):
        global _CAPTURE_ACTIVE
        hs = kwargs.get("hidden_states", args[0] if args else None)
        seq_len = hs.shape[1] if hs is not None else "?"
        has_cache = "cache_params" in kwargs
        print(f"[LAYER WRAPPER] enter seq_len={seq_len} has_cache={has_cache}", flush=True)
        prev = _CAPTURE_ACTIVE
        _CAPTURE_ACTIVE = True
        try:
            out = orig_forward(*args, **kwargs)
            print(f"[LAYER WRAPPER] exit", flush=True)
            return out
        finally:
            _CAPTURE_ACTIVE = prev

    attn.forward = forward
    return orig_forward


def collect(args):
    global STEP_DATA, _CAPTURE_ACTIVE
    STEP_DATA.clear()
    _CAPTURE_ACTIVE = False

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    target_layer = model.model.layers[args.layer_idx]
    attn = target_layer.linear_attn
    print(f"Target layer class: {type(attn).__name__}")

    # The active forward callable (may be accelerate partial).
    raw_forward = attn.forward
    print("[RAW FORWARD]", raw_forward)
    f = raw_forward
    while hasattr(f, "func"):
        f = f.func
        print("  -> partial func:", f, "module:", getattr(f, "__module__", "?"))
    if hasattr(f, "__globals__"):
        print("  -> globals module:", f.__globals__.get("__name__"))
        print(
            "  -> globals names containing 'recurrent':",
            [k for k in f.__globals__ if "recurrent" in k],
        )

    old = getattr(attn, "_old_forward", None)
    print("[OLD FORWARD]", old)
    if old is not None:
        f = old
        while hasattr(f, "func"):
            f = f.func
            print("  -> old partial func:", f, getattr(f, "__module__", "?"))
        if hasattr(f, "__globals__"):
            print("  -> old globals module:", f.__globals__.get("__name__"))
            print(
                "  -> old globals has recurrent:",
                "torch_recurrent_gated_delta_rule" in f.__globals__,
            )
        try:
            import inspect

            cv = inspect.getclosurevars(f)
            print("  -> closure vars:", list(cv.nonlocals.keys()) if hasattr(cv, "nonlocals") else cv)
        except Exception as e:
            print("  -> closure inspect error:", e)
        if f.__closure__:
            freevar_names = f.__code__.co_freevars
            cells = {name: cell for name, cell in zip(freevar_names, f.__closure__)}
            for name in ("forward_func",):
                if name in cells:
                    orig = cells[name].cell_contents
                    print(f"  -> closure {name}:", orig, getattr(orig, "__module__", "?"))
                    if hasattr(orig, "__globals__"):
                        print(f"    -> globals module:", orig.__globals__.get("__name__"))
                        print(
                            f"    -> has recurrent:",
                            "torch_recurrent_gated_delta_rule" in orig.__globals__,
                        )
                    if hasattr(orig, "__code__"):
                        names = [
                            n
                            for n in orig.__code__.co_names
                            if "delta" in n or "recurrent" in n or "chunk" in n
                        ]
                        print(f"    -> relevant co_names:", names)
        wrapped = getattr(f, "__wrapped__", None)
        print("  -> __wrapped__:", wrapped)
        if wrapped is not None and hasattr(wrapped, "__globals__"):
            print("    -> wrapped globals module:", wrapped.__globals__.get("__name__"))
            print(
                "    -> wrapped has recurrent:",
                "torch_recurrent_gated_delta_rule" in wrapped.__globals__,
            )

    # Patch target layer forward to toggle capture flag.
    _patch_target_layer(target_layer, raw_forward)

    # The forward may reference either `torch_recurrent_gated_delta_rule` or
    # `recurrent_gated_delta_rule` (alias in generated MoE files). Patch both.
    candidate_modules = [
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "transformers.models.qwen3.modeling_qwen3",
    ]
    name_pairs = [
        ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"),
        ("recurrent_gated_delta_rule", "chunk_gated_delta_rule"),
    ]
    patched_modules = []
    orig_functions = {}
    for mod_name in candidate_modules:
        try:
            gdn_module = importlib.import_module(mod_name)
            for recurrent_attr, chunk_attr in name_pairs:
                if hasattr(gdn_module, recurrent_attr):
                    orig_recurrent = getattr(gdn_module, recurrent_attr)
                    orig_chunk = getattr(gdn_module, chunk_attr)
                    setattr(gdn_module, recurrent_attr, _make_hooked_recurrent_fn(orig_recurrent))
                    setattr(gdn_module, chunk_attr, _make_hooked_recurrent_fn(orig_chunk))
                    patched_modules.append(f"{mod_name}.{recurrent_attr}")
                    orig_functions[f"{mod_name}.{recurrent_attr}"] = (orig_recurrent, orig_chunk)
                    print(f"  patched {mod_name}.{recurrent_attr}")
        except Exception as e:
            print(f"  skipped {mod_name}: {e}")

    if not orig_functions:
        raise RuntimeError("No GDN kernel module found to patch")

    # Some forwards resolve the kernel as a class attribute (e.g. hub-kernel wrapper).
    cls = type(attn)
    for recurrent_attr, chunk_attr in name_pairs:
        for attr in (recurrent_attr, chunk_attr):
            fn = getattr(cls, attr, None)
            if fn is not None and callable(fn):
                hooked_fn = _make_hooked_recurrent_fn(fn)
                setattr(cls, attr, hooked_fn)
                print(f"[PATCH CLASS ATTR] {cls.__name__}.{attr} id={id(fn)} -> {id(hooked_fn)}", flush=True)
            fn_inst = getattr(attn, attr, None)
            if fn_inst is not None and callable(fn_inst):
                hooked_fn = _make_hooked_recurrent_fn(fn_inst)
                setattr(attn, attr, hooked_fn)
                print(f"[PATCH INSTANCE ATTR] {type(attn).__name__}.{attr} id={id(fn_inst)} -> {id(hooked_fn)}", flush=True)

    try:
        # Prefill.
        enc = tokenizer(
            args.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prefill_length,
        )
        input_ids = enc["input_ids"].to(device)
        print(f"Prefill length: {input_ids.shape[1]}")

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True, return_dict=True)
        cache = outputs.past_key_values

        # First decode token comes from the prefill logits, not the last prompt token.
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for step in range(args.num_decode_steps):
            with torch.no_grad():
                outputs = model(next_id, past_key_values=cache, use_cache=True, return_dict=True)
            logits = outputs.logits[:, -1, :]
            cache = outputs.past_key_values
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if step % 10 == 0:
                print(f"  decode step {step}/{args.num_decode_steps}")

        # Save.
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(STEP_DATA, out_path)
        meta = {
            "model_path": args.model_path,
            "layer_idx": args.layer_idx,
            "patched_modules": patched_modules,
            "prompt": args.prompt,
            "num_decode_steps": args.num_decode_steps,
            "output_path": str(out_path),
            "num_steps_collected": len(STEP_DATA),
        }
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(STEP_DATA)} steps to {out_path}")
    finally:
        for key, (orig_recurrent, orig_chunk) in orig_functions.items():
            mod_name, attr_base = key.rsplit(".", 1)
            gdn_module = importlib.import_module(mod_name)
            recurrent_attr = attr_base
            chunk_attr = attr_base.replace("recurrent", "chunk")
            setattr(gdn_module, recurrent_attr, orig_recurrent)
            setattr(gdn_module, chunk_attr, orig_chunk)


def main():
    parser = argparse.ArgumentParser(description="Collect per-token GDN step data")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--num_decode_steps", type=int, default=1)
    parser.add_argument("--max_prefill_length", type=int, default=256)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_path", default="data/gdn_step_data/layer20.pt")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
