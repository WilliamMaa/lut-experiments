#!/usr/bin/env python3
"""Collect Gated DeltaNet recurrent states for compressibility analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer
from common.prompts import load_eval_texts


def _extract_states(cache, layer_indices):
    out = {}
    for idx in layer_indices:
        layer_cache = cache.layers[idx]
        if not hasattr(layer_cache, "recurrent_states"):
            continue
        state = layer_cache.recurrent_states.get(0)
        if torch.is_tensor(state):
            out[idx] = state.detach().cpu().squeeze(0)
    return out


def _run_forward(model, tokenizer, text, max_length, positions, layer_indices, device):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        return {}, 0

    results = {}
    effective_positions = sorted(set(p for p in positions if 0 < p <= seq_len))
    if not effective_positions or effective_positions[-1] != seq_len:
        effective_positions.append(seq_len)

    for pos in effective_positions:
        ids = input_ids[:, :pos].to(device)
        with torch.no_grad():
            outputs = model(ids, use_cache=True, return_dict=True)
        results[pos] = _extract_states(outputs.past_key_values, layer_indices)
        del outputs
    return results, seq_len


def collect(args):
    texts = load_eval_texts(args.eval_file, args.max_prompts)
    print(f"Loaded {len(texts)} texts")

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_path": args.model_path,
        "layers": args.layers,
        "max_length": args.max_length,
        "requested_positions": args.positions,
        "samples": [],
    }

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] collecting states...")
        states_by_pos, seq_len = _run_forward(
            model, tokenizer, text, args.max_length, args.positions, args.layers, device
        )
        if not states_by_pos:
            continue

        sample_path = output_dir / f"sample_{i:04d}.pt"
        torch.save(states_by_pos, sample_path)

        manifest["samples"].append({
            "sample_idx": i,
            "text_preview": text[:200],
            "length": seq_len,
            "positions": sorted(states_by_pos.keys()),
            "output_path": sample_path.name,
        })
        print(f"  saved {sample_path}; positions={sorted(states_by_pos.keys())}; length={seq_len}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[Manifest] {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect GDN recurrent states")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 20, 26, 34])
    parser.add_argument("--positions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--max_prompts", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="data/gdn_states")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
