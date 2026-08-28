#!/usr/bin/env python3
"""Evaluate compact attention memory schemes.

Stage 1: compare three baselines on collected attention states.
  B0: Full Attention
  B1: Oracle Top-M
  B2: Segment Mean Merge

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/evaluate_compact_attention.py \
    --data_dir data/attention_states \
    --num_heads 32 \
    --head_dim 128 \
    --memory_budgets 2048 1024 512 256 128 64 \
    --output_json results/v8_attention_compact_stage1.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reshape_to_heads(x, num_heads, head_dim):
    """x: (seq_len, hidden_size) -> (num_heads, seq_len, head_dim)"""
    seq_len, hidden = x.shape
    assert hidden == num_heads * head_dim
    return x.view(seq_len, num_heads, head_dim).transpose(0, 1)


def scaled_dot_product(q, k, v, causal=False):
    """q,k,v: (num_heads, len_q, head_dim), (num_heads, len_kv, head_dim)"""
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        # causal mask: q_i only attends to k_j for j <= i (relative positions)
        # This assumes q and k share the same sequence and q is at the end.
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        mask = torch.arange(seq_k).unsqueeze(0).expand(seq_q, -1)
        row_idx = torch.arange(seq_q).unsqueeze(-1)
        scores = scores.masked_fill(mask > row_idx, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v), probs


def full_attention(q_all, k_all, v_all, position=None):
    """Compute baseline attention at a single position using full history."""
    if position is None:
        position = q_all.shape[1] - 1
    q = q_all[:, position:position + 1, :]
    k_hist = k_all[:, :position + 1, :]
    v_hist = v_all[:, :position + 1, :]
    out, probs = scaled_dot_product(q, k_hist, v_hist, causal=False)
    return out.squeeze(1), probs.squeeze(1)


def oracle_top_m(q_all, k_all, v_all, position, m):
    """Keep M KV with highest attention score."""
    q = q_all[:, position:position + 1, :]
    k_hist = k_all[:, :position + 1, :]
    v_hist = v_all[:, :position + 1, :]
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k_hist.transpose(-2, -1)).squeeze(1) / math.sqrt(head_dim)
    top_k = min(m, scores.shape[-1])
    _, top_idx = torch.topk(scores, top_k, dim=-1)
    # Gather selected K/V per head.
    num_heads = k_all.shape[0]
    k_sel = torch.zeros(num_heads, top_k, k_all.shape[-1])
    v_sel = torch.zeros(num_heads, top_k, v_all.shape[-1])
    for h in range(num_heads):
        k_sel[h] = k_hist[h, top_idx[h]]
        v_sel[h] = v_hist[h, top_idx[h]]
    # Recompute attention only over selected KV.
    q_h = q_all[:, position:position + 1, :]
    out, probs = scaled_dot_product(q_h, k_sel, v_sel, causal=False)
    return out.squeeze(1)


def segment_mean_merge(q_all, k_all, v_all, position, m):
    """Merge consecutive segments of KV into M compact states."""
    seq_len = position + 1
    if m >= seq_len:
        return full_attention(q_all, k_all, v_all, position)[0]
    segment_size = seq_len // m
    remainder = seq_len % m
    # Split into m segments, distribute remainder to first few.
    segments = []
    start = 0
    for i in range(m):
        size = segment_size + (1 if i < remainder else 0)
        segments.append((start, start + size))
        start += size

    num_heads, _, head_dim = k_all.shape
    k_merged = torch.zeros(num_heads, m, head_dim)
    v_merged = torch.zeros(num_heads, m, head_dim)
    for h in range(num_heads):
        for j, (s, e) in enumerate(segments):
            k_merged[h, j] = k_all[h, s:e, :].mean(dim=0)
            v_merged[h, j] = v_all[h, s:e, :].mean(dim=0)

    q_h = q_all[:, position:position + 1, :]
    out, _ = scaled_dot_product(q_h, k_merged, v_merged, causal=False)
    return out.squeeze(1)


def cosine_similarity(a, b):
    a_f = a.float()
    b_f = b.float()
    a_n = F.normalize(a_f, dim=-1, eps=1e-8)
    b_n = F.normalize(b_f, dim=-1, eps=1e-8)
    return (a_n * b_n).sum(dim=-1).mean().item()


def relative_error(a, b):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).pow(2).mean().sqrt().item()
    denom = a_f.pow(2).mean().sqrt().item()
    if denom == 0:
        return float("inf")
    return diff / denom


def evaluate_sample(q_all, k_all, v_all, memory_budgets, position=None):
    """Evaluate all compression schemes at one position."""
    if position is None:
        position = q_all.shape[1] - 1
    baseline, _ = full_attention(q_all, k_all, v_all, position)

    results = {"full": {"cos": 1.0, "rel_err": 0.0}}
    for m in memory_budgets:
        if m > position + 1:
            continue
        oracle_out = oracle_top_m(q_all, k_all, v_all, position, m)
        seg_out = segment_mean_merge(q_all, k_all, v_all, position, m)
        results[f"oracle_{m}"] = {
            "cos": cosine_similarity(baseline, oracle_out),
            "rel_err": relative_error(baseline, oracle_out),
        }
        results[f"segment_{m}"] = {
            "cos": cosine_similarity(baseline, seg_out),
            "rel_err": relative_error(baseline, seg_out),
        }
    return results


def evaluate_all(args):
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    layers = manifest["layers"]
    summary = {
        "manifest": manifest_path,
        "layers": layers,
        "memory_budgets": args.memory_budgets,
        "results": {layer: {} for layer in layers},
    }

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Evaluating layer {layer}")
        print(f"{'='*60}")

        metric_sums = {}
        metric_counts = {}

        for sample_info in manifest["samples"]:
            sample_idx = sample_info["sample_idx"]
            pt_path = data_dir / sample_info["output_path"]
            if not pt_path.exists():
                continue
            data = torch.load(pt_path, map_location="cpu")

            q = reshape_to_heads(data[f"q_{layer}"], args.num_heads, args.head_dim)
            k = reshape_to_heads(data[f"k_{layer}"], args.num_heads, args.head_dim)
            v = reshape_to_heads(data[f"v_{layer}"], args.num_heads, args.head_dim)
            seq_len = q.shape[1]

            # Evaluate at a subset of positions to save time.
            positions = list(range(128, seq_len, args.position_stride))
            if not positions:
                positions = [seq_len - 1]

            for pos in positions:
                r = evaluate_sample(q, k, v, args.memory_budgets, position=pos)
                for method, metrics in r.items():
                    for metric_name, val in metrics.items():
                        key = f"{method}_{metric_name}"
                        metric_sums[key] = metric_sums.get(key, 0.0) + val
                        metric_counts[key] = metric_counts.get(key, 0) + 1

            if (sample_idx + 1) % 10 == 0:
                print(f"  processed {sample_idx + 1} samples")

        # Aggregate
        aggregated = {}
        for key in metric_sums:
            aggregated[key] = metric_sums[key] / metric_counts[key]

        # Reorganize by method.
        methods = set("_".join(k.split("_")[:-1]) for k in aggregated.keys())
        for method in methods:
            summary["results"][layer][method] = {
                "cos": aggregated.get(f"{method}_cos", 0.0),
                "rel_err": aggregated.get(f"{method}_rel_err", 0.0),
            }

        # Print compact table.
        print(f"\nLayer {layer} summary:")
        print(f"{'Method':<20} {'Cos':>8} {'RelErr':>10}")
        for method in sorted(summary["results"][layer].keys(), key=lambda x: (x.split("_")[0], int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)):
            r = summary["results"][layer][method]
            print(f"{method:<20} {r['cos']:>8.4f} {r['rel_err']:>10.4f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate compact attention memory baselines")
    parser.add_argument("--data_dir", required=True, help="Directory containing collected .pt files")
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--memory_budgets", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--position_stride", type=int, default=64, help="Evaluate every N positions")
    parser.add_argument("--output_json", default="results/v8_attention_compact_stage1.json")
    args = parser.parse_args()
    evaluate_all(args)


if __name__ == "__main__":
    main()
