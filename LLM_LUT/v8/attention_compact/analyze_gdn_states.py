#!/usr/bin/env python3
"""Analyze collected GDN recurrent states for compressibility.

Reports:
  - per-head / per-layer effective rank
  - low-rank reconstruction error at several ranks
  - head energy distribution
  - inter-position delta statistics (if multiple positions were collected)

Usage:
  cd LLM_LUT/v8
  python -u attention_compact/analyze_gdn_states.py \
    --data_dir data/gdn_states \
    --output_json results/gdn_state_analysis.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _relative_frobenius_error(orig, approx):
    diff = (orig - approx).float()
    denom = orig.float().pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    num = diff.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    # average over heads and samples
    valid = denom > 0
    ratios = (num / denom.clamp_min(1e-12))[valid]
    return ratios.mean().item() if ratios.numel() else float("nan")


def _low_rank_error_for_stack(stack, rank):
    """stack: (N, H, K, V) -> approximate each H matrix with rank `rank`."""
    # Move to float32 for SVD.
    x = stack.float()
    # svd of each (K, V) matrix.
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    # truncate
    if rank >= S.shape[-1]:
        return 0.0
    U_r = U[..., :rank]
    S_r = S[..., :rank]
    Vh_r = Vh[..., :rank, :]
    # Scale columns of U by singular values: S_r shape (..., rank) -> (..., 1, rank)
    approx = (U_r * S_r.unsqueeze(-2)) @ Vh_r
    return _relative_frobenius_error(x, approx)


def analyze(data_dir, output_json):
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    layers = manifest["layers"]
    positions = manifest.get("requested_positions", [])
    summary = {}

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Group states by position.
        states_by_pos = {p: [] for p in positions}
        for sample in manifest["samples"]:
            # The manifest may store either a file name or a path relative to
            # the original collection working directory. Try both.
            raw_path = Path(sample["output_path"])
            if raw_path.exists():
                pt_path = raw_path
            else:
                pt_path = data_dir / raw_path.name
            if not pt_path.exists():
                print(f"  warning: missing {pt_path}")
                continue
            data = torch.load(pt_path, map_location="cpu")
            for pos, layer_states in data.items():
                if pos not in states_by_pos:
                    continue
                st = layer_states.get(layer)
                if torch.is_tensor(st):
                    states_by_pos[pos].append(st)

        layer_summary = {}
        for pos in sorted(states_by_pos.keys()):
            states = states_by_pos[pos]
            if not states:
                continue
            stack = torch.stack(states, dim=0)  # (N, H, K, V)
            n, h, k, v = stack.shape
            print(f"\nPosition {pos}: {n} samples, shape {tuple(stack.shape)}")

            # Per-head singular values.
            flat = stack.reshape(-1, k, v).float()
            U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
            S = S.reshape(n, h, -1)  # (N, H, min(K,V))
            mean_s = S.mean(dim=(0, 1))  # (min(K,V),)
            total_energy = mean_s.sum()
            cum_energy = mean_s.cumsum(dim=0)
            effective_ranks = []
            for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
                # number of singular values needed to reach threshold of energy
                idx = (cum_energy >= threshold * total_energy).nonzero(as_tuple=True)[0]
                eff = idx[0].item() + 1 if idx.numel() else mean_s.numel()
                effective_ranks.append((threshold, eff))

            # Low-rank reconstruction errors.
            rank_errors = {}
            for r in [1, 2, 4, 8, 16, 32, 64]:
                if r >= min(k, v):
                    continue
                err = _low_rank_error_for_stack(stack, r)
                rank_errors[r] = err
                print(f"  rank {r:2d} rel err: {err:.4f}")

            # Head energy distribution: spectral norm per head averaged over samples.
            spectral_norms = S.max(dim=-1)[0]  # (N, H)
            mean_spec = spectral_norms.mean(dim=0)  # (H,)
            total_spec = mean_spec.sum()
            head_energy_share = (mean_spec / total_spec).numpy().tolist()
            top1_share = head_energy_share[0]
            top4_share = sum(sorted(head_energy_share, reverse=True)[:4])
            top8_share = sum(sorted(head_energy_share, reverse=True)[:8])

            print(f"  effective ranks (energy): {effective_ranks}")
            print(f"  top-1 head energy share: {top1_share:.4f}, top-4: {top4_share:.4f}, top-8: {top8_share:.4f}")

            pos_summary = {
                "num_samples": n,
                "shape": [n, h, k, v],
                "mean_singular_values": mean_s.numpy().tolist(),
                "effective_ranks": {f"{int(t*100)}%": r for t, r in effective_ranks},
                "low_rank_rel_err": rank_errors,
                "head_energy_share": head_energy_share,
                "top1_head_energy_share": top1_share,
                "top4_head_energy_share": top4_share,
                "top8_head_energy_share": top8_share,
            }

            # Delta w.r.t. previous position, if available.
            prev_pos = max((p for p in states_by_pos if p < pos and states_by_pos[p]), default=None)
            if prev_pos is not None:
                prev_stack = torch.stack(states_by_pos[prev_pos], dim=0)
                # Align samples by sample index order? states_by_pos lists are in manifest order.
                if prev_stack.shape[0] == stack.shape[0]:
                    delta = (stack - prev_stack).float()
                    rel = delta.pow(2).sum(dim=(-2, -1)).sqrt() / stack.float().pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(1e-12)
                    rel_mean = rel.mean().item()
                    rel_max = rel.max().item()
                    # Per-head entry-wise delta stats.
                    delta_abs = delta.abs()
                    mean_abs_delta = delta_abs.mean().item()
                    max_abs_delta = delta_abs.max().item()
                    pos_summary["delta"] = {
                        "from_position": prev_pos,
                        "relative_frob_mean": rel_mean,
                        "relative_frob_max": rel_max,
                        "mean_abs_element": mean_abs_delta,
                        "max_abs_element": max_abs_delta,
                    }
                    print(f"  delta from pos {prev_pos}: rel_frob_mean={rel_mean:.4f}, "
                          f"rel_frob_max={rel_max:.4f}, mean_abs_elem={mean_abs_delta:.6f}")

            layer_summary[str(pos)] = pos_summary

        summary[str(layer)] = layer_summary

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "manifest": str(manifest_path),
            "layers": layers,
            "positions": positions,
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GDN recurrent states")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_json", default="results/gdn_state_analysis.json")
    args = parser.parse_args()
    analyze(args.data_dir, args.output_json)


if __name__ == "__main__":
    main()
