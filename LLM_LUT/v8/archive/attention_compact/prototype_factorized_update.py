#!/usr/bin/env python3
"""Prototype native factorized GDN recurrent update.

This script tests the core idea: instead of maintaining the dense recurrent
state S_t, maintain its low-rank factors U_t, V_t such that S_t ≈ U_t V_t^T,
and apply the Gated DeltaNet update directly in factorized form.

It can run on:
  1. Synthetic data (for quick math checks), or
  2. Real per-token step data collected by collect_gdn_step_data.py.

Usage (synthetic):
  python -u attention_compact/prototype_factorized_update.py --mode synthetic --rank 32

Usage (real):
  python -u attention_compact/prototype_factorized_update.py \
    --mode real \
    --step_data data/gdn_step_data/layer20.pt \
    --rank 32 \
    --output_json results/factorized_update_prototype.json
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


def _relative_error(a, b, eps=1e-12):
    diff = (a - b).float().pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    denom = a.float().pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)
    return (diff / denom).squeeze(-1).squeeze(-1)


def _apply_qk_norm(q, k, eps=1e-6):
    """Match the official GDN kernel's use_qk_l2norm_in_kernel behavior exactly.

    The kernel only scales query by 1/sqrt(head_dim); key is only l2-normalized.
    """
    q_inv_norm = torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + eps)
    k_inv_norm = torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + eps)
    q = q * q_inv_norm
    k = k * k_inv_norm
    scale = 1.0 / (q.shape[-1] ** 0.5)
    return q * scale, k


def _dense_update_step(S, q, k, v, g, beta):
    """Single-token GDN update using dense state.

    Shapes:
        S: (B, H, K, V)
        q, k: (B, H, K)
        v, out: (B, H, V)
        g, beta: (B, H)
    """
    q, k = _apply_qk_norm(q, k)
    decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    S_decayed = S * decay
    # kv_mem = k^T S -> (B, H, V)
    kv_mem = (S_decayed * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)  # (B, H, V)
    S_new = S_decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, K, V)
    out = (S_new * q.unsqueeze(-1)).sum(dim=-2)  # (B, H, V)
    return S_new, out


def _factorized_update_step(U, V, q, k, v, g, beta, return_dense=False):
    """Single-token GDN update using low-rank factors S = U V^T.

    U: (B, H, K, r)
    V: (B, H, V, r)
    Returns new U, V. Optionally returns dense S_new for comparison.
    """
    q, k = _apply_qk_norm(q, k)
    B, H, K, r = U.shape
    V_dim = V.shape[2]
    decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    sqrt_decay = decay.sqrt()

    # Decay factors: S_decayed = (sqrt_decay * U) (sqrt_decay * V)^T
    U_decayed = U * sqrt_decay
    V_decayed = V * sqrt_decay

    # kv_mem = k^T S = k^T U V^T = (k^T U) V^T
    kt_U = torch.matmul(k.unsqueeze(-2), U).squeeze(-2)  # (B, H, r)
    kv_mem = torch.matmul(kt_U.unsqueeze(-2), V.transpose(-2, -1)).squeeze(-2)  # (B, H, V)

    delta = (v - kv_mem) * beta.unsqueeze(-1)  # (B, H, V)

    # Augmented factors: [U_decayed, k] [V_decayed, delta]^T
    U_aug = torch.cat([U_decayed, k.unsqueeze(-1)], dim=-1)  # (B, H, K, r+1)
    V_aug = torch.cat([V_decayed, delta.unsqueeze(-1)], dim=-1)  # (B, H, V, r+1)

    # Truncate back to rank r by SVD on the small reconstructed matrix.
    S_aug = U_aug @ V_aug.transpose(-2, -1)  # (B, H, K, V)
    u, s, vh = torch.linalg.svd(S_aug, full_matrices=False)
    u_r = u[..., :r]
    s_r = s[..., :r]
    vh_r = vh[..., :r, :]
    sqrt_s = s_r.sqrt().clamp_min(1e-12)
    U_new = u_r * sqrt_s.unsqueeze(-2)
    V_new = vh_r.transpose(-2, -1) * sqrt_s.unsqueeze(-2)

    # Output without reconstructing full S: q^T S_new = (q^T U_new) V_new^T
    qt_U = torch.matmul(q.unsqueeze(-2), U_new).squeeze(-2)  # (B, H, r)
    out = torch.matmul(qt_U.unsqueeze(-2), V_new.transpose(-2, -1)).squeeze(-2)  # (B, H, V)

    if return_dense:
        S_new = U_new @ V_new.transpose(-2, -1)
        return U_new, V_new, out, S_new
    return U_new, V_new, out


def _squeeze_decode(t):
    """Drop the singleton sequence dimension from collected step tensors."""
    if t is None:
        return None
    if t.ndim == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    if t.ndim == 3 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def _factorize(S, rank):
    """Return U, V such that U @ V^T approximates S."""
    u, s, vh = torch.linalg.svd(S, full_matrices=False)
    u_r = u[..., :rank]
    s_r = s[..., :rank]
    vh_r = vh[..., :rank, :]
    sqrt_s = s_r.sqrt().clamp_min(1e-12)
    U = u_r * sqrt_s.unsqueeze(-2)
    V = vh_r.transpose(-2, -1) * sqrt_s.unsqueeze(-2)
    return U, V


def _synthetic_experiment(rank, num_steps=64, seed=42):
    torch.manual_seed(seed)
    B, H, K, V = 1, 32, 128, 128
    S0 = torch.randn(B, H, K, V) * 0.01
    U0, V0 = _factorize(S0, rank)

    errs_S = []
    errs_out = []
    dense_S_norms = []
    for i in range(num_steps):
        q = torch.randn(B, H, K)
        k = torch.randn(B, H, K)
        v = torch.randn(B, H, V)
        g = -torch.rand(B, H) * 0.1  # negative log-decay
        beta = torch.sigmoid(torch.randn(B, H))

        S_dense, out_dense = _dense_update_step(S0 if i == 0 else S_dense, q, k, v, g, beta)

        if i == 0:
            U, V = U0, V0
        U, V, out_low, S_low = _factorized_update_step(U, V, q, k, v, g, beta, return_dense=True)

        errs_S.append(_relative_error(S_dense, S_low).mean().item())
        errs_out.append(_relative_error(out_dense, out_low).mean().item())
        dense_S_norms.append(S_dense.float().pow(2).sum(dim=(-2, -1)).sqrt().mean().item())

    return {
        "mode": "synthetic",
        "rank": rank,
        "num_steps": num_steps,
        "mean_rel_err_S": sum(errs_S) / len(errs_S),
        "max_rel_err_S": max(errs_S),
        "mean_rel_err_out": sum(errs_out) / len(errs_out),
        "max_rel_err_out": max(errs_out),
        "final_rel_err_S": errs_S[-1],
        "final_rel_err_out": errs_out[-1],
    }


def _real_experiment(step_data_path, rank):
    data = torch.load(step_data_path, map_location="cpu")
    if not data:
        raise ValueError(f"No step data in {step_data_path}")

    errs_S = []
    errs_out = []
    errs_S_accum = []
    errs_S_dense_teacher = []

    # Use first step's S_in as initial state, factorize it.
    S0 = data[0]["S_in"]
    if S0 is None:
        raise ValueError("First step has no S_in; need a previous state.")
    U, V = _factorize(S0, rank)

    S_dense = S0
    for i, step in enumerate(data):
        q = _squeeze_decode(step["q"])
        k = _squeeze_decode(step["k"])
        v = _squeeze_decode(step["v"])
        g = _squeeze_decode(step["g"])
        beta = _squeeze_decode(step["beta"])
        S_dense_teacher = step["S_out"]
        out_teacher = _squeeze_decode(step["out"])

        if S_dense_teacher is None or out_teacher is None:
            continue

        S_dense, out_dense = _dense_update_step(S_dense, q, k, v, g, beta)
        U, V, out_low, S_low = _factorized_update_step(U, V, q, k, v, g, beta, return_dense=True)

        errs_S.append(_relative_error(S_dense_teacher, S_low).mean().item())
        errs_out.append(_relative_error(out_teacher, out_low).mean().item())
        errs_S_accum.append(_relative_error(S_dense, S_low).mean().item())
        errs_S_dense_teacher.append(_relative_error(S_dense_teacher, S_dense).mean().item())

    return {
        "mode": "real",
        "rank": rank,
        "num_steps": len(errs_S),
        "mean_rel_err_S": sum(errs_S) / len(errs_S),
        "max_rel_err_S": max(errs_S),
        "mean_rel_err_out": sum(errs_out) / len(errs_out),
        "max_rel_err_out": max(errs_out),
        "mean_rel_err_S_accum": sum(errs_S_accum) / len(errs_S_accum),
        "max_rel_err_S_accum": max(errs_S_accum),
        "mean_rel_err_S_dense_teacher": sum(errs_S_dense_teacher) / len(errs_S_dense_teacher),
        "max_rel_err_S_dense_teacher": max(errs_S_dense_teacher),
        "final_rel_err_S": errs_S[-1],
        "final_rel_err_out": errs_out[-1],
        "final_rel_err_S_accum": errs_S_accum[-1],
        "final_rel_err_S_dense_teacher": errs_S_dense_teacher[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Prototype factorized GDN update")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--step_data", default="data/gdn_step_data/layer20.pt")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--output_json", default="results/factorized_update_prototype.json")
    args = parser.parse_args()

    if args.mode == "synthetic":
        result = _synthetic_experiment(args.rank, args.num_steps)
    else:
        result = _real_experiment(args.step_data, args.rank)

    print(json.dumps(result, indent=2))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
