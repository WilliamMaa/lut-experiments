#!/usr/bin/env python3
"""
diagnose_pairwise_interaction.py

诊断 B：检查 group 之间的残差交互。

对 v3 shared-coarse + residual 的 checkpoint，在 calibration 数据上：
  1. 计算 LUT 预测 y_lut = coarse + residual
  2. 计算残差 r = y_teacher - y_lut
  3. 对每个 group g，取 r_g（64-dim）
  4. 检查 group h 的 residual leaf 能在多大程度上解释 r_g 的方差
  5. 输出解释方差比例最高的 group 对

如果某些 (g, h) 对的解释比例很高，说明 group g 的误差受 group h 状态影响大，
值得加入 pairwise 交互表。

用法：
  python -u diagnose_pairwise_interaction.py \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
    --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/output \
    --max_samples 100000 \
    --top_k 20 \
    --device cuda:0
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut_ffn_output_v3_shared_coarse as v3


def _inject_v3_classes():
    import __main__ as _main_mod
    for _name in ("AddressGreedyTree", "_TreeNode", "LUTGroup", "QwenMoEExpert"):
        _cls = getattr(v3, _name, None)
        if _cls is not None and not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, _cls)


def load_v3_base(ckpt_dir, hidden_size, group_size, device):
    _inject_v3_classes()
    ckpt_dir = Path(ckpt_dir)
    coarse_ckpt = torch.load(ckpt_dir / "shared_coarse.pt", map_location="cpu", weights_only=False)
    coarse_address = coarse_ckpt["address"]
    coarse_lut = v3.LUTGroup(
        num_tables=coarse_address.num_tables,
        num_entries=coarse_address.num_entries,
        output_dim=hidden_size,
        init_table=coarse_ckpt["table"],
        device=device,
    )
    residual_addresses = {}
    residual_luts = {}
    max_group = hidden_size // group_size
    for gid in range(max_group):
        residual_path = ckpt_dir / f"residual_g{gid}.pt"
        if not residual_path.exists():
            continue
        res_ckpt = torch.load(residual_path, map_location="cpu", weights_only=False)
        residual_addresses[gid] = res_ckpt["address"]
        residual_luts[gid] = v3.LUTGroup(
            num_tables=residual_addresses[gid].num_tables,
            num_entries=residual_addresses[gid].num_entries,
            output_dim=group_size,
            init_table=res_ckpt["table"],
            device=device,
        )
    group_ids = sorted(residual_luts.keys())
    return coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids


@torch.no_grad()
def predict_base(coarse_address, coarse_lut, residual_addresses, residual_luts,
                 group_ids, group_size, x, device):
    was_2d = (x.dim() == 2)
    if was_2d:
        x = x.unsqueeze(0)
    x = x.to(device)
    coarse_lut.to(device)
    coarse_indices = coarse_address.compute_indices(x).view(-1, coarse_address.num_tables)
    coarse_full = coarse_lut(coarse_indices)
    pred_y = torch.zeros_like(coarse_full)
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        residual_luts[gid].to(device)
        residual_indices = residual_addresses[gid].compute_indices(x).view(
            -1, residual_addresses[gid].num_tables
        )
        residual_group = residual_luts[gid](residual_indices)
        pred_y[:, g_start:g_end] = coarse_full[:, g_start:g_end] + residual_group
    pred_y = pred_y.view(x.shape[0], x.shape[1], -1)
    if was_2d:
        pred_y = pred_y.squeeze(0)
    return pred_y


def load_data(input_files, output_files, max_samples):
    xs, ys = [], []
    n = 0
    for inp_path, out_path in zip(input_files, output_files):
        if n >= max_samples:
            break
        x = torch.load(inp_path, map_location="cpu", weights_only=False)
        y = torch.load(out_path, map_location="cpu", weights_only=False)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if x.shape != y.shape:
            continue
        xs.append(x)
        ys.append(y)
        n += x.shape[0]
    x = torch.cat(xs, dim=0)[:max_samples]
    y = torch.cat(ys, dim=0)[:max_samples]
    return x, y


def explained_variance_by_leaf_index(target, leaf_idx, dim_size):
    """
    target: [N, D]
    leaf_idx: [N] long tensor, values in [0, dim_size)
    返回：用 leaf_idx 分组均值能解释的 target 方差比例。
    """
    N, D = target.shape
    device = target.device
    dtype = target.dtype

    # sum_per_leaf[d, leaf] = sum_i target[i, d] * (leaf_idx[i] == leaf)
    sum_per_leaf = torch.zeros(dim_size, D, device=device, dtype=dtype)
    sum_per_leaf.scatter_add_(0, leaf_idx.unsqueeze(1).expand(-1, D), target)

    count_per_leaf = torch.zeros(dim_size, device=device, dtype=dtype)
    count_per_leaf.scatter_add_(0, leaf_idx, torch.ones(N, device=device, dtype=dtype))

    # 避免除零，只保留 count > 0 的 leaf
    safe_count = count_per_leaf.clamp(min=1.0)
    mean_per_leaf = sum_per_leaf / safe_count.unsqueeze(1)

    # E[E[target_d | leaf]^2] = sum_leaf count_leaf/N * mean_leaf_d^2
    #                       = sum_leaf sum_leaf_d^2 / (N * count_leaf)
    squared_mean = (sum_per_leaf ** 2) / (safe_count.unsqueeze(1) * N)
    expected_cond_mean_sq = squared_mean.sum(dim=0)  # [D]

    expected_sq = (target ** 2).mean(dim=0)  # [D]
    total_var = target.var(dim=0, unbiased=False)  # [D]

    cond_var = expected_sq - expected_cond_mean_sq  # [D]
    cond_var = cond_var.clamp(min=0.0)

    explained_ratio = torch.where(
        total_var > 1e-12,
        1.0 - cond_var / total_var,
        torch.zeros_like(total_var)
    )
    return explained_ratio.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dataset_dir", required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    _inject_v3_classes()

    # Load v3 base and infer sizes
    ckpt_dir = Path(args.checkpoint_dir)
    first_residual = next(ckpt_dir.glob("residual_g*.pt"))
    res_ckpt = torch.load(first_residual, map_location="cpu", weights_only=False)
    group_size = res_ckpt["table"].shape[-1]
    max_group = int(first_residual.stem.split("g")[-1]) + 1
    while (ckpt_dir / f"residual_g{max_group}.pt").exists():
        max_group += 1
    hidden_size = max_group * group_size
    print(f"Inferred hidden_size={hidden_size}, group_size={group_size}")

    coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids = load_v3_base(
        args.checkpoint_dir, hidden_size, group_size, device
    )

    # Load data
    input_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    output_files_map = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.output_dataset_dir, "*.pt"))
    }
    paired = [(inp, output_files_map.get(os.path.basename(inp))) for inp in input_files]
    paired = [(inp, out) for inp, out in paired if out is not None]
    if not paired:
        raise FileNotFoundError("No paired input/output files")

    print(f"Loading up to {args.max_samples} samples ...")
    x, y = load_data([p[0] for p in paired], [p[1] for p in paired], args.max_samples)
    print(f"Data: {tuple(x.shape)}")

    # Compute LUT prediction and residual
    all_pred = []
    for start in range(0, x.shape[0], args.batch_size):
        xb = x[start:start + args.batch_size]
        pred = predict_base(
            coarse_address, coarse_lut, residual_addresses, residual_luts,
            group_ids, group_size, xb, device
        ).float().cpu()
        all_pred.append(pred)
    pred_y = torch.cat(all_pred, dim=0)
    residual = (y - pred_y).to(device)

    # Compute residual leaf indices for each group
    print("Computing residual leaf indices ...")
    residual_leaf_indices = {}
    for gid in group_ids:
        indices = residual_addresses[gid].compute_indices(x.unsqueeze(0).to(device)).view(-1).cpu().to(device)
        residual_leaf_indices[gid] = indices

    num_entries = residual_addresses[group_ids[0]].num_entries

    # For each group pair (g, h), compute explained variance
    print("\nComputing pairwise interactions ...")
    results = []
    for g in group_ids:
        g_start = g * group_size
        g_end = g_start + group_size
        r_g = residual[:, g_start:g_end]
        total_var_g = r_g.var(unbiased=False).mean().item()
        if total_var_g < 1e-12:
            continue

        for h in group_ids:
            if g == h:
                continue
            leaf_h = residual_leaf_indices[h]
            explained = explained_variance_by_leaf_index(r_g, leaf_h, num_entries)
            results.append({
                "g": g,
                "h": h,
                "explained_ratio": explained,
                "total_var_g": total_var_g,
            })

    results.sort(key=lambda x: x["explained_ratio"], reverse=True)

    print(f"\nTop-{args.top_k} interacting group pairs:")
    print(f"{'rank':>4} {'g':>3} {'h':>3} {'explained_ratio':>16} {'total_var_g':>12}")
    print("-" * 45)
    for i, r in enumerate(results[:args.top_k], 1):
        print(f"{i:4d} {r['g']:3d} {r['h']:3d} {r['explained_ratio']:16.4f} {r['total_var_g']:12.6e}")

    # Aggregate statistics
    explained_values = [r["explained_ratio"] for r in results]
    ev_tensor = torch.tensor(explained_values)
    print(f"\nExplained ratio distribution across all {len(results)} pairs:")
    print(f"  mean: {ev_tensor.mean():.4f}")
    print(f"  p50:  {torch.quantile(ev_tensor, 0.50):.4f}")
    print(f"  p90:  {torch.quantile(ev_tensor, 0.90):.4f}")
    print(f"  p95:  {torch.quantile(ev_tensor, 0.95):.4f}")
    print(f"  max:  {ev_tensor.max():.4f}")

    n_strong = int((ev_tensor > 0.20).sum().item())
    n_moderate = int((ev_tensor > 0.10).sum().item())
    print(f"\nPairs with explained_ratio > 0.20: {n_strong}")
    print(f"Pairs with explained_ratio > 0.10: {n_moderate}")

    report = {
        "n_samples": x.shape[0],
        "n_groups": len(group_ids),
        "n_pairs": len(results),
        "top_pairs": results[:args.top_k],
        "statistics": {
            "mean": float(ev_tensor.mean()),
            "p50": float(torch.quantile(ev_tensor, 0.50)),
            "p90": float(torch.quantile(ev_tensor, 0.90)),
            "p95": float(torch.quantile(ev_tensor, 0.95)),
            "max": float(ev_tensor.max()),
            "n_strong_gt_0.20": n_strong,
            "n_moderate_gt_0.10": n_moderate,
        },
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
