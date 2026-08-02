#!/usr/bin/env python3
"""
diagnose_leaf_residual_pca.py

诊断 A：检查 FFN 残差在每个 coarse leaf 内部是否低秩。

对 v3 shared-coarse + residual 的 checkpoint，在 calibration 数据上：
  1. 计算 LUT 预测 y_lut = coarse + residual
  2. 计算残差 r = y_teacher - y_lut
  3. 按 coarse leaf 分组
  4. 对每个有足够样本的 leaf 的残差做 PCA
  5. 输出 rank-1/4/8/16/32/64 的解释方差比例

用法：
  python -u diagnose_leaf_residual_pca.py \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
    --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/output \
    --max_samples 100000 \
    --min_leaf_samples 32 \
    --device cuda:0
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

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


def pca_explained_ratio(residual, ranks=(1, 4, 8, 16, 32, 64)):
    """对 [N, D] 残差做 PCA，返回各 rank 的累计解释方差比例。"""
    N, D = residual.shape
    if N < 2:
        return {r: float("nan") for r in ranks}
    centered = residual - residual.mean(dim=0, keepdim=True)
    # SVD: centered = U S V^T, singular values in descending order
    try:
        _, s, _ = torch.svd(centered.float())
    except RuntimeError:
        # fall back to CPU if CUDA SVD fails
        _, s, _ = torch.svd(centered.float().cpu())
        s = s.to(centered.device)
    total_var = (s ** 2).sum().item()
    if total_var < 1e-12:
        return {r: 0.0 for r in ranks}
    cumsum = (s ** 2).cumsum(dim=0)
    result = {}
    for r in ranks:
        k = min(r, len(s))
        result[r] = cumsum[k - 1].item() / total_var
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dataset_dir", required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--min_leaf_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    _inject_v3_classes()

    # Load v3 base
    # Infer hidden_size/group_size from first residual ckpt
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

    # Compute LUT prediction in batches
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

    # Compute coarse leaf indices
    coarse_indices = coarse_address.compute_indices(x.unsqueeze(0).to(device)).view(-1).cpu()

    # Group residuals by coarse leaf
    leaf_to_residuals = defaultdict(list)
    for i, leaf in enumerate(coarse_indices.tolist()):
        leaf_to_residuals[leaf].append(i)

    print(f"\nTotal samples: {x.shape[0]}")
    print(f"Unique coarse leaves: {len(leaf_to_residuals)}")

    # Compute per-leaf PCA
    ranks = [1, 4, 8, 16, 32, 64]
    leaf_results = []
    weighted_sum = {r: 0.0 for r in ranks}
    total_weight = 0
    n_evaluated_leaves = 0

    for leaf, idx_list in leaf_to_residuals.items():
        if len(idx_list) < args.min_leaf_samples:
            continue
        leaf_residual = residual[idx_list]
        explained = pca_explained_ratio(leaf_residual, ranks=ranks)
        weight = len(idx_list)
        for r in ranks:
            weighted_sum[r] += explained[r] * weight
        total_weight += weight
        n_evaluated_leaves += 1
        leaf_results.append({
            "leaf": int(leaf),
            "n_samples": len(idx_list),
            "explained": {str(k): float(v) for k, v in explained.items()},
        })

    print(f"Leaves with >= {args.min_leaf_samples} samples: {n_evaluated_leaves}")
    print(f"Samples covered by evaluated leaves: {total_weight} / {x.shape[0]} ({total_weight / x.shape[0]:.1%})")

    if total_weight > 0:
        print("\nWeighted average explained variance ratio:")
        for r in ranks:
            avg = weighted_sum[r] / total_weight
            print(f"  rank-{r:2d}: {avg:.4f}")

    # Distribution of rank-8 explained ratio across leaves
    if leaf_results:
        rank8_values = [lr["explained"]["8"] for lr in leaf_results]
        rank8_tensor = torch.tensor(rank8_values)
        print(f"\nRank-8 explained ratio distribution across leaves:")
        print(f"  mean: {rank8_tensor.mean():.4f}")
        print(f"  p10:  {torch.quantile(rank8_tensor, 0.10):.4f}")
        print(f"  p25:  {torch.quantile(rank8_tensor, 0.25):.4f}")
        print(f"  p50:  {torch.quantile(rank8_tensor, 0.50):.4f}")
        print(f"  p75:  {torch.quantile(rank8_tensor, 0.75):.4f}")
        print(f"  p90:  {torch.quantile(rank8_tensor, 0.90):.4f}")

    report = {
        "n_samples": x.shape[0],
        "n_unique_leaves": len(leaf_to_residuals),
        "n_evaluated_leaves": n_evaluated_leaves,
        "samples_covered": int(total_weight),
        "coverage_ratio": total_weight / x.shape[0] if x.shape[0] > 0 else 0.0,
        "weighted_avg_explained": {str(r): weighted_sum[r] / total_weight if total_weight > 0 else 0.0 for r in ranks},
        "leaf_results": leaf_results,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
