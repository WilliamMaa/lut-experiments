#!/usr/bin/env python3
"""
快速验证 residual 表是否可被两级可加表近似。

不依赖新的树训练，直接对已有 v3 residual table 做两种压缩：
  1. 机械 8+8 bit 拆分（按 leaf ID 高低位）
  2. 基于 k-means 的 256+256 聚类拆分

如果机械拆分很差但 k-means 拆分不错，说明 leaf 输出本身有可加结构，
只是 leaf ID 的 bit 没有语义，值得做“基于树路径的可分解地址”。
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Dummy classes so checkpoints that pickled tree objects can be loaded.
# Only the table tensor is used by this diagnostic.
class _TreeNode:
    def __init__(self, *args, **kwargs):
        pass


class AddressGreedyTree:
    def __init__(self, *args, **kwargs):
        pass


def mechanical_split(table, coarse_bits, fine_bits):
    """
    把 leaf ID 按二进制高低位机械拆分。
    table: [num_leaves, group_size]
    返回 (T_c, T_f) 使得 table[leaf] ≈ T_c[coarse_code] + T_f[fine_code]
    """
    num_leaves, group_size = table.shape
    total_bits = coarse_bits + fine_bits
    assert num_leaves <= 2 ** total_bits, (
        f"num_leaves {num_leaves} > 2^{total_bits}; leaf IDs must fit in {total_bits} bits"
    )

    num_c = 2 ** coarse_bits
    num_f = 2 ** fine_bits

    T_c = torch.zeros(num_c, group_size)
    T_f = torch.zeros(num_f, group_size)
    counts_c = torch.zeros(num_c, group_size)
    counts_f = torch.zeros(num_f, group_size)

    for leaf in range(num_leaves):
        coarse = leaf >> fine_bits
        fine = leaf & (num_f - 1)
        T_c[coarse] += table[leaf]
        T_f[fine] += table[leaf]
        counts_c[coarse] += 1
        counts_f[fine] += 1

    T_c = T_c / counts_c.clamp(min=1)
    T_f = T_f / counts_f.clamp(min=1)

    # 两者有重复计数，取平均并去均值化
    # 更稳的做法：T_c 存 leaf 均值按 coarse 聚类，T_f 存残差按 fine 聚类
    T_c = torch.zeros(num_c, group_size)
    T_f = torch.zeros(num_f, group_size)
    counts_c = torch.zeros(num_c, 1)
    for leaf in range(num_leaves):
        coarse = leaf >> fine_bits
        T_c[coarse] += table[leaf]
        counts_c[coarse] += 1
    T_c = T_c / counts_c.clamp(min=1)

    residual = torch.zeros_like(table)
    for leaf in range(num_leaves):
        coarse = leaf >> fine_bits
        residual[leaf] = table[leaf] - T_c[coarse]

    counts_f = torch.zeros(num_f, 1)
    for leaf in range(num_leaves):
        fine = leaf & (num_f - 1)
        T_f[fine] += residual[leaf]
        counts_f[fine] += 1
    T_f = T_f / counts_f.clamp(min=1)

    return T_c, T_f


def kmeans_split(table, num_c, num_f, n_iter=20):
    """
    基于 residual table 的向量做 k-means，分成 coarse + fine 两组。
    table: [num_leaves, group_size]
    返回 (T_c, cluster_c), (T_f, cluster_f)
    """
    num_leaves, group_size = table.shape
    # coarse 聚类
    T_c, cluster_c = kmeans(table, num_c, n_iter)
    residual = table - T_c[cluster_c]
    T_f, cluster_f = kmeans(residual, num_f, n_iter)
    return T_c, cluster_c, T_f, cluster_f


def kmeans(x, k, n_iter=20):
    """简单 k-means，返回 (centers, assignments)。"""
    n, d = x.shape
    # k-means++ 初始化
    centers = torch.zeros(k, d, dtype=x.dtype)
    centers[0] = x[torch.randint(0, n, (1,)).item()]
    for i in range(1, k):
        dist = torch.cdist(x, centers[:i]).min(dim=1).values
        dist_sum = dist.sum()
        if dist_sum <= 0 or not torch.isfinite(dist_sum):
            # Fallback: random initialization when all points are identical
            # or distances collapse.
            idx = torch.randint(0, n, (1,)).item()
        else:
            prob = dist / dist_sum
            idx = torch.multinomial(prob, 1).item()
        centers[i] = x[idx]

    for _ in range(n_iter):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, dtype=torch.long)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
                counts[j] = mask.sum()
        centers = torch.where(counts.unsqueeze(1) > 0, new_centers, centers)

    dist = torch.cdist(x, centers)
    labels = dist.argmin(dim=1)
    return centers, labels


def evaluate_approximation(table, pred):
    # Filter rows where table has NaN/Inf or zero norm; these are empty/corrupt leaves.
    valid = torch.isfinite(table).all(dim=-1) & (table.norm(dim=-1) > 1e-12)
    table_v = table[valid]
    pred_v = pred[valid]
    n_valid = valid.sum().item()
    n_total = table.shape[0]

    mse = F.mse_loss(pred_v, table_v).item()
    rmse = mse ** 0.5
    cos = F.cosine_similarity(pred_v, table_v, dim=-1, eps=1e-8).mean().item()
    norm_ratio = (pred_v.norm(dim=-1) / table_v.norm(dim=-1).clamp(min=1e-12)).mean().item()
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_ratio": n_valid / n_total if n_total > 0 else 0.0,
        "mse": mse,
        "rmse": rmse,
        "cosine_similarity": cos,
        "norm_ratio": norm_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_checkpoint_dir", required=True)
    parser.add_argument("--coarse_bits", type=int, default=8)
    parser.add_argument("--fine_bits", type=int, default=8)
    parser.add_argument("--output_json", default="bitwise_compressibility.json")
    args = parser.parse_args()

    ckpt_dir = Path(args.v3_checkpoint_dir)
    # v3 checkpoints use residual_g*.pt; converted v4 checkpoints use replacement_g*.pt.
    v3_paths = sorted(ckpt_dir.glob("residual_g*.pt"))
    v4_paths = sorted(ckpt_dir.glob("replacement_g*.pt"))
    if v3_paths:
        residual_paths = v3_paths
        ckpt_format = "v3"
    else:
        residual_paths = v4_paths
        ckpt_format = "v4"
    print(f"Found {len(residual_paths)} checkpoints (format={ckpt_format})")
    if not residual_paths:
        print("No residual_g*.pt or replacement_g*.pt files found; aborting.")
        return

    results = {}
    for res_path in residual_paths:
        gid = int(res_path.stem.split("g")[-1])
        ckpt = torch.load(res_path, map_location="cpu", weights_only=False)
        if ckpt_format == "v4":
            # v4 checkpoint: lut_tables[0]=coarse, lut_tables[1]=residual
            table = ckpt["lut_tables"][1]
        else:
            table = ckpt["table"]
        if table.dim() == 3:
            table = table[0]  # squeeze num_tables dimension
        # Drop NaN/Inf rows so means and k-means don't get poisoned.
        valid_mask = torch.isfinite(table).all(dim=-1)
        table_clean = table[valid_mask].contiguous()
        num_leaves, group_size = table_clean.shape
        n_invalid = table.shape[0] - num_leaves
        print(f"\nGroup {gid}: table shape {tuple(table.shape)}, valid rows={num_leaves}, invalid rows={n_invalid}")

        # 机械 8+8 拆分
        T_c_mech, T_f_mech = mechanical_split(table_clean, args.coarse_bits, args.fine_bits)
        pred_mech = torch.zeros_like(table_clean)
        for leaf in range(num_leaves):
            coarse = leaf >> args.fine_bits
            fine = leaf & (2 ** args.fine_bits - 1)
            pred_mech[leaf] = T_c_mech[coarse] + T_f_mech[fine]
        mech_metrics = evaluate_approximation(table_clean, pred_mech)
        print(f"  mechanical {args.coarse_bits}+{args.fine_bits}: cos={mech_metrics['cosine_similarity']:.4f}, rmse={mech_metrics['rmse']:.4f}")

        # k-means 256+256
        k_total = (2 ** args.coarse_bits) + (2 ** args.fine_bits)
        if num_leaves >= k_total:
            T_c_km, _, T_f_km, _ = kmeans_split(table_clean, 2 ** args.coarse_bits, 2 ** args.fine_bits)
            # 对每个 leaf 找最近 coarse 和 fine 中心组合
            pred_km = torch.zeros_like(table_clean)
            dist_c = torch.cdist(table_clean, T_c_km)
            dist_f = torch.cdist(table_clean, T_f_km)
            for leaf in range(num_leaves):
                best_c = dist_c[leaf].argmin().item()
                best_f = dist_f[leaf].argmin().item()
                pred_km[leaf] = T_c_km[best_c] + T_f_km[best_f]
            km_metrics = evaluate_approximation(table_clean, pred_km)
            print(f"  k-means {2**args.coarse_bits}+{2**args.fine_bits}: cos={km_metrics['cosine_similarity']:.4f}, rmse={km_metrics['rmse']:.4f}")
        else:
            km_metrics = {
                "n_total": num_leaves,
                "n_valid": num_leaves,
                "valid_ratio": 1.0,
                "mse": float("nan"),
                "rmse": float("nan"),
                "cosine_similarity": float("nan"),
                "norm_ratio": float("nan"),
            }
            print(f"  k-means skipped: only {num_leaves} rows, need >= {k_total}")

        results[f"group_{gid}"] = {
            "mechanical": mech_metrics,
            "kmeans": km_metrics,
        }

    if not results:
        print("No groups were successfully processed; aborting.")
        return

    avg_mech_cos = sum(v["mechanical"]["cosine_similarity"] for v in results.values()) / len(results)
    avg_km_cos = sum(v["kmeans"]["cosine_similarity"] for v in results.values()) / len(results)
    print(f"\nAverage mechanical cos: {avg_mech_cos:.4f}")
    print(f"Average k-means cos:    {avg_km_cos:.4f}")

    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "coarse_bits": args.coarse_bits,
        "fine_bits": args.fine_bits,
        "avg_mech_cos": avg_mech_cos,
        "avg_kmeans_cos": avg_km_cos,
        "per_group": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
