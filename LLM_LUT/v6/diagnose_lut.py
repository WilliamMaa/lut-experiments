#!/usr/bin/env python3
"""
diagnose_lut.py

加载 build_lut_ffn_output.py 生成的 checkpoint，做详细诊断：

- 每组均值 baseline
- coarse / residual / combined 单独指标
- 每个 address 的 bin 占用情况、bin 内方差占比

用法：
  python diagnose_lut.py \
    --output_root ./outputs_ffn_lut_layer1_4groups_... \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_1000w_0711 \
    --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_1000w_0711 \
    --calib_size 600000 \
    --eval_size 20000 \
    --device cuda:0
"""

import os
import glob
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lut_ffn_output import (
    estimate_eval_files_needed,
    collect_calibration_and_eval,
    LUTGroup,
    AddressHighOrderRandom,
    Address2D,
    AddressGreedyTree,
    _TreeNode,
)


class DummyTeacher(nn.Module):
    """collect_calibration_and_eval 需要 teacher 对象，但只用来取 dtype；
    如果提供预计算输出，不会调用 forward。"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        raise RuntimeError("DummyTeacher forward should not be called when output files are provided")


def fmt_metric(metrics):
    return (
        f"cos={metrics['cosine_similarity']:.4f}, "
        f"rel_mse={metrics['relative_mse']:.4f}, "
        f"rel_l2={metrics['relative_l2']:.2%}, "
        f"rmse={metrics['rmse']:.6f}"
    )


def compute_metrics(pred, true):
    mse = F.mse_loss(pred, true).item()
    rmse = math.sqrt(mse)
    var = true.var().item()
    rel_mse = mse / (var + 1e-8)
    rel_l2 = torch.norm(pred - true).item() / (torch.norm(true).item() + 1e-8)
    cos_sim = F.cosine_similarity(pred, true, dim=-1).mean().item()
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
    }


def bin_diagnostics(address, calib_x, group_target, device):
    """计算某个 address 的 bin 占用和 bin 内方差占比。"""
    with torch.no_grad():
        # torch.bincount 在 CPU 上更稳
        calib_x_cpu = calib_x.cpu()
        group_target_cpu = group_target.cpu().float()
        indices = address.compute_indices(calib_x_cpu.unsqueeze(0)).view(-1, address.num_tables)

        # 对每个 table 单独诊断（通常 num_tables=1）
        results = []
        for m in range(address.num_tables):
            idx = indices[:, m].long().cpu()
            E = address.num_entries
            N = group_target_cpu.shape[0]
            gs = group_target_cpu.shape[1]

            counts = torch.bincount(idx, minlength=E).float()
            used = (counts > 0).sum().item()
            empty = E - used

            total_var = group_target_cpu.var(dim=0).sum().item()
            sum_table = torch.zeros(E, gs, dtype=torch.float32)
            sumsq_table = torch.zeros(E, gs, dtype=torch.float32)
            for d in range(gs):
                sum_table[:, d] = torch.bincount(idx, weights=group_target_cpu[:, d], minlength=E)
                sumsq_table[:, d] = torch.bincount(idx, weights=(group_target_cpu[:, d] ** 2), minlength=E)

            mean = sum_table / counts.clamp(min=1).unsqueeze(1)
            var = (sumsq_table / counts.clamp(min=1).unsqueeze(1)) - mean ** 2
            var.clamp_(min=0.0)
            weighted_var = (counts.unsqueeze(1) * var).sum().item()
            intra_var_ratio = weighted_var / (N * total_var + 1e-12)

            results.append({
                "table_idx": m,
                "num_entries": E,
                "used_bins": used,
                "empty_bins": empty,
                "used_ratio": used / E,
                "samples_per_bin_mean": counts.mean().item(),
                "samples_per_bin_median": counts[counts > 0].median().item() if used > 0 else 0.0,
                "samples_per_bin_min": counts[counts > 0].min().item() if used > 0 else 0.0,
                "samples_per_bin_max": counts.max().item(),
                "intra_bin_var_ratio": intra_var_ratio,
                "variance_reduction": 1.0 - intra_var_ratio,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose LUT checkpoints built by build_lut_ffn_output.py")
    parser.add_argument("--output_root", required=True, help="Directory containing summary.json and checkpoints/")
    parser.add_argument("--dataset_dir", required=True, help="Input .pt directory")
    parser.add_argument("--output_dataset_dir", required=True, help="Precomputed output .pt directory")
    parser.add_argument("--calib_size", type=int, default=65536)
    parser.add_argument("--eval_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_root)
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    group_ids = summary["group_ids"]
    group_size = summary["group_size"]
    hidden_size = summary["hidden_size"]
    target_mode = summary["target_mode"]

    ckpt_dir = out_dir / "checkpoints"

    # -------------------------------------------------------------------------
    # 配对数据
    # -------------------------------------------------------------------------
    input_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    output_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(args.output_dataset_dir, "*.pt"))
    }
    paired = [(inp, output_files.get(os.path.basename(inp))) for inp in input_files]
    paired = [pair for pair in paired if pair[1] is not None]
    if not paired:
        raise FileNotFoundError(
            f"No matching .pt files between {args.dataset_dir} and {args.output_dataset_dir}"
        )
    input_files = [p[0] for p in paired]
    output_files = [p[1] for p in paired]

    n_eval = estimate_eval_files_needed(input_files, args.eval_size)
    train_input_files = input_files[:-n_eval]
    test_input_files = input_files[-n_eval:]
    train_output_files = output_files[:-n_eval]
    test_output_files = output_files[-n_eval:]

    print(f"Found {len(paired)} paired files")
    print(f"  calibration files: {len(train_input_files)}")
    print(f"  eval files: {len(test_input_files)}")

    print("\nLoading calibration / eval data ...")
    teacher = DummyTeacher()
    calib_x, calib_y, eval_x, eval_y = collect_calibration_and_eval(
        train_input_files, test_input_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
        train_output_files, test_output_files,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    # 从 calibration 计算 group mean（作为 baseline 用）
    group_means_calib = {}
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        group_means_calib[gid] = calib_y[:, g_start:g_end].mean(dim=0)

    # -------------------------------------------------------------------------
    # 逐组诊断
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Target mode: {target_mode}")
    print("=" * 70)

    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        true_group = eval_y[:, g_start:g_end].to(device)
        eval_x_device = eval_x.to(device)
        calib_x_device = calib_x.to(device)

        ckpt = torch.load(ckpt_dir / f"replacement_g{gid}.pt", map_location="cpu", weights_only=False)
        addresses = ckpt["addresses"]
        tables = ckpt["lut_tables"]
        group_mean_ckpt = ckpt.get("group_mean")

        # 创建 LUTGroup 对象
        luts = []
        for t in tables:
            t = t.float()
            lut = LUTGroup(
                num_tables=t.shape[0],
                num_entries=t.shape[1],
                group_size=t.shape[2],
                init_table=t,
                device=device,
            )
            luts.append(lut)

        # 从 checkpoint 的 group_mean（如果 target_mode 是 residual_mean）
        gm = group_mean_ckpt.float() if group_mean_ckpt is not None else group_means_calib[gid]

        print(f"\n[Group {gid}] channels [{g_start}:{g_end}]")
        print(f"  num_luts={len(addresses)}, "
              f"coarse_entries={addresses[0].num_entries}, "
              f"coarse_num_tables={addresses[0].num_tables}")
        if len(addresses) > 1:
            print(f"  residual_entries={addresses[1].num_entries}, "
                  f"residual_num_tables={addresses[1].num_tables}")

        # 1. 均值 baseline
        if target_mode == "residual_input":
            baseline_pred = eval_x_device[:, g_start:g_end]
        elif target_mode == "residual_mean":
            baseline_pred = gm.to(device).unsqueeze(0).expand_as(true_group)
        else:
            baseline_pred = group_means_calib[gid].to(device).unsqueeze(0).expand_as(true_group)
        baseline_metrics = compute_metrics(baseline_pred, true_group)
        print(f"  baseline    : {fmt_metric(baseline_metrics)}")

        # 2. coarse 单独
        coarse_indices = addresses[0].compute_indices(eval_x_device.unsqueeze(0)).view(-1, addresses[0].num_tables)
        coarse_pred = luts[0](coarse_indices).squeeze(1)
        if target_mode == "residual_mean":
            coarse_pred_full = coarse_pred + gm.to(device)
        elif target_mode == "residual_input":
            coarse_pred_full = coarse_pred + eval_x_device[:, g_start:g_end]
        else:
            coarse_pred_full = coarse_pred
        coarse_metrics = compute_metrics(coarse_pred_full, true_group)
        print(f"  coarse only : {fmt_metric(coarse_metrics)}")

        # 3. residual 单独
        if len(addresses) > 1:
            residual_indices = addresses[1].compute_indices(eval_x_device.unsqueeze(0)).view(-1, addresses[1].num_tables)
            residual_pred = luts[1](residual_indices).squeeze(1)

            # residual 目标：coarse 之后的剩余误差
            if target_mode == "residual_mean":
                residual_target = true_group - coarse_pred_full
            elif target_mode == "residual_input":
                residual_target = true_group - coarse_pred_full
            else:
                residual_target = true_group - coarse_pred

            residual_metrics = compute_metrics(residual_pred, residual_target)
            print(f"  residual tgt: {fmt_metric(residual_metrics)}")

            # 4. combined
            if target_mode == "residual_mean":
                combined_pred = coarse_pred + residual_pred + gm.to(device)
            elif target_mode == "residual_input":
                combined_pred = coarse_pred + residual_pred + eval_x_device[:, g_start:g_end]
            else:
                combined_pred = coarse_pred + residual_pred
            combined_metrics = compute_metrics(combined_pred, true_group)
            print(f"  combined    : {fmt_metric(combined_metrics)}")
        else:
            combined_metrics = coarse_metrics

        # 5. bin 诊断（基于 calibration 数据）
        calib_group_target = calib_y[:, g_start:g_end].to(device)
        # 对 direct / residual_mean / residual_input 使用 group_target_for_lut 更合适
        if target_mode == "residual_mean":
            calib_group_target_for_lut = calib_group_target - gm.to(device)
        elif target_mode == "residual_input":
            calib_group_target_for_lut = calib_group_target - calib_x_device[:, g_start:g_end]
        else:
            calib_group_target_for_lut = calib_group_target

        for desc, addr in [("coarse", addresses[0])] + ([("residual", addresses[1])] if len(addresses) > 1 else []):
            bd_list = bin_diagnostics(addr, calib_x_device, calib_group_target_for_lut, device)
            for bd in bd_list:
                print(f"  {desc} bin-diag: entries={bd['num_entries']}, "
                      f"used={bd['used_bins']}, empty={bd['empty_bins']}, "
                      f"used_ratio={bd['used_ratio']:.2%}, "
                      f"samples/bin median={bd['samples_per_bin_median']:.1f}, "
                      f"max={bd['samples_per_bin_max']:.0f}, "
                      f"intra_var_ratio={bd['intra_bin_var_ratio']:.4f}")

    print("\n" + "=" * 70)
    print("Diagnosis done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
