#!/usr/bin/env python3
"""
run_phase0.py

Phase 0: 数据固化与 30k 基线复现。
- 收集 activation 数据
- 构建 30k anchor (k-means++)
- 复现暴力 NN + Jacobian 基线
- 复现裸 anchor 基线
- 固化评估脚本

参考: v6/build_lut_ffn_output.py 的参数风格
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.activation_collector import (
    load_expert_from_state_dict,
    prepare_data_paths,
    collect_from_pt_files,
)
from src.data.anchor_builder import AnchorBuilder, compute_anchor_coverage_stats
from src.teacher.exact_search import (
    TeacherOracle,
    evaluate_bare_anchor,
    evaluate_with_jacobian,
)
from src.evaluation.metrics import (
    compute_output_metrics,
    compute_storage_metrics,
    print_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0: Data collection and 30k anchor baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型和数据路径 (参考 v6 风格)
    parser.add_argument(
        "--teacher_weight_path",
        type=str,
        required=True,
        help="Path to expert .pt weights",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing .pt input tensors",
    )
    parser.add_argument(
        "--output_dataset_dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed .pt FFN output tensors",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for checkpoints and summary",
    )

    # 设备设置 (关键：明确单卡，禁止 auto)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., 'cuda:0', 'cpu'). "
             "IMPORTANT: Must specify single device, no auto mapping!",
    )

    # 数据收集参数
    parser.add_argument(
        "--calib_size",
        type=int,
        default=65536,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=8192,
        help="Number of evaluation samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for data collection",
    )

    # Anchor 参数
    parser.add_argument(
        "--n_anchors",
        type=int,
        default=30000,
        help="Number of anchors (30k/100k/300k)",
    )
    parser.add_argument(
        "--anchor_method",
        type=str,
        default="kmeans++",
        choices=["random", "reservoir", "kmeans++"],
        help="Anchor selection method",
    )
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        default=True,
        help="Remove near-duplicate anchors",
    )

    # Jacobian 参数
    parser.add_argument(
        "--jacobian_method",
        type=str,
        default="finite_diff",
        choices=["finite_diff", "autograd"],
        help="Jacobian computation method",
    )
    parser.add_argument(
        "--jacobian_epsilon",
        type=float,
        default=1e-4,
        help="Finite difference epsilon for Jacobian",
    )
    parser.add_argument(
        "--skip_jacobian",
        action="store_true",
        help="Skip Jacobian computation (faster, but no JVP baseline)",
    )

    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # 参数验证
    if args.calib_size <= 0:
        raise ValueError(f"--calib_size must be positive, got {args.calib_size}")
    if args.eval_size <= 0:
        raise ValueError(f"--eval_size must be positive, got {args.eval_size}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.n_anchors <= 0:
        raise ValueError(f"--n_anchors must be positive, got {args.n_anchors}")

    return args


def main():
    args = parse_args()

    print("=" * 70)
    print("Phase 0: Data Collection and 30k Anchor Baseline")
    print("=" * 70)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    print(f"  WARNING: Explicit single device. No auto multi-GPU mapping allowed.")

    # 创建输出目录
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 加载模型
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 1: Loading teacher model")
    print("-" * 70)

    teacher, hidden_size, intermediate_size = load_expert_from_state_dict(
        args.teacher_weight_path,
        device,
    )

    # -------------------------------------------------------------------------
    # 2. 准备数据路径
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 2: Preparing data paths")
    print("-" * 70)

    train_input, test_input, train_output, test_output = prepare_data_paths(
        args.dataset_dir,
        args.output_dataset_dir,
        args.eval_size,
    )

    # -------------------------------------------------------------------------
    # 3. 收集数据
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 3: Collecting activation data")
    print("-" * 70)

    print("\nCollecting calibration data...")
    calib_x, calib_y = collect_from_pt_files(
        train_input,
        train_output,
        args.calib_size,
        teacher,
        args.batch_size,
        device,
        desc="calibration",
    )

    print("\nCollecting evaluation data...")
    eval_x, eval_y = collect_from_pt_files(
        test_input,
        test_output,
        args.eval_size,
        teacher,
        args.batch_size,
        device,
        desc="evaluation",
    )

    print(f"\nData collected:")
    print(f"  Calibration: x={calib_x.shape}, y={calib_y.shape}")
    print(f"  Evaluation:  x={eval_x.shape}, y={eval_y.shape}")

    # 保存数据元信息
    data_info = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "calib_samples": calib_x.shape[0],
        "eval_samples": eval_x.shape[0],
        "calib_x_mean": calib_x.mean().item(),
        "calib_x_std": calib_x.std().item(),
        "calib_y_mean": calib_y.mean().item(),
        "calib_y_std": calib_y.std().item(),
    }

    # -------------------------------------------------------------------------
    # 4. 构建 Anchors
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"Step 4: Building {args.n_anchors} anchors ({args.anchor_method})")
    print("-" * 70)

    builder = AnchorBuilder(
        n_anchors=args.n_anchors,
        method=args.anchor_method,
        remove_duplicates=args.remove_duplicates,
    )

    anchors_x, anchors_y = builder.build(calib_x, calib_y)
    n_anchors_actual = anchors_x.shape[0]

    print(f"\nAnchors built:")
    print(f"  Target: {args.n_anchors}")
    print(f"  Actual: {n_anchors_actual}")
    print(f"  Shape: x={anchors_x.shape}, y={anchors_y.shape}")

    # 计算覆盖统计
    coverage_stats = compute_anchor_coverage_stats(calib_x, anchors_x)
    print(f"\nCoverage statistics (on calibration set):")
    for key, value in coverage_stats.items():
        print(f"  {key}: {value:.6f}")

    # 保存 anchors
    anchor_ckpt = {
        "anchors_x": anchors_x.half(),  # FP16 for storage
        "anchors_y": anchors_y.half(),
        "n_anchors": n_anchors_actual,
        "method": args.anchor_method,
        "coverage_stats": coverage_stats,
    }
    torch.save(anchor_ckpt, ckpt_dir / "anchors.pt")
    print(f"\nSaved anchors to {ckpt_dir / 'anchors.pt'}")

    # -------------------------------------------------------------------------
    # 5. 评估：裸 Anchor 基线
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 5: Evaluating bare anchor baseline (no Jacobian)")
    print("-" * 70)

    bare_metrics = evaluate_bare_anchor(
        eval_x.to(device),
        eval_y.to(device),
        anchors_x.to(device),
        anchors_y.to(device),
    )
    print_metrics(bare_metrics, "Bare Anchor Baseline")

    # -------------------------------------------------------------------------
    # 6. 评估：Anchor + Jacobian 基线 (Teacher Path)
    # -------------------------------------------------------------------------
    jacobian_metrics = None
    if not args.skip_jacobian:
        print("\n" + "-" * 70)
        print("Step 6: Evaluating anchor + Jacobian baseline (Teacher Path)")
        print("-" * 70)
        print("  WARNING: This is expensive! Computing Jacobians for all anchors...")

        jacobian_metrics = evaluate_with_jacobian(
            eval_x.to(device),
            eval_y.to(device),
            anchors_x.to(device),
            anchors_y.to(device),
            teacher,
            args.jacobian_method,
        )
        print_metrics(jacobian_metrics, "Anchor + Jacobian Baseline")
    else:
        print("\n" + "-" * 70)
        print("Step 6: Skipping Jacobian baseline (as requested)")
        print("-" * 70)

    # -------------------------------------------------------------------------
    # 7. 存储分析
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 7: Storage analysis")
    print("-" * 70)

    storage_metrics = compute_storage_metrics(
        n_anchors=n_anchors_actual,
        d_in=hidden_size,
        d_out=hidden_size,
        per_anchor_code_dim=0,  # Phase 0 没有 code
        n_banks=1,
        rank=0,
    )
    print_metrics(storage_metrics, "Storage Metrics")

    # -------------------------------------------------------------------------
    # 8. 保存 Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 8: Saving summary")
    print("-" * 70)

    summary = {
        # 配置
        "teacher_weight_path": args.teacher_weight_path,
        "dataset_dir": args.dataset_dir,
        "output_dataset_dir": args.output_dataset_dir,
        "output_root": args.output_root,
        "device": str(device),
        "seed": args.seed,

        # 模型信息
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,

        # 数据信息
        "data_info": data_info,

        # Anchor 信息
        "n_anchors_target": args.n_anchors,
        "n_anchors_actual": n_anchors_actual,
        "anchor_method": args.anchor_method,
        "coverage_stats": coverage_stats,

        # 评估指标
        "bare_anchor_metrics": bare_metrics,
        "jacobian_metrics": jacobian_metrics,

        # 存储
        "storage_metrics": storage_metrics,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # -------------------------------------------------------------------------
    # 9. 打印最终对比
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 0 Complete: Baseline Comparison")
    print("=" * 70)
    print(f"\nBare Anchor (no correction):")
    print(f"  Cosine Sim: {bare_metrics['cosine_similarity']:.4f}")
    print(f"  Relative L2: {bare_metrics['relative_l2']:.2%}")

    if jacobian_metrics:
        print(f"\nAnchor + Jacobian (Teacher Oracle):")
        print(f"  Cosine Sim: {jacobian_metrics['cosine_similarity']:.4f}")
        print(f"  Relative L2: {jacobian_metrics['relative_l2']:.2%}")
        print(f"\nGap (room for improvement):")
        print(f"  Cosine Sim: {jacobian_metrics['cosine_similarity'] - bare_metrics['cosine_similarity']:.4f}")
        print(f"  Relative L2: {bare_metrics['relative_l2'] - jacobian_metrics['relative_l2']:.2%}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  - Phase 1: 300k anchor scaling")
    print("  - Phase 2: Jacobian action distillation (shared low-rank basis)")
    print("  - Phase 3: Direct functional addressing")
    print("=" * 70)


if __name__ == "__main__":
    main()
