#!/usr/bin/env python3
"""
inspect_onpolicy_data.py

检查 collect_onpolicy_data.py 产出的 train_samples.pt 质量：
  - 样本数量
  - LUT 输出 vs teacher 输出的 cosine 分布
  - relative L2 / relative MSE 分布
  - 低 cosine 样本比例
  - 输出范数比

用法：
  python inspect_onpolicy_data.py \
    --train_samples ./onpolicy_layer39_v4_tail_test/train_samples.pt \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints \
    [--device cuda:0] [--output_json ./quality_report.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_replacement_engine import V6ReplacementEngine


def compute_metrics(y_teacher: torch.Tensor, y_lut: torch.Tensor) -> dict:
    """计算 teacher 与 LUT 输出之间的各项距离/相似度指标。"""
    cos = F.cosine_similarity(y_teacher, y_lut, dim=-1)  # [N]

    norm_teacher = torch.norm(y_teacher, dim=-1)
    norm_lut = torch.norm(y_lut, dim=-1)
    rel_l2 = torch.norm(y_teacher - y_lut, dim=-1) / (norm_teacher + 1e-12)
    rel_mse = ((y_teacher - y_lut) ** 2).sum(dim=-1) / ((y_teacher ** 2).sum(dim=-1) + 1e-12)

    return {
        "cos": cos,
        "rel_l2": rel_l2,
        "rel_mse": rel_mse,
        "norm_ratio": norm_lut / (norm_teacher + 1e-12),
    }


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def main():
    parser = argparse.ArgumentParser(description="Inspect on-policy training data quality")
    parser.add_argument("--train_samples", required=True, help="Path to train_samples.pt")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to LUT checkpoint dir")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for running LUT forward")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载训练样本
    data = torch.load(args.train_samples, map_location="cpu", weights_only=False)
    x = data["x"].float()
    y_teacher = data["y_teacher"].float()
    print(f"Loaded train_samples: x={tuple(x.shape)}, y_teacher={tuple(y_teacher.shape)}")

    # 加载 LUT engine（不需要 hook 到模型，只用来 forward）
    # 构造一个 dummy model，engine 只用于查表
    class DummyMLP:
        pass
    class DummyLayer:
        def __init__(self):
            self.mlp = DummyMLP()
    class DummyModel:
        def __init__(self):
            self.model = type("Inner", (), {"layers": [DummyLayer()]})()
    dummy_model = DummyModel()
    engine = V6ReplacementEngine(
        model=dummy_model,
        layer_idx=0,
        checkpoint_dir=args.checkpoint_dir,
        hook_path=None,
        device=device,
    )

    # 批量跑 LUT forward
    y_luts = []
    for start in range(0, x.shape[0], args.batch_size):
        xb = x[start:start + args.batch_size].to(device)
        with torch.no_grad():
            yb = engine.lut_forward(xb)
        y_luts.append(yb.cpu())
    y_lut = torch.cat(y_luts, dim=0).float()

    metrics = compute_metrics(y_teacher, y_lut)

    # 转成 numpy 做统计
    def summarize(arr: torch.Tensor):
        a = arr.detach().cpu().numpy()
        a = a[np.isfinite(a)]
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p1": percentile(a, 1),
            "p5": percentile(a, 5),
            "p10": percentile(a, 10),
            "p25": percentile(a, 25),
            "p50": percentile(a, 50),
            "p75": percentile(a, 75),
            "p90": percentile(a, 90),
        }

    report = {
        "n_samples": x.shape[0],
        "hidden_size": x.shape[-1],
        "cosine": summarize(metrics["cos"]),
        "relative_l2": summarize(metrics["rel_l2"]),
        "relative_mse": summarize(metrics["rel_mse"]),
        "norm_ratio": summarize(metrics["norm_ratio"]),
        "bad_ratios": {
            "cos_lt_0.5": float((metrics["cos"] < 0.5).float().mean()),
            "cos_lt_0.6": float((metrics["cos"] < 0.6).float().mean()),
            "cos_lt_0.7": float((metrics["cos"] < 0.7).float().mean()),
            "rel_l2_gt_0.5": float((metrics["rel_l2"] > 0.5).float().mean()),
            "rel_l2_gt_1.0": float((metrics["rel_l2"] > 1.0).float().mean()),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
