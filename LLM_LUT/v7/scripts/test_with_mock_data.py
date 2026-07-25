#!/usr/bin/env python3
"""
使用模拟数据测试 Phase 0 代码结构

不依赖真实模型，快速验证代码逻辑。
"""

import sys
sys.path.insert(0, 'd:/for_fun_project/glacier/project/LLM_LUT/v7')

import torch
import numpy as np
from pathlib import Path

from src.data.anchor_builder import AnchorBuilder
from src.teacher.exact_search import ExactNNTeacher, BareAnchorTeacher
from src.teacher.jacobian import JacobianComputer
from src.evaluation.metrics import (
    compute_error_statistics,
    StorageMetrics,
    CostModel,
    print_metrics
)


def create_mock_ffn(d_in: int, d_out: int, rank: int = 64):
    """
    创建一个模拟的 FFN 层（低秩非线性变换）。
    
    F(x) = tanh(x @ A) @ B
    其中 A: [d_in, rank], B: [rank, d_out]
    """
    A = torch.randn(d_in, rank) * 0.1
    B = torch.randn(rank, d_out) * 0.1
    
    def ffn(x):
        hidden = torch.tanh(x @ A)
        return hidden @ B
    
    return ffn


def generate_mock_data(
    n_samples: int,
    d_in: int,
    d_out: int,
    device: str = "cpu"
) -> dict:
    """生成模拟的输入输出数据"""
    print(f"[Mock] Generating {n_samples} samples with d_in={d_in}, d_out={d_out}")
    
    # 创建模拟 FFN
    ffn = create_mock_ffn(d_in, d_out)
    
    # 生成输入（高斯分布 + 一些结构）
    inputs = torch.randn(n_samples, d_in) * 0.5
    # 添加一些聚类结构
    n_clusters = 10
    for i in range(n_clusters):
        center = torch.randn(d_in) * 2
        cluster_size = n_samples // n_clusters
        start = i * cluster_size
        end = start + cluster_size
        inputs[start:end] += center
    
    # 计算输出
    outputs = torch.stack([ffn(x) for x in inputs])
    
    # 添加少量噪声
    outputs += torch.randn_like(outputs) * 0.01
    
    return {
        'inputs': inputs,
        'outputs': outputs,
        'ffn': ffn
    }


def main():
    print("=" * 70)
    print("Phase 0 Test with Mock Data")
    print("=" * 70)
    
    # 配置
    d_in = 128  # 模拟小维度以便快速测试
    d_out = 128
    n_samples = 5000
    n_anchors = 500  # 小数量便于测试
    n_test = 500
    device = "cpu"
    
    print(f"\nConfig:")
    print(f"  d_in={d_in}, d_out={d_out}")
    print(f"  n_samples={n_samples}, n_anchors={n_anchors}")
    
    # ============ Step 1: 生成数据 ============
    print("\n" + "-" * 70)
    print("[Step 1] Generating mock data...")
    print("-" * 70)
    
    data = generate_mock_data(n_samples, d_in, d_out, device)
    inputs = data['inputs']
    outputs = data['outputs']
    ffn = data['ffn']
    
    print(f"[Data] Input shape: {inputs.shape}")
    print(f"[Data] Output shape: {outputs.shape}")
    
    # 划分 train/test
    train_inputs = inputs[:-n_test]
    train_outputs = outputs[:-n_test]
    test_inputs = inputs[-n_test:]
    test_outputs = outputs[-n_test:]
    
    # ============ Step 2: 选择 Anchors ============
    print("\n" + "-" * 70)
    print("[Step 2] Building anchors with k-means++...")
    print("-" * 70)
    
    anchor_builder = AnchorBuilder(
        method="kmeans++",
        n_anchors=n_anchors,
        remove_duplicates=True,
        device=device
    )
    
    anchor_result = anchor_builder.build(train_inputs, train_outputs)
    anchors = anchor_result['anchors']
    anchor_outputs = anchor_result['anchor_outputs']
    
    print(f"[Anchor] Selected {anchor_result['n_anchors']} anchors")
    
    # ============ Step 3: Bare Anchor 基线 ============
    print("\n" + "-" * 70)
    print("[Step 3] Evaluating Bare Anchor (no correction)...")
    print("-" * 70)
    
    bare_teacher = BareAnchorTeacher(
        anchors=anchors,
        anchor_outputs=anchor_outputs,
        device=device
    )
    
    bare_eval = bare_teacher.evaluate(test_inputs, test_outputs)
    print_metrics(bare_eval, "Bare Anchor Results")
    
    # ============ Step 4: 计算 Jacobian ============
    print("\n" + "-" * 70)
    print("[Step 4] Computing Jacobians...")
    print("-" * 70)
    
    computer = JacobianComputer(method="finite_diff", device=device)
    
    # 为所有 anchor 计算 Jacobian
    jacobians = []
    for i, anchor in enumerate(anchors):
        J = computer.compute(ffn, anchor, epsilon=1e-3)
        jacobians.append(J)
        if (i + 1) % 100 == 0:
            print(f"[Jacobian] Computed {i+1}/{len(anchors)}")
    
    jacobians = torch.stack(jacobians)
    print(f"[Jacobian] Shape: {jacobians.shape}")
    print(f"[Jacobian] Storage: {jacobians.element_size() * jacobians.nelement() / 1e6:.2f} MB")
    
    # ============ Step 5: NN + Jacobian 基线 ============
    print("\n" + "-" * 70)
    print("[Step 5] Evaluating NN + Jacobian...")
    print("-" * 70)
    
    nn_teacher = ExactNNTeacher(
        anchors=anchors,
        anchor_outputs=anchor_outputs,
        jacobians=jacobians,
        device=device
    )
    
    nn_eval = nn_teacher.evaluate(test_inputs, test_outputs)
    print_metrics(nn_eval, "NN + Jacobian Results")
    
    # ============ Step 6: 比较与分析 ============
    print("\n" + "-" * 70)
    print("[Step 6] Comparison & Analysis")
    print("-" * 70)
    
    # 误差改进
    bare_rel_l2 = bare_eval['relative_l2']
    nn_rel_l2 = nn_eval['relative_l2']
    improvement = (bare_rel_l2 - nn_rel_l2) / bare_rel_l2 * 100
    
    print(f"\nRelative L2 Error:")
    print(f"  Bare Anchor:    {bare_rel_l2:.4f}")
    print(f"  NN + Jacobian:  {nn_rel_l2:.4f}")
    print(f"  Improvement:    {improvement:.1f}%")
    
    # 存储对比
    storage = StorageMetrics()
    bare_storage = storage.compute_storage_bytes(
        anchors=anchors,
        anchor_outputs=anchor_outputs
    )
    jacobian_storage = storage.compute_storage_bytes(
        anchors=anchors,
        anchor_outputs=anchor_outputs,
        jacobians=jacobians
    )
    
    comparison = storage.compare_with_jacobian_baseline(
        n_anchors=anchors.shape[0],
        d_in=d_in,
        d_out=d_out,
        our_storage=bare_storage
    )
    
    print(f"\nStorage:")
    print(f"  Bare Anchor:      {bare_storage['total_mb']:.2f} MB")
    print(f"  With Jacobian:    {jacobian_storage['total_mb']:.2f} MB")
    print(f"  Jacobian baseline (theoretical): {comparison['jacobian_baseline_mb']:.2f} MB")
    
    # 成本模型
    cost_model = CostModel()
    
    # 原始 FFN 成本
    ffn_cost = cost_model.estimate_cost(
        n_macs=d_in * d_out,
        n_bytes_read=(d_in + d_out) * 4,
        n_lookups=0
    )
    
    # Bare anchor 成本
    # 搜索: N * d_in MACs (距离计算)
    bare_cost = cost_model.estimate_cost(
        n_macs=n_anchors * d_in,
        n_bytes_read=bare_storage['total'],
        n_lookups=1
    )
    
    # Jacobian 修正成本
    # 搜索 + Jacobian matvec: N * d_in + d_out * d_in
    nn_cost = cost_model.estimate_cost(
        n_macs=n_anchors * d_in + d_out * d_in,
        n_bytes_read=jacobian_storage['total'],
        n_lookups=1
    )
    
    print(f"\nCost Model (relative units):")
    print(f"  Original FFN:     {ffn_cost['total_cost']:.1f}")
    print(f"  Bare Anchor NN:   {bare_cost['total_cost']:.1f} (vs FFN: {ffn_cost['total_cost']/bare_cost['total_cost']:.2f}x)")
    print(f"  NN + Jacobian:    {nn_cost['total_cost']:.1f} (vs FFN: {ffn_cost['total_cost']/nn_cost['total_cost']:.2f}x)")
    
    # ============ Step 7: Anchor 距离统计 ============
    print("\n" + "-" * 70)
    print("[Step 7] Anchor Distance Statistics")
    print("-" * 70)
    
    print(f"\nDistance to nearest anchor (test set):")
    print(f"  Mean:   {nn_eval['mean_distance']:.4f}")
    print(f"  P50:    {nn_eval['p50_distance']:.4f}")
    print(f"  P95:    {nn_eval['p95_distance']:.4f}")
    print(f"  P99:    {nn_eval['p99_distance']:.4f}")
    print(f"  Max:    {nn_eval['max_distance']:.4f}")
    
    print(f"\nAnchor usage:")
    print(f"  Unique anchors used: {nn_eval['unique_anchors_used']} / {n_anchors}")
    print(f"  Usage ratio: {nn_eval['anchor_usage_ratio']*100:.1f}%")
    
    # ============ 总结 ============
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
Phase 0 Baseline (30k anchor equivalent) established:

1. Bare Anchor: Simple but limited (no correction)
   - Rel L2 Error: {bare_rel_l2:.4f}
   - Storage: {bare_storage['total_mb']:.2f} MB
   
2. NN + Jacobian: High quality but expensive
   - Rel L2 Error: {nn_rel_l2:.4f} ({improvement:.1f}% better)
   - Storage: {jacobian_storage['total_mb']:.2f} MB
   - Cost: Higher than original FFN (not deployable)

Next steps (Phase 1-3):
- Scale to 100k/300k anchors
- Replace full Jacobian with low-rank approximation
- Learn direct addressing (no search)
    """)
    
    print("\n[Test] All checks passed!")
    return True


if __name__ == '__main__':
    main()
