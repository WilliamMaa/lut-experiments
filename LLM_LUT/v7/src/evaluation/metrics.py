"""
metrics.py

评估指标：输出质量、路由质量、存储、成本。
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_output_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """
    计算输出质量指标。

    Args:
        y_pred: [N, d] 预测输出
        y_true: [N, d] 真实输出

    Returns:
        metrics: 指标字典
    """
    mse = F.mse_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)
    var = y_true.var().item()
    rel_mse = mse / (var + 1e-8)
    rel_l2 = torch.norm(y_pred - y_true).item() / (torch.norm(y_true).item() + 1e-8)

    # Cosine similarity
    cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1)
    cos_mean = cos_sim.mean().item()
    cos_p10 = torch.quantile(cos_sim, 0.10).item()
    cos_p50 = torch.quantile(cos_sim, 0.50).item()
    cos_p90 = torch.quantile(cos_sim, 0.90).item()

    # Norm ratio
    pred_norm = torch.norm(y_pred, dim=-1)
    true_norm = torch.norm(y_true, dim=-1)
    norm_ratio = pred_norm / (true_norm + 1e-8)
    norm_mean = norm_ratio.mean().item()
    norm_p10 = torch.quantile(norm_ratio, 0.10).item()
    norm_p50 = torch.quantile(norm_ratio, 0.50).item()
    norm_p90 = torch.quantile(norm_ratio, 0.90).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_mean,
        "cosine_similarity_p10": cos_p10,
        "cosine_similarity_p50": cos_p50,
        "cosine_similarity_p90": cos_p90,
        "norm_ratio": norm_mean,
        "norm_ratio_p10": norm_p10,
        "norm_ratio_p50": norm_p50,
        "norm_ratio_p90": norm_p90,
    }


def compute_routing_metrics(
    predicted_indices: torch.Tensor,
    optimal_indices: torch.Tensor,
    losses: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    计算路由质量指标。

    Args:
        predicted_indices: [N] 预测地址索引
        optimal_indices: [N] 最优地址索引（如暴力 NN）
        losses: [N, M] 每个查询到每个 anchor 的损失（可选）

    Returns:
        metrics: 指标字典
    """
    N = predicted_indices.shape[0]

    # Exact-address rate
    exact_match = (predicted_indices == optimal_indices).float()
    exact_rate = exact_match.mean().item()

    metrics = {
        "exact_address_rate": exact_rate,
    }

    if losses is not None:
        # 计算 misrouting regret
        predicted_loss = losses[torch.arange(N), predicted_indices]
        optimal_loss = losses[torch.arange(N), optimal_indices]
        regret = predicted_loss - optimal_loss

        metrics["mean_regret"] = regret.mean().item()
        metrics["median_regret"] = regret.median().item()
        metrics["p95_regret"] = torch.quantile(regret, 0.95).item()
        metrics["p99_regret"] = torch.quantile(regret, 0.99).item()
        metrics["max_regret"] = regret.max().item()

        # Catastrophic routing rate (regret > threshold)
        threshold = optimal_loss.mean().item()
        catastrophic_rate = (regret > threshold).float().mean().item()
        metrics["catastrophic_routing_rate"] = catastrophic_rate

    return metrics


def compute_storage_metrics(
    n_anchors: int,
    d_in: int,
    d_out: int,
    per_anchor_code_dim: int = 0,
    n_banks: int = 1,
    rank: int = 0,
    dtype_bytes: int = 2,  # FP16
) -> Dict[str, float]:
    """
    计算存储开销。

    Args:
        n_anchors: anchor 数量
        d_in: 输入维度
        d_out: 输出维度
        per_anchor_code_dim: 每个 anchor 的 code 维度
        n_banks: bank 数量
        rank: 低秩修正的 rank
        dtype_bytes: 数据类型字节数

    Returns:
        metrics: 存储指标字典（字节数）
    """
    # Anchor 输入存储
    anchor_input_bytes = n_anchors * d_in * dtype_bytes

    # Anchor 输出存储
    anchor_output_bytes = n_anchors * d_out * dtype_bytes

    # Per-anchor code 存储
    code_bytes = n_anchors * per_anchor_code_dim * dtype_bytes

    # 共享低秩基存储 (U, V 每 bank 一套)
    shared_basis_bytes = n_banks * rank * (d_in + d_out) * dtype_bytes

    # 对比：完整 Jacobian 存储
    full_jacobian_bytes = n_anchors * d_out * d_in * dtype_bytes

    total_bytes = anchor_input_bytes + anchor_output_bytes + code_bytes + shared_basis_bytes

    return {
        "anchor_input_bytes": anchor_input_bytes,
        "anchor_output_bytes": anchor_output_bytes,
        "code_bytes": code_bytes,
        "shared_basis_bytes": shared_basis_bytes,
        "total_bytes": total_bytes,
        "total_mib": total_bytes / (1024 * 1024),
        "full_jacobian_bytes": full_jacobian_bytes,
        "full_jacobian_mib": full_jacobian_bytes / (1024 * 1024),
        "compression_ratio": full_jacobian_bytes / total_bytes if total_bytes > 0 else 0,
    }


def compute_cost_metrics(
    address_encoder_macs: int,
    n_lookups: int,
    record_bytes: int,
    correction_macs: int,
    original_ffn_macs: int,
) -> Dict[str, float]:
    """
    计算计算成本指标。

    Args:
        address_encoder_macs: 地址编码器 MAC 数
        n_lookups: 查表次数
        record_bytes: 每次查表读取字节数
        correction_macs: 修正计算 MAC 数
        original_ffn_macs: 原始 FFN MAC 数

    Returns:
        metrics: 成本指标字典
    """
    total_macs = address_encoder_macs + correction_macs
    total_bytes = n_lookups * record_bytes

    return {
        "address_encoder_macs": address_encoder_macs,
        "n_lookups": n_lookups,
        "record_bytes_per_lookup": record_bytes,
        "total_bytes_per_query": total_bytes,
        "correction_macs": correction_macs,
        "total_macs": total_macs,
        "original_ffn_macs": original_ffn_macs,
        "mac_reduction_ratio": 1.0 - (total_macs / original_ffn_macs) if original_ffn_macs > 0 else 0,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """打印指标。"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                print(f"  {key}: {value:.4e}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
