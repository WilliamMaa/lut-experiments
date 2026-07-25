"""
anchor_builder.py

Anchor 选择策略：从大规模激活值中选择代表性 anchor。
Phase 0-1: 30k -> 100k -> 300k anchor scaling
"""

from typing import Tuple, Optional
import math

import torch
from tqdm import tqdm


def reservoir_sampling(
    x: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    水库采样：从流式数据中均匀采样。

    Args:
        x: [N, d] 输入数据
        n_samples: 采样数量

    Returns:
        samples: [n_samples, d] 采样结果
    """
    N = x.shape[0]
    if N <= n_samples:
        return x

    # 初始化 reservoir
    reservoir = x[:n_samples].clone()

    # 替换概率
    for i in range(n_samples, N):
        j = torch.randint(0, i + 1, (1,)).item()
        if j < n_samples:
            reservoir[j] = x[i]

    return reservoir


def kmeans_plus_plus(
    x: torch.Tensor,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    batch_size: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mini-batch k-means++ 初始化 + 迭代。

    Args:
        x: [N, d] 输入数据
        n_clusters: 聚类中心数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        batch_size: mini-batch 大小

    Returns:
        centroids: [n_clusters, d] 聚类中心
        labels: [N] 每个点的类别
    """
    N, d = x.shape
    device = x.device

    # k-means++ 初始化
    centroids = torch.zeros(n_clusters, d, device=device)
    centroids[0] = x[torch.randint(0, N, (1,))]

    for i in range(1, n_clusters):
        # 计算每个点到最近中心的距离
        dists = torch.cdist(x, centroids[:i]).min(dim=1)[0]
        probs = dists / dists.sum()
        next_idx = torch.multinomial(probs, 1).item()
        centroids[i] = x[next_idx]

    # Mini-batch k-means 迭代
    labels = torch.zeros(N, dtype=torch.long, device=device)

    for iteration in range(max_iter):
        # 随机采样 mini-batch
        if N > batch_size:
            batch_idx = torch.randperm(N)[:batch_size]
            x_batch = x[batch_idx]
        else:
            batch_idx = torch.arange(N, device=device)
            x_batch = x

        # 分配标签
        dists = torch.cdist(x_batch, centroids)
        new_labels = dists.argmin(dim=1)

        # 更新中心
        old_centroids = centroids.clone()
        for k in range(n_clusters):
            mask = new_labels == k
            if mask.any():
                centroids[k] = x_batch[mask].mean(dim=0)

        # 检查收敛
        change = torch.norm(centroids - old_centroids)
        if change < tol:
            print(f"  k-means converged at iteration {iteration}")
            break

    # 最终分配
    dists = torch.cdist(x, centroids)
    labels = dists.argmin(dim=1)

    return centroids, labels


def remove_near_duplicates(
    x: torch.Tensor,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """
    移除近似重复点。

    Args:
        x: [N, d] 输入数据
        threshold: 距离阈值

    Returns:
        filtered: [M, d] 过滤后的数据
    """
    if x.shape[0] <= 1:
        return x

    keep = []
    for i in range(x.shape[0]):
        xi = x[i:i+1]
        if not keep:
            keep.append(i)
        else:
            kept = x[keep]
            dists = torch.cdist(xi, kept).min()
            if dists > threshold:
                keep.append(i)

    return x[keep]


class AnchorBuilder:
    """
    Anchor 构建器：支持多种采样策略。
    """

    def __init__(
        self,
        n_anchors: int,
        method: str = "kmeans++",
        remove_duplicates: bool = True,
        duplicate_threshold: float = 1e-6,
    ):
        """
        Args:
            n_anchors: 目标 anchor 数量
            method: 采样方法 ("random", "reservoir", "kmeans++")
            remove_duplicates: 是否移除重复点
            duplicate_threshold: 重复判定阈值
        """
        self.n_anchors = n_anchors
        self.method = method
        self.remove_duplicates = remove_duplicates
        self.duplicate_threshold = duplicate_threshold

    def build(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        构建 anchors。

        Args:
            x: [N, d_in] 输入数据
            y: [N, d_out] 输出数据（可选）

        Returns:
            anchors_x: [n_anchors, d_in] anchor 输入
            anchors_y: [n_anchors, d_out] anchor 输出（如果提供了 y）
        """
        N = x.shape[0]

        if self.method == "random":
            # 随机采样
            if N <= self.n_anchors:
                idx = torch.arange(N)
            else:
                idx = torch.randperm(N)[:self.n_anchors]
            anchors_x = x[idx]
            anchors_y = y[idx] if y is not None else None

        elif self.method == "reservoir":
            # 水库采样
            anchors_x = reservoir_sampling(x, self.n_anchors)
            if y is not None:
                # 同时采样对应的 y
                y_reservoir = reservoir_sampling(
                    torch.cat([x, y], dim=-1),
                    self.n_anchors
                )
                anchors_x = y_reservoir[:, :x.shape[-1]]
                anchors_y = y_reservoir[:, x.shape[-1]:]
            else:
                anchors_y = None

        elif self.method == "kmeans++":
            # k-means++
            print(f"Running k-means++ for {self.n_anchors} clusters...")
            anchors_x, labels = kmeans_plus_plus(x, self.n_anchors)

            if y is not None:
                # 计算每个 cluster 的 y 均值
                anchors_y = torch.zeros(self.n_anchors, y.shape[-1], device=y.device)
                for k in range(self.n_anchors):
                    mask = labels == k
                    if mask.any():
                        anchors_y[k] = y[mask].mean(dim=0)
            else:
                anchors_y = None

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 移除重复点
        if self.remove_duplicates:
            original_count = anchors_x.shape[0]
            anchors_x = remove_near_duplicates(anchors_x, self.duplicate_threshold)
            if anchors_y is not None:
                anchors_y = anchors_y[:anchors_x.shape[0]]
            final_count = anchors_x.shape[0]
            if final_count < original_count:
                print(f"  Removed {original_count - final_count} near-duplicate anchors")

        return anchors_x, anchors_y


def compute_anchor_coverage_stats(
    x: torch.Tensor,
    anchors: torch.Tensor,
) -> dict:
    """
    计算 anchor 覆盖统计信息。

    Args:
        x: [N, d] 查询点
        anchors: [M, d] anchor 点

    Returns:
        stats: 统计信息字典
    """
    # 计算每个查询点到最近 anchor 的距离
    dists = torch.cdist(x, anchors)
    min_dists = dists.min(dim=1)[0]

    return {
        "mean_distance": min_dists.mean().item(),
        "median_distance": min_dists.median().item(),
        "p90_distance": torch.quantile(min_dists, 0.90).item(),
        "p95_distance": torch.quantile(min_dists, 0.95).item(),
        "p99_distance": torch.quantile(min_dists, 0.99).item(),
        "max_distance": min_dists.max().item(),
    }
