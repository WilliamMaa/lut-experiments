#!/usr/bin/env python3
"""Clustered VQK Linear for v8 Stage 4.

Clusters weight blocks by distribution statistics, then assigns each cluster
its own clipping threshold / scale.  Activation quantization is identical to
IntegerVQKLinear.

Plan C simplest version:
    weight bits = 4
    K clusters  = 2 / 4 / 8
    per-cluster clipping threshold learned by 1-D grid search on the blocks
    belonging to that cluster.
"""

from typing import Tuple

import torch
import torch.nn as nn

from vqk.integer_vqk_linear import IntegerVQKLinear
from vqk.activation_quantizer import quantize_activation


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def _block_features(weight_blocks: torch.Tensor) -> torch.Tensor:
    """Extract per-block distribution features for clustering.

    Args:
        weight_blocks: (out_features, num_blocks, block_size)

    Returns:
        features: (num_blocks, n_features)
    """
    *_, num_blocks, block_size = weight_blocks.shape
    w = weight_blocks.transpose(0, 1).reshape(num_blocks, -1).float()

    mean_abs = w.abs().mean(dim=1)
    std = w.std(dim=1, unbiased=False)
    max_abs = w.abs().max(dim=1).values
    # outlier ratio relative to 99.9 percentile within the block
    p999 = torch.quantile(w.abs(), 0.999, dim=1)
    outlier_ratio = (w.abs() > p999.unsqueeze(1)).float().mean(dim=1)

    features = torch.stack([mean_abs, std, max_abs, outlier_ratio], dim=1)
    features = torch.log1p(features)
    # Standardize for k-means.
    mean = features.mean(dim=0, keepdim=True)
    stdv = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (features - mean) / stdv


def _kmeans(features: torch.Tensor, k: int, max_iters: int = 50) -> torch.Tensor:
    """Lloyd's k-means on (num_blocks, n_features).

    Returns cluster assignment indices of shape (num_blocks,).
    """
    num_blocks, feat_dim = features.shape
    device = features.device
    if k >= num_blocks:
        return torch.arange(num_blocks, device=device)

    # k-means++ initialization
    centers = [features[torch.randint(num_blocks, (1,), device=device)].squeeze(0)]
    for _ in range(1, k):
        dists = torch.stack([((features - c) ** 2).sum(dim=1) for c in centers], dim=1)
        min_dists, _ = dists.min(dim=1)
        probs = min_dists / min_dists.sum()
        next_idx = torch.multinomial(probs, 1)
        centers.append(features[next_idx].squeeze(0))
    centers = torch.stack(centers, dim=0)

    assignments = torch.zeros(num_blocks, dtype=torch.long, device=device)
    for _ in range(max_iters):
        # Assign
        dists = torch.cdist(features, centers)
        new_assignments = dists.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        # Recompute centers
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centers[c] = features[mask].mean(dim=0)

    return assignments


def _optimal_threshold(w: torch.Tensor, bits: int, n_candidates: int = 21) -> float:
    """Find clipping threshold T minimizing MSE for symmetric INT quantization.

    Args:
        w: 1-D tensor of weight values from one cluster.
        bits: target bit width.

    Returns:
        Best clipping threshold T.
    """
    qmax = _qmax(bits)
    w = w.float()
    max_abs = w.abs().max()
    if max_abs < 1e-8:
        return 1e-8

    # Search thresholds from 0.5*max_abs up to max_abs.
    # Also include a percentile-based candidate for heavy tails.
    percentile_t = torch.quantile(w.abs(), 0.999)
    ts = torch.linspace(0.5, 1.0, n_candidates, device=w.device) * max_abs
    ts = torch.unique(torch.cat([ts, percentile_t.unsqueeze(0)]).clamp_min(1e-8))

    best_mse = float("inf")
    best_t = max_abs.item()
    for t in ts:
        scale = t / qmax
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        w_hat = q * scale
        mse = (w - w_hat).pow(2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_t = t.item()
    return best_t


def quantize_symmetric_clustered(
    weight: torch.Tensor,
    bits: int,
    block_size: int,
    num_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster weight blocks and quantize with per-cluster clipping threshold.

    Args:
        weight: (out_features, in_features)
        bits: weight bit width.
        block_size: block size along input dimension.
        num_clusters: number of clusters K.

    Returns:
        (weight_q, cluster_scales, cluster_assignments)
        weight_q: (out_features, in_features) INT8 storage of quantized values.
        cluster_scales: (num_clusters,) float scales = T_c / qmax.
        cluster_assignments: (num_blocks,) long, block -> cluster mapping.
    """
    out_features, in_features = weight.shape
    assert in_features % block_size == 0
    num_blocks = in_features // block_size
    qmax = _qmax(bits)

    weight_blocks = weight.view(out_features, num_blocks, block_size)
    features = _block_features(weight_blocks)
    cluster_assignments = _kmeans(features, num_clusters)

    cluster_scales = torch.zeros(num_clusters, device=weight.device, dtype=weight.dtype)
    w_q = torch.zeros_like(weight, dtype=torch.int8)

    for c in range(num_clusters):
        block_mask = cluster_assignments == c
        if not block_mask.any():
            cluster_scales[c] = 1e-8
            continue
        # Gather all weights in this cluster.
        cluster_weight_blocks = weight_blocks[:, block_mask, :]
        w_cluster = cluster_weight_blocks.reshape(-1)
        T_c = _optimal_threshold(w_cluster, bits)
        scale_c = T_c / qmax
        cluster_scales[c] = scale_c

        # Quantize all blocks in this cluster with the shared scale.
        q_cluster = torch.clamp(torch.round(cluster_weight_blocks / scale_c), -qmax, qmax).to(torch.int8)
        w_q_blocks = w_q.view(out_features, num_blocks, block_size)
        w_q_blocks[:, block_mask, :] = q_cluster

    return w_q, cluster_scales, cluster_assignments


class ClusteredVQKLinear(IntegerVQKLinear):
    """Integer VQK Linear with per-cluster clipping thresholds / scales.

    Keeps the same forward path as IntegerVQKLinear; only the weight
    representation changes:
        W_B ≈ S_{c(B)} Q_B
    where c(B) is the cluster assignment of input block B.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        weight_bits: int = 4,
        activation_bits: int = 4,
        block_size: int = 128,
        activation_mode: str = "per-token-per-block",
        num_clusters: int = 4,
    ):
        # Bypass parent __init__ and set buffers ourselves.
        nn.Module.__init__(self)
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.block_size = block_size
        self.activation_mode = activation_mode
        self.num_clusters = num_clusters
        self.num_blocks = self.in_features // self.block_size

        weight = base_linear.weight.detach()
        w_q, s_c, assignments = quantize_symmetric_clustered(
            weight, weight_bits, block_size, num_clusters
        )
        self.register_buffer("weight_q", w_q)
        self.register_buffer("cluster_scales", s_c.to(base_linear.weight.dtype))
        self.register_buffer("cluster_assignments", assignments)

        # Build a per-block scale lookup table from cluster scales.
        per_block_scales = s_c[assignments]
        self.register_buffer("weight_scales", per_block_scales)

        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach())
        else:
            self.bias = None

    def get_weight_storage_stats(self, scale_bits: int = 16) -> dict:
        """Return storage statistics including cluster assignment overhead."""
        weight_bytes = self.out_features * self.in_features * self.weight_bits / 8
        scale_bytes = self.num_clusters * scale_bits / 8
        # ceil(log2(K)) bits per block assignment
        import math
        assignment_bits = self.num_blocks * math.ceil(math.log2(max(2, self.num_clusters)))
        assignment_bytes = assignment_bits / 8
        total_bytes = weight_bytes + scale_bytes + assignment_bytes
        effective_bits = total_bytes * 8 / (self.out_features * self.in_features)
        return {
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "assignment_bytes": assignment_bytes,
            "total_bytes": total_bytes,
            "effective_bits_per_weight": effective_bits,
            "scale_count": self.num_clusters,
            "num_clusters": self.num_clusters,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
            f"block_size={self.block_size}, activation_mode={self.activation_mode}, "
            f"num_clusters={self.num_clusters}"
        )
