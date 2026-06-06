"""Learnable Codebook Table for LLM-LUT v1.1.

Soft-to-hard assignment with learned centroids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kmeans_torch(data: torch.Tensor, k: int, max_iter: int = 30, seed: int = 42):
    """
    Simple k-means initialization in PyTorch.

    Args:
        data: [N, D]
        k: number of clusters
    Returns:
        centroids: [k, D]
        labels: [N]
    """
    data = data.float()
    N, D = data.shape
    if k > N:
        k = N
    torch.manual_seed(seed)
    indices = torch.randperm(N)[:k]
    centroids = data[indices].clone()

    for _ in range(max_iter):
        dist = torch.cdist(data, centroids, p=2)  # [N, k]
        labels = dist.argmin(dim=-1)  # [N]
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(dim=0)
            else:
                # Re-initialize empty centroid
                new_centroids[i] = data[torch.randint(0, N, (1,))]
        centroids = new_centroids

    return centroids, labels


class LearnableCodebookTable(nn.Module):
    """
    Learnable codebook with soft-to-hard assignment.

    Args:
        num_centroids: K, number of learned centers
        group_size: dimension of target group
        address_dim: dimension of address vector (typically 2 for 2-head)
        init_centroids: [K, address_dim] or None
        init_table: [K, group_size] or None
        init_temperature: initial softmax temperature
        min_temperature: floor for temperature
    """

    def __init__(
        self,
        num_centroids: int,
        group_size: int,
        address_dim: int = 2,
        init_centroids: torch.Tensor = None,
        init_table: torch.Tensor = None,
        init_temperature: float = 1.0,
        min_temperature: float = 0.1,
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.group_size = group_size
        self.address_dim = address_dim
        self.min_temperature = min_temperature

        if init_centroids is not None:
            self.centroids = nn.Parameter(init_centroids.clone())
        else:
            self.centroids = nn.Parameter(torch.randn(num_centroids, address_dim) * 0.1)

        if init_table is not None:
            self.table = nn.Parameter(init_table.clone())
        else:
            self.table = nn.Parameter(torch.zeros(num_centroids, group_size))

        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

    def forward(self, addresses: torch.Tensor, hard: bool = False):
        """
        Args:
            addresses: [N, address_dim] float
            hard: if True, use argmin assignment (inference)
        Returns:
            If hard: [N, group_size]
            If not hard: ([N, group_size], [N, K]) where second is assignment weights
        """
        # Ensure float32 for stable distance computation
        addresses = addresses.float()
        centroids = self.centroids.float()
        table = self.table.float()

        # Squared Euclidean distance: [N, K]
        dist = torch.cdist(addresses, centroids, p=2)  # [N, K]
        dist_sq = dist ** 2

        if hard or not self.training:
            indices = dist_sq.argmin(dim=-1)  # [N]
            return table[indices]  # [N, group_size]
        else:
            temp = self.temperature.clamp_min(self.min_temperature)
            weights = F.softmax(-dist_sq / temp, dim=-1)  # [N, K]
            output = torch.matmul(weights, table)  # [N, group_size]
            return output, weights

    def get_temperature(self) -> float:
        return self.temperature.clamp_min(self.min_temperature).item()

    def get_usage_stats(self, addresses: torch.Tensor) -> dict:
        """Compute centroid usage statistics for a batch of addresses."""
        with torch.no_grad():
            addresses = addresses.float()
            centroids = self.centroids.float()
            dist = torch.cdist(addresses, centroids, p=2)
            dist_sq = dist ** 2
            temp = self.temperature.clamp_min(self.min_temperature)
            weights = F.softmax(-dist_sq / temp, dim=-1)  # [N, K]
            usage = weights.mean(dim=0)  # [K]
            hard_labels = dist_sq.argmin(dim=-1)  # [N]
            hard_counts = torch.bincount(hard_labels, minlength=self.num_centroids).float()
            hard_usage = hard_counts / hard_counts.sum()
            return {
                "soft_entropy": -(usage * torch.log(usage + 1e-8)).sum().item(),
                "hard_entropy": -(hard_usage * torch.log(hard_usage + 1e-8)).sum().item(),
                "hard_coverage": (hard_counts > 0).sum().item() / self.num_centroids,
            }
