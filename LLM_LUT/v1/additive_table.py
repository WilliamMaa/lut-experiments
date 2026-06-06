"""Additive LUT Table for LLM-LUT v1.2.

Supports:
  A: additive only (LUT1[a1] + LUT2[a2] + b)
  B: additive with ANOVA init
  C: additive + small coarse interaction table
"""

import torch
import torch.nn as nn


class AdditiveLUTTable(nn.Module):
    """
    Args:
        num_bins: number of bins per head (e.g. 64)
        group_size: dimension of target group
        init_lut1: [num_bins, group_size] or None
        init_lut2: [num_bins, group_size] or None
        init_bias: [group_size] or None
        interaction_table: [coarse_bins, coarse_bins, group_size] or None
        interaction_bins: coarse resolution (e.g. 8)
    """

    def __init__(
        self,
        num_bins: int,
        group_size: int,
        init_lut1: torch.Tensor = None,
        init_lut2: torch.Tensor = None,
        init_bias: torch.Tensor = None,
        interaction_table: torch.Tensor = None,
        interaction_bins: int = 8,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.group_size = group_size
        self.has_interaction = interaction_table is not None
        self.interaction_bins = interaction_bins

        self.lut1 = nn.Parameter(torch.zeros(num_bins, group_size))
        self.lut2 = nn.Parameter(torch.zeros(num_bins, group_size))
        self.bias = nn.Parameter(torch.zeros(group_size))

        if init_lut1 is not None:
            with torch.no_grad():
                self.lut1.copy_(init_lut1)
        if init_lut2 is not None:
            with torch.no_grad():
                self.lut2.copy_(init_lut2)
        if init_bias is not None:
            with torch.no_grad():
                self.bias.copy_(init_bias)

        if self.has_interaction:
            self.lut_c = nn.Parameter(torch.zeros(interaction_bins, interaction_bins, group_size))
            if interaction_table is not None:
                with torch.no_grad():
                    self.lut_c.copy_(interaction_table)

    def forward(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bin_indices: [N, 2] long, each in [0, num_bins-1]
        Returns:
            [N, group_size]
        """
        b1 = bin_indices[:, 0]
        b2 = bin_indices[:, 1]

        out = self.lut1[b1] + self.lut2[b2] + self.bias

        if self.has_interaction:
            stride = self.num_bins // self.interaction_bins
            c1 = (b1 // stride).clamp(0, self.interaction_bins - 1)
            c2 = (b2 // stride).clamp(0, self.interaction_bins - 1)
            out = out + self.lut_c[c1, c2]

        return out


def anova_decompose(joint_table: torch.Tensor) -> tuple:
    """
    Decompose joint 2D table into additive components.

    Args:
        joint_table: [B, B, group_size]
    Returns:
        lut1: [B, group_size]
        lut2: [B, group_size]
        bias: [group_size]
    """
    B = joint_table.shape[0]
    bias = joint_table.mean(dim=(0, 1))  # [group_size]

    lut1 = torch.zeros(B, joint_table.shape[2])
    lut2 = torch.zeros(B, joint_table.shape[2])

    for i in range(B):
        lut1[i] = joint_table[i, :].mean(dim=0) - bias
    for j in range(B):
        lut2[j] = joint_table[:, j].mean(dim=0) - bias

    return lut1, lut2, bias


def build_coarse_interaction(joint_table: torch.Tensor, coarse_bins: int = 8) -> torch.Tensor:
    """
    Build coarse interaction table from residual of ANOVA decomposition.

    Args:
        joint_table: [B, B, group_size] where B should be divisible by coarse_bins
    Returns:
        lut_c: [coarse_bins, coarse_bins, group_size]
    """
    B = joint_table.shape[0]
    gs = joint_table.shape[2]
    stride = B // coarse_bins
    assert B % coarse_bins == 0, f"num_bins {B} must be divisible by coarse_bins {coarse_bins}"

    lut1, lut2, bias = anova_decompose(joint_table)

    # Reconstruct additive approximation
    approx = bias.unsqueeze(0).unsqueeze(0) + lut1.unsqueeze(1) + lut2.unsqueeze(0)  # [B, B, gs]
    residual = joint_table - approx  # [B, B, gs]

    lut_c = torch.zeros(coarse_bins, coarse_bins, gs)
    for ci in range(coarse_bins):
        i0 = ci * stride
        i1 = (ci + 1) * stride
        for cj in range(coarse_bins):
            j0 = cj * stride
            j1 = (cj + 1) * stride
            lut_c[ci, cj] = residual[i0:i1, j0:j1].mean(dim=(0, 1))

    return lut_c
