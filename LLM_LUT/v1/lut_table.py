"""Trainable LUT Table for LLM-LUT v1.

Supports 1-head and multi-head addressing with nn.Parameter table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableLUTTable(nn.Module):
    """
    Args:
        num_bins: number of bins per head
        group_size: dimension of the target group
        num_heads: number of address heads (1 or 2)
        init_table: optional initial table (from bucket average)
    """

    def __init__(
        self,
        num_bins: int,
        group_size: int,
        num_heads: int = 2,
        init_table: torch.Tensor = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.group_size = group_size
        self.num_heads = num_heads

        if num_heads == 1:
            shape = (num_bins, group_size)
        elif num_heads == 2:
            shape = (num_bins, num_bins, group_size)
        else:
            raise ValueError(f"num_heads must be 1 or 2, got {num_heads}")

        self.table = nn.Parameter(torch.zeros(*shape))

        if init_table is not None:
            with torch.no_grad():
                if init_table.shape == self.table.shape:
                    self.table.copy_(init_table)
                else:
                    # shape mismatch: init from zeros but warn
                    print(f"[WARN] init_table shape {init_table.shape} != expected {self.table.shape}, using zeros")

    def forward(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bin_indices: [N, num_heads] long tensor with values in [0, num_bins-1]
        Returns:
            replacements: [N, group_size]
        """
        if self.num_heads == 1:
            return self.table[bin_indices[:, 0]]
        else:
            return self.table[bin_indices[:, 0], bin_indices[:, 1]]

    def get_flat_table(self) -> torch.Tensor:
        """Return flattened view for regularization / inspection."""
        if self.num_heads == 1:
            return self.table
        else:
            return self.table.view(-1, self.group_size)
