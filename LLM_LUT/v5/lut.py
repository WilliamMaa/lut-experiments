"""
Trainable ensemble LUT tables for v5.

A LUTGroup stores M tables, each with E entries of group_size-dimensional vectors.
Forward pass looks up one entry per table and sums them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LUTGroup(nn.Module):
    """
    Trainable ensemble LUT for one output group.

    Shape: [num_tables, num_entries, group_size]
    Forward: given indices [B, S, num_tables], returns [B, S, group_size]
    """

    def __init__(self, num_tables: int, num_entries: int, group_size: int,
                 init_table: torch.Tensor = None):
        super().__init__()
        self.num_tables = num_tables
        self.num_entries = num_entries
        self.group_size = group_size

        if init_table is not None:
            table = init_table.float().clone()
        else:
            table = torch.zeros(num_tables, num_entries, group_size)
        self.table = nn.Parameter(table)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [B, S, num_tables] long
        Returns:
            out: [B, S, group_size]
        """
        B, S, M = indices.shape
        assert M == self.num_tables
        flat_idx = indices.view(-1, M).to(self.table.device)  # [N, M]
        N = flat_idx.shape[0]

        # Gather per table: table[m, flat_idx[:, m], :]
        outs = []
        for m in range(M):
            t = self.table[m]  # [E, gs]
            idx_m = flat_idx[:, m].clamp(0, self.num_entries - 1)
            outs.append(t[idx_m])  # [N, gs]
        out = torch.stack(outs, dim=1).sum(dim=1)  # [N, gs]
        return out.view(B, S, self.group_size)

    def initialize_from_calibration(self, indices: torch.Tensor, targets: torch.Tensor):
        """
        Initialize table entries by averaging calibration targets per entry.

        Args:
            indices: [N, num_tables]
            targets: [N, group_size]
        """
        with torch.no_grad():
            M = self.num_tables
            E = self.num_entries
            gs = self.group_size
            device = self.table.device
            indices = indices.to(device)
            targets = targets.to(device)
            new_table = torch.zeros(M, E, gs, device=device, dtype=torch.float32)
            counts = torch.zeros(M, E, device=device, dtype=torch.float32)

            for m in range(M):
                idx_m = indices[:, m].clamp(0, E - 1)
                # scatter_add: accumulate targets into table entries
                idx_exp = idx_m.unsqueeze(1).expand(-1, gs)
                new_table[m].scatter_add_(0, idx_exp, targets.float())
                counts[m].scatter_add_(0, idx_m, torch.ones_like(idx_m, dtype=torch.float32))

            counts = counts.clamp_min(1.0).unsqueeze(-1)
            new_table = new_table / counts
            self.table.copy_(new_table)


def build_lut_group(num_tables: int, num_entries: int, group_size: int,
                    indices: torch.Tensor, targets: torch.Tensor) -> LUTGroup:
    """Helper to create and initialize a LUTGroup from calibration data."""
    lut = LUTGroup(num_tables, num_entries, group_size)
    lut.initialize_from_calibration(indices, targets)
    return lut
