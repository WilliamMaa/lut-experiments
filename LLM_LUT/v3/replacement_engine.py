"""Standalone functional replacement engine for v3.

This is a minimal copy of v2/r1_replacement.py::ReplacementEngine so that
v3 does not depend on v2 source files.
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

import torch
import torch.nn as nn
from config import get_hook_target


class ReplacementEngine:
    """
    Functional replacement of a single mlp_delta group via 2D bucket lookup.

    Args:
        model: the LLM (already on target device)
        layer_id: target layer
        group_id: target group within layer output
        group_size: dimension of each group
        addr_idx: [2] — address channel indices
        addr_mean: [2] — per-channel mean from calibration
        addr_std: [2] — per-channel std from calibration
        table: [num_bins, num_bins, group_size] — 2D bucket table
        num_bins: number of bins per head
        addr_clip: address clipping value
    """

    def __init__(
        self,
        model,
        layer_id: int,
        group_id: int,
        group_size: int,
        addr_idx: torch.Tensor,
        addr_mean: torch.Tensor,
        addr_std: torch.Tensor,
        table: torch.Tensor,
        num_bins: int = 64,
        addr_clip: float = 3.0,
    ):
        self.model = model
        self.layer_id = layer_id
        self.group_id = group_id
        self.group_size = group_size
        self.addr_idx = addr_idx
        self.addr_mean = addr_mean
        self.addr_std = addr_std
        self.table = table
        self.num_bins = num_bins
        self.addr_clip = addr_clip
        self._hook_handle = None

    def _compute_bin_indices(self, addr_source: torch.Tensor) -> torch.Tensor:
        """Compute per-token 2D bin indices [B, seq, 2]."""
        addr_flat = self.addr_idx.to(addr_source.device).view(-1)
        addr_acts = addr_source.index_select(-1, addr_flat)

        mean = self.addr_mean.to(addr_source.device, addr_source.dtype).view(1, 1, -1)
        std = self.addr_std.to(addr_source.device, addr_source.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr_acts - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        bin_idx = torch.round(qf).long().clamp(0, self.num_bins - 1)
        return bin_idx

    def _hook(self, module, input, output):
        """Forward hook: replace group mlp_delta with 2D bucket lookup."""
        addr_source = input[0] if isinstance(input, tuple) else input
        out_tensor = output[0] if isinstance(output, tuple) else output

        bin_idx = self._compute_bin_indices(addr_source)  # [B, seq, 2]
        B, seq, _ = bin_idx.shape

        # Lookup 2D table
        b1 = bin_idx[:, :, 0].view(-1)
        b2 = bin_idx[:, :, 1].view(-1)
        table = self.table.to(addr_source.device, addr_source.dtype)
        repl = table[b1, b2]  # [B*seq, group_size]
        repl = repl.view(B, seq, self.group_size)

        # Replace in mlp_delta
        x = addr_source
        delta = out_tensor - x
        num_groups = delta.shape[-1] // self.group_size
        delta_g = delta.view(B, seq, num_groups, self.group_size)
        delta_g[:, :, self.group_id, :] = repl
        modified = x + delta_g.view(B, seq, -1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def install(self):
        """Install replacement hook."""
        if self._hook_handle is not None:
            return
        target_mod = get_hook_target(self.model, self.layer_id, "mlp_delta")
        self._hook_handle = target_mod.register_forward_hook(self._hook)

    def uninstall(self):
        """Remove replacement hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
