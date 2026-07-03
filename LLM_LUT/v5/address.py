"""
Address generators for v5 hybrid LUT.

All address generation is fixed/random (no trainable neural networks), so it stays
within the O(1) LUT-lookup regime.  Table values themselves are trainable.
"""

from typing import Tuple, Optional
import torch


class Address2D:
    """Original v3/v4 style: 2 selected channels -> 2D bins -> flattened index."""

    def __init__(self, addr_idx: torch.Tensor, addr_mean: torch.Tensor,
                 addr_std: torch.Tensor, num_bins: int = 64, addr_clip: float = 3.0):
        """
        Args:
            addr_idx: [2] channel indices in the residual/input tensor
            addr_mean: [2] calibration mean
            addr_std: [2] calibration std
        """
        self.addr_idx = addr_idx.cpu().long()
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu()
        self.num_bins = num_bins
        self.addr_clip = addr_clip
        self.num_entries = num_bins * num_bins
        self.num_tables = 1

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
        Returns:
            indices: [B, S, 1] flattened 2D bin index
        """
        B, S, _ = x.shape
        device = x.device
        idx = self.addr_idx.to(device)
        addr = x.index_select(-1, idx)  # [B, S, 2]

        mean = self.addr_mean.to(device, x.dtype).view(1, 1, -1)
        std = self.addr_std.to(device, x.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        b = torch.round(qf).long().clamp(0, self.num_bins - 1)
        flat = b[:, :, 0] * self.num_bins + b[:, :, 1]
        return flat.unsqueeze(-1)  # [B, S, 1]


class AddressHighOrderRandom:
    """
    Fixed random high-order address.

    For each of M tables and B bits, randomly select K input channels and random
    signs, project x, standardize, and threshold to get a binary bit.
    The B bits form an integer index in [0, 2^B).

    This is inspired by the high-order comparisons in new_lut.py, but kept as
    fixed random LSH with no trainable index generator.
    """

    def __init__(self, input_dim: int, num_tables: int, num_bits: int,
                 channels_per_bit: int = 4, seed: int = 0,
                 addr_mean: Optional[torch.Tensor] = None,
                 addr_std: Optional[torch.Tensor] = None):
        """
        Args:
            input_dim: dimension of x (e.g. hidden_size)
            num_tables: number of independent address tables (ensemble)
            num_bits: bits per address -> 2^num_bits entries per table
            channels_per_bit: how many input channels are combined per bit
            seed: random seed for reproducibility
            addr_mean: [num_tables, num_bits] calibration mean of projection
            addr_std: [num_tables, num_bits] calibration std of projection
        """
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.channels_per_bit = channels_per_bit
        self.num_entries = 2 ** num_bits

        gen = torch.Generator().manual_seed(seed)
        # channel_idx: [num_tables, num_bits, channels_per_bit]
        self.channel_idx = torch.randint(
            0, input_dim, (num_tables, num_bits, channels_per_bit), generator=gen
        ).long()
        # signs: [num_tables, num_bits, channels_per_bit] random +/- 1
        self.signs = torch.randint(0, 2, (num_tables, num_bits, channels_per_bit), generator=gen).float() * 2 - 1

        if addr_mean is None:
            addr_mean = torch.zeros(num_tables, num_bits)
        if addr_std is None:
            addr_std = torch.ones(num_tables, num_bits)
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu().clamp_min(1e-6)

        # powers of two for bit -> integer conversion
        self.register_buffer("powers", (2 ** torch.arange(num_bits)).long())

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, input_dim]
        Returns:
            indices: [B, S, num_tables] integer index per table
        """
        B, S, _ = x.shape
        device = x.device
        N = B * S
        x_flat = x.view(N, self.input_dim)

        # gather selected channels: [N, num_tables, num_bits, channels_per_bit]
        idx = self.channel_idx.to(device)
        selected = x_flat[:, idx]  # advanced indexing -> [N, num_tables, num_bits, channels_per_bit]
        signs = self.signs.to(device, x.dtype)
        proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]

        mean = self.addr_mean.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        std = self.addr_std.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        z = (proj - mean) / std
        bits = (z > 0).long()  # [N, num_tables, num_bits]

        powers = self.powers.to(device).view(1, 1, self.num_bits)
        indices = (bits * powers).sum(dim=-1)  # [N, num_tables]
        return indices.view(B, S, self.num_tables)

    def fit_calibration(self, x: torch.Tensor):
        """Re-compute addr_mean/std from calibration data."""
        with torch.no_grad():
            B, S, _ = x.shape
            N = B * S
            x_flat = x.view(N, self.input_dim)
            idx = self.channel_idx.to(x_flat.device)
            selected = x_flat[:, idx]
            signs = self.signs.to(x_flat.device, x_flat.dtype)
            proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]
            self.addr_mean = proj.mean(dim=0).cpu()
            self.addr_std = proj.std(dim=0).cpu().clamp_min(1e-6)
