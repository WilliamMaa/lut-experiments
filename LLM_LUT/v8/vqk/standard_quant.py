#!/usr/bin/env python3
"""RTN (Round-To-Nearest) INT quantization baseline for VQK comparison.

Implements per-channel symmetric round-to-nearest quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_per_channel_symmetric(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix per output-channel.

    Args:
        weight: (out_features, in_features)
        bits: target bit width

    Returns:
        (weight_q, scales) where scales has shape (out_features,)
    """
    qmax = (1 << (bits - 1)) - 1
    max_abs = weight.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    scales = (max_abs / qmax).squeeze(1)
    w_q = torch.clamp(torch.round(weight / max_abs * qmax), -qmax, qmax).to(torch.int8)
    return w_q, scales


def quantize_per_block_symmetric(
    weight: torch.Tensor,
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix per input block.

    Args:
        weight: (out_features, in_features)
        bits: target bit width
        block_size: partition in_features into blocks of this size.

    Returns:
        (weight_q, scales) where scales has shape (num_blocks,)
    """
    qmax = (1 << (bits - 1)) - 1
    out_features, in_features = weight.shape
    assert in_features % block_size == 0, (
        f"in_features {in_features} not divisible by block_size {block_size}"
    )
    num_blocks = in_features // block_size
    w_blocks = weight.view(out_features, num_blocks, block_size)
    # One scale per input block: max abs over (out_features, block_size).
    max_abs = w_blocks.abs().max(dim=2, keepdim=True).values.max(dim=0, keepdim=True).values.clamp_min(1e-8)
    scales = (max_abs / qmax).squeeze()  # (num_blocks,)
    w_q = torch.clamp(torch.round(w_blocks / scales.view(1, num_blocks, 1) * qmax), -qmax, qmax).to(torch.int8)
    w_q = w_q.view(out_features, in_features)
    return w_q, scales


class StandardQuantLinear(nn.Module):
    """A nn.Linear replaced by INT quantization (per-channel or per-block)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        bits: int = 8,
        block_size: int = None,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bits = bits
        self.block_size = block_size

        weight = base_linear.weight.detach()
        if block_size is None:
            w_q, scales = quantize_per_channel_symmetric(weight, bits)
            self.num_blocks = 1
        else:
            w_q, scales = quantize_per_block_symmetric(weight, bits, block_size)
            self.num_blocks = weight.shape[1] // block_size
        self.register_buffer("weight_q", w_q)
        self.register_buffer("scales", scales.to(base_linear.weight.dtype))
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach())
        else:
            self.bias = None

    def get_dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.block_size is None:
            s = self.scales.view(-1, 1).to(dtype)
            return self.weight_q.to(dtype) * s
        # Per-block: reconstruct via broadcasting.
        out_features, in_features = self.out_features, self.in_features
        num_blocks = self.num_blocks
        block_size = self.block_size
        w_q = self.weight_q.view(out_features, num_blocks, block_size)
        s = self.scales.view(1, num_blocks, 1).to(dtype)
        return (w_q.to(dtype) * s).view(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = self.get_dequantized_weight(x.dtype)
        return F.linear(x, w_hat, self.bias)

    def extra_repr(self) -> str:
        quant = f"per-block-symmetric (bs={self.block_size})" if self.block_size else "per-channel-symmetric"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, quant={quant}"
        )

    def get_weight_storage_stats(self, scale_bits: int = 16) -> dict:
        """Return weight-related storage statistics."""
        weight_bytes = self.out_features * self.in_features * self.bits / 8
        if self.block_size is None:
            scale_count = self.out_features
        else:
            scale_count = self.num_blocks
        scale_bytes = scale_count * scale_bits / 8
        total_bytes = weight_bytes + scale_bytes
        effective_bits = total_bytes * 8 / (self.out_features * self.in_features)
        return {
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": total_bytes,
            "effective_bits_per_weight": effective_bits,
            "scale_count": scale_count,
        }
