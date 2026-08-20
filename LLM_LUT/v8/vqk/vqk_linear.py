#!/usr/bin/env python3
"""VQK Linear layer implementation for v8.

VQK form: W ≈ S ⊙ W_q
- W_q: integer weight tensor, same shape as W
- S: per-block FP/BF16 scale, block partitioned along input dimension
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _qmax(bits: int, symmetric: bool = True) -> int:
    """Return max integer value for a signed quantizer."""
    assert symmetric, "only symmetric quantization supported for now"
    return (1 << (bits - 1)) - 1


def quantize_symmetric(
    weight: torch.Tensor,
    bits: int,
    block_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric round-to-nearest quantization.

    Args:
        weight: float tensor of shape (out_features, in_features).
        bits: target bit width (2/3/4/6/8).
        block_size: if given, partition in_features into blocks of this size.
                      Otherwise quantize per-channel (along out_features).

    Returns:
        (weight_q, scales)
        weight_q: int8 tensor of same shape as weight.
        scales: float tensor of shape (num_blocks,) if block_size given,
                else (out_features,) for per-channel.
    """
    qmax = _qmax(bits)
    if block_size is None:
        # Per-channel quantization: each output channel gets its own scale.
        max_abs = weight.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
        scales = max_abs / qmax
        w_q = torch.clamp(torch.round(weight / scales), -qmax, qmax).to(torch.int8)
        scales = scales.squeeze(1)
    else:
        # Block-wise quantization along input dimension.
        out_features, in_features = weight.shape
        assert in_features % block_size == 0, (
            f"in_features {in_features} must be divisible by block_size {block_size}"
        )
        num_blocks = in_features // block_size
        # (out_features, num_blocks, block_size)
        w_blocks = weight.view(out_features, num_blocks, block_size)
        max_abs = w_blocks.abs().max(dim=2, keepdim=True).values.clamp_min(1e-8)
        block_scales = max_abs / qmax  # (out_features, num_blocks, 1)
        w_q_blocks = torch.clamp(torch.round(w_blocks / block_scales), -qmax, qmax).to(torch.int8)
        w_q = w_q_blocks.view(out_features, in_features)

        # DSConv-style L2-optimal scale per block.
        # For each block B: S_B = sum(W * W_q) / sum(W_q^2)
        w_q_float = w_q_blocks.float()
        numerator = (w_blocks * w_q_float).sum(dim=2)  # (out_features, num_blocks)
        denominator = (w_q_float * w_q_float).sum(dim=2).clamp_min(1e-8)
        scales = (numerator / denominator)  # (out_features, num_blocks)

        # For a single scale per block (not per-output-channel), average over out_features.
        scales = scales.mean(dim=0)  # (num_blocks,)

    return w_q, scales


class VQKLinear(nn.Module):
    """Replace a nn.Linear with VQK-decomposed weight.

    Forward computes y = x @ (S * W_q).T + bias.
    For eval-only use, we dequantize on the fly; no custom kernel required.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        bits: int = 4,
        block_size: int = 64,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bits = bits
        self.block_size = block_size

        weight = base_linear.weight.detach()
        w_q, scales = quantize_symmetric(weight, bits, block_size)
        self.register_buffer("weight_q", w_q)
        self.register_buffer("scales", scales.to(base_linear.weight.dtype))
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach())
        else:
            self.bias = None

    def get_dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct FP/BF16 weight from VQK for forward."""
        out_features, in_features = self.out_features, self.in_features
        num_blocks = in_features // self.block_size
        w_q = self.weight_q.view(out_features, num_blocks, self.block_size)
        # scales: (num_blocks,) -> broadcast to (out_features, num_blocks, block_size)
        s = self.scales.view(1, num_blocks, 1).to(dtype)
        w_hat = w_q.float() * s
        return w_hat.view(out_features, in_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = self.get_dequantized_weight(x.dtype)
        return F.linear(x, w_hat, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, block_size={self.block_size}"
        )
