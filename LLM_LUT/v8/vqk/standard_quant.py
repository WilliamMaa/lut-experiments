#!/usr/bin/env python3
"""Standard INT quantization baseline for VQK comparison.

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


class StandardQuantLinear(nn.Module):
    """A nn.Linear replaced by per-channel INT quantization."""

    def __init__(self, base_linear: nn.Linear, bits: int = 8):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bits = bits

        weight = base_linear.weight.detach()
        w_q, scales = quantize_per_channel_symmetric(weight, bits)
        self.register_buffer("weight_q", w_q)
        self.register_buffer("scales", scales.to(base_linear.weight.dtype))
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach())
        else:
            self.bias = None

    def get_dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        s = self.scales.view(-1, 1).to(dtype)
        return self.weight_q.float() * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = self.get_dequantized_weight(x.dtype)
        return F.linear(x, w_hat, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, quant=per-channel-symmetric"
        )
