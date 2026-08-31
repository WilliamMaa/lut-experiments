#!/usr/bin/env python3
"""Quantization helpers for KV cache compression."""

import torch


def quantize_per_channel(x, bits):
    """Asymmetric per-channel quantization along the token dimension.

    x: (..., T, D) -> quantize each channel (each D) using min/max over T.
    Returns (q, scale, zero_point) where q is uint8 in [0, 2**bits - 1].
    """
    mn = x.min(dim=-2, keepdim=True)[0]
    mx = x.max(dim=-2, keepdim=True)[0]
    qmax = (1 << bits) - 1
    scale = (mx - mn) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round((x - mn) / scale).clamp(0, qmax).to(torch.uint8)
    return q, scale, mn


def dequantize_per_channel(q, scale, mn):
    """Dequantize per-channel quantized tensor."""
    return q.to(scale.dtype) * scale + mn


def quantize_per_token(x, bits):
    """Asymmetric per-token quantization along the channel dimension.

    x: (..., T, D) -> quantize each token (each T) using min/max over D.
    Returns (q, scale, zero_point).
    """
    mn = x.min(dim=-1, keepdim=True)[0]
    mx = x.max(dim=-1, keepdim=True)[0]
    qmax = (1 << bits) - 1
    scale = (mx - mn) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round((x - mn) / scale).clamp(0, qmax).to(torch.uint8)
    return q, scale, mn


def dequantize_per_token(q, scale, mn):
    """Dequantize per-token quantized tensor."""
    return q.to(scale.dtype) * scale + mn
