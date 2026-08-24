#!/usr/bin/env python3
"""Activation quantizer for v8 VQK low-bit arithmetic experiments.

Supports:
  - per-token symmetric quantization
  - per-token-per-block symmetric quantization
  - bit widths: 4, 6, 8

All quantizers return integer tensor + scale, keeping the dequantization
form x ≈ scale * Qx available for the VQK linear layer.
"""

from typing import Literal, Tuple

import torch


ActivationMode = Literal["per-token", "per-token-per-block"]


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_activation_per_token(
    x: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric round-to-nearest quantization.

    Args:
        x: float tensor of shape (..., in_features).  The last dim is treated
           as the feature dimension.
        bits: target bit width.

    Returns:
        (qx, scale)
        qx: int8 tensor of same shape as x.
        scale: float tensor of shape x.shape[:-1] + (1,).
    """
    qmax = _qmax(bits)
    # max abs over the last dimension (features), per token.
    max_abs = x.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    scale = max_abs / qmax
    qx = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return qx, scale


def quantize_activation_per_token_per_block(
    x: torch.Tensor,
    bits: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-per-block symmetric round-to-nearest quantization.

    Args:
        x: float tensor of shape (..., in_features).
        bits: target bit width.
        block_size: block size along the feature dimension.

    Returns:
        (qx, scale)
        qx: int8 tensor of same shape as x.
        scale: float tensor of shape (..., num_blocks, 1).
    """
    qmax = _qmax(bits)
    *prefix, in_features = x.shape
    assert in_features % block_size == 0, (
        f"in_features {in_features} not divisible by block_size {block_size}"
    )
    num_blocks = in_features // block_size

    # (..., num_blocks, block_size)
    x_blocks = x.view(*prefix, num_blocks, block_size)
    max_abs = x_blocks.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    scale = max_abs / qmax
    qx_blocks = torch.clamp(torch.round(x_blocks / scale), -qmax, qmax).to(torch.int8)
    qx = qx_blocks.view(*prefix, in_features)
    return qx, scale


def quantize_activation(
    x: torch.Tensor,
    bits: int,
    mode: str,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified activation quantizer entry."""
    if mode == "per-token":
        return quantize_activation_per_token(x, bits)
    if mode == "per-token-per-block":
        return quantize_activation_per_token_per_block(x, bits, block_size)
    raise ValueError(f"Unknown activation mode: {mode}")


def effective_bits_per_weight(
    weight_bits: int,
    out_features: int,
    in_features: int,
    scale_bits: int,
    num_scales: int,
) -> float:
    """Effective storage bits per weight element including scale overhead."""
    weight_bytes = out_features * in_features * weight_bits / 8
    scale_bytes = num_scales * scale_bits / 8
    return (weight_bytes + scale_bytes) * 8 / (out_features * in_features)
