#!/usr/bin/env python3
"""Integer VQK Linear layer for v8 Stage 1 low-bit arithmetic experiments.

Mathematical path:
    W ≈ s_w ⊙ Q_w
    x ≈ s_x ⊙ Q_x
    y = x @ W.T ≈ (s_x s_w) ⊙ (Q_x @ Q_w.T)

The integer dot product Q_x @ Q_w.T is accumulated in int32 (or simulated in FP),
then scaled back with the per-block weight scale and per-token activation scale.

Stage 1 config:
    weight bits = 4
    activation bits = 8
    weight block size = 32 / 64 / 128
    activation mode = per-token
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vqk.vqk_linear import quantize_symmetric
from vqk.activation_quantizer import quantize_activation


class IntegerVQKLinear(nn.Module):
    """A nn.Linear replaced by integer-only VQK-style low-bit arithmetic.

    Args:
        base_linear: the original nn.Linear.
        weight_bits: bit width for weight quantization.
        activation_bits: bit width for activation quantization.
        block_size: block size along input dimension for weight blocks.
        activation_mode: "per-token" or "per-token-per-block".
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        weight_bits: int = 4,
        activation_bits: int = 8,
        block_size: int = 64,
        activation_mode: str = "per-token",
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.block_size = block_size
        self.activation_mode = activation_mode

        weight = base_linear.weight.detach()
        w_q, s_w = quantize_symmetric(weight, weight_bits, block_size)
        self.register_buffer("weight_q", w_q)
        self.register_buffer("weight_scales", s_w.to(base_linear.weight.dtype))
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach())
        else:
            self.bias = None

        self.num_blocks = self.in_features // self.block_size

    def get_weight_storage_stats(self, scale_bits: int = 16) -> dict:
        """Return weight-related storage statistics."""
        weight_bytes = self.out_features * self.in_features * self.weight_bits / 8
        scale_count = self.num_blocks
        scale_bytes = scale_count * scale_bits / 8
        effective_bits = (weight_bytes + scale_bytes) * 8 / (self.out_features * self.in_features)
        return {
            "weight_bytes": weight_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": weight_bytes + scale_bytes,
            "effective_bits_per_weight": effective_bits,
            "scale_count": scale_count,
        }

    def _quantize_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation and return Q_x, s_x."""
        return quantize_activation(x, self.activation_bits, self.activation_mode, self.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize activation dynamically.
        q_x, s_x = self._quantize_activation(x)

        # 2. Reshape weight and activation into blocks.
        # q_w: (out_features, in_features) -> (out_features, num_blocks, block_size)
        # q_x: (*, in_features) -> (*, num_blocks, block_size)
        q_w = self.weight_q.view(self.out_features, self.num_blocks, self.block_size)
        *prefix, _ = q_x.shape
        q_x_blocks = q_x.view(*prefix, self.num_blocks, self.block_size)

        # 3. Integer dot product per block.
        #    acc[b] = q_x[..., b, :] @ q_w[:, b, :].T  ->  (*, out_features)
        acc = torch.einsum("...bi,obi->...bo", q_x_blocks.to(torch.int32), q_w.to(torch.int32))
        # acc shape: (*, num_blocks, out_features)

        # 4. Apply scales.
        #    weight_scales: (num_blocks,) -> (1, num_blocks, 1)
        s_w = self.weight_scales.view(1, self.num_blocks, 1).to(x.dtype)

        if self.activation_mode == "per-token":
            # s_x: (*, 1) -> (*, 1, 1)
            s_x_b = s_x.to(x.dtype).unsqueeze(-1)
        elif self.activation_mode == "per-token-per-block":
            # s_x: (*, num_blocks) -> (*, num_blocks, 1)
            s_x_b = s_x.to(x.dtype).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown activation mode: {self.activation_mode}")

        # acc shape: (*, num_blocks, out_features)
        # scales broadcast and sum over blocks
        output = (acc.to(x.dtype) * s_w * s_x_b).sum(dim=-2)

        if self.bias is not None:
            output = output + self.bias.to(x.dtype)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
            f"block_size={self.block_size}, activation_mode={self.activation_mode}"
        )
