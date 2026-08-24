#!/usr/bin/env python3
"""Integer VQK EvalPatch for v8 unified evaluator."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

import torch.nn as nn

from common.evaluator import EvalPatch
from vqk.integer_vqk_linear import IntegerVQKLinear


class IntegerVQKPatch(EvalPatch):
    """Replace a single Transformer Linear with integer VQK low-bit arithmetic.

    Args:
        layer_idx: layer index in model.model.layers
        module_path: attribute path within the layer, e.g. "self_attn.o_proj"
        weight_bits: weight quantization bit width
        activation_bits: activation quantization bit width
        block_size: VQK block size along input dimension
        activation_mode: "per-token" or "per-token-per-block"
    """

    def __init__(
        self,
        layer_idx: int,
        module_path: str,
        weight_bits: int = 4,
        activation_bits: int = 8,
        block_size: int = 64,
        activation_mode: str = "per-token",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_path = module_path
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.block_size = block_size
        self.activation_mode = activation_mode
        self._original_module: nn.Module = None
        self._parent: nn.Module = None
        self._attr_name: str = None
        self._replacement: IntegerVQKLinear = None

    def _resolve(self, model: nn.Module):
        layer = model.model.layers[self.layer_idx]
        target = layer
        attrs = self.module_path.split(".")
        for name in attrs[:-1]:
            target = getattr(target, name)
        self._parent = target
        self._attr_name = attrs[-1]
        self._original_module = getattr(target, self._attr_name)
        if not isinstance(self._original_module, nn.Linear):
            raise TypeError(
                f"Target module {self.module_path} in layer {self.layer_idx} is not nn.Linear: "
                f"{type(self._original_module)}"
            )

    def install(self, model: nn.Module) -> None:
        self._resolve(model)
        self._replacement = IntegerVQKLinear(
            self._original_module,
            weight_bits=self.weight_bits,
            activation_bits=self.activation_bits,
            block_size=self.block_size,
            activation_mode=self.activation_mode,
        )
        setattr(self._parent, self._attr_name, self._replacement)

    def uninstall(self, model: nn.Module) -> None:
        if self._parent is not None and self._attr_name is not None:
            setattr(self._parent, self._attr_name, self._original_module)
            self._parent = None
            self._attr_name = None
            self._original_module = None
            self._replacement = None

    def name(self) -> str:
        return (
            f"int_vqk_l{self.layer_idx}_{self.module_path}_"
            f"w{self.weight_bits}a{self.activation_bits}_"
            f"blk{self.block_size}_{self.activation_mode}"
        )

    def config(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "module_path": self.module_path,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "block_size": self.block_size,
            "activation_mode": self.activation_mode,
        }

    def get_storage_stats(self, scale_bits: int = 16) -> Dict[str, Any]:
        """Return storage statistics for the replaced module."""
        if self._replacement is None:
            return {}
        return self._replacement.get_weight_storage_stats(scale_bits)
