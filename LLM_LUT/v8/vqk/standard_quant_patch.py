#!/usr/bin/env python3
"""Standard INT quantization EvalPatch for v8 unified evaluator."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

import torch.nn as nn

from common.evaluator import EvalPatch
from vqk.standard_quant import StandardQuantLinear


class StandardQuantPatch(EvalPatch):
    """Replace a single Transformer Linear with standard RTN INT quantization.

    Args:
        layer_idx: layer index in model.model.layers
        module_path: attribute path within the layer, e.g. "self_attn.o_proj"
        bits: weight quantization bit width
        block_size: if given, per-block quantization along input dimension;
                    otherwise per-channel along output dimension.
    """

    def __init__(
        self,
        layer_idx: int,
        module_path: str,
        bits: int = 4,
        block_size: int = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_path = module_path
        self.bits = bits
        self.block_size = block_size
        self._original_module: nn.Module = None
        self._parent: nn.Module = None
        self._attr_name: str = None
        self._replacement: StandardQuantLinear = None
        self._cached_storage_stats: Dict[str, Any] = {}

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
        self._replacement = StandardQuantLinear(
            self._original_module,
            bits=self.bits,
            block_size=self.block_size,
        )
        setattr(self._parent, self._attr_name, self._replacement)
        self._cached_storage_stats = self._replacement.get_weight_storage_stats()

    def uninstall(self, model: nn.Module) -> None:
        if self._parent is not None and self._attr_name is not None:
            setattr(self._parent, self._attr_name, self._original_module)
            self._parent = None
            self._attr_name = None
            self._original_module = None
            self._replacement = None

    def name(self) -> str:
        quant = f"blk{self.block_size}" if self.block_size else "perchannel"
        return f"rtn_int{self.bits}_l{self.layer_idx}_{self.module_path}_{quant}"

    def config(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "module_path": self.module_path,
            "bits": self.bits,
            "block_size": self.block_size,
        }

    def storage_stats(self, scale_bits: int = 16) -> Dict[str, Any]:
        """Return storage statistics for the replaced module."""
        if self._replacement is not None:
            self._cached_storage_stats = self._replacement.get_weight_storage_stats(scale_bits)
        return self._cached_storage_stats
