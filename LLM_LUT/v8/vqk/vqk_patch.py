#!/usr/bin/env python3
"""VQK / RTN INT EvalPatch implementation for v8 unified evaluator."""

from typing import Any, Dict

import torch.nn as nn

from common.evaluator import EvalPatch
from .vqk_linear import VQKLinear
from .standard_quant import StandardQuantLinear


class VQKPatch(EvalPatch):
    """Replace a single Transformer Linear with VQK or RTN INT quantization.

    Args:
        layer_idx: layer index in model.model.layers
        module_path: attribute path within the layer, e.g. "self_attn.o_proj"
        bits: quantization bit width
        block_size: VQK block size along input dimension; ignored for RTN INT
        quant_method: "vqk" or "int"
    """

    def __init__(
        self,
        layer_idx: int,
        module_path: str,
        bits: int,
        block_size: int = 64,
        quant_method: str = "vqk",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_path = module_path
        self.bits = bits
        self.block_size = block_size
        self.quant_method = quant_method
        self._original_module: nn.Module = None
        self._parent: nn.Module = None
        self._attr_name: str = None

    def _resolve(self, model: nn.Module):
        """Resolve parent module and attribute name for the target Linear."""
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
        base_linear = self._original_module
        if self.quant_method == "vqk":
            replacement = VQKLinear(base_linear, bits=self.bits, block_size=self.block_size)
        elif self.quant_method == "int":
            replacement = StandardQuantLinear(base_linear, bits=self.bits)
        else:
            raise ValueError(f"Unknown quant_method: {self.quant_method}")
        setattr(self._parent, self._attr_name, replacement)

    def uninstall(self, model: nn.Module) -> None:
        if self._parent is not None and self._attr_name is not None:
            setattr(self._parent, self._attr_name, self._original_module)
            self._parent = None
            self._attr_name = None
            self._original_module = None

    def name(self) -> str:
        if self.quant_method == "vqk":
            return f"vqk_l{self.layer_idx}_{self.module_path}_b{self.bits}_blk{self.block_size}"
        return f"int_l{self.layer_idx}_{self.module_path}_b{self.bits}"

    def config(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "module_path": self.module_path,
            "bits": self.bits,
            "block_size": self.block_size,
            "quant_method": self.quant_method,
        }
