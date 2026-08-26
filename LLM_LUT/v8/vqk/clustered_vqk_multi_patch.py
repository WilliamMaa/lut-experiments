#!/usr/bin/env python3
"""Composite patch replacing multiple Transformer Linear layers with Clustered VQK."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List

import torch.nn as nn

from common.evaluator import EvalPatch
from vqk.clustered_vqk_patch import ClusteredVQKPatch


class ClusteredVQKMultiPatch(EvalPatch):
    """Replace multiple Linear modules in one layer with Clustered VQK.

    Args:
        layer_idx: layer index in model.model.layers
        module_paths: list of attribute paths within the layer
        weight_bits, activation_bits, block_size, activation_mode, num_clusters:
            passed through to each per-module ClusteredVQKPatch
    """

    def __init__(
        self,
        layer_idx: int,
        module_paths: List[str],
        weight_bits: int = 4,
        activation_bits: int = 4,
        block_size: int = 128,
        activation_mode: str = "per-token-per-block",
        num_clusters: int = 2,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_paths = module_paths
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.block_size = block_size
        self.activation_mode = activation_mode
        self.num_clusters = num_clusters
        self.patches = [
            ClusteredVQKPatch(
                layer_idx=layer_idx,
                module_path=mp,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                block_size=block_size,
                activation_mode=activation_mode,
                num_clusters=num_clusters,
            )
            for mp in module_paths
        ]
        self.total_weight_elements = 0

    def install(self, model: nn.Module) -> None:
        self.total_weight_elements = 0
        for p in self.patches:
            p.install(model)
            self.total_weight_elements += (
                p._replacement.out_features * p._replacement.in_features
            )

    def uninstall(self, model: nn.Module) -> None:
        for p in self.patches:
            p.uninstall(model)

    def name(self) -> str:
        modules = "+".join(self.module_paths)
        return (
            f"clustered_vqk_l{self.layer_idx}_{modules}_"
            f"w{self.weight_bits}a{self.activation_bits}_"
            f"blk{self.block_size}_k{self.num_clusters}_{self.activation_mode}"
        )

    def config(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "module_paths": self.module_paths,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "block_size": self.block_size,
            "activation_mode": self.activation_mode,
            "num_clusters": self.num_clusters,
        }

    def storage_stats(self, scale_bits: int = 16) -> Dict[str, Any]:
        """Aggregate storage stats across all replaced modules."""
        total = {}
        per_module = []
        for p in self.patches:
            stats = p.storage_stats(scale_bits)
            per_module.append({"module_path": p.module_path, "stats": stats})
            for key in ["weight_bytes", "scale_bytes", "assignment_bytes", "total_bytes"]:
                total[key] = total.get(key, 0.0) + stats.get(key, 0.0)

        if total.get("weight_bytes", 0.0) > 0 and self.total_weight_elements > 0:
            total["effective_bits_per_weight"] = (
                total["total_bytes"] * 8 / self.total_weight_elements
            )
        else:
            total["effective_bits_per_weight"] = 0.0
        total["num_modules"] = len(self.patches)

        return {
            "total": total,
            "per_module": per_module,
        }
