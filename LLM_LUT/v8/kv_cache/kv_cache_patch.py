#!/usr/bin/env python3
"""Base class and first method for KV cache compression patches."""

import sys
import os
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import EvalPatch
from kv_cache.kivi_cache import KIVICache


class KVCachePatch(EvalPatch):
    """Base class for patches that replace the generation-time KV cache."""

    def name(self) -> str:
        return self.__class__.__name__

    def config(self) -> Dict[str, Any]:
        return {}

    def storage_stats(self) -> Dict[str, Any]:
        return {}

    def get_cache(self, device, config=None):
        """Return a fresh cache instance for one generation."""
        raise NotImplementedError

    def install(self, model):
        """Optional in-place model modification. Default does nothing."""
        pass

    def uninstall(self, model):
        """Optional cleanup. Default does nothing."""
        pass


class KIVICachePatch(KVCachePatch):
    """KIVI-style asymmetric KV cache: K per-channel, V per-token."""

    def __init__(self, k_bits: int = 4, v_bits: int = 4):
        self.k_bits = k_bits
        self.v_bits = v_bits

    def name(self) -> str:
        return f"kivi_k{self.k_bits}_v{self.v_bits}"

    def config(self) -> Dict[str, Any]:
        return {"k_bits": self.k_bits, "v_bits": self.v_bits}

    def get_cache(self, device, config=None):
        return KIVICache(k_bits=self.k_bits, v_bits=self.v_bits, config=config).to(device)

    def storage_stats(self) -> Dict[str, Any]:
        # Qwen full-attention: 2 KV heads, head_dim=256, 10 full-attention layers.
        # Per token: 2 heads * 256 dim * (k_bits + v_bits) / 8 bytes.
        kv_heads = 2
        head_dim = 256
        full_attn_layers = 10
        bytes_per_token = kv_heads * head_dim * (self.k_bits + self.v_bits) / 8
        bf16_bytes_per_token = kv_heads * head_dim * 4  # bf16 = 2 bytes each for K and V
        return {
            "kv_bits": self.k_bits + self.v_bits,
            "bytes_per_token": bytes_per_token,
            "bf16_bytes_per_token": bf16_bytes_per_token,
            "compression_ratio": bf16_bytes_per_token / bytes_per_token,
            "estimated_128k_context_bytes": bytes_per_token * 128000 * full_attn_layers,
            "estimated_128k_context_bf16_bytes": bf16_bytes_per_token * 128000 * full_attn_layers,
        }
