#!/usr/bin/env python3
"""KIVI-style KV cache: K per-channel, V per-token quantization."""

import torch
from transformers.cache_utils import DynamicCache

from kv_cache.kv_quantizers import (
    quantize_per_channel,
    dequantize_per_channel,
    quantize_per_token,
    dequantize_per_token,
)


class KIVICache(DynamicCache):
    """DynamicCache that stores quantized K/V.

    K is quantized per-channel (min/max over token dim for each head/channel).
    V is quantized per-token (min/max over channel dim for each head/token).
    On get(), the tensors are dequantized back to the original dtype.

    This is a simplified implementation: after each update, the entire cached
    tensor for that layer is re-quantized using fresh scales. It is sufficient
    for measuring end-to-end quality; a production version would update scales
    incrementally.
    """

    def __init__(self, k_bits=4, v_bits=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self._k_meta = {}  # layer_idx -> (scale, zero_point)
        self._v_meta = {}  # layer_idx -> (scale, zero_point)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Append new key/value states, then re-quantize the full layer cache."""
        # Append original values to raw cache first.
        if layer_idx >= len(self._key_cache):
            self._key_cache.append(key_states)
            self._value_cache.append(value_states)
        else:
            self._key_cache[layer_idx] = torch.cat(
                [self._key_cache[layer_idx], key_states], dim=-2
            )
            self._value_cache[layer_idx] = torch.cat(
                [self._value_cache[layer_idx], value_states], dim=-2
            )

        if self.k_bits >= 16 or self.v_bits >= 16:
            # No quantization; keep raw tensors.
            return self

        # Re-quantize the full layer using current min/max.
        k = self._key_cache[layer_idx]
        v = self._value_cache[layer_idx]
        qk, s_k, z_k = quantize_per_channel(k, self.k_bits)
        qv, s_v, z_v = quantize_per_token(v, self.v_bits)
        self._key_cache[layer_idx] = qk
        self._value_cache[layer_idx] = qv
        self._k_meta[layer_idx] = (s_k, z_k)
        self._v_meta[layer_idx] = (s_v, z_v)
        return self

    def get(self, layer_idx, *args, **kwargs):
        """Return dequantized K/V for attention computation."""
        if self.k_bits >= 16 or self.v_bits >= 16:
            return super().get(layer_idx, *args, **kwargs)
        qk = self._key_cache[layer_idx]
        qv = self._value_cache[layer_idx]
        s_k, z_k = self._k_meta[layer_idx]
        s_v, z_v = self._v_meta[layer_idx]
        k = dequantize_per_channel(qk, s_k, z_k)
        v = dequantize_per_token(qv, s_v, z_v)
        return k, v

    def to(self, device):
        """Move metadata to device. Raw uint8 tensors are moved by base class."""
        super().to(device)
        for layer_idx in self._k_meta:
            s_k, z_k = self._k_meta[layer_idx]
            s_v, z_v = self._v_meta[layer_idx]
            self._k_meta[layer_idx] = (s_k.to(device), z_k.to(device))
            self._v_meta[layer_idx] = (s_v.to(device), z_v.to(device))
        return self
