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
    """DynamicCache that stores quantized K/V for standard attention layers.

    K is quantized per-channel (min/max over token dim for each head/channel).
    V is quantized per-token (min/max over channel dim for each head/token).
    Linear attention / GDN layers are left untouched (original DynamicCache behavior).
    """

    def __init__(self, k_bits=4, v_bits=4, config=None):
        # Initialize DynamicCache with the model config so that layer types
        # (attention / linear_attention) match the model's structure.
        super().__init__(config=config)
        self.k_bits = k_bits
        self.v_bits = v_bits
        self._k_meta = {}  # layer_idx -> (scale, zero_point)
        self._v_meta = {}  # layer_idx -> (scale, zero_point)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Append new key/value states, re-quantize the full layer, return dequantized tensors."""
        # Ensure the layer exists (same logic as DynamicCache.update).
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        # Only quantize standard attention layers (they expose .keys/.values).
        # Linear attention / GDN layers use different internal state and are left untouched.
        if layer_idx < len(self.layers) and not hasattr(self.layers[layer_idx], "keys"):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        layer = self.layers[layer_idx]

        if layer.keys is None or not layer.is_initialized:
            # First update for this layer.
            full_k = key_states
            full_v = value_states
        else:
            # Align cached tensors with incoming states (multi-GPU device_map safety).
            if layer.keys.device != key_states.device:
                layer.keys = layer.keys.to(key_states.device)
                layer.values = layer.values.to(key_states.device)
                if layer_idx in self._k_meta:
                    s_k, z_k = self._k_meta[layer_idx]
                    s_v, z_v = self._v_meta[layer_idx]
                    self._k_meta[layer_idx] = (s_k.to(key_states.device), z_k.to(key_states.device))
                    self._v_meta[layer_idx] = (s_v.to(key_states.device), z_v.to(key_states.device))

            # Dequantize existing cached states if they have been quantized before.
            if layer_idx in self._k_meta:
                old_k = dequantize_per_channel(layer.keys, *self._k_meta[layer_idx])
                old_v = dequantize_per_token(layer.values, *self._v_meta[layer_idx])
            else:
                old_k = layer.keys
                old_v = layer.values

            # Concatenate with new states.
            full_k = torch.cat([old_k, key_states], dim=-2)
            full_v = torch.cat([old_v, value_states], dim=-2)

        if self.k_bits >= 16 or self.v_bits >= 16:
            layer.keys = full_k
            layer.values = full_v
            return full_k, full_v

        # Re-quantize the full layer using current min/max.
        qk, s_k, z_k = quantize_per_channel(full_k, self.k_bits)
        qv, s_v, z_v = quantize_per_token(full_v, self.v_bits)
        layer.keys = qk
        layer.values = qv
        self._k_meta[layer_idx] = (s_k, z_k)
        self._v_meta[layer_idx] = (s_v, z_v)

        # Return dequantized tensors for attention computation.
        return dequantize_per_channel(qk, s_k, z_k), dequantize_per_token(qv, s_v, z_v)

    def to(self, device):
        """No-op: cache tensors are created/moved on the correct device during update()."""
        return self

    def get_max_length(self) -> int | None:
        # KIVI does not bound generation length; it grows as needed.
        return None
