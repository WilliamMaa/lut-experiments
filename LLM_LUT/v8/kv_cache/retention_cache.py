#!/usr/bin/env python3
"""Retention-based KV cache: keep sink tokens + most recent tokens, evict middle."""

import torch
from transformers.cache_utils import DynamicCache


class RetentionCache(DynamicCache):
    """DynamicCache that retains only sink + recent tokens for standard attention layers.

    GDN / linear attention layers are left untouched.
    """

    def __init__(self, max_cache_len=512, sink_tokens=4, config=None):
        super().__init__(config=config)
        self.retention_max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        layer = self.layers[layer_idx]
        if not hasattr(layer, "keys"):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)

        # Align cached tensors with the incoming states (multi-GPU device_map safety).
        if layer.keys.device != key_states.device:
            layer.keys = layer.keys.to(key_states.device)
            layer.values = layer.values.to(key_states.device)

        keys = torch.cat([layer.keys, key_states], dim=-2)
        values = torch.cat([layer.values, value_states], dim=-2)

        total_len = keys.shape[-2]
        if total_len > self.retention_max_cache_len:
            keep_recent = self.retention_max_cache_len - self.sink_tokens
            sink_keys = keys[..., : self.sink_tokens, :]
            sink_values = values[..., : self.sink_tokens, :]
            recent_keys = keys[..., -keep_recent:, :]
            recent_values = values[..., -keep_recent:, :]
            keys = torch.cat([sink_keys, recent_keys], dim=-2)
            values = torch.cat([sink_values, recent_values], dim=-2)

        layer.keys = keys
        layer.values = values
        return keys, values

    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys = layer.keys.to(device)
            if hasattr(layer, "values") and layer.values is not None:
                layer.values = layer.values.to(device)
        return self

    def get_max_length(self) -> int | None:
        # Do not let the cache's own capacity bound generation length.
        # RetentionCache manually evicts; generation `max_length` should govern.
        return None
