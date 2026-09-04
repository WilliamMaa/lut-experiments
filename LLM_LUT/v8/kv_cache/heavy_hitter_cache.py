#!/usr/bin/env python3
"""Heavy-hitter KV cache: keep sink + recent + important middle tokens.

Unlike pure retention (sink + recent), this cache also retains the most
"important" tokens from the evicted middle region. Importance is proxied by
the L2 norm of the key vector, which correlates with attention magnitude.
"""

import torch
from transformers.cache_utils import DynamicCache


class HeavyHitterCache(DynamicCache):
    """DynamicCache that retains sink + recent + heavy-hitter middle tokens."""

    # Override DynamicCache's length bound so generation uses --max_length, not cache capacity.
    @property
    def max_cache_len(self):
        return None

    @max_cache_len.setter
    def max_cache_len(self, value):
        pass

    def __init__(self, max_cache_len=512, sink_tokens=4, recent_tokens=128, config=None):
        super().__init__(config=config)
        self.retention_max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

    def _importance_scores(self, keys):
        """Compute per-token importance from key vectors.

        keys shape: [batch, heads, seq, head_dim]
        Returns: [batch, seq]
        """
        # L2 norm per head, averaged over heads
        return keys.float().pow(2).sum(dim=-1).sqrt().mean(dim=1)

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
            budget = self.retention_max_cache_len
            sink_n = min(self.sink_tokens, total_len)
            recent_n = min(self.recent_tokens, budget - sink_n)
            hh_budget = budget - sink_n - recent_n

            sink_keys = keys[..., :sink_n, :]
            sink_values = values[..., :sink_n, :]
            recent_keys = keys[..., -recent_n:, :]
            recent_values = values[..., -recent_n:, :]

            middle_end = total_len - recent_n
            middle_keys = keys[..., sink_n:middle_end, :]
            middle_values = values[..., sink_n:middle_end, :]
            middle_len = middle_keys.shape[-2]

            if hh_budget > 0 and middle_len > hh_budget:
                B, H, M, D = middle_keys.shape
                scores = self._importance_scores(middle_keys)  # [B, M]
                topk = scores.topk(hh_budget, dim=-1).indices  # [B, hh_budget]
                topk, _ = topk.sort(dim=-1)  # maintain temporal order

                # Gather heavy hitters: [B, H, hh_budget, D]
                topk_expanded = topk.unsqueeze(1).unsqueeze(-1).expand(B, H, hh_budget, D)
                hh_keys = torch.gather(middle_keys, dim=2, index=topk_expanded)
                hh_values = torch.gather(middle_values, dim=2, index=topk_expanded)

                keys = torch.cat([sink_keys, hh_keys, recent_keys], dim=-2)
                values = torch.cat([sink_values, hh_values, recent_values], dim=-2)
            else:
                keys = torch.cat([sink_keys, middle_keys, recent_keys], dim=-2)
                values = torch.cat([sink_values, middle_values, recent_values], dim=-2)

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
