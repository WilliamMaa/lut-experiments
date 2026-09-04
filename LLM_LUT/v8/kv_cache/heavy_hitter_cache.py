#!/usr/bin/env python3
"""Heavy-hitter KV cache: keep sink + recent + important middle tokens.

Two importance signals are supported:

- ``key_norm`` (default, cheap): L2 norm of the key vector, which only
  correlates with attention magnitude. It cannot tell *which* keys the
  queries actually read, so it keeps the wrong tokens in documents whose
  key norms are uninformative.

- ``attn_score``: column sums of the prefill attention probabilities
  (accumulated by ``attention_scores.AttentionScoreBank``). This is the
  H2O/SnapKV-style signal. Because the scores for a sequence are only known
  after attention runs, eviction is deferred: prefill appends without
  eviction, and the first decode step compresses the cache using the scores.
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

    def __init__(self, max_cache_len=512, sink_tokens=4, recent_tokens=128, config=None,
                 importance_mode="key_norm", score_bank=None):
        super().__init__(config=config)
        self.retention_max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.importance_mode = importance_mode
        self.score_bank = score_bank

    def _importance_scores(self, keys):
        """Compute per-token importance from key vectors.

        keys shape: [batch, heads, seq, head_dim]
        Returns: [batch, seq]
        """
        # L2 norm per head, averaged over heads
        return keys.float().pow(2).sum(dim=-1).sqrt().mean(dim=1)

    def _middle_importance(self, layer, layer_idx, sink_n, middle_len, total_len,
                           middle_keys, device):
        """Per-token importance for the evictable middle region: [1, middle_len]."""
        if self.importance_mode == "attn_score" and self.score_bank is not None:
            prefill_scores = self.score_bank.scores.get(layer_idx)
            if prefill_scores is not None:
                if getattr(layer, "_hh_prefill_scores", None) is None:
                    layer._hh_prefill_scores = prefill_scores.to(device)
                prefill = layer._hh_prefill_scores
                # Prefill positions get their measured attention mass; tokens
                # appended after the snapshot (post-compression decode tokens)
                # are protected with +inf so they roll out gradually instead of
                # being evicted before the prefill heavy hitters.
                scores = torch.full((total_len,), float("inf"), device=device)
                scores[:prefill.shape[-1]] = prefill
                middle_scores = scores[sink_n:sink_n + middle_len]
                return middle_scores.unsqueeze(0).expand(middle_keys.shape[0], -1)
        # Fallback: key-L2-norm proxy.
        return self._importance_scores(middle_keys)

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

        incoming_len = key_states.shape[-2]
        keys = torch.cat([layer.keys, key_states], dim=-2)
        values = torch.cat([layer.values, value_states], dim=-2)

        total_len = keys.shape[-2]

        if self.importance_mode == "attn_score" and incoming_len > 1:
            # Prefill: the attention scores for this very sequence only exist
            # after attention runs, so we cannot evict yet. Grow the cache and
            # defer compression to the first decode step.
            layer.keys = keys
            layer.values = values
            return keys, values

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
                scores = self._middle_importance(
                    layer, layer_idx, sink_n, middle_len, total_len,
                    middle_keys, keys.device,
                )  # [B, M]
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
            if getattr(layer, "_hh_prefill_scores", None) is not None:
                layer._hh_prefill_scores = layer._hh_prefill_scores.to(device)
        return self
