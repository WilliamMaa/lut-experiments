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

from kv_cache.kv_quantizers import (
    quantize_per_channel,
    dequantize_per_channel,
    quantize_per_token,
    dequantize_per_token,
)


class HeavyHitterCache(DynamicCache):
    """DynamicCache that retains sink + recent + heavy-hitter middle tokens.

    Optional K/V storage quantization (k_bits/v_bits, 16 = off): the stored
    cache is quantized once the post-eviction state is reached (decode
    steps); update() dequantizes at entry and returns dequantized tensors
    for attention, KIVI-style (per-channel K, per-token V). Prefill growth
    stays bf16 — the compression target is the decode-time cache.
    """

    # Override DynamicCache's length bound so generation uses --max_length, not cache capacity.
    @property
    def max_cache_len(self):
        return None

    @max_cache_len.setter
    def max_cache_len(self, value):
        pass

    def __init__(self, max_cache_len=512, sink_tokens=4, recent_tokens=128, config=None,
                 importance_mode="key_norm", score_bank=None, obs_window=0,
                 merge_evicted=False, k_bits=16, v_bits=16):
        super().__init__(config=config)
        self.retention_max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.importance_mode = importance_mode
        self.score_bank = score_bank
        self.obs_window = obs_window
        # Compensation eviction: fold each evicted token's value (weighted by
        # its per-head attention mass relative to the target kept token,
        # capped at 1) into its nearest kept successor. Selection no longer
        # has to be perfect — a fact living in an evicted token survives in
        # the value of a token that stays.
        self.merge_evicted = merge_evicted
        self.k_bits = k_bits
        self.v_bits = v_bits
        self._k_meta = {}  # layer_idx -> (scale, min) for quantized stored K
        self._v_meta = {}  # layer_idx -> (scale, min) for quantized stored V

    def _maybe_quantize(self, layer, layer_idx, keys, values, incoming_len):
        """Quantize the stored post-eviction state on decode steps.

        Returns the tensors attention should use this forward (dequantized).
        Prefill growth (incoming_len > 1) stays bf16: storage is transient
        there and the compression target is the decode-time cache.
        """
        if (self.k_bits >= 16 and self.v_bits >= 16) or incoming_len > 1:
            return keys, values
        qk, qv = keys, values
        if self.k_bits < 16:
            qk, s_k, m_k = quantize_per_channel(keys, self.k_bits)
            self._k_meta[layer_idx] = (s_k, m_k)
        else:
            self._k_meta.pop(layer_idx, None)
        if self.v_bits < 16:
            qv, s_v, m_v = quantize_per_token(values, self.v_bits)
            self._v_meta[layer_idx] = (s_v, m_v)
        else:
            self._v_meta.pop(layer_idx, None)
        layer.keys = qk
        layer.values = qv
        out_k = dequantize_per_channel(qk, *self._k_meta[layer_idx]) if self.k_bits < 16 else qk
        out_v = dequantize_per_token(qv, *self._v_meta[layer_idx]) if self.v_bits < 16 else qv
        return out_k, out_v

    def _importance_scores(self, keys):
        """Compute per-token importance from key vectors.

        keys shape: [batch, heads, seq, head_dim]
        Returns: [batch, seq]
        """
        # L2 norm per head, averaged over heads
        return keys.float().pow(2).sum(dim=-1).sqrt().mean(dim=1)

    def _position_scores(self, layer, layer_idx, orig_idx, device):
        """Attention-mass score for each currently cached key position.

        Maps every cached key back to its original prefill position via
        ``orig_idx`` (the cache is compressed, so current positions are not
        prefill positions anymore). Keys written after the prefill snapshot
        get +inf so they roll out gradually instead of being evicted before
        the prefill heavy hitters.

        Returns [total_len], or None when no prefill scores are available.
        """
        if self.importance_mode != "attn_score" or self.score_bank is None:
            return None
        prefill = getattr(layer, "_hh_prefill_scores", None)
        if prefill is None:
            bank_scores = self.score_bank.scores.get(layer_idx)
            if bank_scores is None:
                return None
            prefill = bank_scores.to(device)
            layer._hh_prefill_scores = prefill
        plen = prefill.shape[-1]
        valid = orig_idx < plen
        safe_idx = orig_idx.clamp(max=plen - 1)
        # Decode tokens (written after the prefill snapshot) compete at the
        # *mean* attention mass. +inf here would let them flush every prefill
        # heavy hitter out of the middle region within one generation.
        vals = torch.where(valid, prefill[safe_idx], prefill.mean())
        return vals

    def _per_head_scores(self, layer, layer_idx, device, n_heads):
        """Per-kv-head prefill attention mass [H_kv, plen], cached per layer.

        Falls back to spreading the head-summed scores evenly across heads
        when the bank has no per-head data (older stash).
        """
        ph = getattr(layer, "_hh_prefill_scores_per_head", None)
        if ph is None:
            ph = self.score_bank.scores_per_head.get(layer_idx) if self.score_bank else None
            if ph is None:
                summed = getattr(layer, "_hh_prefill_scores", None)
                if summed is None:
                    bank_scores = self.score_bank.scores.get(layer_idx) if self.score_bank else None
                    if bank_scores is None:
                        return None
                    summed = bank_scores.to(device)
                    layer._hh_prefill_scores = summed
                ph = (summed / n_heads).unsqueeze(0).expand(n_heads, -1).contiguous()
            else:
                ph = ph.to(device)
            layer._hh_prefill_scores_per_head = ph
        return ph

    def _fold_evicted_values(self, layer, layer_idx, hh_values, middle_values,
                             topk, middle_orig):
        """Fold evicted middle tokens' values into their nearest kept successor.

        Convex (mass-weighted) folding, NOT additive: for each kept slot t,

            V_t' = (p_t * V_t + sum_i p_i * V_i) / (p_t + sum_i p_i)

        where i ranges over the evicted tokens folded into t. V_t' is a
        weighted average, so magnitudes stay bounded — an earlier additive
        variant (min(1, p_i/p_t) per token, up to ~8 tokens folded per slot)
        inflated kept values 2-9x and caused repetition loops + flipped
        correct turns into loops (measured 2026-09-10).

        Each evicted token folds into the first kept token at a later
        original position (fallback: the last kept token); causally later
        queries look backward.

        hh_values: [B, H, hh, D] gathered heavy-hitter values (fresh tensor,
        safe to modify in place). topk: [B, hh] middle-relative kept indices,
        sorted ascending. middle_orig: [M] original prefill positions of the
        middle region. middle_values: [B, H, M, D] (read-only view).
        """
        B, H, hh, D = hh_values.shape
        M = middle_values.shape[2]
        kept = topk[0]  # [hh], ascending
        ev_mask = torch.ones(M, dtype=torch.bool, device=middle_values.device)
        ev_mask[kept] = False
        ev_pos = ev_mask.nonzero(as_tuple=True)[0]  # [E]
        if ev_pos.numel() == 0:
            return hh_values
        ph = self._per_head_scores(layer, layer_idx, middle_values.device, H)
        if ph is None:
            return hh_values
        # Clamp every index against its table: the score table comes from the
        # prefill snapshot and any length mismatch must never turn into a CUDA
        # device-side assert.
        kept = kept.clamp(max=M - 1).long()
        ev_pos = ev_pos.clamp(max=M - 1).long()
        # First kept slot whose middle position is >= evicted position; an
        # evicted token folds FORWARD (causally later queries look backward).
        tgt_slot = torch.searchsorted(kept, ev_pos, right=True).clamp(max=hh - 1).long()
        ev_orig = middle_orig[ev_pos].clamp(max=ph.shape[-1] - 1).long()           # [E]
        tgt_orig = middle_orig[kept[tgt_slot]].clamp(max=ph.shape[-1] - 1).long()  # [E]
        own_orig = middle_orig[kept].clamp(max=ph.shape[-1] - 1).long()            # [hh]

        # All folding in fp32; result cast back to the cache dtype.
        p_ev = ph[:, ev_orig].float()    # [H, E]
        p_own = ph[:, own_orig].float()  # [H, hh]

        num = hh_values.float() * p_own.unsqueeze(0).unsqueeze(-1)  # [B, H, hh, D]
        num.index_add_(
            2, tgt_slot,
            p_ev.unsqueeze(0).unsqueeze(-1) * middle_values[:, :, ev_pos, :].float(),
        )
        den = p_own.clone()  # [H, hh]
        den.index_add_(1, tgt_slot, p_ev)
        folded = num / den.clamp(min=1e-8).unsqueeze(0).unsqueeze(-1)
        if not getattr(layer, "_hh_merge_logged", False):
            layer._hh_merge_logged = True
            print(f"[heavy_hitter] layer {layer_idx}: convex-folded {ev_pos.numel()} "
                  f"evicted values into {hh} kept slots")
        return folded.to(hh_values.dtype)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        layer = self.layers[layer_idx]
        if not hasattr(layer, "keys"):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # Only take over layers that hold a standard 4-D KV state. v5 hybrid
        # caches can route other state (recurrent/conv placeholders) through
        # update as well; those must go through the base class untouched.
        standard_kv = (
            torch.is_tensor(layer.keys) and layer.keys.dim() == 4
            and torch.is_tensor(layer.values) and layer.values.dim() == 4
        )
        if not standard_kv:
            if layer.is_initialized and not getattr(layer, "_hh_bypass_warned", False):
                print(f"[heavy_hitter] layer {layer_idx}: non-4D KV state "
                      f"(keys shape={getattr(layer.keys, 'shape', None)}, "
                      f"type={type(layer.keys).__name__}), delegating to base cache")
                layer._hh_bypass_warned = True
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)

        # Align cached tensors with the incoming states (multi-GPU device_map safety).
        if layer.keys.device != key_states.device:
            layer.keys = layer.keys.to(key_states.device)
            layer.values = layer.values.to(key_states.device)
            if getattr(layer, "_hh_orig_idx", None) is not None:
                layer._hh_orig_idx = layer._hh_orig_idx.to(key_states.device)
            if getattr(layer, "_hh_prefill_scores", None) is not None:
                layer._hh_prefill_scores = layer._hh_prefill_scores.to(key_states.device)
            if getattr(layer, "_hh_prefill_scores_per_head", None) is not None:
                layer._hh_prefill_scores_per_head = layer._hh_prefill_scores_per_head.to(key_states.device)
            if layer_idx in self._k_meta:
                self._k_meta[layer_idx] = tuple(t.to(key_states.device) for t in self._k_meta[layer_idx])
            if layer_idx in self._v_meta:
                self._v_meta[layer_idx] = tuple(t.to(key_states.device) for t in self._v_meta[layer_idx])

        # Stored state is quantized after the first post-eviction decode step;
        # dequantize back to bf16 for this step's concat/eviction/fold math.
        if layer_idx in self._k_meta:
            layer.keys = dequantize_per_channel(layer.keys, *self._k_meta[layer_idx])
        if layer_idx in self._v_meta:
            layer.values = dequantize_per_token(layer.values, *self._v_meta[layer_idx])

        incoming_len = key_states.shape[-2]
        prev_len = layer.keys.shape[-2]
        keys = torch.cat([layer.keys, key_states], dim=-2)
        values = torch.cat([layer.values, value_states], dim=-2)

        # Track original prefill positions of cached keys. After eviction the
        # current positions no longer match prefill positions, so importance
        # scores are looked up through this index.
        orig_idx = getattr(layer, "_hh_orig_idx", None)
        if orig_idx is None:
            orig_idx = torch.arange(prev_len, device=keys.device)
        orig_idx = torch.cat([
            orig_idx,
            torch.arange(prev_len, prev_len + incoming_len, device=keys.device),
        ])

        total_len = keys.shape[-2]

        if self.importance_mode == "attn_score" and incoming_len > 1:
            # Prefill: the attention scores for this very sequence only exist
            # after attention runs, so we cannot evict yet. Grow the cache and
            # defer compression to the first decode step.
            layer.keys = keys
            layer.values = values
            layer._hh_orig_idx = orig_idx
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
                pos_scores = self._position_scores(
                    layer, layer_idx, orig_idx, keys.device,
                )
                if pos_scores is not None:
                    scores = pos_scores[sink_n:sink_n + middle_len]
                    if self.obs_window > 0:
                        # SnapKV: tokens inside the observation window are
                        # mostly chat-template tokens that attract sink-like
                        # attention; they are already protected by recency, so
                        # exclude them from heavy-hitter candidacy instead of
                        # letting them eat the whole hh budget.
                        plen = layer._hh_prefill_scores.shape[-1]
                        cand_orig = orig_idx[sink_n:sink_n + middle_len]
                        scores = torch.where(
                            cand_orig >= plen - self.obs_window,
                            torch.full_like(scores, float("-inf")), scores,
                        )
                else:
                    # Fallback: key-L2-norm proxy. Loud about it: silently
                    # degrading to the proxy would produce results
                    # indistinguishable from the key_norm variant.
                    if not getattr(layer, "_hh_fallback_warned", False):
                        print(f"[heavy_hitter] layer {layer_idx}: no prefill "
                              f"attention scores available, falling back to key-norm")
                        layer._hh_fallback_warned = True
                    scores = self._importance_scores(middle_keys)[0]
                scores = scores.unsqueeze(0).expand(B, -1)  # [B, M]
                topk = scores.topk(hh_budget, dim=-1).indices  # [B, hh_budget]
                topk, _ = topk.sort(dim=-1)  # maintain temporal order

                # Keep original-position index in sync with the compressed keys.
                middle_orig = orig_idx[sink_n:middle_end]

                # Gather heavy hitters: [B, H, hh_budget, D]
                topk_expanded = topk.unsqueeze(1).unsqueeze(-1).expand(B, H, hh_budget, D)
                hh_keys = torch.gather(middle_keys, dim=2, index=topk_expanded)
                hh_values = torch.gather(middle_values, dim=2, index=topk_expanded)

                if self.merge_evicted and pos_scores is not None:
                    hh_values = self._fold_evicted_values(
                        layer, layer_idx, hh_values, middle_values,
                        topk, middle_orig,
                    )

                sel_orig = torch.gather(
                    middle_orig.unsqueeze(0).expand(B, -1), 1, topk,
                )[0]  # batch is always 1 in this eval
                if not getattr(layer, "_hh_selection_logged", False):
                    layer._hh_selection_logged = True
                    plen = (layer._hh_prefill_scores.shape[-1]
                            if getattr(layer, "_hh_prefill_scores", None) is not None
                            else -1)
                    in_win = (sel_orig >= plen - self.obs_window).sum().item() if plen > 0 else -1
                    print(f"[heavy_hitter] layer {layer_idx}: first eviction "
                          f"plen={plen} kept={sel_orig.numel()} "
                          f"in_obs_window={in_win} "
                          f"min_pos={sel_orig.min().item()} max_pos={sel_orig.max().item()}")
                orig_idx = torch.cat([
                    orig_idx[:sink_n], sel_orig, orig_idx[-recent_n:],
                ])

                keys = torch.cat([sink_keys, hh_keys, recent_keys], dim=-2)
                values = torch.cat([sink_values, hh_values, recent_values], dim=-2)
            else:
                if hh_budget <= 0 and not getattr(layer, "_hh_nobudget_warned", False):
                    print(f"[heavy_hitter] layer {layer_idx}: hh_budget=0 "
                          f"(budget={budget}, sink={sink_n}, recent={recent_n}), "
                          f"eviction disabled — cache grows unbounded, "
                          f"result will equal baseline")
                    layer._hh_nobudget_warned = True
                keys = torch.cat([sink_keys, middle_keys, recent_keys], dim=-2)
                values = torch.cat([sink_values, middle_values, recent_values], dim=-2)

        layer.keys = keys
        layer.values = values
        layer._hh_orig_idx = orig_idx
        keys, values = self._maybe_quantize(layer, layer_idx, keys, values, incoming_len)
        return keys, values

    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys = layer.keys.to(device)
            if hasattr(layer, "values") and layer.values is not None:
                layer.values = layer.values.to(device)
            if getattr(layer, "_hh_prefill_scores", None) is not None:
                layer._hh_prefill_scores = layer._hh_prefill_scores.to(device)
            if getattr(layer, "_hh_orig_idx", None) is not None:
                layer._hh_orig_idx = layer._hh_orig_idx.to(device)
        return self
