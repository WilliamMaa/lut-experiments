#!/usr/bin/env python3
"""Attention-score collection for importance-aware KV cache eviction.

Wraps the eager attention kernel so that during prefill we accumulate, per
full-attention layer, the total attention mass each key position receives
(column sum of attention probabilities over all heads and query positions).
This is the H2O/SnapKV-style importance signal, replacing the key-L2-norm
proxy (which cannot see *which* keys the queries actually attend to).
"""

import torch
import transformers.modeling_utils as modeling_utils


class AttentionScoreBank:
    """Per-layer per-position accumulated attention mass from the last prefill."""

    def __init__(self):
        self.scores = {}  # layer_idx -> FloatTensor [seq_len]

    def clear(self):
        self.scores.clear()


class _StashState:
    bank = None


_ORIG_EAGER = modeling_utils.ALL_ATTENTION_FUNCTIONS["eager"]


def _stashing_eager(module, query, key, value, attention_mask, *args, **kwargs):
    out, attn_weights = _ORIG_EAGER(
        module, query, key, value, attention_mask, *args, **kwargs
    )
    bank = _StashState.bank
    if bank is None or module.training:
        return out, attn_weights
    if query.shape[-2] <= 1:
        # Decode step: attention over the compressed cache is not a reliable
        # importance signal and is not needed anyway (eviction is driven by
        # the prefill scores).
        return out, attn_weights
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        return out, attn_weights
    # weights: [batch, heads, q_len, k_len] -> column sum over heads/queries.
    bank.scores[layer_idx] = attn_weights.detach().float().sum(dim=(0, 1, 2))
    return out, attn_weights


def install_eager_score_stash(model, bank):
    """Force eager attention on `model` and stash prefill column sums into `bank`."""
    _StashState.bank = bank
    modeling_utils.ALL_ATTENTION_FUNCTIONS["eager"] = _stashing_eager
    model.config._attn_implementation = "eager"


def uninstall_eager_score_stash(model, prev_impl):
    _StashState.bank = None
    modeling_utils.ALL_ATTENTION_FUNCTIONS["eager"] = _ORIG_EAGER
    model.config._attn_implementation = prev_impl
