#!/usr/bin/env python3
"""Attention-score collection for importance-aware KV cache eviction.

Wraps the eager attention kernel so that during prefill we accumulate, per
full-attention layer, the total attention mass each key position receives
(column sum of attention probabilities over all heads and query positions).
This is the H2O/SnapKV-style importance signal, replacing the key-L2-norm
proxy (which cannot see *which* keys the queries actually attend to).

The eager kernel is imported from its defining module (location differs
across transformers versions) and injected into the attention-function
registry, whose concrete type also differs across versions.
"""

import torch
import transformers.modeling_utils as modeling_utils

try:  # transformers >= 4.56
    from transformers.attention.interface import eager_attention_forward as _EAGER_FN
except Exception:  # transformers 4.48 - 4.55
    from transformers.modeling_utils import eager_attention_forward as _EAGER_FN


class AttentionScoreBank:
    """Per-layer per-position accumulated attention mass from the last prefill."""

    def __init__(self):
        self.scores = {}  # layer_idx -> FloatTensor [seq_len]

    def clear(self):
        self.scores.clear()


class _StashState:
    bank = None


def _stashing_eager(module, query, key, value, attention_mask, *args, **kwargs):
    out, attn_weights = _EAGER_FN(
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


def _registry_set(fn):
    """Install `fn` as the 'eager' attention implementation; return previous."""
    reg = modeling_utils.ALL_ATTENTION_FUNCTIONS
    if hasattr(reg, "_global_mapping"):
        prev = reg._global_mapping.get("eager")
        reg._global_mapping["eager"] = fn
    else:
        prev = reg.get("eager")
        reg["eager"] = fn
    return prev


def _registry_restore(prev):
    reg = modeling_utils.ALL_ATTENTION_FUNCTIONS
    target = reg._global_mapping if hasattr(reg, "_global_mapping") else reg
    if prev is None:
        target.pop("eager", None)
    else:
        target["eager"] = prev


class _InstallState:
    prev_impl = None
    prev_eager = None


def install_eager_score_stash(model, bank):
    """Force eager attention on `model` and stash prefill column sums into `bank`."""
    _StashState.bank = bank
    _InstallState.prev_eager = _registry_set(_stashing_eager)
    _InstallState.prev_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"


def uninstall_eager_score_stash(model):
    _StashState.bank = None
    if _InstallState.prev_eager is not None or True:
        _registry_restore(_InstallState.prev_eager)
    model.config._attn_implementation = _InstallState.prev_impl
    _InstallState.prev_eager = None
    _InstallState.prev_impl = None
