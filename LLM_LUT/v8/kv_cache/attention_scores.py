#!/usr/bin/env python3
"""Attention-score collection for importance-aware KV cache eviction.

Wraps the eager attention kernel so that during prefill we accumulate, per
full-attention layer, the total attention mass each key position receives
(column sum of attention probabilities over all heads and query positions).
This is the H2O/SnapKV-style importance signal, replacing the key-L2-norm
proxy (which cannot see *which* keys the queries actually attend to).

The kernel is resolved at install time (after model load) from the attention
registry, because eager is registered lazily in some transformers versions
and is not importable from a fixed module path.
"""

import torch
import transformers
import transformers.modeling_utils as modeling_utils


class AttentionScoreBank:
    """Per-layer per-position accumulated attention mass from the last prefill."""

    def __init__(self):
        self.scores = {}  # layer_idx -> FloatTensor [seq_len]

    def clear(self):
        self.scores.clear()


class _StashState:
    bank = None
    orig_eager = None


def _stashing_eager(module, query, key, value, attention_mask, *args, **kwargs):
    out, attn_weights = _StashState.orig_eager(
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


def _registry_target():
    reg = modeling_utils.ALL_ATTENTION_FUNCTIONS
    return reg._global_mapping if hasattr(reg, "_global_mapping") else reg


class _InstallState:
    prev_impl = None
    prev_eager = None
    installed = False


def install_eager_score_stash(model, bank):
    """Force eager attention on `model` and stash prefill column sums into `bank`."""
    target = _registry_target()
    orig = target.get("eager")
    if orig is None:
        raise RuntimeError(
            "eager attention kernel is not registered in "
            f"ALL_ATTENTION_FUNCTIONS (transformers {transformers.__version__}). "
            "Registry keys: "
            + ", ".join(sorted(str(k) for k in target.keys()))
        )
    _StashState.bank = bank
    _StashState.orig_eager = orig
    _InstallState.prev_eager = target.get("eager")
    target["eager"] = _stashing_eager
    _InstallState.prev_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    _InstallState.installed = True


def uninstall_eager_score_stash(model):
    if not _InstallState.installed:
        return
    _StashState.bank = None
    _StashState.orig_eager = None
    target = _registry_target()
    if _InstallState.prev_eager is None:
        target.pop("eager", None)
    else:
        target["eager"] = _InstallState.prev_eager
    model.config._attn_implementation = _InstallState.prev_impl
    _InstallState.installed = False
