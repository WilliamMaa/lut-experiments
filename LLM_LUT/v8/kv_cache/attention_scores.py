#!/usr/bin/env python3
"""Attention-score collection for importance-aware KV cache eviction.

Provides an "eager" attention kernel (Q@K^T -> mask -> softmax -> @V, with
GQA head replication) that additionally accumulates, per full-attention
layer, the total attention mass each key position receives during prefill
(column sum of attention probabilities over heads and query positions).

This is the H2O/SnapKV-style importance signal, replacing the key-L2-norm
proxy (which cannot see *which* keys the queries actually attend to).

Why a self-contained kernel: transformers v5 registers implementations
lazily and "eager" is absent from ALL_ATTENTION_FUNCTIONS at patch time;
importing the internal kernel is not reliable across versions. The math is
identical to the reference eager attention.
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


def _hh_eager_attention(module, query, key, value, attention_mask,
                        dropout=0.0, scaling=None, sliding_window=None,
                        head_mask=None, **kwargs):
    """Eager attention with GQA replication; returns (output, attn_weights).

    Importance stash: during prefill (q_len > 1) we column-sum the softmax
    probabilities and store them per layer in the shared bank.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    # GQA: replicate k/v heads to match query heads.
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if isinstance(attention_mask, dict):
        # v5-style packed/boolean mask dict: True means "keep".
        bool_mask = attention_mask.get("bool_mask")
        if bool_mask is not None:
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(~bool_mask, min_val)
    elif attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(~attention_mask, min_val)
        else:
            # Additive float mask (0 / -inf), possibly 2D or 4D.
            attn_weights = attn_weights + attention_mask[..., : key.shape[-2]]
    else:
        # No mask provided: apply causal mask ourselves (prefill case).
        q_len, k_len = query.shape[-2], key.shape[-2]
        if q_len > 1:
            min_val = torch.finfo(attn_weights.dtype).min
            causal = torch.full((q_len, k_len), min_val,
                                device=attn_weights.device, dtype=attn_weights.dtype)
            causal = torch.triu(causal, diagonal=1 + (k_len - q_len))
            attn_weights = attn_weights + causal

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    if dropout and dropout > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout,
                                                   training=module.training)
    attn_output = torch.matmul(attn_weights.to(value.dtype), value)
    return attn_output, attn_weights


def _registry_target():
    reg = modeling_utils.ALL_ATTENTION_FUNCTIONS
    return reg._global_mapping if hasattr(reg, "_global_mapping") else reg


class _InstallState:
    prev_impl = None
    prev_eager = None
    installed = False


def install_eager_score_stash(model, bank):
    """Route the model's full-attention layers through the stashing kernel."""
    if getattr(model.config, "sliding_window", None):
        raise RuntimeError(
            "model uses sliding_window attention; the stashing eager kernel "
            "does not handle sliding windows. Aborting instead of running silently wrong."
        )
    target = _registry_target()
    _InstallState.prev_eager = target.get("eager")  # None on transformers v5
    target["eager"] = _stashing_wrapper
    _StashState.bank = bank
    _InstallState.prev_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    _InstallState.installed = True
    print("[heavy_hitter_attn] eager kernel installed "
          f"(transformers eager registered: {_InstallState.prev_eager is not None})")


def uninstall_eager_score_stash(model):
    if not _InstallState.installed:
        return
    _StashState.bank = None
    target = _registry_target()
    if _InstallState.prev_eager is None:
        target.pop("eager", None)
    else:
        target["eager"] = _InstallState.prev_eager
    model.config._attn_implementation = _InstallState.prev_impl
    _InstallState.installed = False


# Attach the stash logic to the kernel via a thin wrapper registered below.
# (Kept separate so the kernel stays readable.)
_ORIG_KERNEL = _hh_eager_attention


def _stashing_wrapper(module, query, key, value, attention_mask, *args, **kwargs):
    out, attn_weights = _ORIG_KERNEL(module, query, key, value, attention_mask,
                                     *args, **kwargs)
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
