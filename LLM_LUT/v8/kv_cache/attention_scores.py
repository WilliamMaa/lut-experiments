#!/usr/bin/env python3
"""Attention-score collection via an sdpa wrapper (no math changes).

Replaces the registry entry of the model's active attention implementation
("sdpa") with a wrapper that:

  1. calls torch's scaled_dot_product_attention for the output, so the
     patched forward is numerically identical to baseline. A previous version
     implemented a custom eager kernel; even with float32 internals the
     full-model logits drifted from baseline (measured max diff 16.3, top-1
     agreement 75-80% over 40 bf16 layers) because any rounding difference
     vs sdpa compounds chaotically through MoE routing. Do NOT reintroduce
     custom attention math here.
  2. stashes per-key attention mass from the last prefill's observation
     window (last W query rows), computed in fp32 on the side, for
     importance-aware KV cache eviction (HeavyHitterCache,
     importance_mode="attn_score").

The config's attention implementation is left untouched; only the registry
entry is swapped, so mask creation and backend selection follow the exact
baseline path.
"""

import torch
import torch.nn.functional as F
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
    obs_window = 64  # SnapKV-style observation window (last W query rows)


def set_observation_window(w: int):
    _StashState.obs_window = int(w)


def _causal_rows(w, k_len, device, dtype):
    """Additive causal mask rows for the LAST w query positions of a prefill
    where q_len == k_len (row i at global position k_len - w + i)."""
    rows = torch.arange(k_len - w, k_len, device=device)[:, None]
    cols = torch.arange(k_len, device=device)[None, :]
    keep = cols <= rows  # [W, K]
    mask = torch.zeros(w, k_len, device=device, dtype=dtype)
    mask.masked_fill_(~keep, torch.finfo(dtype).min)
    return mask


def _stash(module, query, key, value, attention_mask, scaling):
    bank = _StashState.bank
    if bank is None or module.training:
        return
    q_len = query.shape[-2]
    if q_len <= 1:
        # Decode step: eviction is driven by the prefill snapshot.
        return
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        return
    w = min(_StashState.obs_window, q_len)
    k_len = key.shape[-2]
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    q = query[..., -w:, :].detach().float()
    k = key.detach().float()
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scaling  # [B, H, W, K]
    if torch.is_tensor(attention_mask):
        if attention_mask.dtype == torch.bool:
            keep = attention_mask[..., -w:, :k_len].bool()
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask[..., -w:, :k_len].float()
    else:
        scores = scores + _causal_rows(w, k_len, scores.device, scores.dtype)[None, None]
    probs = torch.softmax(scores, dim=-1)
    bank.scores[layer_idx] = probs.sum(dim=(0, 1, 2))  # [K] float32


def _sdpa_stash(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, **kwargs):
    _stash(module, query, key, value, attention_mask, scaling)
    if isinstance(attention_mask, dict):
        raise RuntimeError(
            "sdpa stash wrapper received a dict mask; the model's attention "
            "implementation is not the plain sdpa path this wrapper assumes."
        )
    is_causal = attention_mask is None and query.shape[-2] > 1
    try:
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=is_causal,
            scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    except TypeError:  # torch < 2.5: no enable_gqa
        n_rep = query.shape[1] // key.shape[1]
        k, v = key, value
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            query, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=is_causal,
            scale=scaling,
        )
    return out


def _registry_target():
    reg = modeling_utils.ALL_ATTENTION_FUNCTIONS
    return reg._global_mapping if hasattr(reg, "_global_mapping") else reg


class _InstallState:
    prev_sdpa = None
    installed = False


def install_eager_score_stash(model, bank):
    """Wrap the active 'sdpa' registry entry with the score-stashing wrapper.

    The model config is NOT modified: mask creation and backend selection
    follow the exact baseline path, so the patched forward is numerically
    identical to baseline (asserted by inspect_kernel_ab.py).
    """
    impl = model.config._attn_implementation
    target = _registry_target()
    orig = target.get(impl)
    if orig is None:
        raise RuntimeError(
            f"attention implementation '{impl}' is not registered in "
            f"ALL_ATTENTION_FUNCTIONS (transformers {transformers.__version__}). "
            "Registry keys: " + ", ".join(sorted(str(k) for k in target.keys()))
        )
    if impl != "sdpa":
        raise RuntimeError(
            f"score stash wrapper assumes the model uses 'sdpa' (got '{impl}'). "
            "Extend _sdpa_stash to wrap the active implementation instead."
        )
    _InstallState.prev_sdpa = orig
    target["sdpa"] = _sdpa_stash
    _StashState.bank = bank
    _InstallState.installed = True
    print(f"[heavy_hitter_attn] sdpa stash wrapper installed over '{impl}'")


def uninstall_eager_score_stash(model):
    if not _InstallState.installed:
        return
    _StashState.bank = None
    _registry_target()["sdpa"] = _InstallState.prev_sdpa
    _InstallState.prev_sdpa = None
    _InstallState.installed = False
