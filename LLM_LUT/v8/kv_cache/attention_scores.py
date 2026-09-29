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
    """Per-layer per-position accumulated attention mass from the last prefill.

    scores:           layer_idx -> [batch, seq_len] float32, summed over all
                      q-heads and obs-window rows (used for eviction
                      selection). Batch-indexed: each concurrent session keeps
                      its own scores so per-session selection is independent.
    scores_per_head:  layer_idx -> [batch, kv_heads, seq_len] float32, summed
                      over the q-heads each kv-head serves and over obs-window
                      rows (used for per-head merge weights).
    """

    def __init__(self):
        self.scores = {}
        self.scores_per_head = {}

    def clear(self):
        self.scores.clear()
        self.scores_per_head.clear()


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
    q = query[..., -w:, :].detach().float()      # [B, H, W, D]
    k = key.detach().float()                     # [B, H_kv, K, D]
    if q.ndim != 4 or k.ndim != 4 or q.shape[1] % k.shape[1] != 0:
        # Exotic sdpa caller (e.g. the linear-attention torch fallback
        # reaches the registry with 5-D inputs) — NOT a KV-cache attention
        # layer. The old full-tensor code silently stored a wrongly-shaped
        # [B, W, K] "score" bank for these; skip instead of crashing.
        print(f"[attn_scores] skip layer {layer_idx}: non-standard q/k dims "
              f"{tuple(query.shape)}/{tuple(key.shape)}")
        return
    B, H, _, _ = q.shape
    H_kv = k.shape[1]
    n_rep = H // H_kv

    # Per-q-head accumulation: the naive repeat_interleave + full softmax
    # materializes B*H*W*K fp32 TWICE (scores + probs) — >100GB at N=64/128k,
    # OOMing before eviction ever runs. Looping heads keeps only [B, W, K]
    # fp32 alive at a time (~1-2GB). Sum order over heads differs from a
    # single tensor sum only by float associativity.
    total = torch.zeros(B, k_len, device=q.device, dtype=torch.float32)
    per_head = torch.zeros(B, H_kv, k_len, device=q.device, dtype=torch.float32)
    if torch.is_tensor(attention_mask):
        if attention_mask.dtype == torch.bool:
            keep = attention_mask[..., -w:, :k_len].bool()      # [B, W, K]
            add = None
        else:
            keep = None
            add = attention_mask[..., -w:, :k_len].float()      # [B, W, K]
    else:
        keep = None
        add = _causal_rows(w, k_len, q.device, torch.float32)[None]
    for h in range(H):
        s = torch.matmul(q[:, h], k[:, h // n_rep].transpose(-1, -2)) * scaling
        if keep is not None:
            s = s.masked_fill(~keep, torch.finfo(s.dtype).min)
        elif add is not None:
            s = s + add
        p = torch.softmax(s, dim=-1)     # [B, W, K], freed before next head
        ph = p.sum(dim=1)                # [B, K]
        if ph.shape != total.shape:
            print(f"[attn_scores] SKIP layer {layer_idx}: shape mismatch "
                  f"query{tuple(query.shape)} key{tuple(key.shape)} w={w} "
                  f"k_len={k_len} B={B} H={H} H_kv={H_kv} "
                  f"total{tuple(total.shape)} ph{tuple(ph.shape)}")
            return
        total += ph                      # [B, K]
        per_head[:, h // n_rep] += ph
        del s, p, ph
    bank.scores[layer_idx] = total
    bank.scores_per_head[layer_idx] = per_head


def _sdpa_stash(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, **kwargs):
    _stash(module, query, key, value, attention_mask, scaling)
    # Delegate to the exact function the baseline run uses: identical call,
    # identical backend, identical rounding. Any hand-rolled sdpa call
    # (enable_gqa layout, is_causal, scale passing) diverges from it and the
    # drift compounds across 40 bf16 layers (measured max logit diff 16-17).
    return _InstallState.prev_sdpa(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, **kwargs,
    )


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
    if _InstallState.installed:
        # Idempotent re-install: just re-point the bank. Re-wrapping would
        # make prev_sdpa the wrapper itself and recurse infinitely (hit when
        # two HeavyHitterAttnScorePatch instances install in one process,
        # e.g. icn_proto's per-representation factories).
        _StashState.bank = bank
        return
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
