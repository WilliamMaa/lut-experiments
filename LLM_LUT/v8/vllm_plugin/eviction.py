#!/usr/bin/env python3
"""v8 heavy-hitter eviction, ported to pure functions from
kv_cache/heavy_hitter_cache.py (attn_score + convex-fold variant).

Single-request versions: the vLLM impl loops requests, so no batch dim.

DO NOT "improve": additive folding inflated values 2-9x and caused
repetition loops (2026-09-10); CUDA topk is not run-to-run stable
(2026-09-12); index_add_ atomics flip near-tie decode tokens
(2026-09-15). All three are forbidden here.
"""
import torch

NEG_INF = float("-inf")


def obs_window_scores(q_lastW, k_comp, k_chunk, scale, causal_mask):
    """Column attention mass for the last W query rows of a prefill chunk.

    q_lastW: [H, W, D] fp32 (detached). k_comp: [H_kv, L, D] or None.
    k_chunk: [H_kv, C, D]. causal_mask: additive [W, C] fp32 (0 / -inf),
    applied to chunk columns only; compact columns are unmasked (invariant:
    prefill evicts BEFORE writing the chunk, so every compact key predates
    every chunk query and is causally visible to all of them).
    Returns total [L+C] fp32, per_head [H_kv, L+C] fp32.

    Loops q-heads, keeping only [W, L+C] fp32 alive at a time
    (attention_scores.py lesson: single-tensor B*H*W*K fp32 >100GB at
    N=64/128k).
    """
    H, W, _ = q_lastW.shape
    L = 0 if k_comp is None else k_comp.shape[1]
    C = k_chunk.shape[1]
    H_kv = k_chunk.shape[0]
    n_rep = H // H_kv
    k_all = k_chunk if k_comp is None else torch.cat([k_comp, k_chunk], dim=1)
    device = q_lastW.device
    total = torch.zeros(L + C, device=device, dtype=torch.float32)
    per_head = torch.zeros(H_kv, L + C, device=device, dtype=torch.float32)
    for h in range(H):
        s = torch.matmul(q_lastW[h], k_all[h // n_rep].transpose(-1, -2)) * scale
        if C > 0:
            s[:, L:] += causal_mask
        p = torch.softmax(s, dim=-1)
        ph = p.sum(dim=0)
        total += ph
        per_head[h // n_rep] += ph
        del s, p, ph
    return total, per_head


def make_causal_add(w, c, device):
    """Additive causal rows [w, c] fp32 for the last w query positions."""
    rows = torch.arange(c - w, c, device=device)[:, None]
    cols = torch.arange(c, device=device)[None, :]
    mask = torch.zeros(w, c, device=device, dtype=torch.float32)
    mask.masked_fill_(cols > rows, torch.finfo(torch.float32).min)
    return mask


def select_kept(scores_middle, middle_orig, snap_len, hh_budget,
                obs_window=0, span_window=0):
    """Deterministic heavy-hitter selection (single request).

    scores_middle: [M] fp32. middle_orig: [M] long (original positions).
    Returns kept middle-relative indices [<=hh_budget], ascending.
    """
    scores = scores_middle
    if obs_window > 0:
        # SnapKV: obs-window tokens are chat-template tokens that attract
        # sink-like attention; recency already protects them, exclude from
        # heavy-hitter candidacy.
        scores = torch.where(
            middle_orig >= snap_len - obs_window,
            torch.full_like(scores, NEG_INF), scores)
    if span_window > 0:
        # Max-pool +-span_window so multi-token facts survive/perish as a
        # unit (tokenizers split digits: '178' -> '1','7','8').
        pooled = torch.nn.functional.max_pool1d(
            scores.unsqueeze(0).unsqueeze(0),
            kernel_size=2 * span_window + 1, stride=1,
            padding=span_window).squeeze(0).squeeze(0)
        scores = pooled
    sorted_idx = torch.argsort(scores, dim=-1, descending=True, stable=True)
    hh_budget = max(0, min(hh_budget, sorted_idx.shape[0]))
    kept = sorted_idx[:hh_budget].sort().values
    return kept


def fold_evicted_values(hh_values, middle_values, kept, middle_orig, per_head):
    """Convex mass-weighted folding of evicted V into nearest kept successor.

    hh_values: [H_kv, hh, D] fresh gathered tensor (modified in place ok).
    middle_values: [H_kv, M, D] read-only. kept: [hh] ascending.
    middle_orig: [M] long. per_head: [H_kv, K] fp32 attention-mass table.
    Returns folded [H_kv, hh, D] in hh_values.dtype.

    V_t' = (p_t * V_t + sum_i p_i * V_i) / (p_t + sum_i p_i), i folded into
    the first kept token at a later original position (fallback: last kept).
    """
    H, hh, D = hh_values.shape
    M = middle_values.shape[1]
    device = middle_values.device
    hh = min(hh, M)
    kept = kept.clamp(max=M - 1).long()
    ev_mask = torch.ones(M, dtype=torch.bool, device=device)
    ev_mask[kept] = False
    ev_pos = ev_mask.nonzero(as_tuple=True)[0]
    E = ev_pos.shape[0]
    if E == 0:
        return hh_values
    ev_pos = ev_pos.clamp(max=M - 1).long()
    tgt_slot = torch.searchsorted(kept, ev_pos, right=True).clamp(max=hh - 1)
    tgt_kept = kept[tgt_slot]
    K = per_head.shape[-1]
    ev_orig = middle_orig[ev_pos].clamp(max=K - 1).long()
    tgt_orig = middle_orig[tgt_kept].clamp(max=K - 1).long()
    own_orig = middle_orig[kept].clamp(max=K - 1).long()

    p_ev = per_head[:, ev_orig].float()
    p_own = per_head[:, own_orig].float()
    ev_vals = middle_values[:, ev_pos, :]
    # Deterministic reduction via one-hot matmul — NOT index_add_ (atomics
    # flip near-tie decode tokens across identical runs, 2026-09-15).
    onehot = torch.zeros(E, hh, dtype=torch.float32, device=device)
    onehot[torch.arange(E, device=device), tgt_slot] = 1.0
    contrib = p_ev.unsqueeze(-1) * ev_vals.float()
    num = hh_values[:, :hh].float() * p_own.unsqueeze(-1)
    num = num + torch.einsum("es,hed->hsd", onehot, contrib)
    den = p_own + torch.einsum("es,he->hs", onehot, p_ev)
    folded = num / den.clamp(min=1e-8).unsqueeze(-1)
    out = hh_values.clone()
    out[:, :hh] = folded.to(hh_values.dtype)
    return out
