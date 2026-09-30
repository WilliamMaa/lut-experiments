#!/usr/bin/env python3
"""Bitwise check: HF's materialized repeat_kv path vs a 5-D stride-0 expanded
GQA layout, on identical inputs, WITHOUT forcing a backend (mirrors runtime
selection — a silent math fallback would explode and is caught by peak).

Context: the harness's high-N OOM wall is dominated by HF's repeat_kv GQA
copies (2*B*H*k*D bf16, ~28GB at N=32/64k). torch 2.6's efficient kernel
rejects enable_gqa=True for dense inputs with an attn_mask, but the rejection
message itself suggests unsqueeze+expand: give the kernel 5-D views
  q5 [B, H_kv, n_rep, L, E]   (view of the materialized q — free)
  k5/v5 [B, H_kv, n_rep, S, E] (stride-0 expand — free)
  m5  [B, 1, 1, L, S]          (stride-0 expand of the 4D additive mask)
Head pairing matches repeat_kv exactly: head (a, b) = a*n_rep+b reads kv-head
a, same as interleaved head i reading kv i//n_rep. If bitwise equal and no
backend fallback, _sdpa_stash can use this layout under SERVE_NO_GQA_COPY=1
and skip the copies entirely.
"""
import warnings
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dt = torch.bfloat16

# (B, H, H_kv, q_len, k_len): prefill chunk, mid prefill, decode step
SHAPES = [(8, 16, 2, 952, 65536),
          (4, 16, 2, 1032, 60888),
          (2, 16, 2, 1, 60888)]

for B, H, H_kv, q, k in SHAPES:
    qh = torch.randn(B, H, q, 256, dtype=dt, device="cuda")
    kh = torch.randn(B, H_kv, k, 256, dtype=dt, device="cuda")
    vh = torch.randn_like(kh)
    m = torch.zeros(B, 1, q, k, dtype=dt, device="cuda")
    m[..., : k // 2] = torch.finfo(dt).min  # exercise the masked region
    n = H // H_kv
    kr = kh[:, :, None, :, :].expand(B, H_kv, n, k, 256).reshape(B, H, k, 256)
    vr = vh[:, :, None, :, :].expand(B, H_kv, n, k, 256).reshape(B, H, k, 256)
    # Path 1: HF — materialize repeat_kv, 4D call (what runs today)
    torch.cuda.reset_peak_memory_stats()
    o1 = F.scaled_dot_product_attention(qh, kr, vr, attn_mask=m)
    torch.cuda.synchronize()
    peak1 = torch.cuda.max_memory_allocated() / 2**30
    # Path 2: 5-D stride-0 views, no backend forcing (mirror runtime)
    q5 = qh.view(B, H_kv, n, q, 256)
    k5 = kh[:, :, None, :, :].expand(B, H_kv, n, k, 256)
    v5 = vh[:, :, None, :, :].expand(B, H_kv, n, k, 256)
    m5 = m[:, None, None]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.reset_peak_memory_stats()
        o2 = F.scaled_dot_product_attention(q5, k5, v5, attn_mask=m5)
        torch.cuda.synchronize()
        peak2 = torch.cuda.max_memory_allocated() / 2**30
        fb = [str(x.message)[:80] for x in w
              if "kernel not used" in str(x.message)
              or "not available" in str(x.message)]
    o2 = o2.reshape(B, H, q, 256)
    d = (o1.float() - o2.float()).abs().max().item()
    eq = bool((o1 == o2).all().item())
    print(f"B={B} q={q} k={k}: max|diff|={d} bitwise_equal={eq} "
          f"peak_repeat={peak1:.1f}GiB peak_5d={peak2:.1f}GiB "
          f"fallback_warn={len(fb)}", flush=True)
    for line in fb:
        print(f"    warn: {line}", flush=True)
    del qh, kh, vh, m, kr, vr, o1, q5, k5, v5, m5, o2
    torch.cuda.empty_cache()
