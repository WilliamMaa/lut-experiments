#!/usr/bin/env python3
"""Bitwise check: HF's materialized repeat_kv path vs 5-D stride-0 expanded
GQA layouts, on identical inputs, WITHOUT forcing a backend (mirrors runtime
selection — a silent math fallback would explode and is caught by peak).

torch 2.6 context: efficient rejects enable_gqa for dense inputs with a mask,
and the naive 5-D call (mask [B,1,1,q,k]) dies in the dispatcher with a 6-D
broadcast error. Variants left: pre-expand the mask stride-0 to
  v2 [B, H_kv, n_rep, q, k]  or  v3 [B, H_kv, 1, q, k]
Head pairing is identical to repeat_kv in all cases: head (a,b)=a*n_rep+b
reads kv-head a == interleaved head i reading kv i//n_rep.

Green-light criteria: a variant that is bitwise_equal, fallback_warn=0, and
peak far below the repeat path. _sdpa_stash may then use it under
SERVE_NO_GQA_COPY=1.
"""
import warnings
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dt = torch.bfloat16

SHAPES = [(8, 16, 2, 952, 65536),
          (4, 16, 2, 1032, 60888),
          (2, 16, 2, 1, 60888)]

for B, H, H_kv, q, k in SHAPES:
    qh = torch.randn(B, H, q, 256, dtype=dt, device="cuda")
    kh = torch.randn(B, H_kv, k, 256, dtype=dt, device="cuda")
    vh = torch.randn_like(kh)
    m = torch.zeros(B, 1, q, k, dtype=dt, device="cuda")
    m[..., : k // 2] = torch.finfo(dt).min
    n = H // H_kv
    kr = kh[:, :, None, :, :].expand(B, H_kv, n, k, 256).reshape(B, H, k, 256)
    vr = vh[:, :, None, :, :].expand(B, H_kv, n, k, 256).reshape(B, H, k, 256)
    o1 = F.scaled_dot_product_attention(qh, kr, vr, attn_mask=m)
    torch.cuda.synchronize()
    peak1 = torch.cuda.max_memory_allocated() / 2**30
    q5 = qh.view(B, H_kv, n, q, 256)
    k5 = kh[:, :, None, :, :].expand(B, H_kv, n, k, 256)
    v5 = vh[:, :, None, :, :].expand(B, H_kv, n, k, 256)
    variants = {
        "v2_mask_full": m[:, None, None].expand(B, H_kv, n, q, k),
        "v3_mask_hkv": m[:, None, None].expand(B, H_kv, 1, q, k),
    }
    print(f"--- B={B} q={q} k={k} peak_repeat={peak1:.1f}GiB ---", flush=True)
    for name, m5 in variants.items():
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.cuda.reset_peak_memory_stats()
                o2 = F.scaled_dot_product_attention(q5, k5, v5, attn_mask=m5)
                torch.cuda.synchronize()
                peak2 = torch.cuda.max_memory_allocated() / 2**30
                fb = [str(x.message)[:90] for x in w
                      if "kernel not used" in str(x.message)
                      or "not available" in str(x.message)]
            o2 = o2.reshape(B, H, q, 256)
            d = (o1.float() - o2.float()).abs().max().item()
            eq = bool((o1 == o2).all().item())
            print(f"{name}: max|diff|={d} bitwise_equal={eq} "
                  f"peak={peak2:.1f}GiB fallback_warn={len(fb)}", flush=True)
            for line in fb:
                print(f"    warn: {line}", flush=True)
            del o2
        except Exception as e:
            print(f"{name}: FAIL {str(e)[:120]}", flush=True)
        torch.cuda.empty_cache()
    del qh, kh, vh, m, kr, vr, q5, k5, v5, o1
    torch.cuda.empty_cache()
