#!/usr/bin/env python3
"""Bitwise check: sdpa with materialized repeat_kv (HF's exact path) vs
enable_gqa=True on identical inputs.

Context: the harness's high-N OOM wall is dominated by HF's repeat_kv GQA
copies (2*B*H*k*D bf16, chunk-independent: 28GB at N=32/64k per the memlog
ratchet). enable_gqa=True makes the kernel read K/V strided instead of
materializing the 8x interleaved copy — same math, zero copies. This probe
verifies the two paths are bitwise identical BEFORE the harness is allowed
to bypass repeat_kv (the score-stash wrapper delegates to HF's
sdpa_attention_forward for bit-exactness; any nonzero diff here vetoes it).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.manual_seed(0)
dt = torch.bfloat16

# (B, H, H_kv, q_len, k_len): prefill chunk, bigger prefill, decode step
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
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        o1 = F.scaled_dot_product_attention(qh, kr, vr, attn_mask=m)
        o2 = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m,
                                            enable_gqa=True)
    d = (o1.float() - o2.float()).abs().max().item()
    eq = bool((o1 == o2).all().item())
    print(f"B={B} q={q} k={k}: max|diff|={d} bitwise_equal={eq}", flush=True)
    del qh, kh, vh, m, kr, vr, o1, o2
    torch.cuda.empty_cache()
