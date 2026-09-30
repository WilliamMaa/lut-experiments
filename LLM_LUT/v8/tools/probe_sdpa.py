#!/usr/bin/env python3
"""Probe which sdpa backend accepts the harness's chunked-prefill mask.

Context: every additive 4D mask we pass seems to land on the math fallback
backend, which materializes B*H*chunk*K scores on ONE GPU (~64GB at N=32,
64k). This probe forces each backend in turn and reports accept/reject plus
the peak memory each one needs — the smoking gun for the high-N OOMs.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B, H, H_kv, S, D = 8, 16, 2, 65536, 256
CHUNK = 952  # 8-aligned, matches the harness's N=32/64k chunk

q = torch.randn(B, H, CHUNK, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
k = k.repeat_interleave(H // H_kv, dim=1)
v = torch.randn_like(k)
m = torch.zeros(B, 1, CHUNK, S, dtype=torch.bfloat16, device="cuda")

for name, be in [("efficient", SDPBackend.EFFICIENT_ATTENTION),
                 ("flash", SDPBackend.FLASH_ATTENTION),
                 ("math", SDPBackend.MATH)]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{name}: OK out={tuple(o.shape)} peak={peak:.1f}GiB")
    except Exception as e:
        print(f"{name}: FAIL {str(e)[:150]}")
