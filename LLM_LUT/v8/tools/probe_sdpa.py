#!/usr/bin/env python3
"""Probe sdpa backend acceptance + peak memory for the exact chunked-prefill
shapes, to locate the 64k high-concurrency OOM (N=32/N=64 OOM by ~260MiB with
the card otherwise at ~78.6GiB).

Per-backend forced trials at the (B, chunk, k_total) the harness's chunk
formula picks for each concurrency N at 64k, including the GQA repeat_kv copy
(HF's sdpa_attention_forward expands K/V 2->16 heads before calling sdpa —
transient 2*B*16*k*256*2B, chunk-independent: 31.7GB at N=32/64k, 63GB at
N=64/64k).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

H, H_kv, D = 16, 2, 256
S_REAL = 60380  # 64k dataset prefill length (from the run logs)


def repeat_kv(t, n):
    """Exactly transformers.integrations.sdpa_attention_forward.repeat_kv."""
    b, h, s, d = t.shape
    return t[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def trial(B, chunk):
    k_total = -(-S_REAL // chunk) * chunk  # ceil to chunk multiple
    q = torch.randn(B, H, chunk, D, dtype=torch.bfloat16, device="cuda")
    kr = repeat_kv(torch.randn(B, H_kv, k_total, D, dtype=torch.bfloat16,
                               device="cuda"), H // H_kv)
    vr = repeat_kv(torch.randn(B, H_kv, k_total, D, dtype=torch.bfloat16,
                               device="cuda"), H // H_kv)
    m = torch.zeros(B, 1, chunk, k_total, dtype=torch.bfloat16, device="cuda")
    for name, be in [("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("math", SDPBackend.MATH)]:
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, kr, vr, attn_mask=m)
            torch.cuda.synchronize()
            print(f"B={B} chunk={chunk} k={k_total} {name}: OK "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
                  flush=True)
        except Exception as e:
            print(f"B={B} chunk={chunk} k={k_total} {name}: "
                  f"FAIL {str(e)[:110]}", flush=True)
        if "o" in dir():
            del o
        torch.cuda.empty_cache()
    del q, kr, vr, m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"torch={torch.__version__} free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB",
          flush=True)
    # (B, chunk) per the harness's 2e9 activation cap at 64k:
    # N=8 -> 4136, N=16 -> 2064, N=32 -> 1032, N=64 -> 512
    for B, chunk in [(8, 4136), (16, 2064), (32, 1032), (64, 512)]:
        trial(B, chunk)
