#!/usr/bin/env python
"""Microbenchmark: shared_expert dense FFN (GEMM) vs flattened LUT lookup.

Purpose (docs/40): predict, with ONE number, whether the v6 LUT line can
ever beat the GPU-native GEMM it replaces on Qwen3.6-35B-A3B serving.
Runs on the remote GPU box (torch only, no transformers/vllm import).

What it measures, per batch size N (decode N=1..64, prefill N=8192):
  dense : gate/up GEMM + SiLU + down GEMM (real module dims from config,
          random weights -- values do not affect GEMM speed)
  naive : v6-as-is lookup: bit-serial tree traversal in Python loops
          (the current v6 engine's structure, lower bound of badness)
  flat  : the "flattened ideal" this project could ship: batched tree
          traversal (14/16 sequential levels, one tensor op per level over
          all N x 33 trees) + one fused row gather per table
  graph : P1.1 experiment (v8/docs/43): the flat path captured in a single
          CUDA graph -- if naive~=flat but graph<<flat, the 55ms floor was
          pure kernel-launch dispatch and no triton is needed yet

Tables are sized exactly like v6's L37 checkpoint (~320 MiB fp16):
  coarse    [16384, 2048]      (14-bit shared tree, full output dim)
  residual  [32, 65536, 64]    (16-bit per-group tree, 2048 = 32 g x 64 ch)

Usage (remote, vllm_py310):
  python v6/scripts/benchmarks/micro_lut_vs_gemm.py \
      --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F


def load_dims(model_path: str):
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    hidden = int(text.get("hidden_size", 2048))
    inter = int(text.get("shared_expert_intermediate_size",
                         text.get("moe_intermediate_size", 1024)))
    return hidden, inter


class Tree:
    """Oblique binary tree: internal node = (random proj of a random input
    channel, threshold). Bit-serial traversal is what v6 does; we also
    provide a flattened level-by-level batched traversal."""

    def __init__(self, depth: int, dim: int, device, dtype=torch.float32):
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.ch = torch.randint(0, dim, (self.n_internal,), device=device)
        self.w = torch.randn(self.n_internal, device=device, dtype=dtype)
        self.b = torch.randn(self.n_internal, device=device, dtype=dtype)

    def naive_indices(self, x):  # x [N, dim] -> leaf idx [N]
        node = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        ar = torch.arange(x.shape[0], device=x.device)
        for lvl in range(self.depth):
            sel = node * 2
            # at the last level the children are leaves: no params exist
            # for them, clamp the param gather (the child index itself is
            # computed from unclamped sel and stays correct)
            pc = sel.clamp(max=self.n_internal - 1)
            c = self.ch[pc]
            go_right = (x[ar, c] * self.w[pc] + self.b[pc]) > 0
            node = sel + 1 + go_right.long()
        return node - self.n_internal  # leaf index [0, 2^d)

    def flat_indices(self, x):  # batched over N: same math, tensor ops
        N = x.shape[0]
        ar = torch.arange(N, device=x.device)
        node = torch.zeros(N, dtype=torch.long, device=x.device)
        for lvl in range(self.depth):
            sel = node * 2
            pc = sel.clamp(max=self.n_internal - 1)
            c = self.ch[pc]
            go_right = (x[ar, c] * self.w[pc] + self.b[pc]) > 0
            node = sel + 1 + go_right.long()
        return node - self.n_internal


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def make_graphed(fn, example):
    """CUDA-graph capture of fn (static shapes, no CPU sync inside).

    Returns a zero-arg replay callable. This is the P1.1 experiment from
    v8/docs/43: naive ~= flat ~= 55ms screams launch-overhead floor, so
    capturing the whole flat path in one graph should collapse dispatch
    cost. If replay is still slow, the time is in the kernels themselves
    (memory-bound gathers) and only a triton rewrite (P1.2) can help.
    """
    static_in = example.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(static_in)  # captured; output buffer owned by the graph
    return graph.replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--Ns", default="1,8,16,32,64,8192")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    hidden, inter = load_dims(args.model_path)
    print(f"[dims] hidden={hidden} shared_expert_inter={inter}")

    torch.manual_seed(0)
    w_gate = torch.randn(inter, hidden, device=device, dtype=dtype)
    w_up = torch.randn(inter, hidden, device=device, dtype=dtype)
    w_down = torch.randn(hidden, inter, device=device, dtype=dtype)

    # LUT tables, v6-L37 sizes (fp16)
    G, CH = 32, 64
    coarse = torch.randn(16384, hidden, device=device, dtype=torch.float16)
    resid = torch.randn(G, 65536, CH, device=device, dtype=torch.float16)
    coarse_t = Tree(14, hidden, device)
    resid_ts = [Tree(16, hidden, device) for _ in range(G)]

    def dense(x):
        h = F.silu(x @ w_gate.t()) * (x @ w_up.t())
        return h @ w_down.t()

    def lut_naive(x):
        out = coarse[coarse_t.naive_indices(x.float())]
        for g in range(G):
            idx = resid_ts[g].naive_indices(x.float())
            out[:, g * CH:(g + 1) * CH] += resid[g][idx]
        return out

    def lut_flat(x):
        xf = x.float()
        out = coarse[coarse_t.flat_indices(xf)]
        # per-group indices stacked as one batched traversal
        idxs = torch.stack([t.flat_indices(xf) for t in resid_ts])  # [G, N]
        rows = resid[torch.arange(G, device=device).unsqueeze(1), idxs]
        return out + rows.transpose(0, 1).reshape(x.shape[0], -1)

    print(f"{'N':>6} {'dense ms':>10} {'naive ms':>10} {'flat ms':>10} "
          f"{'graph ms':>10} {'graph/dense':>12}")
    verdict = []
    for N in [int(n) for n in args.Ns.split(",")]:
        x = torch.randn(N, hidden, device=device, dtype=dtype)
        d = bench(lambda: dense(x), args.iters)
        n = bench(lambda: lut_naive(x), max(10, args.iters // 10))
        f = bench(lambda: lut_flat(x), args.iters)
        replay = make_graphed(lut_flat, x)
        g = bench(replay, args.iters)
        ratio = f / d
        gratio = g / d
        verdict.append((N, d, n, f, ratio, g, gratio))
        print(f"{N:>6} {d:>10.4f} {n:>10.4f} {f:>10.4f} "
              f"{g:>10.4f} {gratio:>11.2f}x")
        del x, replay
        torch.cuda.empty_cache()

    worst = max(v[4] for v in verdict if v[0] <= 64)
    worst_g = max(v[6] for v in verdict if v[0] <= 64)
    print(f"\n[verdict] worst flat/dense at decode sizes: {worst:.2f}x")
    print(f"[verdict] worst graph/dense at decode sizes: {worst_g:.2f}x")
    print("[verdict P1.1] graph~flat  => launch-bound confirmed; time is in "
          "the kernels themselves, triton (P1.2) required.")
    print("[verdict P1.1] graph<<flat => dispatch was the floor; LUT "
          "execution may be production-viable without triton.")
    print("[verdict] route viable ONLY if 'graph' (or a fused kernel) beats "
          "'flat' significantly; if even 'graph' is >2x slower than dense, "
          "kernelization cannot save it.")


if __name__ == "__main__":
    main()
