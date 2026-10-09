#!/usr/bin/env python3
"""P1.3 numerical test for the shared_expert LUT forward (docs/43).

Builds a small synthetic bundle (hidden=64, 2 groups x 32 channels,
early-stopped random trees in the padded self-looping format produced by
v6/scripts/conversion/prep_lut_bundle.py), then checks the triton kernels
inside LutFfnReplacer against its pure-torch reference path.

CPU anywhere: the reference path is exercised (shape/dtype/plumbing).
Kernel-vs-reference allclose needs triton + CUDA; it SKIPs cleanly where
either is missing (this box has neither — the real comparison runs on
the remote GPU machine).

Run anywhere:
    python vllm_plugin/tests/test_lut_ffn.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm_plugin.lut_ffn import HAS_TRITON, LutFfnReplacer

HID = 64
GS = 32
G = HID // GS
CB = 3          # coarse bits (8 entries)
RB = 4          # residual bits (16 entries)
E_C = 2 ** CB
E_R = 2 ** RB


def _random_tree(num_bits, seed):
    """Random early-stopped tree in padded self-looping node format.

    Starts from the full-depth binary tree (BFS ids), prunes random
    internal nodes into leaves, and rewrites leaf rows to self-loop.
    Unreachable rows stay self-looped too — exactly the invariants the
    prep tool and the kernels rely on.
    """
    g = torch.Generator().manual_seed(seed)
    n_max = 2 ** (num_bits + 1) - 1
    depth0 = 2 ** num_bits - 1          # ids [0, depth0) are internal
    is_branch = torch.arange(n_max) < depth0
    # prune: random subset of non-root internal nodes becomes leaves
    prune_p = torch.rand(depth0, generator=g) < 0.35
    prune_p[0] = False
    is_branch[:depth0] &= ~prune_p
    idx = torch.arange(n_max)
    left = torch.where(is_branch, 2 * idx + 1, idx)
    right = torch.where(is_branch, 2 * idx + 2, idx)
    ch = torch.randint(0, HID, (n_max, 4), generator=g)
    signs = torch.where(torch.rand(n_max, 4, generator=g) < 0.5, -1.0, 1.0)
    thr = torch.randn(n_max, generator=g) * 0.5
    leaf = torch.randint(0, 2 ** num_bits, (n_max,), generator=g)
    leaf = torch.where(is_branch, torch.full_like(leaf, -1), leaf)
    return {"ch": ch, "signs": signs, "thr": thr, "leaf": leaf,
            "left": left, "right": right}


def synth_bundle():
    """Mirror prep_lut_bundle.py's output on tiny dims."""
    coarse = _random_tree(CB, seed=1)
    resid = {k: torch.stack([_random_tree(RB, seed=10 + gid)[k]
                             for gid in range(G)]) for k in
             ("ch", "signs", "thr", "leaf", "left", "right")}
    g = torch.Generator().manual_seed(99)
    return {
        "hidden": HID, "group_size": GS, "num_groups": G,
        "coarse_num_bits": CB, "residual_num_bits": RB,
        "coarse": coarse, "resid": resid,
        "coarse_table": (torch.randn(E_C, HID, generator=g) * 0.3).half(),
        "resid_table": (torch.randn(G, E_R, GS, generator=g) * 0.3).half(),
        "has_lowrank": False, "has_pairwise": False,
    }


def main():
    torch.manual_seed(0)
    rep = LutFfnReplacer(synth_bundle(), torch.device("cpu"))

    x = torch.randn(37, HID, dtype=torch.bfloat16)
    ref = rep.forward_reference(x)
    assert ref.shape == (37, HID), ref.shape
    assert ref.dtype == x.dtype, ref.dtype
    # 3D input is flattened and reshaped back
    ref3 = rep.forward_reference(x.view(1, 37, HID))
    assert ref3.shape == (1, 37, HID), ref3.shape
    assert torch.equal(ref3.view(37, HID), ref)
    print("[lut_ffn] reference path OK (CPU)")

    if not HAS_TRITON or not torch.cuda.is_available():
        print("[lut_ffn] triton/CUDA unavailable — kernel comparison "
              "SKIPPED (run on the remote GPU machine)")
        sys.exit(0)

    rep.to(torch.device("cuda"))
    out = rep(x.cuda())
    if not torch.allclose(out.float(), ref.float().cuda(), atol=0.05):
        diff = (out.float() - ref.float().cuda()).abs().max().item()
        print(f"[lut_ffn] FAIL: kernel vs reference max|diff|={diff}")
        sys.exit(1)
    print("[lut_ffn] kernel matches reference (atol=0.05)")
    sys.exit(0)


if __name__ == "__main__":
    main()
