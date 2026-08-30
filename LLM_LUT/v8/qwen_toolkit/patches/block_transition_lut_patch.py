#!/usr/bin/env python3
"""Method 4: Block Transition LUT (crude first cut).

A head's dense recurrent state S in R^{K x V} is split into small blocks,
e.g. 16x16. Each block is represented by a code from a fixed codebook.
At runtime only the block codes are stored.

This first version is intentionally crude: it decodes the full state from
codes, performs the exact dense GDN update, and re-encodes the new state.
It does NOT yet implement a true transition LUT that maps
(old_block_code, k_block, delta_block, gate) -> new_block_code.
The crude version lets us validate whether block-wise VQ of the recurrent
state can survive end-to-end before we build the real transition table.
"""

import sys
import os
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_toolkit.patches.gdn_replacement_base import GDNReplacementPatch


class BlockTransitionLUTPatch(GDNReplacementPatch):
    """Replace one GDN layer with a block-quantized recurrent state."""

    def __init__(self, layer_idx: int = 20, block_size: int = 16, num_codes: int = 256):
        super().__init__(layer_idx=layer_idx)
        self.block_size = block_size
        self.num_codes = num_codes
        self._codebook = None  # (num_codes, block_size, block_size)

    def name(self) -> str:
        return f"block_vq_state_l{self.layer_idx}_b{self.block_size}_c{self.num_codes}"

    def config(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "block_size": self.block_size,
            "num_codes": self.num_codes,
        }

    def _init_codebook(self, device):
        if self._codebook is None:
            torch.manual_seed(42)
            self._codebook = torch.randn(
                self.num_codes,
                self.block_size,
                self.block_size,
                device=device,
                dtype=torch.float32,
            ) * 0.01

    def _split_into_blocks(self, S: torch.Tensor):
        # S: (B, H, K, V), assume K,V divisible by block_size.
        B, H, K, V = S.shape
        bs = self.block_size
        nK = K // bs
        nV = V // bs
        blocks = S.view(B, H, nK, bs, nV, bs)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        return blocks.reshape(B, H, nK, nV, bs * bs)

    def _merge_blocks(self, blocks: torch.Tensor, B: int, H: int, K: int, V: int):
        bs = self.block_size
        nK = K // bs
        nV = V // bs
        blocks = blocks.view(B, H, nK, nV, bs, bs)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        return blocks.reshape(B, H, K, V)

    def _encode(self, S: torch.Tensor):
        self._init_codebook(S.device)
        B, H, K, V = S.shape
        flat = self._split_into_blocks(S).reshape(-1, self.block_size * self.block_size)
        cb_flat = self._codebook.view(self.num_codes, -1)
        # Euclidean distance to each code.
        dist = torch.cdist(flat, cb_flat, p=2)
        codes = dist.argmin(dim=-1)
        return codes.view(B, H, K // self.block_size, V // self.block_size)

    def _decode(self, codes: torch.Tensor, K: int, V: int):
        self._init_codebook(codes.device)
        B, H, nK, nV = codes.shape
        entries = self._codebook[codes.flatten()]  # (B*H*nK*nV, bs, bs)
        blocks = entries.view(B, H, nK, nV, self.block_size, self.block_size)
        return self._merge_blocks(blocks, B, H, K, V)

    def init_state(self, S0: torch.Tensor):
        codes = self._encode(S0.float())
        return {"codes": codes}

    def step(self, q, k, v, g, beta, state):
        codes = state["codes"]
        K = q.shape[-1]
        Vdim = v.shape[-1]
        S = self._decode(codes, K, Vdim)  # (B, H, K, V)

        # Standard dense GDN update.
        decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S_decayed = S * decay
        m = torch.matmul(k.unsqueeze(-2), S_decayed).squeeze(-2)  # (B, H, V)
        delta = (v - m) * beta.unsqueeze(-1)
        S_new = S_decayed + torch.matmul(k.unsqueeze(-1), delta.unsqueeze(-2))
        y = torch.matmul(q.unsqueeze(-2), S_new).squeeze(-2)

        codes_new = self._encode(S_new)
        return {"codes": codes_new}, y

    def final_state(self, state) -> torch.Tensor:
        codes = state["codes"]
        # We do not know K/V from codes alone; infer from codebook dims.
        bs = self.block_size
        # Qwen GDN uses square K=V=128 for the recurrent state.
        K = V = 128
        return self._decode(codes, K, V)

    def storage_stats(self) -> Dict[str, Any]:
        full_bytes = 1 * 32 * 128 * 128 * 4
        num_blocks = (128 // self.block_size) ** 2
        code_bytes = 1 * 32 * num_blocks * 4  # store code as int32
        return {
            "full_state_bytes_per_layer": full_bytes,
            "factor_state_bytes_per_layer": code_bytes,
            "compression_ratio": full_bytes / code_bytes if code_bytes else 1.0,
            "block_size": self.block_size,
            "num_codes": self.num_codes,
        }
