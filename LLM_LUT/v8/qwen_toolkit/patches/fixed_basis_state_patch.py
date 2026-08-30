#!/usr/bin/env python3
"""Method 1: Fixed-basis recurrent state.

Runtime state is C in R^{r x V} per head; S is approximated as B @ C where
B in R^{K x r} is a fixed basis.

Read:    m = (B^T k)^T C
Decay:   C <- exp(g) * C
Delta:   delta = beta * (v - m)
Write:   C <- C + (B^T k) delta^T
Output:  y = (B^T q)^T C
"""

import sys
import os
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_toolkit.patches.gdn_replacement_base import GDNReplacementPatch


class FixedBasisStatePatch(GDNReplacementPatch):
    """Replace one GDN layer with a fixed-basis recurrent state."""

    def __init__(self, layer_idx: int = 20, rank: int = 32):
        super().__init__(layer_idx=layer_idx)
        self.rank = rank
        self._basis = None
        self._K = None

    def name(self) -> str:
        return f"fixed_basis_state_l{self.layer_idx}_r{self.rank}"

    def config(self) -> Dict[str, Any]:
        return {"layer_idx": self.layer_idx, "rank": self.rank}

    def _get_basis(self, K: int, device, dtype):
        if self._basis is None or self._K != K:
            B = torch.randn(K, self.rank, device=device, dtype=dtype)
            B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
            self._basis = B
            self._K = K
        return self._basis

    def init_state(self, S0: torch.Tensor):
        # S0: (B, H, K, V)
        B, H, K, V = S0.shape
        basis = self._get_basis(K, S0.device, S0.dtype)
        # Project S onto basis: C = B^T S, shape (B, H, r, V)
        C = torch.matmul(basis.t(), S0)  # (B, H, r, V)
        return {"C": C}

    def step(self, q, k, v, g, beta, state):
        # q,k: (B,H,K); v,y: (B,H,V); g,beta: (B,H); C: (B,H,r,V)
        C = state["C"]
        K = k.shape[-1]
        B = self._get_basis(K, C.device, C.dtype)
        # Project keys/queries: (B,H,K) @ (K,r) -> (B,H,r)
        k_proj = torch.matmul(k, B)
        q_proj = torch.matmul(q, B)

        # Decay state
        decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
        C_decayed = C * decay

        # Read: m = (B^T k)^T C = k_proj^T C -> (B,H,V)
        m = torch.matmul(k_proj.unsqueeze(-2), C_decayed).squeeze(-2)

        # Delta
        delta = (v - m) * beta.unsqueeze(-1)  # (B,H,V)

        # Write: C <- C + (B^T k) delta^T
        C_new = C_decayed + torch.matmul(k_proj.unsqueeze(-1), delta.unsqueeze(-2))

        # Output: y = (B^T q)^T C_new -> (B,H,V)
        y = torch.matmul(q_proj.unsqueeze(-2), C_new).squeeze(-2)

        return {"C": C_new}, y

    def final_state(self, state) -> torch.Tensor:
        # Reconstruct dense S = B @ C for cache compatibility.
        C = state["C"]
        B = self._get_basis(self._K, C.device, C.dtype)
        return torch.matmul(B, C)  # (B,H,K,V)

    def storage_stats(self) -> Dict[str, Any]:
        full_bytes = 1 * 32 * 128 * 128 * 4
        factor_bytes = 1 * 32 * self.rank * 128 * 4
        return {
            "full_state_bytes_per_layer": full_bytes,
            "factor_state_bytes_per_layer": factor_bytes,
            "compression_ratio": full_bytes / factor_bytes if factor_bytes else 1.0,
            "rank": self.rank,
        }
