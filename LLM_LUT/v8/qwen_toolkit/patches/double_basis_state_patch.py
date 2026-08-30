#!/usr/bin/env python3
"""Method 2: Double-basis recurrent state.

S is approximated as B_k @ C @ B_v^T, where B_k and B_v are fixed bases.
Runtime state is C in R^{r_k x r_v}.

Read:    m = (B_k^T k)^T C (B_v^T v?? no, v stays)
Actually output: y = q^T S = q^T B_k C B_v^T = (B_k^T q)^T C B_v^T.
For the delta write k delta^T: S += k delta^T => C += (B_k^T k)(B_v^T delta)^T.

So:
  k' = B_k^T k   (B,H,r_k)
  q' = B_k^T q   (B,H,r_k)
  delta' = B_v^T delta  (B,H,r_v)
  m  = k'^T C    (B,H,r_v)  -- used to approximate v-space via B_v? Wait.

Original GDN:
  m = k^T S       (shape V)
  With S = B_k C B_v^T:
  m = k^T B_k C B_v^T = (B_k^T k)^T C B_v^T = k'^T C B_v^T -> shape V.
  So m = (k'^T C) @ B_v^T.

Output:
  y = q^T S = q'^T C B_v^T.

Delta:
  delta = beta * (v - m)  (V space)
  write to C:  C_new = C + k' (B_v^T delta)^T = C + k' delta'^T.

State dimension: r_k x r_v per head. With r_k=r_v=32, 16x reduction.
"""

import sys
import os
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_toolkit.patches.gdn_replacement_base import GDNReplacementPatch


class DoubleBasisStatePatch(GDNReplacementPatch):
    """Replace one GDN layer with a double-basis recurrent state."""

    def __init__(self, layer_idx: int = 20, rank_k: int = 32, rank_v: int = 32):
        super().__init__(layer_idx=layer_idx)
        self.rank_k = rank_k
        self.rank_v = rank_v
        self._basis_k = None
        self._basis_v = None
        self._K = None
        self._V = None

    def name(self) -> str:
        return f"double_basis_state_l{self.layer_idx}_rk{self.rank_k}_rv{self.rank_v}"

    def config(self) -> Dict[str, Any]:
        return {"layer_idx": self.layer_idx, "rank_k": self.rank_k, "rank_v": self.rank_v}

    def _get_basis_k(self, K: int, device):
        if self._basis_k is None or self._K != K:
            B = torch.randn(K, self.rank_k, device=device, dtype=torch.float32)
            B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
            self._basis_k = B
            self._K = K
        return self._basis_k

    def _get_basis_v(self, V: int, device):
        if self._basis_v is None or self._V != V:
            B = torch.randn(V, self.rank_v, device=device, dtype=torch.float32)
            B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
            self._basis_v = B
            self._V = V
        return self._basis_v

    def init_state(self, S0: torch.Tensor):
        B, H, K, V = S0.shape
        Bk = self._get_basis_k(K, S0.device)
        Bv = self._get_basis_v(V, S0.device)
        # C = Bk^T S Bv -> (B,H,r_k,r_v)
        C = torch.matmul(torch.matmul(Bk.t(), S0.float()), Bv)
        return {"C": C}

    def step(self, q, k, v, g, beta, state):
        C = state["C"]  # (B,H,r_k,r_v)
        K = k.shape[-1]
        V = v.shape[-1]
        Bk = self._get_basis_k(K, C.device)
        Bv = self._get_basis_v(V, C.device)

        k_proj = torch.matmul(k, Bk)  # (B,H,r_k)
        q_proj = torch.matmul(q, Bk)

        # Decay
        decay = g.exp().unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
        C_decayed = C * decay

        # Read: m = (k_proj^T C) @ Bv^T -> (B,H,V)
        tmp = torch.matmul(k_proj.unsqueeze(-2), C_decayed).squeeze(-2)  # (B,H,r_v)
        m = torch.matmul(tmp, Bv.t())  # (B,H,V)

        # Delta
        delta = (v - m) * beta.unsqueeze(-1)  # (B,H,V)

        # Write: C += k_proj @ (Bv^T delta)^T
        delta_proj = torch.matmul(delta, Bv)  # (B,H,r_v)
        C_new = C_decayed + torch.matmul(k_proj.unsqueeze(-1), delta_proj.unsqueeze(-2))

        # Output: y = (q_proj^T C) @ Bv^T -> (B,H,V)
        out_tmp = torch.matmul(q_proj.unsqueeze(-2), C_new).squeeze(-2)  # (B,H,r_v)
        y = torch.matmul(out_tmp, Bv.t())  # (B,H,V)

        return {"C": C_new}, y

    def final_state(self, state) -> torch.Tensor:
        C = state["C"]
        Bk = self._get_basis_k(self._K, C.device)
        Bv = self._get_basis_v(self._V, C.device)
        return torch.matmul(torch.matmul(Bk, C), Bv.t())  # (B,H,K,V)

    def storage_stats(self) -> Dict[str, Any]:
        full_bytes = 1 * 32 * 128 * 128 * 4
        factor_bytes = 1 * 32 * self.rank_k * self.rank_v * 4
        return {
            "full_state_bytes_per_layer": full_bytes,
            "factor_state_bytes_per_layer": factor_bytes,
            "compression_ratio": full_bytes / factor_bytes if factor_bytes else 1.0,
            "rank_k": self.rank_k,
            "rank_v": self.rank_v,
        }
