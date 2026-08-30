#!/usr/bin/env python3
"""Method 3: Atom / dictionary recurrent state.

Runtime state is a small coefficient vector c in R^M per head.
The dense recurrent state is approximated as a linear combination of
fixed rank-1 memory atoms:

    S ~= sum_j c_j * (u_j v_j^T)

where U in R^{K x M} and V in R^{V x M} are fixed atom dictionaries.

GDN update in coefficient space:
    c_decayed = exp(g) * c
    q_atoms = U^T q                  # (B,H,M)
    k_atoms = U^T k                  # (B,H,M)
    m       = (c_decayed * q_atoms)^T V   -> actually m = sum_j c_j (k^T u_j) v_j
    delta   = beta * (v - m)
    delta_atoms = V^T delta          # (B,H,M)
    c_new   = c_decayed + k_atoms * delta_atoms   # element-wise, rank-1 write
    y       = (c_new * q_atoms)^T V

This is the first design that is explicitly structured for a future
LUT read/write: q^T u_j and k^T u_j are scalar atom responses, v_j are
fixed output directions, and the coefficient update is a simple per-atom
operation.
"""

import sys
import os
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_toolkit.patches.gdn_replacement_base import GDNReplacementPatch


class AtomStatePatch(GDNReplacementPatch):
    """Replace one GDN layer with an atom/dictionary state."""

    def __init__(self, layer_idx: int = 20, num_atoms: int = 512):
        super().__init__(layer_idx=layer_idx)
        self.num_atoms = num_atoms
        self._U = None  # (K, M)
        self._V = None  # (V, M)
        self._K = None
        self._Vdim = None

    def name(self) -> str:
        return f"atom_state_l{self.layer_idx}_a{self.num_atoms}"

    def config(self) -> Dict[str, Any]:
        return {"layer_idx": self.layer_idx, "num_atoms": self.num_atoms}

    def _init_atoms(self, K: int, Vdim: int, device):
        if self._U is None or self._K != K or self._Vdim != Vdim:
            M = self.num_atoms
            U = torch.randn(K, M, device=device, dtype=torch.float32)
            V = torch.randn(Vdim, M, device=device, dtype=torch.float32)
            # Normalize each atom direction.
            U = U / (U.norm(dim=0, keepdim=True) + 1e-12)
            V = V / (V.norm(dim=0, keepdim=True) + 1e-12)
            self._U = U
            self._V = V
            self._K = K
            self._Vdim = Vdim

    def init_state(self, S0: torch.Tensor):
        # S0: (B, H, K, V)
        B, H, K, Vdim = S0.shape
        self._init_atoms(K, Vdim, S0.device)
        # Coefficients: c_j = u_j^T S0 v_j for each atom j.
        # einsum: bhkv,kj,vj -> bhj
        c0 = torch.einsum("bhkv,kj,vj->bhj", S0.float(), self._U, self._V)
        return {"c": c0}

    def step(self, q, k, v, g, beta, state):
        # q,k: (B,H,K); v,y: (B,H,V); g,beta: (B,H); c: (B,H,M)
        c = state["c"]
        K = k.shape[-1]
        Vdim = v.shape[-1]
        self._init_atoms(K, Vdim, c.device)
        U = self._U
        Vt = self._V.t()  # (M, V)

        # Atom responses for current q/k.
        # (B,H,K) @ (K,M) -> (B,H,M)
        q_atoms = torch.matmul(q, U)
        k_atoms = torch.matmul(k, U)

        # Decay coefficients.
        decay = g.exp().unsqueeze(-1)  # (B,H,1)
        c_decayed = c * decay

        # Read: m = sum_j c_j (k^T u_j) v_j = (c_decayed * k_atoms) @ V^T
        m = torch.matmul(c_decayed * k_atoms, Vt)  # (B,H,V)

        # Delta write direction.
        delta = (v - m) * beta.unsqueeze(-1)  # (B,H,V)
        delta_atoms = torch.matmul(delta, self._V)  # (B,H,M)

        # Rank-1 write projected onto atoms: element-wise outer-product coefficients.
        c_new = c_decayed + k_atoms * delta_atoms

        # Output from the updated state.
        y = torch.matmul(c_new * q_atoms, Vt)  # (B,H,V)

        return {"c": c_new}, y

    def final_state(self, state) -> torch.Tensor:
        # Reconstruct dense S = U @ diag(c) @ V^T for each head.
        c = state["c"]  # (B,H,M)
        # einsum: bhj (atom coeffs), kj (atom K direction), vj (atom V direction)
        S = torch.einsum("bhj,kj,vj->bhkv", c, self._U, self._V)  # (B,H,K,V)
        return S

    def storage_stats(self) -> Dict[str, Any]:
        full_bytes = 1 * 32 * 128 * 128 * 4
        atom_bytes = 1 * 32 * self.num_atoms * 4
        return {
            "full_state_bytes_per_layer": full_bytes,
            "factor_state_bytes_per_layer": atom_bytes,
            "compression_ratio": full_bytes / atom_bytes if atom_bytes else 1.0,
            "num_atoms": self.num_atoms,
        }
