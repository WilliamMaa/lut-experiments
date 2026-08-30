#!/usr/bin/env python3
"""Base class for replacing a single GDN layer's recurrent step.

The common interface is exactly the GDN single-step operator:

    (q_t, k_t, v_t, g_t, beta_t, S_{t-1}) -> (S_t, y_t)

where shapes are:
    q, k: (B, H, K)
    v, y: (B, H, V)
    g, beta: (B, H)
    S:    (B, H, K, V)

Subclasses implement:
    - init_state(self, S0): create a compressed state object from dense initial state.
    - step(self, q, k, v, g, beta, state): return (new_state, output).
    - final_state(self, state): return a dense S for cache compatibility (if needed).
"""

import sys
import os
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.evaluator import EvalPatch


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _prepare_gdn_inputs(query, key, value, g, beta):
    """Squeeze singleton sequence dim and apply the official q/k normalization."""
    # Inputs arrive as [B, seq_len=1, H, D]; squeeze to [B, H, D].
    q = query.squeeze(1)
    k = key.squeeze(1)
    v = value.squeeze(1)
    g = g.squeeze(1)
    beta = beta.squeeze(1)

    # Official kernel normalization: l2norm + query scaled by 1/sqrt(head_dim).
    q = _l2norm(q) / (q.shape[-1] ** 0.5)
    k = _l2norm(k)
    return q, k, v, g, beta


class GDNReplacementPatch(EvalPatch):
    """Base class: replace one GDN layer's recurrent function with a custom step."""

    def __init__(self, layer_idx: int = 20):
        self.layer_idx = layer_idx
        self._orig_fn = None
        self._state = None

    # ----- subclass interface -----

    def init_state(self, S0: torch.Tensor):
        """Create compressed state from dense initial state S0 (B, H, K, V)."""
        raise NotImplementedError

    def step(self, q, k, v, g, beta, state) -> tuple:
        """One decode step. Return (new_state, output) where output is (B, H, V)."""
        raise NotImplementedError

    def final_state(self, state) -> torch.Tensor:
        """Return dense S for cache compatibility. Default reconstructs."""
        raise NotImplementedError

    def storage_stats(self) -> Dict[str, Any]:
        return {}

    # ----- install / uninstall -----

    def install(self, model: torch.nn.Module) -> None:
        layer = model.model.layers[self.layer_idx]
        if not hasattr(layer, "linear_attn"):
            raise ValueError(f"Layer {self.layer_idx} is not a GDN layer")
        self._orig_fn = layer.linear_attn.recurrent_gated_delta_rule
        self_model = self

        def replacement(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            **kwargs,
        ):
            # Only intercept single-token decode.
            if query is None or query.shape[1] != 1:
                return self_model._orig_fn(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    **kwargs,
                )

            q, k, v, g, beta = _prepare_gdn_inputs(query, key, value, g, beta)

            if self_model._state is None:
                if initial_state is None:
                    raise RuntimeError("No initial state for first decode step")
                self_model._state = self_model.init_state(initial_state)

            new_state, output = self_model.step(q, k, v, g, beta, self_model._state)
            self_model._state = new_state

            final = self_model.final_state(new_state) if output_final_state else None
            # Output expected as [B, seq_len=1, H, V].
            output = output.unsqueeze(1)
            return output, final

        layer.linear_attn.recurrent_gated_delta_rule = replacement

    def uninstall(self, model: torch.nn.Module) -> None:
        if self._orig_fn is not None:
            layer = model.model.layers[self.layer_idx]
            layer.linear_attn.recurrent_gated_delta_rule = self._orig_fn
            self._orig_fn = None
        self._state = None
