#!/usr/bin/env python3
"""Low-rank GDN recurrent state patch."""

from typing import Any, Dict, List, Optional

import torch

from common.evaluator import EvalPatch


class LowRankStateContainer:
    """Dict-like container that stores SVD factors instead of a dense tensor."""

    def __init__(self, rank: int, keys, dtype=None, device=None):
        self.rank = rank
        self.dtype = dtype
        self.device = device
        self._keys = list(keys)
        self._data: Dict[int, Any] = {k: None for k in self._keys}

    def _clear(self):
        for k in self._keys:
            self._data[k] = None

    def __getitem__(self, key):
        val = self._data[key]
        if val is None:
            return None
        if torch.is_tensor(val):
            return val
        u, v = val
        return u @ v.transpose(-2, -1)

    def __setitem__(self, key, tensor: torch.Tensor):
        if tensor is None:
            self._data[key] = None
            return
        if self.rank <= 0:
            self._data[key] = tensor
            return
        min_dim = min(tensor.shape[-2], tensor.shape[-1])
        if self.rank >= min_dim:
            self._data[key] = tensor
            return
        u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
        u_r = u[..., : self.rank]
        s_r = s[..., : self.rank]
        vh_r = vh[..., : self.rank, :]
        sqrt_s = s_r.sqrt().clamp_min(1e-12)
        u_store = u_r * sqrt_s.unsqueeze(-2)
        v_store = vh_r.transpose(-2, -1) * sqrt_s.unsqueeze(-2)
        self._data[key] = (u_store, v_store)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return iter(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self[k]


class LowRankStatePatch(EvalPatch):
    """Apply low-rank SVD compression to GDN recurrent states."""

    def __init__(self, rank: int = 32, layer_indices: Optional[List[int]] = None):
        self.rank = rank
        self.layer_indices = set(layer_indices) if layer_indices else None
        self._orig_init = None
        self._orig_update = None
        self._orig_reset = None

    def name(self) -> str:
        return f"low_rank_state_r{self.rank}"

    def config(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "layer_indices": sorted(self.layer_indices) if self.layer_indices else "all",
        }

    def install(self, model: torch.nn.Module) -> None:
        from transformers.cache_utils import LinearAttentionLayer

        orig_init = LinearAttentionLayer.__init__
        orig_update = LinearAttentionLayer.update_recurrent_state
        orig_reset = LinearAttentionLayer.reset
        self._orig_init = orig_init
        self._orig_update = orig_update
        self._orig_reset = orig_reset
        rank = self.rank

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.recurrent_states = LowRankStateContainer(
                rank, list(self.recurrent_states.keys()), self.dtype, self.device
            )

        def new_update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs):
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            self.recurrent_states[state_idx] = recurrent_states
            return self.recurrent_states[state_idx]

        def new_reset(self):
            for i in range(self.number_of_states):
                if self.is_conv_states_initialized[i]:
                    self.conv_states[i].zero_()
                self.has_previous_state[i] = False
                if isinstance(self.recurrent_states, LowRankStateContainer):
                    self.recurrent_states._clear()
                elif self.is_recurrent_states_initialized[i]:
                    self.recurrent_states[i].zero_()

        LinearAttentionLayer.__init__ = new_init
        LinearAttentionLayer.update_recurrent_state = new_update_recurrent_state
        LinearAttentionLayer.reset = new_reset

    def uninstall(self, model: torch.nn.Module) -> None:
        from transformers.cache_utils import LinearAttentionLayer

        if self._orig_init is not None:
            LinearAttentionLayer.__init__ = self._orig_init
        if self._orig_update is not None:
            LinearAttentionLayer.update_recurrent_state = self._orig_update
        if self._orig_reset is not None:
            LinearAttentionLayer.reset = self._orig_reset
        self._orig_init = None
        self._orig_update = None
        self._orig_reset = None

    def storage_stats(self) -> Dict[str, Any]:
        full_bytes = 1 * 32 * 128 * 128 * 4
        if 0 < self.rank < 128:
            factor_bytes = 2 * 1 * 32 * 128 * self.rank * 4
        else:
            factor_bytes = full_bytes
        return {
            "full_state_bytes_per_layer": full_bytes,
            "low_rank_state_bytes_per_layer": factor_bytes,
            "compression_ratio": full_bytes / factor_bytes if factor_bytes else 1.0,
            "rank": self.rank,
        }
