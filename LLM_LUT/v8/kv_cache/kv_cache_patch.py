#!/usr/bin/env python3
"""Base class and first method for KV cache compression patches."""

import sys
import os
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluator import EvalPatch
from kv_cache.kivi_cache import KIVICache
from kv_cache.retention_cache import RetentionCache
from kv_cache.heavy_hitter_cache import HeavyHitterCache
from kv_cache.attention_scores import (
    AttentionScoreBank,
    install_eager_score_stash,
    uninstall_eager_score_stash,
    set_observation_window,
)


class KVCachePatch(EvalPatch):
    """Base class for patches that replace the generation-time KV cache."""

    def name(self) -> str:
        return self.__class__.__name__

    def config(self) -> Dict[str, Any]:
        return {}

    def storage_stats(self) -> Dict[str, Any]:
        return {}

    def get_cache(self, device, config=None):
        """Return a fresh cache instance for one generation."""
        raise NotImplementedError

    def install(self, model):
        """Optional in-place model modification. Default does nothing."""
        pass

    def uninstall(self, model):
        """Optional cleanup. Default does nothing."""
        pass


class KIVICachePatch(KVCachePatch):
    """KIVI-style asymmetric KV cache: K per-channel, V per-token."""

    def __init__(self, k_bits: int = 4, v_bits: int = 4):
        self.k_bits = k_bits
        self.v_bits = v_bits

    def name(self) -> str:
        return f"kivi_k{self.k_bits}_v{self.v_bits}"

    def config(self) -> Dict[str, Any]:
        return {"k_bits": self.k_bits, "v_bits": self.v_bits}

    def get_cache(self, device, config=None):
        return KIVICache(k_bits=self.k_bits, v_bits=self.v_bits, config=config).to(device)

    def storage_stats(self) -> Dict[str, Any]:
        kv_heads = 2
        head_dim = 256
        full_attn_layers = 10
        bytes_per_token = kv_heads * head_dim * (self.k_bits + self.v_bits) / 8
        bf16_bytes_per_token = kv_heads * head_dim * 4
        return {
            "kv_bits": self.k_bits + self.v_bits,
            "bytes_per_token": bytes_per_token,
            "bf16_bytes_per_token": bf16_bytes_per_token,
            "compression_ratio": bf16_bytes_per_token / bytes_per_token,
            "estimated_128k_context_bytes": bytes_per_token * 128000 * full_attn_layers,
            "estimated_128k_context_bf16_bytes": bf16_bytes_per_token * 128000 * full_attn_layers,
        }


class RetentionCachePatch(KVCachePatch):
    """Retention-based KV cache: keep sink + recent tokens."""

    def __init__(self, max_cache_len: int = 512, sink_tokens: int = 4):
        self.max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens

    def name(self) -> str:
        return f"retention_l{self.max_cache_len}_s{self.sink_tokens}"

    def config(self) -> Dict[str, Any]:
        return {"max_cache_len": self.max_cache_len, "sink_tokens": self.sink_tokens}

    def get_cache(self, device, config=None):
        return RetentionCache(
            max_cache_len=self.max_cache_len,
            sink_tokens=self.sink_tokens,
            config=config,
        ).to(device)

    def storage_stats(self) -> Dict[str, Any]:
        kv_heads = 2
        head_dim = 256
        full_attn_layers = 10
        bf16_bytes_per_token = kv_heads * head_dim * 4
        effective_bytes_per_token = bf16_bytes_per_token * (self.max_cache_len / 128000)
        return {
            "max_cache_len": self.max_cache_len,
            "sink_tokens": self.sink_tokens,
            "bf16_bytes_per_token": bf16_bytes_per_token,
            "effective_bytes_per_token_128k": effective_bytes_per_token,
            "compression_ratio": bf16_bytes_per_token / effective_bytes_per_token,
            "estimated_128k_context_bytes": self.max_cache_len * bf16_bytes_per_token * full_attn_layers,
            "estimated_128k_context_bf16_bytes": bf16_bytes_per_token * 128000 * full_attn_layers,
        }


class HeavyHitterCachePatch(KVCachePatch):
    """Heavy-hitter KV cache: sink + recent + important middle tokens."""

    def __init__(self, max_cache_len: int = 512, sink_tokens: int = 4, recent_tokens: int = 128):
        self.max_cache_len = max_cache_len
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

    def name(self) -> str:
        return f"heavy_hitter_l{self.max_cache_len}_s{self.sink_tokens}_r{self.recent_tokens}"

    def config(self) -> Dict[str, Any]:
        return {
            "max_cache_len": self.max_cache_len,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
        }

    def get_cache(self, device, config=None):
        return HeavyHitterCache(
            max_cache_len=self.max_cache_len,
            sink_tokens=self.sink_tokens,
            recent_tokens=self.recent_tokens,
            config=config,
        ).to(device)

    def storage_stats(self) -> Dict[str, Any]:
        kv_heads = 2
        head_dim = 256
        full_attn_layers = 10
        bf16_bytes_per_token = kv_heads * head_dim * 4
        effective_bytes_per_token = bf16_bytes_per_token * (self.max_cache_len / 128000)
        return {
            "max_cache_len": self.max_cache_len,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
            "bf16_bytes_per_token": bf16_bytes_per_token,
            "effective_bytes_per_token_128k": effective_bytes_per_token,
            "compression_ratio": bf16_bytes_per_token / effective_bytes_per_token,
            "estimated_128k_context_bytes": self.max_cache_len * bf16_bytes_per_token * full_attn_layers,
            "estimated_128k_context_bf16_bytes": bf16_bytes_per_token * 128000 * full_attn_layers,
        }


class HeavyHitterAttnScorePatch(HeavyHitterCachePatch):
    """Heavy-hitter KV cache with prefill attention-score importance.

    Same retention structure as HeavyHitterCachePatch (sink + recent + heavy
    hitters), but importance comes from accumulated prefill attention mass
    (H2O/SnapKV-style) instead of the key-L2-norm proxy. install() forces
    eager attention on the student and routes it through a stashing kernel
    that column-sums the last ``obs_window`` query rows per key position.
    """

    def __init__(self, max_cache_len: int = 512, sink_tokens: int = 4,
                 recent_tokens: int = 128, obs_window: int = 64):
        super().__init__(max_cache_len, sink_tokens, recent_tokens)
        self._bank = AttentionScoreBank()
        self.obs_window = obs_window

    def name(self) -> str:
        return (f"heavy_hitter_attn_l{self.max_cache_len}_s{self.sink_tokens}"
                f"_r{self.recent_tokens}_w{self.obs_window}")

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg["importance"] = "prefill_attn_score"
        cfg["obs_window"] = self.obs_window
        return cfg

    def get_cache(self, device, config=None):
        # Fresh scores per generation: the bank only reflects the current prefill.
        self._bank.clear()
        return HeavyHitterCache(
            max_cache_len=self.max_cache_len,
            sink_tokens=self.sink_tokens,
            recent_tokens=self.recent_tokens,
            config=config,
            importance_mode="attn_score",
            score_bank=self._bank,
            obs_window=self.obs_window,
        ).to(device)

    def install(self, model):
        set_observation_window(self.obs_window)
        install_eager_score_stash(model, self._bank)

    def uninstall(self, model):
        uninstall_eager_score_stash(model)
