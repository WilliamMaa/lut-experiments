#!/usr/bin/env python3
"""定型 representation presets (docs/19-final-summary.md 方法栈).

m_sp4: heavy_hitter_attn, budget 128 = sink 4 + recent 32 + hh 92,
obs_window 64, merge_evicted (convex fold), span_window 4, bf16 storage.
k8v8: m_sp4 + K per-channel INT8 / V per-token INT8 storage.
bf16: plain DynamicCache.

The worker in Step 2 builds its per-session cache through these factories so
that what the prototype ships between workers is exactly what v8's eval
measured at 1000x / 2000x.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.cache_utils import DynamicCache

from kv_cache.kv_cache_patch import HeavyHitterAttnScorePatch
from kv_cache.attention_scores import AttentionScoreBank, set_observation_window


def m_sp4_patch() -> HeavyHitterAttnScorePatch:
    return HeavyHitterAttnScorePatch(
        max_cache_len=128, sink_tokens=4, recent_tokens=32,
        obs_window=64, merge_evicted=True,
        k_bits=16, v_bits=16, span_window=4,
    )


def k8v8_patch() -> HeavyHitterAttnScorePatch:
    return HeavyHitterAttnScorePatch(
        max_cache_len=128, sink_tokens=4, recent_tokens=32,
        obs_window=64, merge_evicted=True,
        k_bits=8, v_bits=8, span_window=4,
    )


def bf16_factory(config=None) -> DynamicCache:
    return DynamicCache(config=config)


def cache_factory(repr: str, config=None, device="cpu"):
    """Return (make_cache, install, uninstall) for a representation name.

    make_cache() builds a fresh per-session cache (the attn_score patches
    also reset their score bank, which is the per-turn semantics v8 uses).
    install(model) must be called once before the first forward for the
    m_sp4/k8v8 importance path; uninstall(model) reverses it.
    """
    if repr == "bf16":
        return (lambda: DynamicCache(config=config)), lambda m: None, lambda m: None
    patch = m_sp4_patch() if repr == "m_sp4" else k8v8_patch()
    return (lambda: patch.get_cache(device, config=config)), \
        patch.install, patch.uninstall
