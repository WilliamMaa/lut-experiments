"""CompressedKVSpec: per-request KV spec (vLLM 0.19.1 API).

Differences from the 0.30-era draft (docs/28 history):
- No @register_kv_cache_spec registry in 0.19.1: KVCacheSpec.merge asserts
  all-equal + deepcopy, so merge fidelity is handled by overriding
  FullAttentionSpec.merge below.
- No customize_spec hook: the spec is converted in Attention.get_kv_cache_spec
  by the monkeypatch in vllm_plugin/__init__.py.
- No prefix_cacheable property in 0.19.1: prefix caching is force-disabled on
  the CLI by vllm_plugin.serve (block reuse would alias per-request compact
  regions).

The allocator-side cap (blocks per request) comes from the
SingleTypeKVCacheManager monkeypatch: 0.19.1's
get_num_blocks_to_allocate is cdiv(num_tokens, block_size) with no spec hook.

v2026-10-04l semantics change: eviction is DEFERRED to the first decode
step (see impl.py header), so a request's blocks must cover its whole
prompt up to V8_MAX_SEQ_TOKENS — the constant-footprint claim now holds
for the decode steady state only, same memory profile as the old harness
and the full-KV baseline during prefill. Mid-request block freeing is a
separate phase.
"""
import copy
from dataclasses import dataclass

from vllm.v1.kv_cache_interface import FullAttentionSpec

from . import config


@dataclass(frozen=True, kw_only=True)
class CompressedKVSpec(FullAttentionSpec):
    retention_tokens: int = config.V8_COMPRESS_SLOTS
    sink_tokens: int = config.V8_SINK_TOKENS
    recent_tokens: int = config.V8_RECENT_TOKENS
    obs_window: int = config.V8_OBS_WINDOW
    span_window: int = config.V8_SPAN_WINDOW
    blocks_per_request: int = 0  # derived in __post_init__

    def __post_init__(self):
        super().__post_init__()
        bs = self.block_size
        # v2026-10-04l: deferred eviction — blocks must hold the WHOLE
        # prompt (up to V8_MAX_SEQ_TOKENS), not just the retention budget.
        # Blocks are allocated lazily by the scheduler as prefill advances;
        # this is only the per-request cap.
        n = (config.V8_MAX_SEQ_TOKENS + bs - 1) // bs + config.BLOCK_MARGIN
        object.__setattr__(self, "blocks_per_request", n)

    def max_memory_usage_bytes(self, vllm_config) -> int:
        # Upper bound for the max-concurrency estimate: a fully-grown
        # request (max_seq_tokens). Actual usage tracks prompt length and
        # shrinks to the retention budget's worth of blocks only after
        # mid-request freeing lands (separate phase); decode attention
        # itself only ever touches the compact region.
        return self.blocks_per_request * self.page_size_bytes

    @classmethod
    def merge(cls, specs):
        # FullAttentionSpec.merge would rebuild with only base fields and
        # silently drop ours. All 10 full-attn layers produce identical
        # specs (same env config), so equality + deepcopy is correct.
        assert all(isinstance(s, CompressedKVSpec) for s in specs), (
            "All layers in a CompressedKVSpec group must be CompressedKVSpec")
        assert all(s == specs[0] for s in specs[1:]), (
            "All layers in a CompressedKVSpec group must be identical")
        return copy.deepcopy(specs[0])
