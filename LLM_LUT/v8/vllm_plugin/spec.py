"""CompressedKVSpec: per-request constant-footprint KV spec (vLLM 0.19.1 API).

Differences from the 0.30-era draft (docs/28 history):
- No @register_kv_cache_spec registry in 0.19.1: KVCacheSpec.merge asserts
  all-equal + deepcopy, so merge fidelity is handled by overriding
  FullAttentionSpec.merge below.
- No customize_spec hook: the spec is converted in Attention.get_kv_cache_spec
  by the monkeypatch in vllm_plugin/__init__.py.
- No prefix_cacheable property in 0.19.1: prefix caching is force-disabled on
  the CLI by vllm_plugin.serve (block reuse would alias per-request compact
  regions).

The allocator-side cap (blocks per request independent of token count) comes
from the SingleTypeKVCacheManager monkeypatch: 0.19.1's
get_num_blocks_to_allocate is cdiv(num_tokens, block_size) with no spec hook.
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
        n = (self.retention_tokens + bs - 1) // bs + config.BLOCK_MARGIN
        object.__setattr__(self, "blocks_per_request", n)

    def max_memory_usage_bytes(self, vllm_config) -> int:
        # Constant per request: the whole point. Used by kv_cache_utils for
        # the max-concurrency estimate; the hard per-request cap comes from
        # the allocator monkeypatch (see __init__.patch_allocator).
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
