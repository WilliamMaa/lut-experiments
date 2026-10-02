"""CompressedKVSpec: per-request constant-footprint KV spec.

Structure mirrors HiSparseHotSpec (vllm/v1/kv_cache_interface.py): the
allocator reads max_memory_usage_bytes() / max_num_blocks_per_req() and
budgets a FIXED number of blocks per request, independent of max_model_len.
page_size_bytes is inherited from AttentionSpec (full block page), so a
request's steady-state KV footprint is

    blocks_per_request * block_size * num_kv_heads * 2 * head_size * dtype

e.g. 33 blocks * 16 tokens * 2 heads * 512 * 2B = 10.8MB per request per
layer at 512 slots bf16 (vs ~2.6GB at 64k full) — the product value.
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

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def block_table_token_alignment(self):
        return None

    def max_memory_usage_bytes(self, vllm_config) -> int:
        return self.blocks_per_request * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        return self.blocks_per_request

    @classmethod
    def merge(cls, specs):
        # All 10 full-attn layers produce identical specs (same env config);
        # unlike FullAttentionSpec.merge, keep the subclass fields.
        assert all(isinstance(s, CompressedKVSpec) for s in specs), (
            "All layers in a CompressedKVSpec group must be CompressedKVSpec")
        assert all(s == specs[0] for s in specs[1:]), (
            "All layers in a CompressedKVSpec group must be identical")
        return copy.deepcopy(specs[0])


# Required by KVCacheSpec.is_uniform_with_collection (kv_cache_spec_registry).
# TODO(remote-1): if the decorator signature differs on the deployed vLLM,
# this line raises at import; the fix is a one-liner at
# vllm/v1/kv_cache_spec_registry.py.
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec
register_kv_cache_spec(CompressedKVSpec)
