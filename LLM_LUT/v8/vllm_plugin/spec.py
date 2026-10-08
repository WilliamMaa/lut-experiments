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

v2026-10-08j fixed-budget semantics (docs/37): blocks_per_request is a
FIXED small budget — cdiv(SLOTS + STAGING, B_g) + BLOCK_MARGIN — with no
dependence on the request's logical length. Physically the pool per
request is SLOTS + STAGING tokens: slots [0, SLOTS) hold the steady-state
compact layout [sink | heavy-hitter | recent]; slots
[SLOTS, SLOTS+STAGING) are prefill staging that deferred eviction may
still grow into. Prompts outgrowing SLOTS+STAGING trigger pressure
eviction (allowance = min(deferred, capacity - C), blockplan.py). The
whole-prompt reservation of v2026-10-04l is gone: memory per request is
constant from admission to finish.
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
        # v2026-10-08j (docs/37): FIXED budget — SLOTS + STAGING tokens of
        # physical pool, independent of logical length. bs here IS the
        # group manager block size B_g (1056 for the hybrid target, set by
        # HybridAttentionMambaModelConfig before any spec is built) —
        # manager units, docs/31 §1.1. The kernel page P never appears in
        # this formula. Blocks are still allocated lazily as prefill
        # advances, but stop at this cap instead of growing with the
        # prompt.
        n = (-(-(config.V8_COMPRESS_SLOTS + config.V8_STAGING_TOKENS)
               // bs)) + config.BLOCK_MARGIN
        object.__setattr__(self, "blocks_per_request", n)

    def max_memory_usage_bytes(self, vllm_config) -> int:
        # Upper bound for the max-concurrency estimate: the fixed per-
        # request budget (docs/37), paid in full once the staging region
        # is touched; unlike stock full attention it does NOT scale with
        # --max-model-len, which is exactly the admission-path win.
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
