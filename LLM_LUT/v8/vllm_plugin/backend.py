"""CompressedKVBackend + metadata builder + allocator cap patch.

Integration facts (vLLM v0.19.1, verified against spike/vllm-src-019):
- Attention.__init__ accepts attn_backend=... (attention.py:202) — injected
  by monkeypatch (vllm_plugin/__init__.py).
- No customize_spec in 0.19.1: Attention.get_kv_cache_spec returns
  FullAttentionSpec directly (attention.py:537) — the monkeypatch converts it.
- Builder signature in 0.19.1: (kv_cache_spec, layer_names, vllm_config,
  device) — our __init__ takes (*args, **kwargs).
- Builder.build(common_prefix_len, common_attn_metadata, fast_build=False)
  consumes CommonAttentionMetadata (query_start_loc_cpu, seq_lens,
  block_table_tensor, ...). No req_ids: per-request state is keyed by the
  request's block-id tuple (unique while alive; reuse => recreate).
- Allocator: single_type_kv_cache_manager.get_num_blocks_to_allocate
  computes cdiv(num_tokens, block_size) with no spec hook — monkeypatched
  to clamp at spec.blocks_per_request (see patch_allocator).
- v1 is eager-only: per-request Python state cannot be CUDA-graph captured.
  --enforce-eager is mandatory (enforced in serve.py).
"""
import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .spec import CompressedKVSpec
from . import config


class RequestKVState:
    """Per layer, per request. K/V live in the pool; this is the control
    plane: compact layout bookkeeping + the attention-mass snapshot table.

    Layout invariant: compact slots [0, L) of the request's private blocks
    always hold [sink | heavy-hitter | recent] in temporal (orig) order.
    """

    __slots__ = ("blocks", "compact_len", "orig", "snap", "snap_per_head",
                 "snap_len", "_gpu")

    def __init__(self, blocks):
        self.blocks = blocks            # tuple[int], len == blocks_per_request
        self.compact_len = 0
        self.orig = None                # LongTensor [cap], lazily on GPU
        self.snap = None                # fp32 [cap], attention-mass snapshot
        self.snap_per_head = None       # fp32 [H_kv, cap]
        self.snap_len = 0
        self._gpu = None                # (device, blk_tensor, arange_cache)

    def ensure_gpu(self, device):
        if self._gpu is not None and self._gpu[0] == device:
            return
        blk = torch.tensor(self.blocks, dtype=torch.long, device=device)
        self._gpu = (device, blk, None)

    def grow(self, needed, device, h_kv):
        """Lazily allocate the orig/snap tables, doubling as needed."""
        cap = 0 if self.orig is None else self.orig.numel()
        if needed <= cap:
            return
        new_cap = max(needed, 2 * cap, 4096)
        orig = torch.zeros(new_cap, dtype=torch.long, device=device)
        snap = torch.zeros(new_cap, dtype=torch.float32, device=device)
        snap_ph = torch.zeros(h_kv, new_cap, dtype=torch.float32,
                              device=device)
        if self.orig is not None:
            orig[:cap] = self.orig
            snap[:cap] = self.snap
            snap_ph[:, :cap] = self.snap_per_head
        self.orig, self.snap, self.snap_per_head = orig, snap, snap_ph

    @property
    def blk_tensor(self):
        return self._gpu[1]


class CompressedKVMetadata(FlashAttentionMetadata):
    """Adds per-request state to the standard metadata.

    req_states[i] corresponds to request i of the batch (same order as
    query_start_loc / seq_lens). Attached post-construction by the builder.
    """

    req_states: list = None
    qsl_cpu: object = None
    seq_lens_cpu: object = None


class CompressedKVBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "V8_COMPRESSED"

    @staticmethod
    def get_impl_cls():
        from .impl import CompressedKVImpl  # lazy: avoid circular import
        return CompressedKVImpl

    @staticmethod
    def get_builder_cls():
        return CompressedKVMetadataBuilder


class CompressedKVMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, *args, **kwargs):
        # 0.19.1 signature: (kv_cache_spec, layer_names, vllm_config, device)
        super().__init__(*args, **kwargs)
        self.spec = args[0] if args else kwargs["kv_cache_spec"]
        self.states: dict[int, RequestKVState] = {}

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata,
                           fast_build)
        nblk = self.spec.blocks_per_request
        # block_table_tensor: [num_reqs, max_blocks] (GPU). Our group's row
        # holds the request's private blocks at [0, blocks_per_request);
        # entries beyond are padding and never touched by us.
        bt = common_attn_metadata.block_table_tensor[:, :nblk].cpu()
        req_states = []
        for i in range(common_attn_metadata.num_reqs):
            blocks = tuple(int(x) for x in bt[i].tolist())
            st = self.states.get(blocks[0])
            if st is None or st.blocks != blocks:
                # New request, or finished-and-blocks-reused: (re)create.
                st = RequestKVState(blocks)
                self.states[blocks[0]] = st
            req_states.append(st)
        md.req_states = req_states
        md.qsl_cpu = common_attn_metadata.query_start_loc_cpu
        md.seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        return md


def register_backend_enum() -> None:
    """Attention.__init__ (attention.py:350) resolves
    ``AttentionBackendEnum[self.attn_backend.get_name()]``; the enum is
    closed, so inject a V8_COMPRESSED member at runtime. The value follows
    the enum's convention (default class path), so get_path()/get_class()
    resolve without register_backend overrides.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if "V8_COMPRESSED" in AttentionBackendEnum._member_map_:
        return
    member = object.__new__(AttentionBackendEnum)
    member._name_ = "V8_COMPRESSED"
    member._value_ = "vllm_plugin.backend.CompressedKVBackend"
    AttentionBackendEnum._member_map_["V8_COMPRESSED"] = member
    AttentionBackendEnum._value2member_map_[member._value_] = member
    setattr(AttentionBackendEnum, "V8_COMPRESSED", member)


def patch_allocator() -> None:    """Clamp per-request block allocation to blocks_per_request.

    0.19.1 has no spec hook here: get_num_blocks_to_allocate derives the
    requirement from the token count, which would demand 4096 blocks for a
    64k request and defeat the compression. Clamping num_tokens passed to
    the original implementation caps both the initial allocation and all
    growth, keeping the request's footprint at blocks_per_request blocks.
    """
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager)

    if getattr(SingleTypeKVCacheManager, "_v8_patched", False):
        return
    orig = SingleTypeKVCacheManager.get_num_blocks_to_allocate

    def patched(self, request_id, num_tokens, new_computed_blocks,
                total_computed_tokens, num_tokens_main_model):
        spec = self.kv_cache_spec
        if isinstance(spec, CompressedKVSpec):
            cap_tokens = spec.blocks_per_request * spec.block_size
            if num_tokens > cap_tokens:
                num_tokens = cap_tokens
        return orig(self, request_id, num_tokens, new_computed_blocks,
                    total_computed_tokens, num_tokens_main_model)

    SingleTypeKVCacheManager.get_num_blocks_to_allocate = patched
    SingleTypeKVCacheManager._v8_patched = True


def spec_from_full(spec: FullAttentionSpec) -> CompressedKVSpec:
    """Convert a layer's FullAttentionSpec (built by Attention layer)."""
    return CompressedKVSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
        dtype=spec.dtype,
        retention_tokens=config.V8_COMPRESS_SLOTS,
        sink_tokens=config.V8_SINK_TOKENS,
        recent_tokens=config.V8_RECENT_TOKENS,
        obs_window=config.V8_OBS_WINDOW,
        span_window=config.V8_SPAN_WINDOW,
    )
