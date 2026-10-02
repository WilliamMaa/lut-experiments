"""CompressedKVBackend + metadata builder.

Integration facts (vLLM commit 58b32984, verified against source):
- customize_spec is applied centrally per attention module at
  vllm/v1/worker/gpu/attn_utils.py, after the model builds the layer spec.
- The builder sees CommonAttentionMetadata (query_start_loc_cpu, seq_lens,
  block_table_tensor, ...). There is no req_id here, so per-request state is
  keyed by the request's block ids (unique while alive; a reused id means the
  old request finished -> mismatch check recreates the state).
- v1 is eager-only: per-request state is a Python object with GPU tensors,
  which cannot be CUDA-graph captured. --enforce-eager is mandatory.
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

    def ensure_gpu(self, device, h_kv, cap_hint):
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
        snap_ph = torch.zeros(h_kv, new_cap, dtype=torch.float32, device=device)
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
    # CPU copies for chunk bookkeeping (one small .cpu() per forward).
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

    @classmethod
    def customize_spec(cls, spec):
        # Only full-attention layers of the target model; Mamba/GDN specs and
        # already-compressed specs pass through untouched.
        if isinstance(spec, CompressedKVSpec):
            return spec
        if not isinstance(spec, FullAttentionSpec):
            return spec
        return CompressedKVSpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode,
            retention_tokens=config.V8_COMPRESS_SLOTS,
            sink_tokens=config.V8_SINK_TOKENS,
            recent_tokens=config.V8_RECENT_TOKENS,
            obs_window=config.V8_OBS_WINDOW,
            span_window=config.V8_SPAN_WINDOW,
        )


class CompressedKVMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, vllm_config):
        super().__init__(kv_cache_spec, vllm_config)
        self.spec = kv_cache_spec
        self.states: dict[int, RequestKVState] = {}

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata,
                           fast_build)
        nblk = self.spec.blocks_per_request
        # block_table_tensor: [num_reqs, max_blocks] (GPU). Row width for our
        # group is exactly nblk (max_num_blocks_per_req is constant).
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
