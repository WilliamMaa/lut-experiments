"""CompressedKVBackend + metadata builder (docs/31 contract implementation).

Integration facts (vLLM 0.19.1, spike/vllm-src-019):
- Attention.__init__ accepts attn_backend=... (attention.py) — injected by
  monkeypatch (vllm_plugin/__init__.py).
- No customize_spec in 0.19.1: Attention.get_kv_cache_spec returns
  FullAttentionSpec directly — the monkeypatch converts it.
- Builder signature: (kv_cache_spec, layer_names, vllm_config, device);
  build(common_prefix_len, common_attn_metadata, fast_build=False) consumes
  CommonAttentionMetadata (NO request identity in it, backend.py:323 — the
  identity truth comes from our patched _update_states instead, I4).
- Block table rows are KERNEL-unit ids (block_table.py expands manager
  blocks, B_g//P each); row padding has NO sentinel (zeros/stale), so valid
  length is never read from the row — only derived from the scheduler's
  per-step token frontier (blockplan.py, I3).
- v1 is eager-only: per-request Python state cannot be CUDA-graph captured.
  --enforce-eager is mandatory (enforced in serve.py).

Per-request lifecycle (I4/I5):
- identity: request_id only, from the step context (input_batch.req_ids +
  SchedulerOutput num_computed/num_scheduled), set by patch_step_context.
- rewind: if the scheduler's frontier (computed+scheduled) falls BEHIND
  v8's processed record (snap_len), the request was preempted/recomputed
  (or the timeline otherwise rewound): the state is dropped and rebuilt
  from the scheduler truth. This is T-unit frontier arithmetic on the same
  step's scheduler output — not a lifecycle heuristic.
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
from . import identity as idn
from . import units
from .blockplan import deferred_allowance, plan_write_span


class RequestKVState:
    """Per layer-group, per request. K/V live in the pool; this is the
    control plane: compact layout bookkeeping + the attention-mass snapshot
    table. Block rows are NOT stored here — each step's addressable row is
    certified by the BlockPlan built from that step's scheduler truth.

    Layout invariant: compact slots [0, L) of the request's certified row
    prefix always hold [sink | heavy-hitter | recent] in temporal order.
    """

    __slots__ = ("request_id", "compact_len", "orig", "snap",
                 "snap_per_head", "snap_len", "applied", "_arange")

    def __init__(self, request_id):
        self.request_id = request_id
        self.compact_len = 0
        self.orig = None                # LongTensor [cap], lazily on GPU
        self.snap = None                # fp32 [cap], attention-mass snapshot
        self.snap_per_head = None       # fp32 [H_kv, cap]
        self.snap_len = 0               # tokens processed (T), v8's record
        self.applied = None             # (chunk_start, C) marker: which step
                                        # last ran the full update; other
                                        # layers of the group attend only
        self._arange = None             # (device, arange cache)

    def arange(self, n, device):
        if self._arange is not None and self._arange[0] == device \
                and self._arange[1].numel() >= n:
            return self._arange[1]
        t = torch.arange(n, device=device)
        self._arange = (device, t)
        return t

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


class CompressedKVMetadata(FlashAttentionMetadata):
    """Standard metadata plus the contract attachments:
    req_states[i] / req_ids[i] / computed_t[i] / scheduled_t[i] correspond
    to batch row i (same order as query_start_loc). Built by the builder
    from the step context — never reconstructed from lag-prone fields."""

    req_states: list = None
    req_ids: list = None
    computed_t: list = None
    scheduled_t: list = None
    chunk_start: list = None
    block_plans: list = None      # docs/32 §2: single plan per request,
                                  # built here, consumed by impl verbatim
    qsl_cpu: object = None
    mgr_block_size: int = 0


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
        layer_names = args[1] if len(args) > 1 else \
            kwargs.get("layer_names", [])
        # Per KV-cache-group identity registry (I4). One builder instance
        # per group; layers in a group share it.
        self.registry = idn.IdentityRegistry(RequestKVState)
        self.group_label = layer_names[0] if layer_names else "?"

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata,
                           fast_build)
        ctx = idn.get_step_context()
        if ctx is None:
            raise idn.IdentityError(
                "[v8_plugin] step context missing: the _update_states patch "
                "did not run before metadata build (patch_step_context "
                "installed? worker inherited the patch?)")
        qsl = common_attn_metadata.query_start_loc_cpu.tolist()
        n_reqs = common_attn_metadata.num_reqs
        if len(ctx.req_ids) < n_reqs:
            raise idn.IdentityError(
                f"[v8_plugin] step context has {len(ctx.req_ids)} req_ids "
                f"for a batch of {n_reqs}")

        # I4: batch order is the worker's own req order for this step.
        req_ids = list(ctx.req_ids[:n_reqs])
        states = self.registry.sync(req_ids)
        self.registry.drop_finished(ctx.finished)

        computed_t, scheduled_t, chunk_starts, plans = [], [], [], []
        b_g = int(self.spec.block_size)
        # docs/37 fixed budget: B_target manager blocks per request, from
        # the spec itself. 0/missing = legacy uncapped behavior.
        budget_blocks = int(getattr(self.spec, "blocks_per_request", 0) or 0)
        capacity_tokens = budget_blocks * b_g if budget_blocks > 0 else None
        for i, rid in enumerate(req_ids):
            C = int(qsl[i + 1] - qsl[i])
            # Scheduler-truth frontier for THIS step (T units). A request
            # missing from the maps defaults to (0, C): first appearance.
            comp = int(ctx.computed.get(rid, 0))
            sched = int(ctx.scheduled.get(rid, C))
            st = states[i]
            # Rewind detection (preemption/recompute): the scheduler's
            # computed frontier must never run BEHIND v8's processed
            # record. The record is max(snap_len, compact_len): snap_len
            # alone is incomplete because C==1 chunks never enter the
            # scoring gate (snap_len stays 0 while compact_len grows —
            # property test R211), and the old comp+C test missed
            # preempted requests whose first re-scheduled chunk was
            # larger than the old snap_len (property test R21). Normal
            # flow never triggers: compact_len <= snap_len <= comp at
            # every step entry (decode steps: snap_len frozen at the
            # prefill total, compact_len <= allowance, both <= comp).
            if comp < max(st.snap_len, st.compact_len):
                print(f"[v8_plugin] rewind detected: {rid} computed {comp} "
                      f"fell behind snap_len {st.snap_len} (preemption/"
                      "recompute); state reset", flush=True)
                st = RequestKVState(rid)
                self.registry.states[rid] = st
                states[i] = st
            computed_t.append(comp)
            scheduled_t.append(sched)
            # Layer-consistency: the first token of THIS chunk is at
            # original position comp (scheduler truth), NOT st.snap_len —
            # the impl mutates snap_len during layer 0's forward, so any
            # layer reading it as chunk_start would double-count (all
            # layers of a group must see the identical value, built once
            # per step here).
            chunk_starts.append(comp)
            # docs/32 §2: the single write-span plan for this request this
            # step. The impl consumes it verbatim (certify_kernel does the
            # one legal P conversion) and never re-derives token counts.
            # 2026-10-08j: the allowance goes through the SAME
            # blockplan.deferred_allowance the impl uses, with the fixed-
            # budget capacity (docs/37) — builder and impl must consume
            # one function (docs/32 铁律).
            plans.append(plan_write_span(
                request_id=rid, group_id=0, chunk_start=comp,
                num_new_tokens=C, compact_len_before=st.compact_len,
                computed=comp, scheduled=sched,
                mgr_block_size=b_g,
                allowance=deferred_allowance(
                    C, config.V8_COMPRESS_SLOTS,
                    config.V8_MAX_SEQ_TOKENS, capacity_tokens),
                block_budget=budget_blocks))

        md.req_states = states
        md.req_ids = req_ids
        md.computed_t = computed_t
        md.scheduled_t = scheduled_t
        md.chunk_start = chunk_starts
        md.block_plans = plans
        md.qsl_cpu = common_attn_metadata.query_start_loc_cpu
        md.mgr_block_size = b_g
        return md


def patch_step_context() -> None:
    """Install the per-step truth: wrap GPUModelRunner._update_states so
    that after the persistent batch reflects the scheduler output, the
    step context (req_ids + per-request computed/scheduled + finished)
    is published for the metadata builders of this step (I4/I5).

    Runs in every worker process (patch propagates via fork from the
    parent, same as the other patches in vllm_plugin/__init__.py)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_v8_stepctx_patched", False):
        return
    orig = GPUModelRunner._update_states

    def patched(self, scheduler_output):
        ret = orig(self, scheduler_output)
        req_ids = list(self.input_batch.req_ids[:self.input_batch.num_reqs])
        computed, scheduled = {}, {}
        for new in scheduler_output.scheduled_new_reqs:
            computed[new.req_id] = new.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for rid, comp in zip(cached.req_ids, cached.num_computed_tokens):
            computed[rid] = comp
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            scheduled[rid] = n
        idn.set_step_context(idn.StepContext(
            req_ids, computed, scheduled,
            scheduler_output.finished_req_ids))
        return ret

    GPUModelRunner._update_states = patched
    GPUModelRunner._v8_stepctx_patched = True


def register_spec_manager() -> None:
    """Map CompressedKVSpec -> FullAttentionManager in the coordinator's
    spec_manager_map (engine-core proc). Behaviorally we are full attention
    with a clamped per-request budget (patch_allocator), so the stock
    manager fits; without this, get_manager_for_kv_cache_spec KeyErrors.
    """
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager, spec_manager_map)

    spec_manager_map.setdefault(CompressedKVSpec, FullAttentionManager)


def register_backend_enum() -> None:
    """Attention.__init__ resolves AttentionBackendEnum[get_name()]; the
    enum is closed, so inject a V8_COMPRESSED member at runtime."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if "V8_COMPRESSED" in AttentionBackendEnum._member_map_:
        return
    member = object.__new__(AttentionBackendEnum)
    member._name_ = "V8_COMPRESSED"
    member._value_ = "vllm_plugin.backend.CompressedKVBackend"
    AttentionBackendEnum._member_map_["V8_COMPRESSED"] = member
    AttentionBackendEnum._value2member_map_[member._value_] = member
    try:
        type.__setattr__(AttentionBackendEnum, "V8_COMPRESSED", member)
    except AttributeError:
        pass


def patch_allocator() -> None:
    """Clamp per-request block allocation at the fixed budget (docs/37).

    0.19.1 has no spec hook here: get_num_blocks_to_allocate derives the
    requirement from the token count in manager units
    (cdiv(tokens, B_g)) — units.scheduler_mgr_blocks is the same math and
    the only place outside the scheduler where BG-unit division is legal
    (docs/31 I2). The function returns the required TOTAL minus blocks
    already owned (incremental), so the correct fixed-budget semantics is
    to clamp the token count such that cdiv(tokens, B_g) <=
    spec.blocks_per_request — cdiv is monotonic, so clamping
    num_tokens to blocks_per_request * block_size caps the required
    total at B_target and the stock code returns the remaining increment.
    Returning a fixed value per call would be wrong (double-booking).
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
