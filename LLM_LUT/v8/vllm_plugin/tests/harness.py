"""Fake-scheduler integration harness (docs/32): drives the REAL
CompressedKVMetadataBuilder + CompressedKVImpl (via fake_vllm shims) with
scheduler-truth step inputs on CPU, and verifies the integration contract
invariants after every step. No 35B model, no GPU — seconds per thousand
state combinations.

What is real: blockplan.plan_write_span / certify_kernel, the builder's
identity/rewind/plan logic, and the impl's whole
update+evict+write-back+attend path.
What is fake: the vllm API surface (fake_vllm), the scheduler (this
file), and the tensors (small random CPU tensors standing in for Q/K/V).
"""
import torch

from vllm_plugin import identity as idn
from vllm_plugin.backend import (CompressedKVMetadataBuilder,
                                 RequestKVState)
from vllm_plugin.impl import CompressedKVImpl


class FakeCommon:
    """CommonAttentionMetadata stand-in (only what the builder reads)."""

    def __init__(self, qsl):
        self.query_start_loc_cpu = qsl
        self.num_reqs = len(qsl) - 1


class RequestRec:
    """Scheduler-side per-request truth (the harness plays the scheduler,
    so this IS the truth the invariants are checked against).

    k_store/v_store hold every token's K/V in orig order: the prompt at
    creation, then decode-generated tokens appended as steps consume
    beyond the prompt (mirroring vLLM: decode K/V come from the model,
    not from a fixed prompt)."""

    def __init__(self, rid, prompt_len, prompt_k, prompt_v):
        self.rid = rid
        self.prompt_len = prompt_len
        self.k_store = prompt_k            # grows: [total_tokens, H_kv, D]
        self.v_store = prompt_v
        self.computed = 0                  # scheduler num_computed
        self.kernel_ids = []               # allocated row (kernel ids)
        self.active = True


class _Spec:
    """Minimal stand-in for CompressedKVSpec (only block_size is read)."""

    def __init__(self, block_size):
        self.block_size = block_size


class FakeWorld:
    """One KV-cache group: builder + impl + pool + block allocator.

    Mirrors the scheduler side of docs/31: manager blocks of B_g tokens,
    each expanded to B_g // P kernel ids in the block table row
    (block_table.py), allocated lazily as the frontier advances, freed on
    finish/preemption and recycled by later requests."""

    def __init__(self, *, P=32, B_g=64, H_kv=2, D=8, pool_blocks=256,
                 seed=0, layers=1):
        assert B_g % P == 0, "manager block must be a multiple of P"
        self.P, self.B_g, self.H_kv, self.D = P, B_g, H_kv, D
        self.kpr = B_g // P                     # kernel ids per manager block
        self.g = torch.Generator().manual_seed(seed)
        self.pool = torch.zeros(2, pool_blocks, P, H_kv, D)
        self.free_mgr = list(range(pool_blocks // self.kpr))
        self.reqs = {}
        self.builder = CompressedKVMetadataBuilder(
            _Spec(block_size=B_g), ["g0.layer0"], None, "cpu")
        # layers impls share ONE builder/state — the real topology (all
        # full-attn layers of a group). Single-layer worlds cannot see
        # shared-state non-idempotency (v2026-10-07 lesson).
        self.impls = [CompressedKVImpl(num_kv_heads=H_kv, scale=D ** -0.5)
                      for _ in range(layers)]
        self.n_steps = 0

    # ---- scheduler ops -------------------------------------------------
    def add_request(self, rid, prompt_len):
        assert rid not in self.reqs
        pk = torch.randn(prompt_len, self.H_kv, self.D, generator=self.g)
        pv = torch.randn(prompt_len, self.H_kv, self.D, generator=self.g)
        self.reqs[rid] = RequestRec(rid, prompt_len, pk, pv)
        return self.reqs[rid]

    def _allocate(self, rec, frontier):
        """Lazily allocate manager blocks so the row covers the frontier
        (SingleTypeKVCacheManager semantics)."""
        need = -(-frontier // self.B_g)  # cdiv
        while len(rec.kernel_ids) < need * self.kpr:
            m = self.free_mgr.pop(0)
            rec.kernel_ids.extend(range(m * self.kpr, (m + 1) * self.kpr))

    def _free(self, rec):
        for j in range(0, len(rec.kernel_ids), self.kpr):
            self.free_mgr.append(rec.kernel_ids[j] // self.kpr)
        rec.kernel_ids = []

    def preempt(self, rid):
        """vLLM preemption: blocks freed, num_computed resets to 0."""
        rec = self.reqs[rid]
        self._free(rec)
        rec.computed = 0

    def step(self, schedule, finished=()):
        """Execute one scheduler step.

        schedule: list of (rid, n_tokens) in batch order (chunked prefill
        or decode). finished: req_ids finished this step.
        Returns (md, errs): the metadata and the invariant violations
        found for this step (errs is empty when the contract holds)."""
        pre_computed = {}
        computed, scheduled = {}, {}
        for rid, n in schedule:
            rec = self.reqs[rid]
            pre_computed[rid] = rec.computed
            computed[rid] = rec.computed
            scheduled[rid] = n
            self._allocate(rec, rec.computed + n)
        ctx = idn.StepContext([rid for rid, _ in schedule], computed,
                              scheduled, finished)
        idn.set_step_context(ctx)

        qsl = [0]
        for _, n in schedule:
            qsl.append(qsl[-1] + n)
        common = FakeCommon(torch.tensor(qsl, dtype=torch.int32))
        md = self.builder.build(0, common)

        # The real super().build() produces block_table; the fake does
        # not, so publish rows exactly as the worker sees them
        # (kernel-unit ids, zero padding, NO in-band sentinel).
        max_len = max((len(self.reqs[rid].kernel_ids) for rid, _ in
                       schedule), default=0)
        bt = torch.zeros(len(schedule), max(max_len, 1), dtype=torch.int64)
        for i, (rid, _) in enumerate(schedule):
            ids = self.reqs[rid].kernel_ids
            bt[i, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
        md.block_table = bt

        T = qsl[-1]
        H = self.H_kv * 2
        q = torch.randn(T, H, self.D, generator=self.g)
        k = torch.zeros(T, self.H_kv, self.D)
        v = torch.zeros(T, self.H_kv, self.D)
        off = 0
        rows_at_serve = {}
        for rid, n in schedule:
            rec = self.reqs[rid]
            rows_at_serve[rid] = list(rec.kernel_ids)
            s = rec.computed
            if s + n > len(rec.k_store):
                # decode-generated tokens: extend the store the way the
                # model would produce new K/V beyond the prompt
                extra = s + n - len(rec.k_store)
                rec.k_store = torch.cat([
                    rec.k_store,
                    torch.randn(extra, self.H_kv, self.D,
                                generator=self.g)])
                rec.v_store = torch.cat([
                    rec.v_store,
                    torch.randn(extra, self.H_kv, self.D,
                                generator=self.g)])
            k[off:off + n] = rec.k_store[s:s + n]
            v[off:off + n] = rec.v_store[s:s + n]
            off += n
        out = torch.zeros(T, H, self.D)
        for impl in self.impls:
            impl.forward(None, q, k, v, self.pool, md, out)

        for rid, n in schedule:
            self.reqs[rid].computed += n
        for rid in finished:
            self._free(self.reqs[rid])
            self.reqs[rid].active = False
        self.n_steps += 1
        return md, self.verify(md, schedule, pre_computed, out, finished,
                               rows_at_serve)

    # ---- invariant verification (docs/32) ------------------------------
    def verify(self, md, schedule, pre_computed, out, finished=(),
               rows_at_serve=None):
        errs = []
        finished = set(finished)
        if not torch.isfinite(out).all():
            errs.append("non-finite attention output")
        for i, (rid, n) in enumerate(schedule):
            rec = self.reqs[rid]
            # I4 batch-order tie-in: the served state object must BE the
            # registry's state for a live request, and finished requests
            # must be gone from the registry (drop_finished, I5).
            st = md.req_states[i]
            live = self.builder.registry.states.get(rid)
            if rid in finished:
                if live is not None:
                    errs.append(f"{rid}: finished but still in registry")
            elif live is not st:
                errs.append(f"{rid}: served state is not the registry "
                            f"state (identity desync)")
            if st.request_id != rid:
                errs.append(f"{rid}: state identity {st.request_id}")
            plan = md.block_plans[i]
            if plan.request_id != rid:
                errs.append(f"{rid}: plan identity {plan.request_id}")
            if plan.chunk_start_t != pre_computed[rid]:
                errs.append(f"{rid}: chunk_start {plan.chunk_start_t} != "
                            f"scheduler computed {pre_computed[rid]}")
            if plan.num_new_tokens != n:
                errs.append(f"{rid}: plan C {plan.num_new_tokens} != {n}")
            if plan.span_end_t > plan.frontier_t:
                errs.append(f"{rid}: span {plan.span_end_t} > frontier "
                            f"{plan.frontier_t}")
            if plan.write_start != 0 or plan.write_end != plan.span_end_t:
                errs.append(f"{rid}: write span fields inconsistent")
            if plan.mgr_blocks_allocated * self.B_g < plan.frontier_t:
                errs.append(f"{rid}: mgr allocation does not cover frontier")
            certified = plan.certify_kernel(self.P)
            if certified.required_pool_blocks > \
                    certified.available_pool_blocks:
                errs.append(f"{rid}: required "
                            f"{certified.required_pool_blocks} > available "
                            f"{certified.available_pool_blocks}")
            if st.compact_len > plan.span_end_t:
                errs.append(f"{rid}: compact_len {st.compact_len} exceeded "
                            f"plan span {plan.span_end_t}")
            if st.snap_len > rec.computed:
                errs.append(f"{rid}: snap_len {st.snap_len} ahead of "
                            f"scheduler computed {rec.computed}")
            self._verify_layout(rid, rows_at_serve[rid], rec.computed,
                                rec.k_store, st, errs)
        return errs

    def _verify_layout(self, rid, kernel_ids, processed, k_store, st, errs):
        """The strong check: compact slot s must hold exactly the K of
        original position st.orig[s] (eviction never modifies K, so the
        read-back must be exact). Catches wrong addressing, cross-request
        leaks and eviction bookkeeping bugs."""
        L = st.compact_len
        if L == 0:
            return
        o = st.orig[:L].long()
        if bool(torch.any(o[1:] <= o[:-1])):
            errs.append(f"{rid}: orig not strictly increasing: "
                        f"{o[:16].tolist()}")
        if int(o[0]) < 0 or int(o[-1]) >= processed:
            errs.append(f"{rid}: orig range [{int(o[0])}, {int(o[-1])}] "
                        f"outside processed [0, {processed})")
        idx = torch.arange(L)
        j, r = idx // self.P, idx % self.P
        if int(j[-1]) >= len(kernel_ids):
            errs.append(f"{rid}: slot addressing leaves allocated row")
            return
        bid = torch.tensor(kernel_ids, dtype=torch.int64)[j]
        got = self.pool[0, bid, r]                        # [L, H_kv, D]
        want = k_store[o]
        if not torch.allclose(got, want, atol=1e-5):
            bad = int((got - want).abs().amax(dim=(1, 2)).argmax())
            errs.append(f"{rid}: K read-back mismatch at slot {bad} "
                        f"(orig {int(o[bad])})")
