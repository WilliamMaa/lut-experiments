"""Per-step write-span plan with fail-closed frontier check (docs/31 I3/I6,
docs/32 §2 single-plan architecture). Pure python.

The metadata builder calls plan_write_span() once per request per step
BEFORE any state is touched or tensor indexed, and attaches the plan to
the metadata; the impl consumes it via certify_kernel() and never
re-derives token counts. build_block_plan() remains as the low-level
fail-closed primitive used by earlier gates/tests.

Why the frontier math is trustworthy (unlike the v2026-10-04q/r/s saga):
- required (numerator) is bounded by the v8 write span in the SAME step the
  builder is constructing metadata for;
- available (denominator) is the scheduler's own allocation guarantee for
  that same step: it allocated cdiv(frontier_tokens, B_g) manager blocks
  before the step executed, which the worker expanded to
  cdiv(frontier_tokens, B_g) * (B_g // P) >= cdiv(frontier_tokens, P)
  kernel blocks in the block table row (block_table.py:110-118).
- Both sides are T units from the same scheduler output; nothing is
  reconstructed from lag-prone fields of previous steps.
"""
from dataclasses import dataclass

from .units import AddrCtx, Qty, Unit, UnitError, as_, cdiv, required_mgr_blocks


@dataclass(frozen=True)
class BlockPlan:
    """docs/31 I3 + docs/32 §2. All token fields in T.

    Produced ONCE per request per step by the metadata builder
    (plan_write_span below); the impl consumes it via certify_kernel() and
    must NOT re-derive any token count from tensor shapes or mutable state.
    """
    request_id: str
    group_id: int
    span_end_t: int            # write span bound = [0, span_end) in T/S
    token_computed_t: int      # this step's metadata (may lag; see frontier_t)
    token_scheduled_t: int
    frontier_t: int            # computed + scheduled: the allocation frontier
    pool_page_size: int        # P (0 = not yet probed; see certify_kernel)
    required_pool_blocks: int  # cdiv(span_end, P); valid only if pool_page>0
    available_pool_blocks: int  # cdiv(frontier, P); scheduler guarantee
    block_row: tuple           # kernel-unit ids, valid prefix ONLY
    # --- docs/32 §2 single-plan fields (builder-written, impl-checked) ---
    chunk_start_t: int = 0        # first orig position of THIS chunk (T)
    num_new_tokens: int = 0       # C, this step's scheduled tokens
    compact_len_before: int = 0   # L, compact slots at step entry
    compact_len_after_max: int = 0  # == span_end_t (post-write bound)
    mgr_blocks_allocated: int = 0   # cdiv(frontier, B_g), manager units
    mgr_block_size: int = 0         # B_g (manager tokens per block)
    mgr_block_budget: int = 0       # docs/37: fixed B_target manager
                                    # blocks per request (0 = uncapped
                                    # legacy); capacity = budget x B_g

    @property
    def write_start(self) -> int:
        return 0

    @property
    def write_end(self) -> int:
        return self.span_end_t

    def certify_kernel(self, pool_page: int) -> "BlockPlan":
        """The ONLY legal conversion of this plan into kernel units
        (docs/31 I2 / docs/32 §3): manager allocation -> expanded kernel
        row capacity (block_table.py expansion), then fail-closed
        required <= available. Runs in the impl after it probes P from
        the pool; raises BEFORE any tensor indexing. Returns a copy with
        pool_page_size / required / available filled in."""
        if pool_page <= 0:
            raise UnitError(f"pool page must be positive, got {pool_page}")
        bg = self.mgr_block_size
        if bg <= 0 or bg % pool_page != 0:
            raise UnitError(
                f"B_g {bg} is not a positive multiple of P {pool_page}; "
                f"req={self.request_id} grp={self.group_id}")
        # block_table.py expansion: each allocated manager block contributes
        # B_g // P kernel ids to the row, so the certified kernel capacity
        # is mgr_blocks_allocated * (B_g // P) (docs/31 §1.2).
        available = self.mgr_blocks_allocated * (bg // pool_page)
        required = (cdiv(self.span_end_t, pool_page)
                    if self.span_end_t > 0 else 0)
        ctx = AddrCtx(request_id=self.request_id, group_id=self.group_id,
                      token_start_t=0, token_end_t=self.span_end_t,
                      token_computed_t=self.token_computed_t,
                      token_scheduled_t=self.token_scheduled_t,
                      mgr_block_size=bg, pool_page_size=pool_page)
        if required > available:
            raise UnitError(
                f"write span exceeds allocation frontier: required "
                f"{required} pool blocks (span_end {self.span_end_t} T, "
                f"P {pool_page}) > available {available} "
                f"({self.mgr_blocks_allocated} mgr blocks x B_g//P "
                f"{bg // pool_page}); {ctx.describe()}")
        return BlockPlan(
            request_id=self.request_id, group_id=self.group_id,
            span_end_t=self.span_end_t, token_computed_t=self.token_computed_t,
            token_scheduled_t=self.token_scheduled_t, frontier_t=self.frontier_t,
            pool_page_size=pool_page, required_pool_blocks=required,
            available_pool_blocks=available, block_row=self.block_row,
            chunk_start_t=self.chunk_start_t,
            num_new_tokens=self.num_new_tokens,
            compact_len_before=self.compact_len_before,
            compact_len_after_max=self.compact_len_after_max,
            mgr_blocks_allocated=self.mgr_blocks_allocated,
            mgr_block_size=bg, mgr_block_budget=self.mgr_block_budget)


def deferred_allowance(num_new_tokens: int, retention_budget: int,
                       max_seq_tokens: int,
                       capacity_tokens: int = None) -> int:
    """Deferred-eviction allowance (single source of truth, docs/31 §4):
    decode steps (C==1) enforce the tight retention budget; prefill chunks
    allow up to max_seq_tokens so scoring happens before any eviction.

    2026-10-08j (docs/37): with fixed-budget allocation the physical pool
    per request is only B_target manager blocks, so when capacity_tokens
    is given the allowance is additionally capped at capacity_tokens - C
    (room must be left for this chunk's own staging). capacity_tokens=None
    keeps the legacy uncapped behavior (harness/old callers).
    """
    base = retention_budget if num_new_tokens == 1 else max(
        retention_budget, max_seq_tokens)
    if capacity_tokens is None:
        return base
    return min(base, capacity_tokens - num_new_tokens)


def plan_write_span(*, request_id: str, group_id: int, chunk_start: int,
                    num_new_tokens: int, compact_len_before: int,
                    computed: int, scheduled: int, mgr_block_size: int,
                    allowance: int, block_budget: int = 0) -> BlockPlan:
    """Builder-side planner (docs/32 §2): turns this step's scheduler truth
    into the single write-span plan the impl will consume.

    All inputs are scheduler-truth ints (T units) or v8's own compact_len
    (S == T positions); P is NOT known at build time, so the frontier check
    happens in pure T units (span_end <= frontier), which implies the
    kernel-block check after expansion (B_g % P == 0, docs/31 §1.1).
    FAIL-CLOSED: never clamps span down to the frontier.

    block_budget (docs/37): fixed per-request manager-block budget
    B_target; the recorded mgr_blocks_allocated is clamped to it so that
    certify_kernel's available reflects the real (capped) allocation.
    0 = legacy uncapped. The eviction allowance itself comes from
    blockplan.deferred_allowance — the caller passes the already-clamped
    value; both sides must consume that one function (docs/32 铁律).
    """
    if chunk_start != computed:
        # The first token of this chunk IS at orig position `computed`
        # (scheduler truth). Any disagreement means the integration fed
        # inconsistent numbers — refuse rather than guess.
        raise UnitError(
            f"chunk_start {chunk_start} != computed {computed} for "
            f"req={request_id}; the plan must come from one scheduler truth")
    total = compact_len_before + num_new_tokens
    # Post-write compact bound: no eviction below the allowance, eviction
    # lands at exactly allowance (sink + hh_budget + recent). Written as
    # an explicit conditional, NOT min(total, allowance) — capacity clamps
    # are banned (docs/32 禁止项).
    span_end = total if total <= allowance else allowance
    frontier = computed + scheduled
    if span_end > frontier:
        raise UnitError(
            f"write span exceeds allocation frontier: span_end {span_end} T "
            f"(L {compact_len_before} + C {num_new_tokens}, allowance "
            f"{allowance}) > frontier {frontier} T (computed {computed} + "
            f"scheduled {scheduled}); req={request_id} B_g={mgr_block_size}")
    mgr_blocks = required_mgr_blocks(
        Qty(frontier, Unit.S), Qty(mgr_block_size, Unit.BG))
    if block_budget > 0 and mgr_blocks > block_budget:
        mgr_blocks = block_budget
    return BlockPlan(
        request_id=request_id, group_id=group_id, span_end_t=span_end,
        token_computed_t=computed, token_scheduled_t=scheduled,
        frontier_t=frontier, pool_page_size=0, required_pool_blocks=0,
        available_pool_blocks=0, block_row=(),
        chunk_start_t=chunk_start, num_new_tokens=num_new_tokens,
        compact_len_before=compact_len_before,
        compact_len_after_max=span_end, mgr_blocks_allocated=mgr_blocks,
        mgr_block_size=mgr_block_size, mgr_block_budget=block_budget)


def build_block_plan(request_id: str, group_id: int,
                     span_end: Qty, computed: Qty, scheduled: Qty,
                     block_row, pool_page: Qty,
                     mgr_block_size: int) -> BlockPlan:
    """Fail-closed (I3): if the write span needs more pool blocks than this
    step's allocation frontier guarantees, raise with the full I6 context.
    NEVER clamps.

    block_row: sequence of kernel-unit block ids (stored verbatim), or an
    int meaning "certified valid length only" — the integration passes an
    int because the real row is a GPU tensor and its ids are consumed
    directly by the impl's gather (no CPU copy needed)."""
    span_end_t = as_(span_end, Unit.S)
    computed_t = as_(computed, Unit.T)
    scheduled_t = as_(scheduled, Unit.T)
    pp = as_(pool_page, Unit.P)
    frontier_t = computed_t + scheduled_t
    required = cdiv(span_end_t, pp) if span_end_t > 0 else 0
    available = cdiv(frontier_t, pp)
    row_len = block_row if isinstance(block_row, int) else len(block_row)
    row_head = () if isinstance(block_row, int) else list(block_row[:16])
    ctx = AddrCtx(request_id=request_id, group_id=group_id,
                  token_start_t=0, token_end_t=span_end_t,
                  token_computed_t=computed_t,
                  token_scheduled_t=scheduled_t,
                  mgr_block_size=mgr_block_size, pool_page_size=pp)
    if required > available:
        raise UnitError(
            f"write span exceeds allocation frontier: required "
            f"{required} pool blocks (span_end {span_end_t} T, P {pp}) > "
            f"available {available} (frontier {frontier_t} T = "
            f"computed {computed_t} + scheduled {scheduled_t}); "
            f"row[:16]={row_head}; {ctx.describe()}")
    if required > row_len:
        # The row the caller certified must cover at least the span;
        # anything shorter is an integration bug, caught before any
        # tensor indexing (rows have NO in-band sentinel, docs/31 §1.2).
        raise UnitError(
            f"block row shorter than plan: required {required}, row has "
            f"{row_len}; row[:16]={row_head}; {ctx.describe()}")
    stored_row = () if isinstance(block_row, int) else tuple(block_row)
    return BlockPlan(request_id=request_id, group_id=group_id,
                     span_end_t=span_end_t, token_computed_t=computed_t,
                     token_scheduled_t=scheduled_t, frontier_t=frontier_t,
                     pool_page_size=pp, required_pool_blocks=required,
                     available_pool_blocks=available,
                     block_row=stored_row)


def scheduler_mgr_blocks(tokens: int, mgr_block_size: int) -> int:
    """Scheduler-facing accounting only (spec.blocks_per_request, the
    allocator cap): manager blocks for a token count. This is the ONLY
    place BG-unit division is legal outside the scheduler itself."""
    if tokens <= 0:
        return 0
    return cdiv(tokens, mgr_block_size)
