"""Per-step write-span plan with fail-closed frontier check (docs/31 I3/I6).

Pure python. The metadata builder calls build_block_plan() once per request
per step BEFORE any state is touched or tensor indexed. The plan is the
single source of truth for how many pool blocks the write-back may span and
which block ids are addressable this step.

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
    """docs/31 I3. All token fields in T; block counts in kernel units (P)."""
    request_id: str
    group_id: int
    span_end_t: int            # write span = [0, span_end) in T/S
    token_computed_t: int      # this step's metadata (may lag; see frontier_t)
    token_scheduled_t: int
    frontier_t: int            # computed + scheduled: the allocation frontier
    pool_page_size: int        # P
    required_pool_blocks: int  # cdiv(span_end, P)
    available_pool_blocks: int  # cdiv(frontier, P); scheduler guarantee
    block_row: tuple           # kernel-unit ids, valid prefix ONLY


def build_block_plan(request_id: str, group_id: int,
                     span_end: Qty, computed: Qty, scheduled: Qty,
                     block_row, pool_page: Qty,
                     mgr_block_size: int) -> BlockPlan:
    """Fail-closed (I3): if the write span needs more pool blocks than this
    step's allocation frontier guarantees, raise with the full I6 context.
    NEVER clamps."""
    span_end_t = as_(span_end, Unit.S)
    computed_t = as_(computed, Unit.T)
    scheduled_t = as_(scheduled, Unit.T)
    pp = as_(pool_page, Unit.P)
    frontier_t = computed_t + scheduled_t
    required = cdiv(span_end_t, pp) if span_end_t > 0 else 0
    available = cdiv(frontier_t, pp)
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
            f"row[:16]={list(block_row[:16])}; {ctx.describe()}")
    if required > len(block_row):
        # The row the caller sliced must cover at least the span; slicing
        # is the builder's job, and an under-sliced row is a builder bug,
        # caught here before any tensor indexing.
        raise UnitError(
            f"block row shorter than plan: required {required}, row has "
            f"{len(block_row)}; row[:16]={list(block_row[:16])}; "
            f"{ctx.describe()}")
    return BlockPlan(request_id=request_id, group_id=group_id,
                     span_end_t=span_end_t, token_computed_t=computed_t,
                     token_scheduled_t=scheduled_t, frontier_t=frontier_t,
                     pool_page_size=pp, required_pool_blocks=required,
                     available_pool_blocks=available,
                     block_row=tuple(block_row))


def scheduler_mgr_blocks(tokens: int, mgr_block_size: int) -> int:
    """Scheduler-facing accounting only (spec.blocks_per_request, the
    allocator cap): manager blocks for a token count. This is the ONLY
    place BG-unit division is legal outside the scheduler itself."""
    if tokens <= 0:
        return 0
    return cdiv(tokens, mgr_block_size)
