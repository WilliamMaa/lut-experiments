#!/usr/bin/env python3
"""Gate 3: builder/impl invariant property test (docs/31 §3 Gate 3). Pure CPU.

Run anywhere:
    python vllm_plugin/tests/test_blockplan.py [-n 20000]

Randomized scheduler-like inputs (span_end, computed, scheduled, block
rows, P, B_g) drive build_block_plan; asserts the I3/I6 contract:

  G3.1  required == cdiv(span_end, P) and available == cdiv(frontier, P)
        exactly (no off-by-one, no clamp).
  G3.2  required <= available  -> plan passes and its block_row is the
        exact valid prefix the caller supplied.
  G3.3  required >  available  -> UnitError carrying EVERY I6 field.
  G3.4  row shorter than the plan -> UnitError (builder-slice bug caught
        before tensor indexing).
  G3.5  manager-unit accounting: scheduler_mgr_blocks == cdiv(t, B_g) and
        the kernel expansion identity
        cdiv(t,B_g)*(B_g//P) >= cdiv(t,P) holds for all probed shapes
        (B_g=1056, P=32 included).
  G3.6  batch-order tie-in with Gate 2: N interleaved requests each get a
        plan keyed by their own request_id, in batch order, regardless of
        row order shuffling.
  G3.7  2026-10-08j deferred_allowance capacity clamp (docs/37): capacity
        >= the legacy value -> identical; capacity small -> pressed to
        capacity - C; C==1 decode -> min(budget, capacity - 1); capacity
        None -> legacy behavior.
  G3.8  2026-10-08j fixed budget accounting: blocks_per_request formula
        cdiv(SLOTS+STAGING, B_g)+BLOCK_MARGIN pinned at the probed
        shapes (B_g=1056 default config and the small test config), and
        plan_write_span clamps mgr_blocks_allocated at block_budget.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm_plugin import config
from vllm_plugin.blockplan import (BlockPlan, build_block_plan,
                                   deferred_allowance, plan_write_span,
                                   scheduler_mgr_blocks)
from vllm_plugin.identity import IdentityRegistry
from vllm_plugin.units import Qty, Unit, UnitError, cdiv

I6_FIELDS = ("required ", "available ", "frontier ", "computed ",
             "scheduled ", "req=", "grp=", "T=[", "B_g=", "P=", "row[:16]=")
P_SIZES = [16, 32]
BG_SIZES = [16, 32, 64, 512, 1056]


def g31_g32_g33(rng, n):
    for _ in range(n):
        pp = rng.choice(P_SIZES)
        bg = rng.choice([b for b in BG_SIZES if b % pp == 0])
        span = rng.choice([0, 1, pp - 1, pp, pp + 1,
                           rng.randint(0, 131072)])
        computed = rng.randint(0, 131072)
        scheduled = rng.choice([1, 8192, rng.randint(1, 16384)])
        frontier = computed + scheduled
        available = cdiv(frontier, pp)
        # valid prefix: exactly `available` kernel blocks (the scheduler
        # guarantee), optionally plus stale garbage after it (real rows
        # have no sentinel — block_table.py:124-171)
        row = list(range(available)) + [rng.randint(0, 999)
                                        for _ in range(rng.randint(0, 8))]
        required = cdiv(span, pp) if span > 0 else 0
        try:
            plan = build_block_plan(
                request_id=f"req-{rng.randint(0, 999)}", group_id=0,
                span_end=Qty(span, Unit.S), computed=Qty(computed, Unit.T),
                scheduled=Qty(scheduled, Unit.T), block_row=row,
                pool_page=Qty(pp, Unit.P), mgr_block_size=bg)
        except UnitError as e:
            assert required > available, \
                f"raised on a satisfiable plan: {e}"
            msg = str(e)
            for f in I6_FIELDS:
                assert f in msg, f"I6 field {f!r} missing: {msg}"
            continue
        assert required <= available, "clamped instead of raising"
        # G3.1 exactness
        assert plan.required_pool_blocks == required
        assert plan.available_pool_blocks == available
        assert plan.frontier_t == frontier and plan.span_end_t == span
        # G3.2: the row handed through is exactly what was supplied
        assert plan.block_row == tuple(row)


def g34_short_row(rng, n):
    for _ in range(n):
        pp = rng.choice(P_SIZES)
        span = rng.randint(1, 65536)
        computed, scheduled = 0, 131072  # frontier always sufficient
        row_len = cdiv(span, pp) - 1     # one block short: builder bug
        if row_len < 0:
            continue
        row = list(range(row_len))
        try:
            build_block_plan("req-x", 0, Qty(span, Unit.S),
                             Qty(computed, Unit.T),
                             Qty(scheduled, Unit.T), row,
                             Qty(pp, Unit.P), mgr_block_size=pp)
        except UnitError as e:
            assert "block row shorter than plan" in str(e), str(e)
            continue
        raise AssertionError("under-sliced row did not raise")


def g35_mgr_accounting(rng, n):
    for _ in range(n):
        pp = rng.choice(P_SIZES)
        bg = rng.choice([b for b in BG_SIZES if b % pp == 0])
        t = rng.randint(0, 131072)
        assert scheduler_mgr_blocks(t, bg) == cdiv(t, bg)
        # the expansion identity the frontier guarantee rests on
        lhs = cdiv(t, bg) * (bg // pp)
        rhs = cdiv(t, pp)
        assert lhs >= rhs, f"t={t} B_g={bg} P={pp}: {lhs} < {rhs}"
    # probed shape, exact numbers
    assert scheduler_mgr_blocks(8192, 1056) == 8
    assert cdiv(8192, 1056) * (1056 // 32) >= cdiv(8192, 32)


def g36_batch_order(rng, n):
    reg = IdentityRegistry()
    for _ in range(n):
        k = rng.randint(1, 4)
        req_ids = [f"req-{i}" for i in rng.sample(range(16), k)]
        states = reg.sync(req_ids)
        rows = {rid: [rng.randint(0, 999) for _ in range(64)]
                for rid in req_ids}
        plans = []
        for rid, st in zip(req_ids, states):
            span = rng.randint(0, 32 * 64)
            plan = build_block_plan(
                request_id=rid, group_id=0,
                span_end=Qty(span, Unit.S), computed=Qty(0, Unit.T),
                scheduled=Qty(span + 32 * 64, Unit.T),
                block_row=rows[rid], pool_page=Qty(32, Unit.P),
                mgr_block_size=1056)
            assert plan.request_id == rid
            assert plan.block_row == tuple(rows[rid])
            plans.append(plan)
        # batch order preserved, nothing cross-contaminated
        assert [p.request_id for p in plans] == req_ids
        for p in plans:
            assert all(b in rows[p.request_id] for b in p.block_row)


def g37_allowance_capacity_clamp(rng, n):
    max_seq = 131072
    for _ in range(n):
        C = rng.choice([1, 2, 8, 8192, rng.randint(1, 16384)])
        budget = rng.choice([24, 512, 1024])
        legacy = budget if C == 1 else max(budget, max_seq)
        # None capacity = legacy behavior (harness/old callers)
        assert deferred_allowance(C, budget, max_seq) == legacy
        assert deferred_allowance(C, budget, max_seq, None) == legacy
        # generous capacity: clamp does not bind
        big = legacy + C + rng.randint(0, 1000)
        assert deferred_allowance(C, budget, max_seq, big) == legacy
        # pressed capacity: allowance = min(legacy, capacity - C); cap
        # reaches all the way down to C, so the binding case is covered
        cap = rng.randint(C, C + legacy + 16)
        assert deferred_allowance(C, budget, max_seq, cap) == min(
            legacy, cap - C)
        # decode with pressed capacity: min(budget, capacity - 1)
        assert deferred_allowance(1, budget, max_seq, cap) == min(
            budget, cap - 1)
    # pinned numbers from docs/37: B_target = 18 manager blocks at
    # SLOTS=1024, STAGING=16384, B_g=1056 -> capacity 19008 tokens
    cap = 18 * 1056
    assert deferred_allowance(8192, 1024, max_seq, cap) == cap - 8192
    assert deferred_allowance(1, 1024, max_seq, cap) == 1024


def g38_fixed_budget_accounting(rng, n):
    # the spec formula, pinned at the probed shapes (spec.py itself needs
    # vllm to import, so the formula is pinned here with exact numbers).
    # With the default env (SLOTS=1024, STAGING=16384, B_g=1056) docs/37
    # fixes B_target at 18 blocks; other env values still must satisfy
    # the formula shape: B_target * B_g >= SLOTS + STAGING.
    if (config.V8_COMPRESS_SLOTS, config.V8_STAGING_TOKENS) == (1024, 16384):
        assert scheduler_mgr_blocks(config.V8_COMPRESS_SLOTS
                                    + config.V8_STAGING_TOKENS, 1056) + 1 == 18
    for _ in range(n):
        bg = rng.choice([b for b in BG_SIZES if b % 32 == 0])
        slots = rng.randint(1, 4096)
        staging = rng.randint(0, 32768)
        budget_blocks = scheduler_mgr_blocks(slots + staging, bg) + 1
        assert budget_blocks * bg >= slots + staging
        frontier = rng.randint(1, 131072)
        C = rng.choice([1, 8192, rng.randint(1, 16384)])
        C = min(C, frontier)
        L = rng.randint(0, min(frontier - C, slots))
        allowance = deferred_allowance(
            C, slots, 131072, budget_blocks * bg)
        plan = plan_write_span(
            request_id="req-b", group_id=0, chunk_start=frontier - C,
            num_new_tokens=C, compact_len_before=L,
            computed=frontier - C, scheduled=C, mgr_block_size=bg,
            allowance=allowance, block_budget=budget_blocks)
        want_blocks = min(cdiv(frontier, bg), budget_blocks)
        assert plan.mgr_blocks_allocated == want_blocks
        assert plan.mgr_block_budget == budget_blocks
        # certified copy carries the budget (impl reads capacity from it)
        pp = 32
        certified = plan.certify_kernel(pp)
        assert certified.mgr_block_budget == budget_blocks
        assert certified.available_pool_blocks == want_blocks * (bg // pp)
        # capacity in tokens: manager and kernel expressions agree
        assert certified.available_pool_blocks * pp == \
            plan.mgr_blocks_allocated * bg
        # span never exceeds what the certified row holds: span <= total
        # <= frontier, and required <= available because B_g % P == 0
        assert plan.span_end_t <= L + C
        assert certified.required_pool_blocks <= \
            certified.available_pool_blocks
        # uncapped legacy path is unchanged
        legacy = plan_write_span(
            request_id="req-b", group_id=0, chunk_start=frontier - C,
            num_new_tokens=C, compact_len_before=L,
            computed=frontier - C, scheduled=C, mgr_block_size=bg,
            allowance=deferred_allowance(C, slots, 131072))
        assert legacy.mgr_blocks_allocated == cdiv(frontier, bg)
        assert legacy.mgr_block_budget == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for name, fn in [("G3.1-3.3 frontier contract", g31_g32_g33),
                     ("G3.4 short row raises", g34_short_row),
                     ("G3.5 manager accounting", g35_mgr_accounting),
                     ("G3.6 batch-order tie-in", g36_batch_order),
                     ("G3.7 allowance capacity clamp", g37_allowance_capacity_clamp),
                     ("G3.8 fixed-budget accounting", g38_fixed_budget_accounting)]:
        fn(rng, args.n)
        print(f"[gate3] PASS {name} (n={args.n})")
    print("[gate3] ALL PASS")


if __name__ == "__main__":
    main()
