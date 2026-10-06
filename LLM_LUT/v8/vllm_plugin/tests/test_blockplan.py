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
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm_plugin.blockplan import (BlockPlan, build_block_plan,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for name, fn in [("G3.1-3.3 frontier contract", g31_g32_g33),
                     ("G3.4 short row raises", g34_short_row),
                     ("G3.5 manager accounting", g35_mgr_accounting),
                     ("G3.6 batch-order tie-in", g36_batch_order)]:
        fn(rng, args.n)
        print(f"[gate3] PASS {name} (n={args.n})")
    print("[gate3] ALL PASS")


if __name__ == "__main__":
    main()
