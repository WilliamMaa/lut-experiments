#!/usr/bin/env python3
"""Gate 1: unit-conversion contract test (docs/31 §3 Gate 1). Pure CPU.

Run anywhere, no GPU/vllm/torch:
    python vllm_plugin/tests/test_units.py [-n 20000]

Property checks (thousands of random cases each):
  P1  Qty arithmetic across units raises UnitError (docs/31 I1).
  P2  slot_address == naive reference, for B_g in {16,32,64,512,1024} and
      token lengths up to 131072 (docs/31 I2 chain).
  P3  short block table raises (fail-closed), and the error message carries
      every I6 field (docs/31 I3/I6). Never clamps.
  P4  required_mgr_blocks == cdiv; 0-length spans need 0 blocks.
  P5  mgr_block_to_pool_pages covers exactly [j*B_g, (j+1)*B_g) in P units,
      both for P == B_g and B_g a multiple of P; non-multiple raises.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from units import (AddrCtx, Qty, Unit, UnitError, as_, cdiv,
                   mgr_block_to_pool_pages, required_mgr_blocks,
                   slot_address)

BG_SIZES = [16, 32, 64, 128, 512, 1024]
CTX = AddrCtx(request_id="gate1-probe", group_id=0, token_start_t=0,
              token_end_t=0, token_computed_t=0, token_scheduled_t=0,
              mgr_block_size=0, pool_page_size=0)


def ctx_with(bg: int) -> AddrCtx:
    return AddrCtx(request_id="gate1-probe", group_id=0, token_start_t=0,
                   token_end_t=16384, token_computed_t=8192,
                   token_scheduled_t=8192, mgr_block_size=bg,
                   pool_page_size=bg)


def naive_ref(s: int, bg: int):
    return s // bg, s % bg


def p1_mixed_units(rng, n):
    for _ in range(n):
        a = Qty(rng.randint(0, 10**6), rng.choice(list(Unit)))
        b = Qty(rng.randint(1, 10**3), rng.choice(list(Unit)))
        for op in (lambda: a + b, lambda: a - b, lambda: a // b,
                   lambda: a % b):
            if a.unit is b.unit:
                continue
            try:
                op()
            except UnitError:
                continue
            raise AssertionError(
                f"mixed-unit {a.unit} vs {b.unit} did not raise")


def p2_chain_matches_reference(rng, n):
    for _ in range(n):
        bg = rng.choice(BG_SIZES)
        length = rng.choice([0, 1, bg - 1, bg, bg + 1,
                             rng.randint(0, 131072)])
        block_ids = list(range(cdiv(max(length, 1), bg) + rng.randint(0, 4)))
        ctx = ctx_with(bg)
        for _ in range(4):
            s = rng.randint(0, max(length - 1, 0))
            b, r = slot_address(Qty(s, Unit.S), block_ids, Qty(bg, Unit.BG),
                                ctx)
            j_ref, r_ref = naive_ref(s, bg)
            assert (b, r) == (block_ids[j_ref], r_ref), \
                f"s={s} bg={bg}: got {(b, r)} want {(block_ids[j_ref], r_ref)}"
            assert 0 <= r < bg


def p3_short_table_raises(rng, n):
    for _ in range(n):
        bg = rng.choice(BG_SIZES)
        length = rng.randint(1, 131072)
        have = cdiv(length, bg)  # exactly enough: must NOT raise at s < length
        block_ids = list(range(have))
        ctx = ctx_with(bg)
        # use the LAST slot: it needs exactly `have` blocks, so removing
        # one block must starve it (any earlier slot might still fit)
        s = length - 1
        slot_address(Qty(s, Unit.S), block_ids, Qty(bg, Unit.BG), ctx)
        # now starve the table by one block: must raise with all I6 fields
        starved = block_ids[:-1]
        try:
            slot_address(Qty(s, Unit.S), starved, Qty(bg, Unit.BG), ctx)
        except UnitError as e:
            msg = str(e)
            for field in ("req=", "grp=", "T=[", "computed=", "scheduled=",
                          "B_g=", "P=", "need ", "have "):
                assert field in msg, f"I6 field {field!r} missing: {msg}"
            continue
        raise AssertionError(f"starved table did not raise (bg={bg} s={s})")


def p4_required_blocks(rng, n):
    for _ in range(n):
        bg = rng.choice(BG_SIZES)
        length = rng.randint(0, 131072)
        assert required_mgr_blocks(Qty(length, Unit.S),
                                   Qty(bg, Unit.BG)) == cdiv(length, bg)


def p5_pool_page_mapping(rng, n):
    for _ in range(n):
        pp = rng.choice([16, 32])
        bg = pp * rng.choice([1, 2, 4, 8, 32])
        j = rng.randint(0, 1000)
        pages = mgr_block_to_pool_pages(j, Qty(bg, Unit.BG), Qty(pp, Unit.P))
        assert list(pages) == list(range(j * bg // pp, (j + 1) * bg // pp))
    try:
        mgr_block_to_pool_pages(0, Qty(48, Unit.BG), Qty(32, Unit.P))
    except UnitError:
        pass
    else:
        raise AssertionError("B_g=48 P=32 (non-multiple) did not raise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for name, fn in [("P1 mixed-units", p1_mixed_units),
                     ("P2 chain==reference", p2_chain_matches_reference),
                     ("P3 short-table raises+I6", p3_short_table_raises),
                     ("P4 required==cdiv", p4_required_blocks),
                     ("P5 pool-page mapping", p5_pool_page_mapping)]:
        fn(rng, args.n)
        print(f"[gate1] PASS {name} (n={args.n})")
    # as_ sanity
    assert as_(Qty(7, Unit.T), Unit.T) == 7
    try:
        as_(Qty(7, Unit.S), Unit.T)
    except UnitError:
        pass
    else:
        raise AssertionError("as_ unit slip did not raise")
    print("[gate1] PASS as_ unit assertion")
    print("[gate1] ALL PASS")


if __name__ == "__main__":
    main()
