#!/usr/bin/env python3
"""Placement-controller unit tests (05 v3 §3): pure Python, no torch/zmq.

Covered:
  G_rep gating   hot tip -> replication planned; cold tip -> none
  policy gate    controller inactive under b3; active under ours/p2
  cooldown       a (tip, target) already copied is not re-planned
  full segment   replication carries the WHOLE chain (ancestors + tip);
                 a segment with no complete resident holder is skipped
  eviction       coldest tip evicted; ancestors shared with a hotter
                 resident tip protected; exclusive cold segments fully
                 dropped; busy/in-flight workers untouched; policy-shared
                 (b3 evicts too) with optimistic byte accounting
  delivered      a replication delivery lands in target residency and
                 counters without touching worker busy state

Run:  python -m icn_proto.test_controller
"""

import os
import sys
import time
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from icn_proto.scheduler import Scheduler, Turn  # noqa: E402

BLOCK = 16
N_TOK = 1024
MB = 1 << 20

ARGS = dict(policy="ours", wait_threshold=2.0, block_tokens=BLOCK,
            decode_steps=4, worker_mem_budget_mb=0.0, repl_min_lambda=0.05,
            repl_cooldown=15.0, max_repl_inflight=2, repl_mem_price=0.0)


def make_sched(policy="ours", **over):
    kw = dict(ARGS, policy=policy, **over)
    return Scheduler(Namespace(**kw), ["w0", "w1"])


def doc_turn(n=N_TOK, seed=3):
    ids = [(i * 7 + seed) % 50257 + 3 for i in range(n)]
    return Turn(f"doc{seed}", -1, list(ids), list(ids))


def hold(sched, w, turn, t_pos=N_TOK, mb=1, lam=2.0):
    """Make w hold turn's chain resumable at t_pos; register the blocks
    in the directory with `mb` MB each and demand EWMA `lam`."""
    names = sched.resume_names(turn, t_pos)
    w.resident |= set(names)
    tip = turn.tip_name_at(t_pos, "bf16", BLOCK)
    w.tips.add(tip)
    for n in names:
        e = sched.dir_add(n, mb * MB)
        e["lambda"] = lam
        e["loc"] = {"w1": lam}
    sched.tips.add(tip)
    return tip, names


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_controller: {name} FAILED {detail}")


def test_repl_gating():
    print("replication planning:")
    turn = doc_turn()
    s = make_sched("ours")
    tip, names = hold(s, s.workers[b"w0"], turn)
    plan = s._plan_repl()
    check("one action", len(plan) == 1, f"n={len(plan)}")
    a = plan[0]
    check("target is cold worker", a["target"] == b"w1", str(a["target"]))
    check("holder is w0", a["holder"] == b"w0")
    check("full segment (chain + tip)",
          set(a["names"]) == set(names) and tip in a["names"],
          f"{len(a['names'])} blocks")
    check("G>0 priced", a["g"] > 0, f"G={a['g']}")

    s2 = make_sched("b3")
    hold(s2, s2.workers[b"w0"], turn)
    check("b3 plans nothing (reactive only)", s2._plan_repl() == [])
    s3 = make_sched("p2")
    hold(s3, s3.workers[b"w0"], turn)
    check("p2 alias plans like ours", len(s3._plan_repl()) == 1)

    s4 = make_sched("ours")
    tip4, _ = hold(s4, s4.workers[b"w0"], turn)
    s4.dir[tip4]["lambda"] = 0.2      # below the G>0 break-even (~1/s)
    s4.dir[tip4]["loc"] = {"w1": 0.2}
    check("cold tip not replicated", s4._plan_repl() == [])

    s5 = make_sched("ours")
    hold(s5, s5.workers[b"w0"], turn)
    s5._apply_repl(None, s5._plan_repl())
    check("apply registers transfer", len(s5._repl) == 1)
    check("cooldown blocks immediate re-plan", s5._plan_repl() == [])

    s6 = make_sched("ours")
    tip6, names6 = hold(s6, s6.workers[b"w0"], turn)
    # no complete resident holder: the segment cannot be copied anywhere
    s6.workers[b"w0"].resident.discard(names6[5])
    check("incomplete holder skipped", s6._plan_repl() == [])


def test_eviction():
    print("eviction planning:")
    # E1: cold NON-aligned tip (510) of the SAME chain as a hot tip —
    # only the partial tip block is exclusive and gets dropped; its
    # ancestors are shared infrastructure and must survive. An
    # ALIGNED cold tip would be an interior block of the hot segment
    # (protection counts full resume sets) and could not be dropped.
    turn = doc_turn()
    s = make_sched("ours", worker_mem_budget_mb=48.0)
    w0 = s.workers[b"w0"]
    hot, _ = hold(s, w0, turn, 1024)
    cold, cold_names = hold(s, w0, turn, 510, lam=0.0)
    w0.resident_bytes = 48e6 + MB      # 1MB over budget == the tip block
    plan = s._plan_evict()
    check("one worker action", len(plan) == 1, f"n={len(plan)}")
    names = set(plan[0]["names"])
    check("cold tip evicted", cold in names, f"n_evicted={len(names)}")
    check("hot tip kept", hot not in names)
    kept_chain = set(cold_names) - {cold}
    check("shared ancestors protected", not (kept_chain & names))
    s._apply_evict(None, plan)
    check("bookkeeping updated", cold not in w0.resident
          and cold not in w0.tips and s.evicted_blocks == len(names))

    # E2: an exclusive cold segment (different doc) is dropped whole
    s2 = make_sched("ours", worker_mem_budget_mb=48.0)
    w0b = s2.workers[b"w0"]
    hot_b, hot_names = hold(s2, w0b, doc_turn(seed=11), 1024)
    cold_b, cold_names_b = hold(s2, w0b, doc_turn(seed=22), 510, lam=0.0)
    w0b.resident_bytes = 48e6 + 32 * MB   # excess == the cold segment
    plan2 = s2._plan_evict()
    names2 = set(plan2[0]["names"]) if plan2 else set()
    check("cold segment fully evicted",
          set(cold_names_b) <= names2, f"{len(names2)} blocks")
    check("hot segment intact",
          not (set(hot_names) & names2) and hot_b not in names2)

    # E3: safety guards
    w0b.busy = True
    check("busy worker untouched", s2._plan_evict() == [])
    w0b.busy = False
    s2._xfer["docX:0"] = {"stage": "fetch", "holder": b"w1",
                          "target": b"w0", "names": ["x"], "t_fetch": 0}
    check("in-flight worker untouched", s2._plan_evict() == [])


def test_eviction_shared_substrate():
    """Eviction under a memory budget is a SHARED substrate: b3 (reactive
    only, no replication) must also evict when over budget — guards the
    v2 change that removed the ours-only policy gate. Also guards the
    optimistic byte accounting in _apply_evict: without it the next
    control cycle re-evicts against the stale pre-ack byte count."""
    print("eviction is policy-shared + optimistic bytes:")
    turn = doc_turn()
    s = make_sched("b3", worker_mem_budget_mb=48.0)
    w0 = s.workers[b"w0"]
    _, names = hold(s, w0, turn)
    total = sum(s.dir[n]["bytes"] for n in names)
    w0.resident_bytes = 48e6 + total
    plan = s._plan_evict()
    check("b3 evicts under budget", len(plan) == 1, f"n={len(plan)}")
    nbytes = sum(s.dir.get(n, {}).get("bytes", 0) for n in plan[0]["names"])
    s._apply_evict(None, plan)
    check("optimistic byte accounting",
          w0.resident_bytes == 48e6, f"{w0.resident_bytes}")
    check("no double eviction next cycle", s._plan_evict() == [])

    # E2 regression (2026-09-25): the slow path must RUN eviction for
    # b3 too. The old controller-level policy gate gave b3 an
    # unbounded residency (800MB vs the 48MB budget in the b3 smoke)
    # while ours fought under the budget — an equal-budget violation.
    s2 = make_sched("b3", worker_mem_budget_mb=48.0)
    w0b = s2.workers[b"w0"]
    _, names_b = hold(s2, w0b, doc_turn(seed=8))
    w0b.resident_bytes = 48e6 + sum(s2.dir[n]["bytes"] for n in names_b)
    s2.controller(None)           # the event-driven path, not _plan_*
    check("controller evicts under b3",
          s2.evictions == 1 and s2.evicted_blocks > 0,
          f"evictions={s2.evictions}")
    # and replication stays gated: b3 controller plans no copies
    check("controller plans no repl under b3",
          s2.repl_planned == 0 and s2._repl == {})


def test_delivered_repl():
    print("replication delivery:")
    turn = doc_turn()
    s = make_sched("ours")
    tip, names = hold(s, s.workers[b"w0"], turn)
    s._apply_repl(None, s._plan_repl())
    rid = next(iter(s._repl))

    class FakeSock:
        def send_multipart(self, frames):
            pass

    # holder replies, scheduler forwards the payload to the target
    s.on_message(FakeSock(), b"w0",
                 {"type": "fetched", "ok": True, "names": names},
                 payload=b"y" * 50)
    check("fetch ack advanced stage", s._repl[rid]["stage"] == "deliver")
    # target stores the payload and acks; no turn may run. The worker's
    # delivered ack carries NO payload (production shape — regression
    # for the len(None) crash of run 20260921 matrix cells)
    s.on_message(None, b"w1",
                 {"type": "delivered", "names": names, "repl": rid},
                 payload=None)
    w1 = s.workers[b"w1"]
    check("blocks resident on target", all(n in w1.resident for n in names))
    check("tip registered on target", tip in w1.tips)
    check("transfer table drained", s._repl == {})
    check("counters", s.replications == 1 and s.replicated_bytes == 50)
    check("worker not marked busy", not w1.busy)


def test_delivered_updates_residency():
    """Regression: a demand fetch delivery must update the TARGET
    worker's residency view immediately. A stale view made the
    controller plan a redundant replication of just-delivered blocks
    (run 20260921_144351), whose ack then crossed with the demand
    fetch's ack."""
    print("delivered-demand residency update:")
    turn = doc_turn()
    s = make_sched("ours")
    tip, names = hold(s, s.workers[b"w0"], turn)
    s._xfer["doc9:0"] = {"stage": "deliver", "holder": b"w0",
                         "target": b"w1", "names": names, "bytes": 100,
                         "t_fetch": time.time(), "turn": turn, "E_loc": 0,
                         "decision": None}
    sent = []

    class FakeSock:
        def send_multipart(self, frames):
            sent.append(frames)

    s.on_message(FakeSock(), b"w1",
                 {"type": "delivered", "names": names}, payload=b"x" * 100)
    w1 = s.workers[b"w1"]
    check("blocks resident on target", all(n in w1.resident for n in names))
    check("tip registered on target", tip in w1.tips)
    check("assign was dispatched", len(sent) == 1)
    check("no redundant replication planned", s._plan_repl() == [])


def test_repl_ack_not_swallowed():
    """A replication ack must be attributed by its echoed rid even when
    a demand xfer with identical names exists."""
    print("replication ack attribution:")
    turn = doc_turn()
    s = make_sched("ours")
    tip, names = hold(s, s.workers[b"w0"], turn)
    s._apply_repl(None, s._plan_repl())
    rid = next(iter(s._repl))
    # concurrent demand xfer, same names, same target, stage deliver
    s._xfer["doc9:0"] = {"stage": "deliver", "holder": b"w0",
                         "target": b"w1", "names": names, "bytes": 100,
                         "t_fetch": time.time(), "turn": turn, "E_loc": 0,
                         "decision": None}

    class FakeSock:
        def send_multipart(self, frames):
            pass

    # holder acks the replication fetch -> stage becomes deliver
    s.on_message(FakeSock(), b"w0",
                 {"type": "fetched", "ok": True, "names": names},
                 payload=b"y" * 50)
    # the delivery ack carries repl rid and must NOT be swallowed by
    # the demand xfer with identical names
    s.on_message(None, b"w1",
                 {"type": "delivered", "names": names, "repl": rid},
                 payload=b"x" * 100)
    check("replication counted", s.replications == 1)
    check("demand xfer untouched", "doc9:0" in s._xfer)
    check("repl table drained", s._repl == {})


def test_status_snapshot_clobber():
    """Regression 2026-09-27 (s0-leg STALE_RESUME hard failures): a
    worker status is a full snapshot tagged with the seq of the last
    command the worker processed. A snapshot whose seq predates one of
    our evict commands was taken before the worker processed that
    eviction and still lists the evicted names — the status handler's
    blind full replace resurrected them into the scheduler's
    (optimistically post-eviction) view, and the next dispatch planned
    a local resume the worker could not serve. Wall-clock timestamps
    did NOT close the hole: a snapshot taken while our evict was still
    in flight to the worker has a fresh timestamp but pre-eviction
    content. Fix: per-worker causal seqs on both sides; the handler
    subtracts evictions with a greater seq."""
    print("status snapshot clobber:")
    turn = doc_turn()
    s = make_sched("b3", worker_mem_budget_mb=48.0)
    w0 = s.workers[b"w0"]
    _, names = hold(s, w0, turn)
    w0.resident_bytes = 48e6 + sum(s.dir[n]["bytes"] for n in names)
    plan = s._plan_evict()
    check("eviction planned", len(plan) == 1, f"n={len(plan)}")
    evicted = set(plan[0]["names"])
    s._apply_evict(None, plan)
    check("optimistic view dropped", not (evicted & w0.resident))

    # stale snapshot: seq 0 — taken BEFORE the worker processed the
    # evict command (the seq-1 command). Covers both the late-queue
    # case and the in-flight-crossing case (fresh wall time, stale
    # causality).
    stale = {"type": "status", "seq": 0,
             "resident": list(w0.resident | evicted),
             "tips": list(w0.tips | evicted),
             "resident_bytes": 1, "spilled": [],
             "spill_bytes": 0, "spill": {}}
    s.on_message(None, b"w0", stale, payload=None)
    check("stale snapshot cannot resurrect evicted names",
          not (evicted & w0.resident))
    check("stale snapshot cannot resurrect evicted tips",
          not (evicted & w0.tips))

    # a snapshot with seq >= the evict command's seq proves the worker
    # processed the eviction: the name is genuinely gone, the log
    # entry retires
    fresh = {"type": "status", "seq": 99, "resident": [],
             "tips": [], "resident_bytes": 0, "spilled": [],
             "spill_bytes": 0, "spill": {}}
    s.on_message(None, b"w0", fresh, payload=None)
    check("fresh snapshot applied", w0.resident == set())
    check("evict log retired", w0.evict_t == {})


def test_watchdog_fail():
    """A worker stall must fail the turn and advance the session (the
    share=8/ours/budget=48 cell of the 20260921 v2 run hung forever on
    a stuck worker). A late result for the failed turn must not advance
    the session twice."""
    print("watchdog hard-fail:")
    s = make_sched("ours")
    turn = doc_turn(seed=5)
    s.turns_of = {"doc5": [turn]}
    w0 = s.workers[b"w0"]
    w0.busy = True
    w0.current = "doc5:-1"
    w0.t_assign = time.time() - 301
    s.records.append({"request_id": "doc5:-1", "worker": "w0",
                      "t_assigned": time.time() - 301, "E": 0, "fp": "x",
                      "xfer_blocks": [], "transfer_bytes": 0,
                      "xfer_s": 0.0, "decision": None})
    s._watchdog_fail(None, time.time())
    rec = s.records[0]
    check("turn failed by watchdog", rec.get("ok") is False)
    check("worker freed", not w0.busy and w0.current is None)
    check("session advanced", s.done_sessions == 1)
    check("error tagged", "watchdog" in (rec.get("error") or ""))

    class FakeSock:
        def send_multipart(self, frames):
            pass

    # late result from the wedged worker must not double-advance
    s.on_message(FakeSock(), b"w0",
                 {"type": "result", "request_id": "doc5:-1", "ok": True},
                 payload=None)
    check("late result ignored", s.done_sessions == 1
          and s.records[0]["ok"] is False)


def test_repl_missing_suffix():
    """E1 (2026-09-23): the copy set is the MISSING SUFFIX of the
    resume set relative to the target, not the whole chain. A 40-turn
    session's full chain (~100MB) exceeds any per-worker budget, so
    full-segment copies can never find a complete holder and the
    controller degenerates to b3. When the target already holds the
    lower chain, only the upper suffix is priced and copied."""
    print("missing-suffix replication:")
    turn = doc_turn()
    s = make_sched("ours")
    w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
    tip, names_full = hold(s, w0, turn, N_TOK)
    # w1 already holds the lower half of the same chain
    low = set(s.resume_names(turn, N_TOK // 2))
    w1.resident |= low
    plan = s._plan_repl()
    check("one action", len(plan) == 1, f"n={len(plan)}")
    a = plan[0]
    expect = set(names_full) - low
    check("only the missing suffix is copied",
          set(a["names"]) == expect, f"{len(a['names'])} blocks "
          f"vs expected {len(expect)}")
    check("suffix bytes priced", a["bytes"] < sum(
        s.dir[n]["bytes"] for n in names_full))
    check("holder is w0 (holds the suffix)", a["holder"] == b"w0")
    # a target holding the WHOLE chain needs nothing
    s2 = make_sched("ours")
    hold(s2, s2.workers[b"w0"], turn)
    s2.workers[b"w1"].resident |= set(names_full)
    check("fully-warmed target skipped", s2._plan_repl() == [])


def test_repl_recompute_pricing():
    """E1 ΔC_future calibration (pre-registered 07 §4): a hit's worth
    is the avoided RE-DERIVATION (missing_tokens/prefill_rate), not the
    transfer time. With byte-heavy blocks (0.5MB/16tok) and lam=0.9/s
    the step-4 placeholder (ΔC = C_copy, break-even 1 hit/s) rejects;
    the calibrated formula accepts: 0.9 * 1024/2500 - 32MB/100MB/s > 0.
    Fresh-sched rates are the EWMA initials (2500 tok/s, 100MB/s)."""
    print("recompute-priced G_rep:")
    turn = doc_turn()
    s = make_sched("ours")
    hold(s, s.workers[b"w0"], turn, N_TOK, mb=0.5, lam=0.9)
    plan = s._plan_repl()
    check("fires below the old 1/s break-even", len(plan) == 1,
          f"n={len(plan)}")
    if plan:
        c_copy = 32 * MB / s.xfer_rate          # 64 blocks x 0.5 MiB
        g_expect = 0.9 * (N_TOK / s.PREFILL_RATE0) - c_copy
        check("g reflects recompute pricing",
              abs(plan[0]["g"] - round(g_expect, 4)) < 1e-3,
              f"G={plan[0]['g']} expected ~{round(g_expect, 4)}")
    # sanity: truly cold demand still rejected under the new formula
    s2 = make_sched("ours")
    hold(s2, s2.workers[b"w0"], turn, N_TOK, mb=0.5, lam=0.5)
    check("0.5/s below calibrated break-even (~0.78/s) too",
          s2._plan_repl() == [])


def main():
    test_repl_gating()
    test_repl_missing_suffix()
    test_repl_recompute_pricing()
    test_eviction()
    test_eviction_shared_substrate()
    test_status_snapshot_clobber()
    test_watchdog_fail()
    test_delivered_repl()
    test_delivered_updates_residency()
    test_repl_ack_not_swallowed()
    print("test_controller: ALL PASS")


if __name__ == "__main__":
    main()
