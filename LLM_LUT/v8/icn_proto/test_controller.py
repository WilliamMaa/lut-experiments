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


def main():
    test_repl_gating()
    test_eviction()
    test_eviction_shared_substrate()
    test_delivered_repl()
    test_delivered_updates_residency()
    test_repl_ack_not_swallowed()
    print("test_controller: ALL PASS")


if __name__ == "__main__":
    main()
