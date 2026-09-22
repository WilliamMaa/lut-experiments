#!/usr/bin/env python3
"""Open-arrival (step 6) unit tests: pure Python, no torch/zmq/GPU.

Covered:
  workload  poisson mode parks doc turns in the arrivals heap (not
            ready), arrival times are non-decreasing, and sessions
            drawing the same zipf slot produce IDENTICAL chains
            (rep parity follows the doc index, not the session)
  release   _release_arrivals moves only due turns and pins t_ready
            to the arrival time (queue_s stays a true queue measure)
  advance   in poisson mode a session's next turn re-enters the
            arrivals stream after a think-time delay, never ready
  closed    closed-loop mode is untouched: ready immediately, no heap

Run:  python -m icn_proto.test_openloop
"""

import json
import os
import sys
import tempfile
import time
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from icn_proto import scheduler as sched_mod  # noqa: E402


class FakeTok:
    """Deterministic ids from text content — identical text in, identical
    ids out (the property content sharing relies on)."""

    class _Ids(list):
        def tolist(self):
            return list(self)

    def __call__(self, text, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._Ids(
            (ord(c) % 500) + 3 for c in text[:128])])


def make_args(**over):
    base = dict(policy="ours", wait_threshold=2.0, block_tokens=16,
                decode_steps=0, worker_mem_budget_mb=0.0,
                repl_min_lambda=0.05, repl_cooldown=15.0,
                max_repl_inflight=2, repl_mem_price=0.0,
                sessions=16, turns_per_session=3, doc_chars=4000,
                doc_repeat=2, doc_repeat_alt=2, doc_share=None,
                arrival="poisson", arrival_rate=1.0, zipf_n=1, zipf_s=1.2,
                think_s=2.0, seed=0)
    base.update(over)
    return Namespace(**base)


def make_sched(tmp_trace, **over):
    s = sched_mod.Scheduler(make_args(**over), ["w0", "w1"])
    s.build_workload(FakeTok())
    return s


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_openloop: {name} FAILED {detail}")


def test_workload_arrivals(tmp_trace):
    print("poisson workload construction:")
    s = make_sched(tmp_trace, zipf_n=1)      # single slot: all identical
    check("doc turns parked in arrivals", len(s.arrivals) == 16,
          f"n={len(s.arrivals)}")
    check("ready starts empty", s.ready == [])
    ts = [s.arrivals[i][0] for i in range(len(s.arrivals))]
    check("arrivals non-decreasing", ts == sorted(ts))
    check("first arrival in the future", ts[0] > time.time() - 1)
    chains = [s.turns_of[f"doc{i}"][0].prefix_ids for i in range(16)]
    check("same zipf slot => identical chains",
          all(c == chains[0] for c in chains))
    check("not finished at start", not s.finished())


def test_release_and_advance(tmp_trace):
    print("release + advance lifecycle:")
    s = make_sched(tmp_trace, zipf_n=1)
    t0 = s.arrivals[0][0]
    s._release_arrivals(t0 - 0.001)
    check("no early release", s.ready == [])
    s._release_arrivals(t0)
    check("due turn released", len(s.ready) == 1)
    turn = s.ready[0]
    check("t_ready pinned to arrival", turn.t_ready == turn.t_arrive)
    # question turn of that session must re-enter the arrivals stream
    # (heap order is arbitrary — find the entry that was just pushed)
    before = {id(x[2]) for x in s.arrivals}
    s.advance(turn.session + ":-1")
    pushed = [x[2] for x in s.arrivals if id(x[2]) not in before]
    check("next turn goes to arrivals", len(pushed) == 1,
          f"n={len(pushed)}")
    nxt = pushed[0]
    check("next turn is the question turn", nxt.turn == 0, str(nxt.turn))
    check("think delay applied", nxt.t_arrive > time.time() - 0.01)
    check("session not done", s.done_sessions == 0)


def test_closed_loop_untouched(tmp_trace):
    print("closed-loop compatibility:")
    s = make_sched(tmp_trace, arrival="none", doc_share=2)
    check("ready starts full", len(s.ready) == 16, f"n={len(s.ready)}")
    check("no arrivals heap", s.arrivals == [])
    turn = s.ready[0]
    s.advance(turn.session + ":-1")
    check("next turn straight to ready", len(s.ready) == 17
          and any(t.turn == 0 for t in s.ready))


def main():
    docs = [{"document": "alpha bravo charlie delta ", "questions": [
        "what is alpha?", "where is bravo?", "why charlie?"]},
        {"document": "echo foxtrot golf hotel ", "questions": [
         "what is echo?", "where is foxtrot?", "why golf?"]}]
    with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        path = f.name
    old_trace = sched_mod.TRACE
    sched_mod.TRACE = path
    try:
        test_workload_arrivals(path)
        test_release_and_advance(path)
        test_closed_loop_untouched(path)
    finally:
        sched_mod.TRACE = old_trace
        os.unlink(path)
    print("test_openloop: ALL PASS")


if __name__ == "__main__":
    main()
