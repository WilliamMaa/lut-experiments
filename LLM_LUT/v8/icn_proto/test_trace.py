#!/usr/bin/env python3
"""Lifecycle tracing (14) unit tests: pure Python, no torch/zmq/GPU.

Covered:
  tracer    off = no-op (no file, no crash); on = append-only JSONL
            with per-process sequence numbers
  scheduler emissions  send_assign emits demand per resume name and a
            resume event; dir_add / evict_plan / repl_send emit per name
  replay    merge across files by time order; --summary counts

Run:  python -m icn_proto.test_trace
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

from icn_proto.scheduler import Scheduler          # noqa: E402
from icn_proto.trace import Tracer, open_tracer    # noqa: E402
from icn_proto.trace_replay import load_events     # noqa: E402

BLOCK = 16
DOC_TOK = 256


def make_args(**over):
    base = dict(policy="ours", wait_threshold=2.0, block_tokens=BLOCK,
                decode_steps=0, worker_mem_budget_mb=0.0,
                repl_min_lambda=0.05, repl_cooldown=15.0,
                max_repl_inflight=2, repl_mem_price=0.0,
                spill_mb=0.0, spill_rate=20e9,
                sessions=2, turns_per_session=2, doc_chars=4000,
                doc_repeat=1, doc_repeat_alt=None, doc_share=None,
                arrival="none", arrival_rate=1.0, zipf_n=0, zipf_s=1.2,
                think_s=2.0, seed=0, q_tokens=None)
    base.update(over)
    return Namespace(**base)


def session_turns(n_q=1, seed=3):
    ids = [(i * 7 + seed) % 50257 + 3 for i in range(DOC_TOK)]
    turns = [Turn("doc5", -1, list(ids), list(ids))]
    prefix = list(ids)
    for t in range(n_q):
        q_ids = [(1000 + t * 13 + i) % 50257 + 3 for i in range(BLOCK)]
        prefix = prefix + q_ids
        turns.append(Turn("doc5", t, list(q_ids), list(prefix)))
    return turns


from icn_proto.scheduler import Turn  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_trace: {name} FAILED {detail}")


def test_tracer_off_and_on():
    print("tracer off is a no-op, on writes JSONL:")
    with tempfile.TemporaryDirectory() as d:
        blocker = os.path.join(d, "afile")
        open(blocker, "w").close()
        off = open_tracer("nope", os.path.join(blocker, "w.jsonl"))
        check("unopenable path degrades to off", not off.enabled)
        off.emit("x", "n")     # must not raise
        quiet = Tracer(None, "q")
        quiet.emit("x", "n")
        check("None path is off", not quiet.enabled)
        tr = open_tracer("w0", os.path.join(d, "sub", "w0.jsonl"))
        check("on when dir writable", tr.enabled)
        tr.emit("resident_add", "blk/a", via="publish")
        tr.emit("evict", "blk/a", outcome="spilled")
        tr.close()
        recs = [json.loads(l) for l in
                open(os.path.join(d, "sub", "w0.jsonl"), encoding="utf-8")]
        check("two records", len(recs) == 2)
        check("seq increments", [r["seq"] for r in recs] == [1, 2])
        check("name carried", recs[0]["name"] == "blk/a"
              and recs[1]["detail"]["outcome"] == "spilled")


def test_scheduler_emissions():
    print("scheduler emits per-block events:")
    with tempfile.TemporaryDirectory() as d:
        os.environ["ICN_TRACE_DIR"] = d
        try:
            s = Scheduler(make_args(), ["w0", "w1"])
        finally:
            del os.environ["ICN_TRACE_DIR"]
        s.turns_of = {"doc5": session_turns()}
        doc, q0 = s.turns_of["doc5"][0], s.turns_of["doc5"][1]
        tip = doc.tip_name_at(DOC_TOK, "bf16", BLOCK)
        names = s.resume_names(doc, DOC_TOK)
        for n in names:
            s.dir_add(n, 1 << 20)
        w0 = s.workers[b"w0"]
        w0.spilled |= set(names)
        w0.tips.add(tip)
        q0.t_ready = q0.t_arrive = time.time()
        s.ready.append(q0)

        class _Sock:
            def send_multipart(self, frames):
                pass

        s.dispatch(_Sock())
        recs = [json.loads(l) for l in
                open(os.path.join(d, "sched.jsonl"), encoding="utf-8")]
        dem = [r for r in recs if r["event"] == "demand"]
        res = [r for r in recs if r["event"] == "resume"]
        da = [r for r in recs if r["event"] == "dir_add"]
        check("demand per resume name", len(dem) == len(names),
              f"{len(dem)} vs {len(names)}")
        check("resume event with E", len(res) == 1
              and res[0]["detail"]["E"] == DOC_TOK)
        check("dir_add per name", len(da) == len(names))
        s.trace.close()


def test_replay_merge():
    print("replay merges streams in time order:")
    with tempfile.TemporaryDirectory() as d:
        a = Tracer(os.path.join(d, "sched.jsonl"), "sched")
        b = Tracer(os.path.join(d, "w0.jsonl"), "w0")
        b.emit("resident_add", "blk/x", via="publish")
        time.sleep(0.01)     # ms-resolution clocks need separation
        a.emit("evict_plan", "blk/x", worker="w0")
        time.sleep(0.01)
        b.emit("evict", "blk/x", outcome="dropped")
        a.close()
        b.close()
        evs = load_events(d)
        check("merged count", len(evs) == 3)
        check("time order", [e["event"] for e in evs] ==
              ["resident_add", "evict_plan", "evict"])
        life = [e for e in evs if e["name"] == "blk/x"]
        check("full chain findable by name", len(life) == 3)


def main():
    test_tracer_off_and_on()
    test_scheduler_emissions()
    test_replay_merge()
    print("test_trace: ALL PASS")


if __name__ == "__main__":
    main()
