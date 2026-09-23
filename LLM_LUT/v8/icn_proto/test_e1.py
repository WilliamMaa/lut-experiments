#!/usr/bin/env python3
"""E1 (07-regime-study §4) unit tests: pure Python, no torch/zmq/GPU.

Covered:
  migration     a session's next turn may be assigned to a DIFFERENT
                worker (the E1 residency-opportunity premise); the
                record carries `migrated`
  opportunity   a question turn assigned away from the session's last
                worker counts a remote-resume opportunity; landing on
                the same worker does not
  repl-served   an opportunity served at the exact latest tip that is
                resident ONLY via controller replication counts
                repl_served_local; a b3 demand-fetch delivery (resident
                but not via_repl) must NOT count
  q-tokens      --q-tokens N pins every question turn's token delta to
                exactly N (tile short / truncate long), questions cycle
                past the end of the sample list
  summary       rederivation_tokens = prefill beyond the turn's genuine
                growth; migration rate; c_recompute buckets;
                state_lifetime percentiles; degrade counter surfaces

Run:  python -m icn_proto.test_e1
"""

import json
import os
import sys
import tempfile
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from icn_proto.scheduler import Scheduler, Turn  # noqa: E402

BLOCK = 16
DOC_TOK = 256          # block-aligned doc prefix


class FakeSock:
    def send_multipart(self, frames):
        pass


class FakeTok:
    """Deterministic ids from text — identical text in, identical ids."""

    class _Ids(list):
        def tolist(self):
            return list(self)

    def __call__(self, text, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._Ids(
            (ord(c) % 500) + 3 for c in text[:128])])


def make_args(**over):
    base = dict(policy="ours", wait_threshold=2.0, block_tokens=BLOCK,
                decode_steps=0, worker_mem_budget_mb=0.0,
                repl_min_lambda=0.05, repl_cooldown=15.0,
                max_repl_inflight=2, repl_mem_price=0.0,
                sessions=2, turns_per_session=2, doc_chars=4000,
                doc_repeat=1, doc_repeat_alt=None, doc_share=None,
                arrival="none", arrival_rate=1.0, zipf_n=0, zipf_s=1.2,
                think_s=2.0, seed=0, q_tokens=None)
    base.update(over)
    return Namespace(**base)


def make_sched(**over):
    return Scheduler(make_args(**over), ["w0", "w1"])


def session_turns(n_q=1, seed=3):
    ids = [(i * 7 + seed) % 50257 + 3 for i in range(DOC_TOK)]
    turns = [Turn("doc5", -1, list(ids), list(ids))]
    prefix = list(ids)
    for t in range(n_q):
        q_ids = [(1000 + t * 13 + i) % 50257 + 3 for i in range(BLOCK)]
        prefix = prefix + q_ids
        turns.append(Turn("doc5", t, list(q_ids), list(prefix)))
    return turns


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_e1: {name} FAILED {detail}")


def test_migration_and_opportunity():
    print("migration + remote-resume opportunity:")
    s = make_sched()
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc, q0 = turns[0], turns[1]

    # doc turn lands on w0
    s.send_assign(FakeSock(), b"w0", doc, 0)
    rec = s.records[-1]
    check("doc turn not migrated", rec.get("migrated") is False)
    check("last worker tracked", s._last_worker["doc5"] == "w0")
    check("doc turn no opportunity", s.remote_resume_opportunities == 0)

    # question turn lands on the OTHER worker, cold (E=0)
    s.send_assign(FakeSock(), b"w1", q0, 0)
    rec = s.records[-1]
    check("question turn migrated", rec.get("migrated") is True)
    check("opportunity counted", s.remote_resume_opportunities == 1
          and rec.get("remote_opp") is True)
    check("not repl-served (tip not resident)", s.repl_served_local == 0
          and "repl_served" not in rec)

    # same-worker landing: no opportunity, no migration
    s2 = make_sched()
    s2.turns_of = {"doc5": session_turns()}
    s2.send_assign(FakeSock(), b"w0", s2.turns_of["doc5"][0], 0)
    s2.send_assign(FakeSock(), b"w0", s2.turns_of["doc5"][1], DOC_TOK)
    rec2 = s2.records[-1]
    check("same-worker not migrated", rec2.get("migrated") is False)
    check("same-worker no opportunity",
          s2.remote_resume_opportunities == 0
          and "remote_opp" not in rec2)


def test_repl_served_local():
    print("replication-served-local attribution:")
    tip_holder = None

    def setup(with_via_repl):
        s = make_sched()
        turns = session_turns()
        s.turns_of = {"doc5": turns}
        doc, q0 = turns[0], turns[1]
        s.send_assign(FakeSock(), b"w0", doc, 0)
        tip = doc.tip_name_at(DOC_TOK, "bf16", BLOCK)
        if with_via_repl:
            s.workers[b"w1"].via_repl.add(tip)
        return s, q0, tip

    # (a) tip resident via REPLICATION, resume reaches it in full
    s, q0, tip = setup(with_via_repl=True)
    s.send_assign(FakeSock(), b"w1", q0, DOC_TOK)
    rec = s.records[-1]
    check("repl-served counted", s.repl_served_local == 1
          and rec.get("repl_served") is True)
    check("opportunity also counted", s.remote_resume_opportunities == 1)

    # (b) tip resident via a b3 DEMAND fetch (not via_repl): must NOT count
    s, q0, tip = setup(with_via_repl=False)
    s.workers[b"w1"].resident.add(tip)      # demand-delivered
    s.send_assign(FakeSock(), b"w1", q0, DOC_TOK)
    rec = s.records[-1]
    check("demand fetch not repl-served", s.repl_served_local == 0
          and "repl_served" not in rec)

    # (c) partial resume below the tip is not "served local"
    s, q0, tip = setup(with_via_repl=True)
    s.send_assign(FakeSock(), b"w1", q0, DOC_TOK - BLOCK)
    rec = s.records[-1]
    check("partial resume not repl-served", s.repl_served_local == 0)

    # eviction forgets the via_repl provenance
    s.dir_add(tip, 1 << 20)
    s._apply_evict(None, [{"worker": b"w1", "names": [tip]}])
    check("evict clears via_repl", tip not in s.workers[b"w1"].via_repl)


def test_q_tokens():
    print("q-tokens workload shaping:")
    docs = [{"document": "alpha bravo charlie delta ",
             "questions": ["q?", "m" * 200]},          # short + long (>128)
            {"document": "echo foxtrot golf hotel ",
             "questions": ["what is echo?"]}]
    with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        path = f.name
    from icn_proto import scheduler as sched_mod
    old_trace = sched_mod.TRACE
    sched_mod.TRACE = path
    try:
        s = make_sched(sessions=2, turns_per_session=5, q_tokens=40)
        s.build_workload(FakeTok())
        for sess, turns in s.turns_of.items():
            deltas = [turns[k + 1].cum_tokens - turns[k].cum_tokens
                      for k in range(len(turns) - 1)]
            check(f"{sess}: every delta pinned to 40",
                  all(d == 40 for d in deltas), str(deltas))
            check(f"{sess}: questions cycled (turn0 == turn2)",
                  turns[1].prefill_ids == turns[3].prefill_ids)
        # natural length preserved when q_tokens is off (FakeTok:
        # ids == chars of text[:128]; doc0 q0="q?" -> 4, q1=200 chars
        # -> truncated to 128)
        s2 = make_sched(sessions=1, turns_per_session=2, q_tokens=None)
        s2.build_workload(FakeTok())
        t2 = s2.turns_of["doc0"]
        check("default keeps natural length (short q)",
              t2[1].cum_tokens - t2[0].cum_tokens == len("\n\nq?"))
        check("default keeps natural length (long q)",
              t2[2].cum_tokens - t2[1].cum_tokens == 128)
    finally:
        sched_mod.TRACE = old_trace
        os.unlink(path)


def test_summary_metrics():
    print("E1 summary metrics:")
    with tempfile.TemporaryDirectory() as out:
        s = make_sched(sessions=1, turns_per_session=2)
        turns = session_turns(n_q=2)
        s.turns_of = {"doc5": turns}
        s.args.out = out
        now = 1_000.0
        s.records = [
            {"request_id": "doc5:-1", "ok": True, "prefill_tokens": DOC_TOK,
             "prefill_s": 0.10, "latency_s": 0.2, "queue_s": 0.0},
            {"request_id": "doc5:0", "ok": True, "prefill_tokens": 40,
             "prefill_s": 0.05, "latency_s": 0.1, "queue_s": 0.0,
             "migrated": True, "remote_opp": True, "repl_served": True},
            {"request_id": "doc5:1", "ok": True, "prefill_tokens": BLOCK,
             "prefill_s": 0.01, "latency_s": 0.1, "queue_s": 0.0,
             "migrated": False},
        ]
        s.remote_resume_opportunities = 1
        s.repl_served_local = 1
        s.degrade_rederiv_tokens = 50
        e = s.dir_add("blk/x/i/0/span/0-16/repr/bf16", 16)
        e["first"], e["last"], e["count"] = now, now + 30.0, 2
        d = s.summary()
        # q0 growth is 16, prefill 40 -> 24 re-derived; q1 exact
        check("rederivation_tokens", d["rederivation_tokens"] == 24,
              str(d["rederivation_tokens"]))
        check("migration rate", d["session_turn_migration_rate"] == 0.5,
              str(d["session_turn_migration_rate"]))
        check("opportunities surfaced",
              d["remote_resume_opportunities"] == 1)
        check("repl-served surfaced",
              d["remote_resume_served_local_due_to_replication"] == 1)
        check("degrade counter surfaced", d["degrade_rederiv_tokens"] == 50)
        check("c_recompute buckets", any(b["bucket_max"] == 64
                                         and b["tok_per_s"] > 0
                                         for b in d["c_recompute"]),
              str(d["c_recompute"]))
        check("state lifetime p50", d["state_lifetime"]["p50"] == 30.0,
              str(d["state_lifetime"]))


def main():
    test_migration_and_opportunity()
    test_repl_served_local()
    test_q_tokens()
    test_summary_metrics()
    print("test_e1: ALL PASS")


if __name__ == "__main__":
    main()
