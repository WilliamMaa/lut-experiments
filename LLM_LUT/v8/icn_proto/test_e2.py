#!/usr/bin/env python3
"""E2 (10-e2-backing-tier §3) unit tests: pure Python, no torch/zmq/GPU.

Covered:
  match         a tip living in the spill tier is matchable — resume
                feasibility is resident ∪ spilled, so eviction no
                longer destroys the resume path
  choose        local recall is priced below cross-worker fetch
                (spill_rate ≫ xfer_rate): the spill-holding worker
                wins a turn that would otherwise ride a demand fetch
  repl          _plan_repl accepts a SPILL holder as the copy source;
                a target holding blocks only in spill is not re-copied
  status        spilled / spill_bytes / spill counters sync from the
                worker's status message
  evict         eviction actions and byte accounting cover RESIDENT
                blocks only — a spilled tip protects its chain but is
                never evicted (or byte-counted) twice
  accounting    dispatch counts fetch_avoided_by_recall; send_assign
                counts per-turn spill recall blocks

Run:  python -m icn_proto.test_e2
"""

import os
import sys
import tempfile
import time
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from icn_proto.scheduler import Scheduler, Turn  # noqa: E402

BLOCK = 16
DOC_TOK = 256          # block-aligned doc prefix -> 17 resume names
MB = 1 << 20


class FakeSock:
    def send_multipart(self, frames):
        pass


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


def make_sched(workers=("w0", "w1"), **over):
    return Scheduler(make_args(**over), list(workers))


def session_turns(n_q=1, seed=3):
    ids = [(i * 7 + seed) % 50257 + 3 for i in range(DOC_TOK)]
    turns = [Turn("doc5", -1, list(ids), list(ids))]
    prefix = list(ids)
    for t in range(n_q):
        q_ids = [(1000 + t * 13 + i) % 50257 + 3 for i in range(BLOCK)]
        prefix = prefix + q_ids
        turns.append(Turn("doc5", t, list(q_ids), list(prefix)))
    return turns


def register(s, turn, t_pos=DOC_TOK, mb=1, lam=2.0):
    """Put the resume set of `turn` at t_pos into the directory with
    `mb` MB per block and demand EWMA `lam` (loc demand on w1)."""
    names = s.resume_names(turn, t_pos)
    for n in names:
        e = s.dir_add(n, mb * MB)
        e["lambda"] = lam
        e["loc"] = {"w1": lam}
    tip = turn.tip_name_at(t_pos, "bf16", BLOCK)
    s.tips.add(tip)
    return tip, names


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_e2: {name} FAILED {detail}")


def test_match_spilled_tip():
    print("spilled tip is matchable:")
    s = make_sched()
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc = turns[0]
    w0 = s.workers[b"w0"]
    tip, names = register(s, doc)
    w0.tips.add(tip)
    # residency without the spill tier: nothing matches
    check("cold worker no match",
          s.match_local(doc, w0, DOC_TOK) == 0)
    # evict-to-spill: blocks leave residency but stay addressable
    w0.spilled |= set(names)
    check("spilled tip matches",
          s.match_local(doc, w0, DOC_TOK) == DOC_TOK)
    # partial spill: an ancestor missing everywhere -> not resumable
    # (the chain must be COMPLETE in resident ∪ spilled)
    w0.spilled.discard(names[3])
    check("incomplete spill no match",
          s.match_local(doc, w0, DOC_TOK) == 0)


def test_choose_recall_beats_fetch():
    print("recall priced below cross-worker fetch:")
    s = make_sched(workers=("w0", "w1", "w2"))
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc, q0 = turns[0], turns[1]
    tip, names = register(s, doc)
    w0, w1, w2 = (s.workers[k] for k in (b"w0", b"w1", b"w2"))
    # w0 holds the chain ONLY in the spill tier; w2 holds it resident
    # but is wedged on a huge turn; w1 is cold (would demand-fetch).
    w0.spilled |= set(names)
    w0.tips.add(tip)
    w2.resident |= set(names)
    w2.tips.add(tip)
    w2.busy = True
    w2.t_assign = time.time()
    w2.cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}
    q0.t_ready = q0.t_arrive = time.time()
    out = s.choose(q0)
    check("choose returns", out is not None)
    ident, e, fetch, decision = out
    check("spill worker wins the turn", ident == b"w0",
          ident.decode() if ident else "None")
    check("full resume via spill", e == DOC_TOK and fetch is None)
    check("mode local", decision["mode"] == "local")
    check("recall priced in decision",
          decision.get("recall_bytes") == len(names) * MB,
          str(decision.get("recall_bytes")))
    # cost ordering: recall (17MB / 20GB/s ≈ 0.9ms) must beat the
    # fetch alternative w1 would have taken (17MB / 100MB/s = 170ms)
    opts = decision["options"]
    check("recall cost below fetch cost",
          opts["w0"]["cost_s"] < opts["w1"]["cost_s"],
          f"w0={opts['w0']['cost_s']} w1={opts['w1']['cost_s']}")
    check("w1 option was a fetch", opts["w1"]["mode"] == "fetch")


def test_plan_repl_spill_holder():
    print("replication from a spill holder:")
    s = make_sched()
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc = turns[0]
    tip, names = register(s, doc, lam=2.0)
    w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
    w0.spilled |= set(names)          # holder lives ONLY in the tier
    w0.tips.add(tip)
    plan = s._plan_repl()
    check("one action", len(plan) == 1, f"n={len(plan)}")
    a = plan[0]
    check("spill holder is the copy source", a["holder"] == b"w0")
    check("cold target", a["target"] == b"w1")
    check("full chain copied", set(a["names"]) == set(names))

    # a target holding the lower chain in the TIER needs only the
    # missing suffix — spill residency counts, nothing is re-copied
    s2 = make_sched()
    s2.turns_of = {"doc5": session_turns()}
    doc2 = s2.turns_of["doc5"][0]
    tip2, names2 = register(s2, doc2, lam=2.0)
    w0b, w1b = s2.workers[b"w0"], s2.workers[b"w1"]
    w0b.spilled |= set(names2)
    w0b.tips.add(tip2)
    low = set(s2.resume_names(doc2, DOC_TOK // 2))
    w1b.spilled |= low
    plan2 = s2._plan_repl()
    check("suffix-only copy to spill-warmed target",
          len(plan2) == 1
          and set(plan2[0]["names"]) == set(names2) - low,
          f"n={len(plan2)}")


def test_status_syncs_spilled():
    print("status syncs the tier view:")
    s = make_sched()
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc = turns[0]
    tip, names = register(s, doc)
    w0 = s.workers[b"w0"]
    s.on_message(None, b"w0",
                 {"type": "status", "resident": [], "tips": [tip],
                  "resident_bytes": 0, "spilled": names,
                  "spill_bytes": len(names) * MB,
                  "spill": {"evicted_bytes": len(names) * MB,
                            "dropped_bytes": 0, "recall_count": 0,
                            "recall_bytes": 0}}, None)
    check("spilled synced", w0.spilled == set(names))
    check("spill bytes synced", w0.spill_bytes == len(names) * MB)
    check("worker counters synced",
          w0.spill_stat.get("evicted_bytes") == len(names) * MB)
    check("match works off the status view",
          s.match_local(doc, w0, DOC_TOK) == DOC_TOK)


def test_evict_resident_only():
    print("eviction touches resident blocks only:")
    s = make_sched(worker_mem_budget_mb=48.0)
    w0 = s.workers[b"w0"]
    # hot tip living in the tier (its whole segment evicted there — the
    # realistic spill shape; its chain registered in the dir); a cold
    # segment is resident
    hot_turn = session_turns(seed=7)[0]
    hot_tip, hot_names = register(s, hot_turn, lam=5.0)
    w0.tips.add(hot_tip)
    w0.spilled |= set(hot_names)
    cold_turn = session_turns(seed=22)[0]
    cold_tip, cold_names = register(s, cold_turn, lam=0.0)
    w0.tips.add(cold_tip)
    w0.resident |= set(cold_names)
    cold_bytes = sum(s.dir[n]["bytes"] for n in cold_names)
    w0.resident_bytes = 48e6 + cold_bytes      # exactly the cold segment over
    plan = s._plan_evict()
    check("one action", len(plan) == 1, f"n={len(plan)}")
    names = set(plan[0]["names"])
    check("cold segment evicted", cold_tip in names)
    check("spilled tip not re-evicted", hot_tip not in names)
    check("no phantom (non-resident) names",
          all(n in w0.resident for n in names), f"{len(names)} blocks")
    s._apply_evict(None, plan)
    check("byte accounting covers resident only",
          s.evicted_bytes == sum(s.dir[n]["bytes"] for n in names),
          str(s.evicted_bytes))


def test_accounting():
    print("spill accounting on the dispatch path:")
    s = make_sched()
    turns = session_turns()
    s.turns_of = {"doc5": turns}
    doc, q0 = turns[0], turns[1]
    tip, names = register(s, doc)
    w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
    w0.spilled |= set(names)
    w0.tips.add(tip)
    w1.resident |= set(names)          # without the tier, w0 would fetch
    w1.tips.add(tip)
    w1.busy = True                     # ... but w1 is wedged, so w0 serves
    w1.t_assign = time.time()
    w1.cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}
    q0.t_ready = q0.t_arrive = time.time()
    s.ready.append(q0)
    check("nothing counted before dispatch", s.spill_fetch_avoided == 0)
    s.dispatch(FakeSock())
    check("fetch avoided by recall", s.spill_fetch_avoided == 1)
    rec = s.records[-1]
    check("recall blocks on the record",
          rec.get("spill_recall_blocks") == len(names),
          str(rec.get("spill_recall_blocks")))
    check("recall counters", s.spill_recall_turns == 1
          and s.spill_recall_blocks == len(names)
          and s.spill_recall_bytes == len(names) * MB)

    # summary surfaces the tier block
    with tempfile.TemporaryDirectory() as out:
        s.args.out = out
        d = s.summary()
        sp = d["spill"]
        check("summary spill block", sp["recall_turns"] == 1
              and sp["recall_blocks"] == len(names)
              and sp["fetch_avoided_by_recall"] == 1
              and "workers" in sp, str(sp))


def main():
    test_match_spilled_tip()
    test_choose_recall_beats_fetch()
    test_plan_repl_spill_holder()
    test_status_syncs_spilled()
    test_evict_resident_only()
    test_accounting()
    print("test_e2: ALL PASS")


if __name__ == "__main__":
    main()
