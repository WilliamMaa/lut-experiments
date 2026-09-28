#!/usr/bin/env python3
"""PIT (interest aggregation) regression + closed-loop hang repro.

Reproduces the 2026-09-28 hang WITHOUT GPU/model: closed-loop,
4 workers, 16 sessions x 4 turns, 8 identical-prefix pairs, --pit.
Drives the scheduler with a perfect-worker FakeSock; if the loop
does not reach finished() the test dumps the live scheduler state
(PIT table, worker busy flags, ready queue) — the exact material
needed to pinpoint a missed wakeup.

Run:  python -m icn_proto.test_pit
"""

import json
import os
import sys
import time
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from icn_proto.scheduler import Scheduler, Turn  # noqa: E402

BLOCK = 16
MB = 1 << 20

ARGS = dict(policy="b3", wait_threshold=2.0, block_tokens=BLOCK,
            decode_steps=4, worker_mem_budget_mb=0.0, repl_min_lambda=0.05,
            repl_cooldown=15.0, max_repl_inflight=2, repl_mem_price=0.0,
            pit=True, arrival="none", seed=0, think_s=2.0)


class FakeSock:
    """Captures scheduler -> worker commands instead of sending them."""

    def __init__(self):
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(list(frames))


def make_sched(n_workers=4, pit=True):
    kw = dict(ARGS, pit=pit)
    return Scheduler(Namespace(**kw), [f"w{i}" for i in range(n_workers)])


def make_session(s, name, seed, n_turns=4, doc=1024, step=64):
    """A session chain: turn -1 (doc) then question turns. `seed` is
    shared between paired sessions so their prefix_ids — and hence
    their block names and fingerprints — are identical."""
    ids = [(i * 7 + seed) % 50257 + 3 for i in range(doc)]
    turns = []
    for t in range(-1, n_turns - 1):
        turns.append(Turn(name, t, list(ids), list(ids),
                          t_ready=0.0, t_arrive=0.0))
        ids = ids + [(i * 13 + seed) % 50257 + 3
                     for i in range(len(ids), len(ids) + step)]
    return turns


def setup_closed_loop(s, n_sessions=16, pairs=8, n_turns=4):
    for i in range(n_sessions):
        seed = i % pairs            # sessions i and i+pairs share a chain
        turns = make_session(s, f"s{i}", seed, n_turns)
        s.turns_of[f"s{i}"] = turns
        s.ready.append(turns[0])    # the doc-turn is ready at t=0
    return s


def drive(s, sock, max_iters=5000, busy_ticks=3):
    """busy_ticks: a worker answers an assign this many scheduler ticks
    after receiving it — real compute time, so concurrency windows
    (parks, fetch merges) actually open. Without it every turn
    completes instantly and no two identical prefixes ever overlap."""
    inflight = []   # (remaining_ticks, ident, hdr)
    for it in range(max_iters):
        s.dispatch(sock)
        # fetch/deliver/evict answers are fast: same tick
        fast = [f for f in list(sock.sent)
                if json.loads(f[1].decode())["type"] != "assign"]
        for f in fast:
            sock.sent.remove(f)
            answer_one(s, sock, f)
        for f in list(sock.sent):
            sock.sent.remove(f)
            inflight.append([busy_ticks, f])
        progressed = False
        for item in inflight:
            item[0] -= 1
            if item[0] <= 0:
                answer_one(s, sock, item[1])
                inflight.remove(item)
                progressed = True
        if not s.ready and not sock.sent and not inflight:
            break
    return it


def answer_one(s, sock, frames):
    ident = frames[0]
    hdr = json.loads(frames[1].decode())
    w = s.workers[ident]
    mtype = hdr["type"]
    if mtype == "assign":
        names = hdr["new_block_names"]
        w.resident |= set(names) | set(hdr["resume_names"])
        for n in hdr["resume_names"]:
            s.dir_add(n, MB)
        pubs = [{"name": n, "bytes": MB} for n in names]
        s.on_message(sock, ident, {
            "type": "result", "request_id": hdr["request_id"],
            "ok": True, "prefill_s": 0.1, "prefill_tokens": 64,
            "decode_s": 0.1, "published": pubs,
            "resident_bytes": MB}, None)
    elif mtype == "fetch":
        for n in hdr["names"]:
            if n in s.dir:
                w.resident.add(n)
        s.on_message(sock, ident, {"type": "fetched", "ok": True,
                                   "names": hdr["names"]},
                     b"payload")
    elif mtype == "deliver":
        xfer = next((x for x in s._xfer.values()
                     if x["target"] == ident
                     and x["stage"] == "deliver"), None)
        names = xfer["names"] if xfer else []
        w.resident |= set(names)
        s.on_message(sock, ident, {"type": "delivered",
                                   "names": list(names)}, None)
    elif mtype == "evict":
        for n in hdr["names"]:
            w.resident.discard(n)
        s.on_message(sock, ident, {"type": "status", "seq": 0,
                                   "resident": list(w.resident),
                                   "tips": [], "resident_bytes": 0,
                                   "spilled": [], "spill_bytes": 0},
                     None)


def dump_state(s, it):
    print(f"  --- dump after {it} iters (NOT finished) ---")
    print(f"  ready: {len(s.ready)} "
          + str([f"{t.session}:{t.turn}" for t in s.ready[:8]]))
    print(f"  done_sessions: {s.done_sessions}/{len(s.turns_of)}")
    for fp, e in s._pit.items():
        sw = s.workers[e["worker"]]
        print(f"  _pit: rid={e['rid']} worker={e['worker'].decode()} "
              f"busy={sw.busy} waiters={len(e['waiters'])}")
    for w in s.workers.values():
        print(f"  {w.ident.decode()}: busy={w.busy} current={w.current} "
              f"grey={w.grey}")
    print(f"  _xfer={len(s._xfer)} _repl={len(s._repl)}")
    tail = [r["request_id"] + (":ok" if r.get("ok") else ":--")
            for r in s.records[-6:]]
    print(f"  records tail: {tail}")


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_pit: {name} FAILED {detail}")


def test_closed_loop_terminates():
    print("closed-loop 16x4, 8 identical-prefix pairs, pit=on:")
    s = make_sched(4, pit=True)
    setup_closed_loop(s)
    sock = FakeSock()
    it = drive(s, sock)
    if not s.finished():
        dump_state(s, it)
    check("terminates", s.finished(), f"iters={it}")
    dupes = {}
    for r in s.records:
        dupes[r["request_id"]] = dupes.get(r["request_id"], 0) + 1
        check("no failed record", r.get("ok") is not False, r["request_id"])
    multi = {k: v for k, v in dupes.items() if v > 1}
    check("exactly one record per rid (no double dispatch)",
          not multi, str(multi))
    check("compute merges happened", s.pit_compute_merged > 0,
          f"merged={s.pit_compute_merged}")
    check("waiters served", s.pit_waiters_served > 0,
          f"served={s.pit_waiters_served}")


def test_pit_off_inert():
    print("same workload, pit=off (inert guard):")
    s = make_sched(4, pit=False)
    setup_closed_loop(s)
    sock = FakeSock()
    it = drive(s, sock)
    check("terminates", s.finished(), f"iters={it}")
    check("counters all zero",
          s.pit_compute_merged == 0 and s.pit_fetch_merged == 0
          and s.pit_waiters_served == 0)


def test_grey_releases_waiter():
    print("grey releases parked waiter:")
    s = make_sched(2, pit=True)
    setup_closed_loop(s, n_sessions=2, pairs=1)
    sock = FakeSock()
    s.dispatch(sock)                    # s0 doc-turn assigned
    s.dispatch(sock)                    # s1 doc-turn parks (same fp)
    fp = next(iter(s._pit))
    check("one waiter parked", len(s._pit[fp]["waiters"]) == 1)
    check("waiter left ready",
          all(t.session != "s1" for t in s.ready))
    victim = s.workers[s._pit[fp]["worker"]]
    s._grey(sock, victim, time.time(), "test")
    check("entry popped", fp not in s._pit)
    check("waiter requeued",
          any(t.session == "s1" for t in s.ready))


def main():
    test_closed_loop_terminates()
    test_pit_off_inert()
    test_grey_releases_waiter()
    print("ALL PASS")


if __name__ == "__main__":
    main()
