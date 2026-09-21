#!/usr/bin/env python3
"""Policy-gating unit tests for Scheduler.choose() (05 v3 §6 step 3).

Pure Python — no torch/transformers/GPU. scheduler only imports zmq at
module load (sockets are unused here), so a stub is injected.

Covered:
  b0   never resumes, never fetches (even with a fully warm worker)
  b1   resumes on a warm idle worker, never fetches cross-worker
  b2   holds a turn for a nearly-done warm busy worker; falls through
       to a cold start once the warm worker's ETA exceeds the threshold
  b3   fetches a contiguous single-holder extension cross-worker
  ours identical to b3 today (placeholder for the step-4 controller);
  p2   legacy alias resolves to ours

Run:  python -m icn_proto.test_policy
"""

import os
import sys
import time
import types
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("zmq", types.ModuleType("zmq"))  # choose() is socket-free

from icn_proto.scheduler import Scheduler, Turn  # noqa: E402

BLOCK = 16
N_TOK = 1024  # 64 complete blocks, block-aligned tip


def make_args(policy):
    return Namespace(policy=policy, wait_threshold=2.0, block_tokens=BLOCK,
                     decode_steps=4)


def make_sched(policy):
    return Scheduler(make_args(policy), ["w0", "w1"])


def doc_turn():
    ids = [(i * 7) % 50257 + 3 for i in range(N_TOK)]
    return Turn("doc0", -1, list(ids), list(ids))


def warm_with(sched, w, turn, t_pos):
    names = sched.resume_names(turn, t_pos)
    w.resident |= set(names)
    w.tips.add(turn.tip_name_at(t_pos, "bf16", BLOCK))


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(f"test_policy: {name} FAILED {detail}")


def test_b0():
    print("b0 (load-only):")
    s = make_sched("b0")
    turn = doc_turn()
    warm_with(s, s.workers[b"w1"], turn, N_TOK)
    out = s.choose(turn)
    check("chooses something", out is not None)
    _, e, fetch, dec = out
    check("never resumes (E=0)", e == 0, f"E={e}")
    check("never fetches", fetch is None)
    check("mode fresh", dec["mode"] == "fresh", dec["mode"])
    check("policy logged", dec["policy"] == "b0")


def test_b1():
    print("b1 (local match, no fetch):")
    s = make_sched("b1")
    turn = doc_turn()
    warm_with(s, s.workers[b"w1"], turn, N_TOK)
    out = s.choose(turn)
    _, e, fetch, dec = out
    check("resumes at chain end", e == N_TOK, f"E={e}")
    check("no fetch", fetch is None)
    check("mode local", dec["mode"] == "local", dec["mode"])

    # chain only remote + no local copy -> cold start, still no fetch
    s2 = make_sched("b1")
    s2.workers[b"w1"].busy = True
    s2.workers[b"w1"].t_assign = 0.0
    s2.workers[b"w1"].cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}
    warm_with(s2, s2.workers[b"w1"], turn, N_TOK)
    out2 = s2.choose(turn)
    _, e2, fetch2, dec2 = out2
    check("cold start when holder busy", e2 == 0, f"E={e2}")
    check("still no fetch", fetch2 is None)
    check("mode fresh", dec2["mode"] == "fresh", dec2["mode"])


def test_b2():
    print("b2 (planned affinity):")
    s = make_sched("b2")
    turn = doc_turn()
    w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
    warm_with(s, w0, turn, N_TOK)
    w0.busy = True
    w0.t_assign = time.time()
    w0.cur_plan = {"prefill_tokens": 100, "decode_steps": 0}  # eta ~0.04s
    check("holds for nearly-done warm worker", s.choose(turn) is None)
    # eta now far above the threshold -> cold start on the idle worker
    w0.t_assign = time.time()
    w0.cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}  # eta ~400s
    out = s.choose(turn)
    check("falls through when warm worker ETA high", out is not None)
    _, e, fetch, _ = out
    check("cold start", e == 0 and fetch is None, f"E={e} fetch={fetch}")


def test_b3_ours():
    print("b3 / ours (cross-worker fetch):")
    for pol in ("b3", "ours", "p2"):
        s = make_sched(pol)
        turn = doc_turn()
        w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
        w1.busy = True
        w1.t_assign = 0.0
        w1.cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}
        warm_with(s, w1, turn, N_TOK)
        out = s.choose(turn)
        check(f"{pol}: picks idle cold worker", out is not None
              and out[0] == b"w0", f"ident={out[0] if out else None}")
        _, e, fetch, dec = out
        check(f"{pol}: fetch planned to chain end", e == N_TOK
              and fetch is not None, f"E={e}")
        check(f"{pol}: holder is w1", fetch[0] == b"w1", str(fetch[0]))
        check(f"{pol}: missing = full resume set",
              set(fetch[1]) == set(s.resume_names(turn, N_TOK)))
        logged = "ours" if pol == "p2" else pol
        check(f"{pol}: decision policy = {logged}", dec["policy"] == logged,
              dec["policy"])
        check(f"{pol}: mode fetch", dec["mode"] == "fetch", dec["mode"])


def test_free_extension_no_fetch():
    """Regression (share=8/ours/budget=48 crash, 20260922): a worker can
    hold EVERY block of a resume set — tip block included — without the
    tip in w.tips (evicted as a tip while the block survived as shared
    infrastructure of a hotter segment). match_local then stops below
    the global tip, and the fetch candidate loop used to see an EMPTY
    `need`, pass the vacuous all(...) holder check, and ship a
    zero-block fetch that crashed the deliver path (names[-1] on []).
    Correct behaviour: a free local extension, no fetch."""
    print("free extension over resident blocks:")
    s = make_sched("ours")
    turn = doc_turn()
    w0, w1 = s.workers[b"w0"], s.workers[b"w1"]
    w1.busy = True
    w1.t_assign = 0.0
    w1.cur_plan = {"prefill_tokens": 10 ** 6, "decode_steps": 0}
    warm_with(s, w1, turn, N_TOK)          # global tip at 1024 lives on w1
    warm_with(s, w0, turn, N_TOK // 2)     # w0's own tip stops at 512
    # the eviction shape: w0 keeps every block up to 1024 (incl. the tip
    # block) but has lost the 1024 tip from its tip set
    for n in s.resume_names(turn, N_TOK):
        w0.resident.add(n)
    w0.tips.discard(turn.tip_name_at(N_TOK, "bf16", BLOCK))
    out = s.choose(turn)
    check("chooses w0", out is not None and out[0] == b"w0")
    _, e, fetch, dec = out
    check("extends to chain end", e == N_TOK, f"E={e}")
    check("no zero-block fetch", fetch is None, str(fetch))
    check("mode local (free extension)", dec["mode"] == "local", dec["mode"])


def main():
    test_b0()
    test_b1()
    test_b2()
    test_b3_ours()
    test_free_extension_no_fetch()
    print("test_policy: ALL PASS")


if __name__ == "__main__":
    main()
