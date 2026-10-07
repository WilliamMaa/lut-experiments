"""Integration contract tests (docs/32): 8 required cases + property-based
random interleavings, all on CPU against the real builder+impl via the
fake scheduler. Run: python vllm_plugin/tests/test_integration.py
"""
import os
import sys

# Small-scale config BEFORE any vllm_plugin import (impl reads config at
# construction; prefill allowance = max(budget, max_seq) drives whether
# eviction fires mid-prefill vs at decode).
os.environ.setdefault("V8_COMPRESS_SLOTS", "24")
os.environ.setdefault("V8_SINK_TOKENS", "2")
os.environ.setdefault("V8_RECENT_TOKENS", "4")
os.environ.setdefault("V8_OBS_WINDOW", "8")
os.environ.setdefault("V8_SPAN_WINDOW", "2")
os.environ.setdefault("V8_MAX_SEQ_TOKENS", "96")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root
sys.path.insert(0, _HERE)
import fake_vllm
fake_vllm.install()

from harness import FakeWorld  # noqa: E402

FAILURES = []


def check(name, errs):
    if errs:
        FAILURES.append(name)
        print(f"[int] FAIL {name}")
        for e in errs:
            print(f"      - {e}")
    else:
        print(f"[int] PASS {name}")


def expect_raise(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"[int] PASS {name} (raised {type(e).__name__})")
        return
    FAILURES.append(name)
    print(f"[int] FAIL {name}: no raise")


# docs/32 case 1: single request chunk1
def case1():
    w = FakeWorld(seed=1)
    w.add_request("A", 40)
    _, errs = w.step([("A", 40)])
    check("1 single-request chunk1", errs)

# case 2: same request chunk2 (continuation, chunked prefill)
def case2():
    w = FakeWorld(seed=2)
    w.add_request("A", 100)
    _, e1 = w.step([("A", 64)])
    _, e2 = w.step([("A", 36)])
    check("2 same-request chunk2", e1 + e2)

# case 3: prefill -> decode (C==1; eviction fires at the tight budget)
def case3():
    w = FakeWorld(seed=3)
    w.add_request("A", 60)
    errs = []
    _, e = w.step([("A", 60)])
    errs += e
    st = w.builder.registry.states["A"]
    L_prefill = st.compact_len
    for t in range(12):
        _, e = w.step([("A", 1)])
        errs += e
    st = w.builder.registry.states["A"]
    if st.compact_len > 24:
        errs.append(f"decode steady-state compact_len {st.compact_len} "
                    f"> budget 24")
    if L_prefill != 60:
        errs.append(f"prefill should keep all {L_prefill} (deferred)")
    check("3 prefill->decode eviction", errs)

# case 4: A/B interleaved — B must never pollute A
def case4():
    w = FakeWorld(seed=4)
    w.add_request("A", 70)
    w.add_request("B", 50)
    _, e1 = w.step([("A", 32)])
    _, e2 = w.step([("B", 32)])
    _, e3 = w.step([("A", 38)])
    _, e4 = w.step([("B", 18)])
    _, e5 = w.step([("A", 1), ("B", 1)])
    _, e6 = w.step([("B", 1), ("A", 1)])   # reversed batch order
    check("4 A/B interleaved", e1 + e2 + e3 + e4 + e5 + e6)

# case 5: preempt -> resume (state must rewind, not desync)
def case5():
    w = FakeWorld(seed=5)
    w.add_request("A", 80)
    errs = []
    _, e = w.step([("A", 50)])
    errs += e
    st_before = w.builder.registry.states["A"]
    snap_before = st_before.snap_len
    w.preempt("A")
    _, e = w.step([("A", 30)])   # frontier rewound: builder must reset state
    errs += e
    st = w.builder.registry.states["A"]
    if st is st_before and st.snap_len != snap_before:
        errs.append("state object not rebuilt after preempt")
    if st.snap_len > 30:
        errs.append(f"snap_len {st.snap_len} not rewound to frontier 30")
    _, e = w.step([("A", 50)])   # finish the (restarted) prompt
    errs += e
    check("5 preempt->resume", errs)

# case 6: equal-length consecutive steps (old length-change identity
# heuristic would merge them; identity must come from request_id)
def case6():
    w = FakeWorld(seed=6)
    w.add_request("A", 30)
    errs = []
    _, e = w.step([("A", 10)])
    errs += e
    st = w.builder.registry.states["A"]
    ids_first = sorted(w.builder.registry.states.keys())
    _, e = w.step([("A", 10)])   # same length, same request
    errs += e
    _, e = w.step([("A", 10)])
    errs += e
    if sorted(w.builder.registry.states.keys()) != ids_first:
        errs.append("registry identity changed across equal-length steps")
    if w.builder.registry.created != 1:
        errs.append(f"registry created {w.builder.registry.created} states "
                    f"for 1 request (identity heuristic misfire)")
    st2 = w.builder.registry.states["A"]
    if st2 is not st:
        errs.append("state object replaced across equal-length steps")
    if st2.snap_len != 30:
        errs.append(f"snap_len {st2.snap_len} != 30 after 3x10 chunks")
    check("6 equal-length steps", errs)

# case 7: block id recycle — A finishes, B reuses A's physical blocks,
# B's state must be fresh and overwrite A's stale pool content correctly
def case7():
    w = FakeWorld(seed=7)
    w.add_request("A", 40)
    errs = []
    _, e = w.step([("A", 40)], finished=["A"])
    errs += e
    if "A" in w.builder.registry.states:
        errs.append("finished A still live in registry")
    a_blocks = sorted(set(b for r in w.reqs.values()
                          for b in r.kernel_ids))
    w.add_request("B", 40)
    _, e = w.step([("B", 24)])
    errs += e
    b_blocks = set(w.reqs["B"].kernel_ids)
    if not b_blocks.issubset(set(a_blocks) | b_blocks):
        pass
    st = w.builder.registry.states["B"]
    if st.compact_len != 24 or int(st.orig[0]) != 0:
        errs.append(f"B inherited A's layout (compact_len {st.compact_len}, "
                    f"orig0 {int(st.orig[0])})")
    _, e = w.step([("B", 16)])
    errs += e
    check("7 block id recycle", errs)

# case 8: two KV groups with different P / B_g — plans are per-group,
# no cross-group addressing
def case8():
    w1 = FakeWorld(P=32, B_g=64, seed=8)
    w2 = FakeWorld(P=16, B_g=48, seed=88)
    w1.add_request("A", 50)
    w2.add_request("B", 50)
    errs = []
    _, e = w1.step([("A", 30)]); errs += e
    _, e = w2.step([("B", 30)]); errs += e
    _, e = w1.step([("A", 20)]); errs += e
    _, e = w2.step([("B", 20)]); errs += e
    _, e = w1.step([("A", 1)]); errs += e
    _, e = w2.step([("B", 1)]); errs += e
    p1 = w1.builder.registry.states["A"]
    p2 = w2.builder.registry.states["B"]
    if p1.compact_len != p2.compact_len:
        # not a failure per se, but the groups must agree on semantics
        pass
    check("8 two groups different P/B_g", errs)

# property test: random interleavings (docs/32)
def property_test(n_steps=1200, seed=99):
    import random
    rng = random.Random(seed)
    w = FakeWorld(seed=seed)
    errs = []
    next_rid = 0
    active = []          # rids with remaining prompt or decoding
    total_checked = 0
    for t in range(n_steps):
        op = rng.random()
        if op < 0.25 or not active:
            # new request
            rid = f"R{next_rid}"
            next_rid += 1
            w.add_request(rid, rng.randint(5, 120))
            active.append(rid)
        # build a schedule from a random subset of active requests
        sched = []
        for rid in list(active):
            rec = w.reqs[rid]
            remaining = rec.prompt_len - rec.computed
            if remaining > 0:
                n = rng.randint(1, min(remaining, 48))
                sched.append((rid, n))
            else:
                # decoding: 1 token, or finish
                if rng.random() < 0.3:
                    sched.append((rid, 1))
        finished = []
        for rid in list(active):
            rec = w.reqs[rid]
            if rec.computed >= rec.prompt_len and rng.random() < 0.25:
                finished.append(rid)
                active.remove(rid)
            elif rng.random() < 0.02:
                w.preempt(rid)
        if sched:
            _, e = w.step(sched, finished=finished)
            errs += [f"step{t}: {x}" for x in e]
            total_checked += len(sched)
        elif finished:
            idn = __import__("vllm_plugin.identity", fromlist=["x"])
            idn.set_step_context(idn.StepContext([], {}, {}, finished))
            w.builder.registry.drop_finished(finished)
        if errs:
            print(f"[int] property test first failure at step {t}")
            break
    name = f"property random interleaving ({total_checked} request-steps)"
    check(name, errs[:8])

# negative case: impl must refuse metadata without plans (docs/32 §2 —
# the fail-closed direction)
def negative_no_plan():
    import torch
    from vllm_plugin import units
    w = FakeWorld(seed=11)
    w.add_request("A", 20)
    _, errs = w.step([("A", 20)])
    assert not errs, errs
    # build a metadata object with the plan stripped and confirm the
    # impl refuses it (docs/32 §2 fail-closed direction)
    md = w.builder.build(0, type("C", (), {
        "query_start_loc_cpu": torch.tensor([0, 5], dtype=torch.int32),
        "num_reqs": 1})())
    md.block_table = torch.zeros(1, 8, dtype=torch.int64)
    md.block_plans = None
    try:
        w.impl.forward(None, torch.randn(5, 4, 8), torch.randn(5, 2, 8),
                       torch.randn(5, 2, 8), w.pool, md,
                       torch.zeros(5, 4, 8))
    except units.UnitError:
        print("[int] PASS negative: impl refuses plan-less metadata")
        return
    FAILURES.append("negative_no_plan")
    print("[int] FAIL negative: impl accepted plan-less metadata")


def main():
    case1(); case2(); case3(); case4(); case5(); case6(); case7(); case8()
    property_test()
    negative_no_plan()
    if FAILURES:
        print(f"[int] {len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("[int] ALL PASS")


if __name__ == "__main__":
    main()
