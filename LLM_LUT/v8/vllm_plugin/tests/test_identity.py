#!/usr/bin/env python3
"""Gate 2: request-identity contract test (docs/31 §3 Gate 2). Pure CPU.

Run anywhere:
    python vllm_plugin/tests/test_identity.py

Simulates the scheduler sequences from docs/31 Gate 2 against
IdentityRegistry and asserts the I4/I5 contract. Block ids appear in the
simulation ONLY as scheduler-side noise (reuse, collision, reshuffle) to
prove they cannot influence identity — the registry never sees them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from vllm_plugin.identity import (IdentityError, IdentityRegistry,
                                  RequestStateStub, StepContext,
                                  get_step_context, set_step_context)


def seq_new_request(r):
    """A request appears for the first time: fresh state, created==1."""
    st, = r.sync(["chatcmpl-A"])
    assert st.request_id == "chatcmpl-A"
    assert r.created == 1 and len(r.states) == 1
    return st


def seq_next_step_same_state(r, st_a):
    """Same request_id on the next scheduler step: SAME object, no create."""
    out = r.sync(["chatcmpl-A"])
    assert out[0] is st_a, "same req_id must return the same state object"
    assert r.created == 1, "no new state for a continuing request"


def seq_equal_length_reschedule(r, st_a):
    """Scheduler re-emits the request with identical metadata shape
    (the v2026-10-04r crash case): identity must not care."""
    for _ in range(3):
        out = r.sync(["chatcmpl-A"])
        assert out[0] is st_a
    assert r.created == 1


def seq_prefix_hit(r, st_a):
    """Another request whose prompt shares chatcmpl-A's whole prefix and
    block list: different request_id -> different state, always."""
    st_b, = r.sync(["chatcmpl-B"])
    assert st_b is not st_a and st_b.request_id == "chatcmpl-B"
    assert r.created == 2


def seq_finish_and_block_reuse(r, st_a, st_b):
    """A finishes; the allocator reissues A's ENTIRE block sequence to a new
    request C (the v2026-10-04s crash case). With block-heuristics this
    inherited A's compact_len; with request_id identity it cannot."""
    r.drop_finished(["chatcmpl-A"])
    assert "chatcmpl-A" not in r.states
    st_c, = r.sync(["chatcmpl-C"])
    assert st_c is not st_a, "recycled blocks must not recycle state"
    assert st_c.payload == {}, "fresh state carries no leftover payload"
    assert r.created == 3
    # A's state object must be unusable now (I5 guard)
    try:
        r.require_live("chatcmpl-A")
    except IdentityError as e:
        assert "after finish" in str(e)
    else:
        raise AssertionError("dropped request_id remained usable")


def seq_two_concurrent_interleaved(r, st_b, st_c):
    """Two live requests across many steps, batch order shuffled: each
    batch slot maps to its own state regardless of order."""
    st_b.payload["mark"] = "B"
    st_c.payload["mark"] = "C"
    orders = [["chatcmpl-B", "chatcmpl-C"],
              ["chatcmpl-C", "chatcmpl-B"],
              ["chatcmpl-C", "chatcmpl-B"],
              ["chatcmpl-B", "chatcmpl-C"]]
    for order in orders:
        out = r.sync(order)
        by_id = {st.request_id: st for st in out}
        assert by_id["chatcmpl-B"].payload["mark"] == "B"
        assert by_id["chatcmpl-C"].payload["mark"] == "C"
        assert by_id["chatcmpl-B"] is st_b and by_id["chatcmpl-C"] is st_c
    assert r.created == 3, "interleaving must never create states"


def seq_step_context_roundtrip():
    set_step_context(None)
    assert get_step_context() is None
    ctx = StepContext(["x", "y"])
    set_step_context(ctx)
    assert get_step_context() == ctx
    assert get_step_context().req_ids == ("x", "y")
    set_step_context(None)


def main():
    r = IdentityRegistry()
    st_a = seq_new_request(r)
    seq_next_step_same_state(r, st_a)
    seq_equal_length_reschedule(r, st_a)
    seq_prefix_hit(r, st_a)
    st_b = r.states["chatcmpl-B"]
    seq_finish_and_block_reuse(r, st_a, st_b)
    st_c = r.states["chatcmpl-C"]
    seq_two_concurrent_interleaved(r, st_b, st_c)
    seq_step_context_roundtrip()
    print("[gate2] PASS new-request / next-step / equal-length / prefix-hit "
          "/ finish+reuse / 2-concurrent-interleaved / step-context")
    print("[gate2] ALL PASS")


if __name__ == "__main__":
    main()
