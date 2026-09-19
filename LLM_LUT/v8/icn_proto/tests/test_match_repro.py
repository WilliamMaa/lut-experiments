#!/usr/bin/env python3
"""Offline reproduction of the scheduler's Match logic (tip semantics).

Mirrors Scheduler.resume_names / Scheduler.match_local in icn_proto.scheduler
without zmq/torch. Simulates: doc-turn publishes complete blocks + tip on
worker A, then (a) the same session's question turn and (b) an identical
second session's doc-turn must both match E=doc_len on A. The doc length is
deliberately NOT a multiple of the block size (regression for the partial-
tip block-name drift bug).

    python -m icn_proto.tests.test_match_repro
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from icn_proto.blkchain import chain_through, derive_chain

BT = 16          # block_tokens
REPR = "bf16"


class FakeWorker:
    def __init__(self):
        self.resident = set()
        self.tips = set()


def tip_end(name):
    parts = name.split("/")
    return int(parts[parts.index("span") + 1].split("-")[1])


def complete_chain(prefix_ids):
    return [b for b in derive_chain(prefix_ids, REPR, BT)
            if b.span_tokens == BT]


def tip_name_at(prefix_ids, n):
    return str(derive_chain(prefix_ids[:n], REPR, BT)[-1])


def publish_names(prefix_ids):
    """Mirror of Turn.publish_names."""
    names = [str(b) for b in complete_chain(prefix_ids)]
    tip = tip_name_at(prefix_ids, len(prefix_ids))
    if tip not in names:
        names.append(tip)
    return names


def resume_names(prefix_ids, t_pos):
    """Mirror of Scheduler.resume_names."""
    floor_t = t_pos - (t_pos % BT)
    names = [str(b) for b in chain_through(complete_chain(prefix_ids),
                                           floor_t)]
    tip = tip_name_at(prefix_ids, t_pos)
    if tip not in names:
        names.append(tip)
    return names


def match_local(prefix_ids, w, tip_bound):
    """Mirror of Scheduler.match_local."""
    best = 0
    for tip_name in w.tips:
        t_pos = tip_end(tip_name)
        if t_pos == 0 or t_pos > tip_bound:
            continue
        if tip_name_at(prefix_ids, t_pos) != tip_name:
            continue
        if not all(n in w.resident
                   for n in resume_names(prefix_ids, t_pos)):
            continue
        best = max(best, t_pos)
    return best


def simulate_doc_turn(doc):
    """Worker state after a doc-turn with prefix == doc (fresh, E=0)."""
    w = FakeWorker()
    names = publish_names(doc)
    w.resident.update(names)
    w.tips.add(names[-1])
    return w


def check(doc_len):
    doc = list(range(doc_len))
    q = [9000, 9001, 9002]
    q_prefix = doc + q

    A = simulate_doc_turn(doc)
    want = doc_len

    # (a) same session, question turn: must resume at doc_len
    e = match_local(q_prefix, A, len(q_prefix))
    print(f"doc_len={doc_len}: match q-turn on A: E={e} (want {want})")
    assert e == want, f"MATCH FAILED: E={e}, want {want}"

    # (b) identical second session, doc-turn: cross-session tip reuse
    e2 = match_local(doc, A, len(doc))
    print(f"doc_len={doc_len}: match doc-turn (session 2) on A: E={e2}")
    assert e2 == want, f"CROSS-SESSION MATCH FAILED: E={e2}, want {want}"

    # publish_names sanity: no duplicate tip when block-aligned
    names = publish_names(doc)
    assert len(names) == len(set(names)), "duplicate names in publish_names"
    assert tip_end(names[-1]) == want
    return A


def main():
    A = check(1603)      # partial tip: 1603 = 100*16 + 3
    check(1600)          # block-aligned: tip == last complete block
    # incomplete worker must not match: drop the tip block
    B = FakeWorker()
    doc = list(range(1603))
    B.resident.update(publish_names(doc)[:-1])
    B.tips.add(tip_name_at(doc, 1603))
    e = match_local(doc + [1, 2, 3], B, 1606)
    print(f"incomplete worker match: E={e} (want 0)")
    assert e == 0, "matched without resident tip block"
    print("match repro OK (tip semantics, partial + aligned + cross-session)")


if __name__ == "__main__":
    main()
