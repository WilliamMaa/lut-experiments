#!/usr/bin/env python3
"""Unit tests for blkchain naming (pure Python, no torch needed).

Run locally or remote:
    python -m icn_proto.tests.test_blkchain
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from icn_proto.blkchain import (BlockName, GENESIS, chain_through,
                                derive_chain, segment_bytes)


def test_chain_deterministic():
    toks = list(range(100))
    c1 = derive_chain(toks, "bf16", 16)
    c2 = derive_chain(toks, "bf16", 16)
    assert [str(b) for b in c1] == [str(b) for b in c2], "same tokens -> same chain"
    assert len(c1) == 7                      # ceil(100/16)
    assert c1[0].span_start == 0 and c1[0].span_end == 16
    assert c1[-1].span_start == 96 and c1[-1].span_end == 100
    return c1


def test_parent_linking(c1):
    assert c1[0].parent_hash != GENESIS      # genesis is hashed with model tag
    for prev, nxt in zip(c1, c1[1:]):
        assert nxt.parent_hash == prev.block_hash, "chain must link"


def test_prefix_extension():
    short = derive_chain(list(range(48)), "bf16", 16)
    long = derive_chain(list(range(64)), "bf16", 16)
    assert [str(b) for b in long[:3]] == [str(b) for b in short], \
        "extending a prefix must not change existing block names"


def test_cross_session_identity():
    a = derive_chain([1, 2, 3, 4, 5], "bf16", 4)
    b = derive_chain([1, 2, 3, 4, 5], "bf16", 4)
    assert [str(x) for x in a] == [str(x) for x in b]


def test_roundtrip_parse(c1):
    for b in c1:
        assert BlockName.parse(str(b)) == b


def test_chain_through():
    c = derive_chain(list(range(100)), "bf16", 16)
    assert len(chain_through(c, 100)) == 7
    assert len(chain_through(c, 33)) == 2    # partial block 3 excluded
    assert len(chain_through(c, 0)) == 0


def test_segment_bytes():
    c = derive_chain(list(range(32)), "bf16", 16)
    assert segment_bytes(c, 20480) == 32 * 20480


def test_repr_changes_name():
    a = derive_chain([1] * 32, "bf16", 16)
    b = derive_chain([1] * 32, "k8v8", 16)
    assert str(a[0]) != str(b[0]), "encoding is part of identity"


def main():
    c1 = test_chain_deterministic()
    test_parent_linking(c1)
    test_prefix_extension()
    test_cross_session_identity()
    test_roundtrip_parse(c1)
    test_chain_through()
    test_segment_bytes()
    test_repr_changes_name()
    print("blkchain: all tests OK")


if __name__ == "__main__":
    main()
