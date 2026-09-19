#!/usr/bin/env python3
"""Prefix block-chain naming (docs/icn-defined-addressing/05-request-lifecycle v3 §1).

Object model: the logical storage unit is an immutable prefix BLOCK of b
tokens, chain-linked by parent hash (vLLM APC style):

    B0 = H(model_version, tokens[0:b])
    Bi = H(B_{i-1}, tokens[i*b:(i+1)*b])

Blocks are the storage/transport/publish granularity. Placement decisions
operate on contiguous prefix SEGMENTS (views over the chain) — see 05 §1.

Identity needs nothing but content: a block name can always be re-derived
from the token prefix (zero-replica, zero-directory-entry safe).

Pure Python (hashlib/array only) so the chain logic is unit-testable
without torch.
"""

import hashlib
from array import array
from dataclasses import dataclass
from typing import List, Sequence

GENESIS = "GENESIS"


def _hash_tokens(parent_hash: str, token_ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    h.update(parent_hash.encode())
    h.update(b"|")
    h.update(array("q", token_ids).tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class BlockName:
    """Identity of one prefix block. parent_hash links the chain."""

    parent_hash: str            # GENESIS for B0, else B_{i-1}.block_hash
    block_hash: str
    block_idx: int
    span_start: int             # token positions [start, end)
    span_end: int
    repr_name: str = "bf16"

    def __str__(self) -> str:
        return (f"/blk/{self.block_hash}/parent/{self.parent_hash}"
                f"/i/{self.block_idx}/span/{self.span_start}-{self.span_end}"
                f"/repr/{self.repr_name}")

    @classmethod
    def parse(cls, s: str) -> "BlockName":
        parts = [p for p in s.split("/") if p]
        if (len(parts) != 10 or parts[0] != "blk" or parts[2] != "parent"
                or parts[4] != "i" or parts[6] != "span"
                or parts[8] != "repr"):
            raise ValueError(f"not a BlockName: {s!r}")
        start, end = parts[7].split("-")
        return cls(parent_hash=parts[3], block_hash=parts[1],
                   block_idx=int(parts[5]), span_start=int(start),
                   span_end=int(end), repr_name=parts[9])

    @property
    def span_tokens(self) -> int:
        return self.span_end - self.span_start


def derive_chain(token_ids: Sequence[int], repr_name: str = "bf16",
                 block_tokens: int = 16,
                 model_tag: str = "qwen35b") -> List[BlockName]:
    """Derive the ordered block-name chain for a token prefix.

    Pure function of the tokens: the same prefix always yields the same
    chain, in any session, on any worker — this is the whole content-
    addressing premise (05 §1, §3 identity invariant).
    """
    genesis = _hash_tokens(f"{GENESIS}:{model_tag}:{repr_name}", [])
    out = []
    parent = genesis
    n = len(token_ids)
    for i in range(0, max(1, (n + block_tokens - 1) // block_tokens)):
        start = i * block_tokens
        end = min(start + block_tokens, n)
        toks = token_ids[start:end]
        bh = _hash_tokens(parent, toks)
        out.append(BlockName(parent_hash=parent, block_hash=bh, block_idx=i,
                             span_start=start, span_end=end,
                             repr_name=repr_name))
        parent = bh
    return out


def chain_through(chain: Sequence[BlockName], n_tokens: int) -> List[BlockName]:
    """Longest prefix of the chain covering n_tokens (contiguous reuse:
    stop at the first incomplete block, vLLM APC semantics)."""
    out = []
    for b in chain:
        if b.span_end <= n_tokens:
            out.append(b)
        else:
            break
    return out


def segment_bytes(chain: Sequence[BlockName], bytes_per_token: int) -> int:
    """Size of a contiguous segment = sum of its blocks' spans."""
    return sum(b.span_tokens for b in chain) * bytes_per_token
