#!/usr/bin/env python3
"""KV object naming and representation definitions (icn_proto Step 1).

A KV object is the transferable unit of the prototype:

    name = /session/<doc_id>/turn/<t>/span/<start>-<end>/repr/<representation>
    payload = per-layer tensor state of one session's prefix span

Naming follows docs/icn-defined-addressing/00-ideas.md §2 (logical identity
decoupled from location). The representation tag carries the
quality/size trade-off that is this project's main lever
(02-real-prototype-plan.md §1): the same logical span can exist as bf16,
m_sp4 (HeavyHitter budget-128 state) or k8v8 (m_sp4 + INT8 storage).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import torch


class Repr(Enum):
    BF16 = "bf16"
    M_SP4 = "m_sp4"       # heavy_hitter_attn budget128 + merge_evicted + span4
    K8V8 = "k8v8"         # m_sp4 storage + K per-channel INT8 / V per-token INT8

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Geometry:
    """KV geometry of the served model, needed for size accounting.

    Derived by probing a cache after one forward (probe_geometry), not from
    config guessing, so the same code fits the 35B hybrid (10 full-attn
    layers) and small dense validation models (all layers full-attn).
    """

    full_attn_layers: int
    kv_heads: int
    head_dim: int

    @property
    def bf16_bytes_per_token(self) -> int:
        # K and V, 2 bytes each.
        return self.kv_heads * self.head_dim * 4

    def slots_bytes(self, repr: Repr, slots_per_layer: int) -> int:
        """Bytes of one object holding `slots_per_layer` slots on every
        full-attn layer, in the given storage representation."""
        per_layer = self.kv_heads * self.head_dim * 2 * slots_per_layer
        if repr is Repr.K8V8:
            per_layer //= 2  # INT8 storage halves K and V
        return per_layer * self.full_attn_layers


# m_sp4 / k8v8 resident budget, from kv_cache docs/19-final-summary.md 定型配置.
MSP4_SLOTS_PER_LAYER = 128  # sink 4 + recent 32 + hh 92

# Measured quality deltas (EOS-rate drop vs bf16 baseline, docs/19-final-summary.md).
# Used by the P2 allocator's quality-penalty term.
EOS_DELTA = {
    Repr.BF16: 0.0,
    Repr.M_SP4: 0.0,
    Repr.K8V8: 0.019,
}


def repr_bytes_per_token(repr: Repr, geometry: Geometry,
                         slots_per_layer: int = MSP4_SLOTS_PER_LAYER) -> float:
    """Effective bytes per context token for sizing/transfer decisions."""
    if repr is Repr.BF16:
        return float(geometry.bf16_bytes_per_token)
    return geometry.slots_bytes(repr, slots_per_layer) / slots_per_layer


@dataclass(frozen=True)
class KVName:
    """Content-addressed identity (docs/icn-defined-addressing/03 §4.2 修正 1).

    The name answers WHAT the content is, not WHO asked: the prefix_hash is
    a hash of (model tag, encoding, prefix token ids). Two requests whose
    token prefixes are identical — across sessions, across users — resolve
    to the same name and share the object through the NRS. The worker id /
    GPU is only a locator tracked by the index, never part of identity."""

    prefix_hash: str
    span_end: int                # number of prefix tokens covered (exclusive)
    repr: Repr = Repr.BF16

    def __str__(self) -> str:
        return (f"/prefix/{self.prefix_hash}/span/0-{self.span_end}"
                f"/repr/{self.repr.value}")

    @classmethod
    def parse(cls, s: str) -> "KVName":
        parts = [p for p in s.split("/") if p]
        if (len(parts) != 6 or parts[0] != "prefix" or parts[2] != "span"
                or parts[4] != "repr"):
            raise ValueError(f"not a KVName: {s!r}")
        start, end = parts[3].split("-")
        return cls(prefix_hash=parts[1], span_end=int(end),
                   repr=Repr(parts[5]))

    @property
    def span_tokens(self) -> int:
        return self.span_end


@dataclass
class LayerPayload:
    """Per-layer state of one object.

    kind == "attn": standard 4-D KV (optionally the post-eviction compact
    state with original-position bookkeeping and quantization metadata).
    kind == "linear": linear-attention (GDN) layer — transformers 5.x stores
    its state in conv_states/recurrent_states dicts plus flag dicts, NOT in
    keys/values; restored verbatim so a migrated session loses nothing.

    Score snapshots (_hh_prefill_scores et al.) are deliberately excluded:
    session-local, recomputable state (see kvcodec module docstring).
    """

    kind: str                                   # "attn" | "linear"
    keys: object = None                         # attn: [B, H, S, D]
    values: object = None
    orig_idx: Optional[object] = None           # attn: _hh_orig_idx
    k_meta: Optional[Tuple] = None              # attn: (scale, min) per-channel K
    v_meta: Optional[Tuple] = None              # attn: (scale, min) per-token V
    conv_states: Optional[Dict[int, object]] = None       # linear
    recurrent_states: Optional[Dict[int, object]] = None  # linear
    state_flags: Optional[Dict[str, Dict[int, object]]] = None  # linear flags + conv_kernel_size

    def nbytes(self) -> int:
        total = 0
        for t in (self.keys, self.values, self.orig_idx):
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
        for meta in (self.k_meta, self.v_meta):
            if meta is not None:
                total += sum(t.numel() * t.element_size() for t in meta
                             if torch.is_tensor(t))
        for d in (self.conv_states, self.recurrent_states):
            if d:
                total += sum(t.numel() * t.element_size()
                             for t in d.values() if torch.is_tensor(t))
        return total


@dataclass
class KVObject:
    name: KVName
    layers: Dict[int, LayerPayload] = field(default_factory=dict)

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self.layers.values())

    def attn_layers(self) -> Dict[int, LayerPayload]:
        return {i: p for i, p in self.layers.items() if p.kind == "attn"}
