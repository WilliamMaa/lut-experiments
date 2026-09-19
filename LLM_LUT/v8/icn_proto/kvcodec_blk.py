#!/usr/bin/env python3
"""Block-level extract / inject / serialization (05-request-lifecycle v3 §1).

Storage unit = immutable prefix block (blkchain.BlockName). This module is
the data-plane twin of blkchain: it slices a live cache into blocks and
rebuilds a cache from a contiguous block list.

Scope decisions (v1 of the block model):
- Attn layers must hold RAW contiguous K,V (plain DynamicCache semantics).
  Heavy-hitter folding destroys positional slicing, so the block chain
  currently pairs with bf16 storage; compressed encodings stay on the old
  snapshot path until a compressed block format exists.
- Linear-attention (GDN) state is a recurrent checkpoint S_t, not a
  per-block function: it is carried ONLY on the LAST block of a turn's
  published set. Inject takes the checkpoint from the latest block that
  carries one.
"""

import io
import json
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

from .blkchain import BlockName
from .kvcodec import _LINEAR_DICT_ATTRS, _is_attn_layer

_MAGIC = b"ICNBLK1\n"


@dataclass
class KVBlockObj:
    """Transferable payload of one block."""

    name: BlockName
    attn: Dict[int, dict] = field(default_factory=dict)
    # attn[layer_idx] = {"keys": [B,H,S,D], "values": [...]}
    linear_checkpoint: bool = False
    conv_states: dict = None
    recurrent_states: dict = None
    state_flags: dict = None

    def nbytes(self) -> int:
        total = 0
        for p in self.attn.values():
            for t in (p["keys"], p["values"]):
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
        for d in (self.conv_states, self.recurrent_states):
            if d:
                total += sum(t.numel() * t.element_size()
                             for t in d.values() if torch.is_tensor(t))
        return total


def extract_blocks(cache, new_blocks: Sequence[BlockName]) -> List[KVBlockObj]:
    """Slice the blocks (all complete, same chain) out of a live cache.

    The linear checkpoint, if any linear layer state exists, is attached
    to the LAST block — the recurrent state describes the whole prefix,
    so it belongs to the chain tip, not to any single block.
    """
    if not new_blocks:
        return []
    objs = [KVBlockObj(name=b) for b in new_blocks]
    linear_payload = None
    for idx, layer in enumerate(cache.layers):
        if _is_attn_layer(layer):
            k, v = layer.keys, layer.values
            for obj in objs:
                s, e = obj.name.span_start, obj.name.span_end
                obj.attn[idx] = {"keys": k[..., s:e, :].detach().to("cpu"),
                                 "values": v[..., s:e, :].detach().to("cpu")}
        elif getattr(layer, "conv_states", None) is not None or \
                getattr(layer, "recurrent_states", None) is not None:
            if linear_payload is None:
                linear_payload = {}
                for attr in ("conv_states", "recurrent_states"):
                    d = getattr(layer, attr, None)
                    linear_payload[attr] = (
                        {i: t.detach().to("cpu") for i, t in d.items()
                         if torch.is_tensor(t)} if d else None)
                flags = {}
                for attr in _LINEAR_DICT_ATTRS:
                    d = getattr(layer, attr, None)
                    if d is not None:
                        flags[attr] = dict(d)
                linear_payload["state_flags"] = flags or None
    if linear_payload is not None:
        tip = objs[-1]
        tip.linear_checkpoint = True
        tip.conv_states = linear_payload["conv_states"]
        tip.recurrent_states = linear_payload["recurrent_states"]
        tip.state_flags = linear_payload["state_flags"]
    return objs


def inject_blocks(cache, blocks: Sequence[KVBlockObj]) -> int:
    """Rebuild cache state from a contiguous block list. Returns the token
    length covered (chain tip). Attn layers concatenate block slices; the
    linear checkpoint comes from the last block that carries one.
    """
    if not blocks:
        return 0
    # group attn payloads per layer in chain order
    per_layer: Dict[int, dict] = {}
    tip_payload = None
    for b in blocks:
        for idx, sl in b.attn.items():
            per_layer.setdefault(idx, {"keys": [], "values": []})
            per_layer[idx]["keys"].append(sl["keys"])
            per_layer[idx]["values"].append(sl["values"])
        if b.linear_checkpoint:
            tip_payload = b
    for idx, parts in per_layer.items():
        layer = cache.layers[idx]
        layer.keys = torch.cat(parts["keys"], dim=-2).clone()
        layer.values = torch.cat(parts["values"], dim=-2).clone()
        layer.is_initialized = True
    if tip_payload is not None:
        for idx, layer in enumerate(cache.layers):
            for attr in ("conv_states", "recurrent_states"):
                src = getattr(tip_payload, attr)
                if not src:
                    continue
                dst = getattr(layer, attr, None)
                if dst is None:
                    continue
                for i, t in src.items():
                    dst[i] = t.clone()
            if tip_payload.state_flags:
                for attr, flags in tip_payload.state_flags.items():
                    dst = getattr(layer, attr, None)
                    if isinstance(dst, dict):
                        dst.update(flags)
    return blocks[-1].name.span_end


def dumps_block(obj: KVBlockObj) -> bytes:
    header = json.dumps({"name": str(obj.name)}).encode()
    buf = io.BytesIO()
    torch.save(obj, buf)
    payload = buf.getvalue()
    return (_MAGIC + json.dumps(len(header)).encode() + b"\n" + header
            + payload + zlib.crc32(payload).to_bytes(4, "big"))


def loads_block(data: bytes) -> KVBlockObj:
    if not data.startswith(_MAGIC):
        raise ValueError("bad magic: not an ICN block object")
    rest = data[len(_MAGIC):]
    hlen_s, rest = rest.split(b"\n", 1)
    body = rest[int(hlen_s):]
    payload, crc = body[:-4], int.from_bytes(body[-4:], "big")
    if zlib.crc32(payload) != crc:
        raise ValueError("crc mismatch: corrupted block object")
    return torch.load(io.BytesIO(payload), weights_only=False)
