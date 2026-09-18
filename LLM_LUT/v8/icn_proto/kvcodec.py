#!/usr/bin/env python3
"""Extract / inject / serialize KV objects from live HF caches.

Faithfulness rules (deviations here silently corrupt transferred sessions):

- Attn layer state moves as-is: keys, values, _hh_orig_idx, and the K/V
  quantization metadata (_k_meta/_v_meta). The stored tensors are copied
  exactly as the cache holds them (post-eviction, possibly INT8-quantized),
  so a resumed session continues bit-identically.
- Per-layer prefill score snapshots (_hh_prefill_scores and friends) are
  NOT part of the object. They are session-local, recomputable state: the
  score bank refills on the next prefill, and update() pulls fresh scores
  when the layer attribute is absent. Transferring them would also ship
  O(seq_len) fp32 tables per layer, defeating the compact representation.
- Non-standard layer state (GDN recurrent state on the 35B hybrid) is
  copied raw so a session migration loses nothing.
"""

import io
import json
import zlib
from typing import Dict

import torch

from .kvname import KVObject, KVName, LayerPayload, Geometry

_MAGIC = b"ICNKVO1\n"


def _is_attn_layer(layer) -> bool:
    return (torch.is_tensor(getattr(layer, "keys", None))
            and layer.keys is not None and layer.keys.dim() == 4
            and torch.is_tensor(getattr(layer, "values", None))
            and layer.values is not None and layer.values.dim() == 4)


def probe_geometry(cache) -> Geometry:
    """Read (n_full_attn_layers, kv_heads, head_dim) from a cache that has
    completed at least one forward. Raises if no 4-D KV layer is found."""
    attn = [l for l in cache.layers if _is_attn_layer(l)]
    if not attn:
        raise RuntimeError("cache has no initialized 4-D KV layer yet")
    k = attn[0].keys
    return Geometry(full_attn_layers=len(attn),
                    kv_heads=k.shape[1], head_dim=k.shape[3])


def extract_object(cache, name: KVName) -> KVObject:
    """Snapshot the cache's full state as a named, location-independent object.

    All tensors are detached and moved to CPU so the object can cross worker
    boundaries. The source cache is not modified.
    """
    obj = KVObject(name=name)
    for idx, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue  # layer never touched (e.g. not yet forwarded)
        keys, values = keys.detach().to("cpu"), values.detach().to("cpu")
        if _is_attn_layer(layer):
            orig_idx = getattr(layer, "_hh_orig_idx", None)
            payload = LayerPayload(
                kind="attn",
                keys=keys,
                values=values,
                orig_idx=(orig_idx.detach().to("cpu")
                          if torch.is_tensor(orig_idx) else None),
                k_meta=_cpu_meta(getattr(cache, "_k_meta", {}).get(idx)),
                v_meta=_cpu_meta(getattr(cache, "_v_meta", {}).get(idx)),
            )
        else:
            payload = LayerPayload(kind="aux", keys=keys, values=values)
        obj.layers[idx] = payload
    return obj


def _cpu_meta(meta):
    if meta is None:
        return None
    return tuple(t.detach().to("cpu") if torch.is_tensor(t) else t for t in meta)


def inject_object(cache, obj: KVObject) -> None:
    """Load an object's state into a fresh cache instance (same class/config
    as the cache the object was extracted from).

    Assumes the cache was constructed with config= so its layer list is
    pre-allocated. Device alignment is deferred to cache.to(device) at the
    receiving worker.
    """
    for idx, payload in obj.layers.items():
        if idx >= len(cache.layers):
            raise RuntimeError(
                f"object has layer {idx} but target cache has {len(cache.layers)}")
        layer = cache.layers[idx]
        layer.keys = payload.keys.clone()
        layer.values = payload.values.clone()
        if payload.kind == "attn":
            if payload.orig_idx is not None:
                layer._hh_orig_idx = payload.orig_idx.clone()
            metas = getattr(cache, "_k_meta", None)
            if metas is not None:
                if payload.k_meta is not None:
                    metas[idx] = tuple(t.clone() for t in payload.k_meta)
                else:
                    metas.pop(idx, None)
            metas = getattr(cache, "_v_meta", None)
            if metas is not None:
                if payload.v_meta is not None:
                    metas[idx] = tuple(t.clone() for t in payload.v_meta)
                else:
                    metas.pop(idx, None)


def dumps(obj: KVObject) -> bytes:
    """Serialize to wire bytes: header + torch payload + crc32."""
    header = json.dumps({
        "name": str(obj.name),
        "layers": {str(i): {"kind": p.kind}
                   for i, p in obj.layers.items()},
    }).encode()
    buf = io.BytesIO()
    torch.save({
        "name": str(obj.name),
        "layers": {i: p for i, p in obj.layers.items()},
    }, buf)
    payload = buf.getvalue()
    return _MAGIC + json.dumps(len(header)).encode() + b"\n" + header + \
        payload + zlib.crc32(payload).to_bytes(4, "big")


def loads(data: bytes) -> KVObject:
    if not data.startswith(_MAGIC):
        raise ValueError("bad magic: not an ICN KV object")
    rest = data[len(_MAGIC):]
    hlen_s, rest = rest.split(b"\n", 1)
    header = json.loads(rest[:int(hlen_s)].decode())
    body = rest[int(hlen_s):]
    payload, crc = body[:-4], int.from_bytes(body[-4:], "big")
    if zlib.crc32(payload) != crc:
        raise ValueError("crc mismatch: corrupted KV object")
    saved = torch.load(io.BytesIO(payload), weights_only=False)
    return KVObject(name=KVName.parse(saved["name"]), layers=saved["layers"])
