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


_LINEAR_DICT_ATTRS = (
    "is_conv_states_initialized",
    "is_recurrent_states_initialized",
    "has_previous_state",
    "conv_kernel_size",  # ints set by lazy_initialization; None here crashes
                         # the model's update_conv_state on the next turn
)


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

    Handles both layer flavors created by DynamicCache(config) on hybrid
    models: 4-D KV attention layers and linear-attention (GDN) layers whose
    state lives in conv_states/recurrent_states dicts. All tensors are
    detached and moved to CPU so the object can cross worker boundaries.
    The source cache is not modified.
    """
    obj = KVObject(name=name)
    for idx, layer in enumerate(cache.layers):
        conv = getattr(layer, "conv_states", None)
        rec = getattr(layer, "recurrent_states", None)
        if conv is not None or rec is not None:
            flags = {}
            for attr in _LINEAR_DICT_ATTRS:
                d = getattr(layer, attr, None)
                if d is not None:
                    flags[attr] = dict(d)

            def _cpu_dict(d):
                if not d:
                    return None
                out = {i: t.detach().to("cpu") for i, t in d.items()
                       if torch.is_tensor(t)}
                return out or None

            obj.layers[idx] = LayerPayload(
                kind="linear",
                conv_states=_cpu_dict(conv),
                recurrent_states=_cpu_dict(rec),
                state_flags=flags or None,
            )
            continue
        if not _is_attn_layer(layer):
            continue  # layer never touched (e.g. not yet forwarded)
        orig_idx = getattr(layer, "_hh_orig_idx", None)
        obj.layers[idx] = LayerPayload(
            kind="attn",
            keys=layer.keys.detach().to("cpu"),
            values=layer.values.detach().to("cpu"),
            orig_idx=(orig_idx.detach().to("cpu")
                      if torch.is_tensor(orig_idx) else None),
            k_meta=_cpu_meta(getattr(cache, "_k_meta", {}).get(idx)),
            v_meta=_cpu_meta(getattr(cache, "_v_meta", {}).get(idx)),
        )
    return obj


def _cpu_meta(meta):
    if meta is None:
        return None
    return tuple(t.detach().to("cpu") if torch.is_tensor(t) else t for t in meta)


def inject_object(cache, obj: KVObject) -> None:
    """Load an object's state into a fresh cache instance (same class/config
    as the cache the object was extracted from).

    Assumes the cache was constructed with config= so its layer list is
    pre-allocated. Device alignment is deferred to place_cache() at the
    receiving worker.
    """
    for idx, payload in obj.layers.items():
        if idx >= len(cache.layers):
            raise RuntimeError(
                f"object has layer {idx} but target cache has {len(cache.layers)}")
        layer = cache.layers[idx]
        if payload.kind == "linear":
            for attr in ("conv_states", "recurrent_states"):
                src = getattr(payload, attr)
                if not src:
                    continue
                dst = getattr(layer, attr, None)
                if dst is None:
                    raise RuntimeError(
                        f"target cache layer {idx} has no {attr} to inject into")
                for i, t in src.items():
                    dst[i] = t.clone()
            if payload.state_flags:
                for attr, flags in payload.state_flags.items():
                    dst = getattr(layer, attr, None)
                    if isinstance(dst, dict):
                        dst.update(flags)
            continue
        if not (torch.is_tensor(payload.keys) and torch.is_tensor(payload.values)):
            raise RuntimeError(
                f"object layer {idx} ({payload.kind}) holds non-tensor state: "
                f"keys={type(payload.keys).__name__} values={type(payload.values).__name__}")
        layer.keys = payload.keys.clone()
        layer.values = payload.values.clone()
        # Without this the next update() treats the layer as fresh and
        # lazy_initialization CLOBBERS the injected state (measured on the
        # 35B hybrid: injected session silently recomputed from scratch).
        layer.is_initialized = True
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


def place_cache(cache, model) -> None:
    """Move each layer's state to the device the model runs that layer on.

    Needed after inject (payloads arrive on CPU). Device resolution order:
    1. the layer submodule's OWN parameters — ground truth for where
       accelerate actually runs the layer (offload hooks included);
    2. model.hf_device_map entries (can be MISSING for offloaded layers
       under uneven tenant GPUs — that gap sent layer 39's KV to CPU
       while the layer computed on cuda:0);
    3. the model's first parameter.
    """
    dev_map = getattr(model, "hf_device_map", None)
    core = getattr(model, "model", model)   # ForCausalLM -> .model
    decoder_layers = getattr(core, "layers", None)

    def layer_device(idx):
        if decoder_layers is not None and idx < len(decoder_layers):
            try:
                return next(decoder_layers[idx].parameters()).device
            except StopIteration:
                pass
        if dev_map:
            for name, dev in dev_map.items():
                if name.endswith(f"layers.{idx}"):
                    return torch.device(dev)
        return next(model.parameters()).device

    tensor_attrs = ("keys", "values", "_hh_orig_idx", "_hh_prefill_scores",
                    "_hh_prefill_scores_per_head", "_hh_shared_scores")
    for idx, layer in enumerate(cache.layers):
        dev = layer_device(idx)
        for attr in tensor_attrs:
            t = getattr(layer, attr, None)
            if torch.is_tensor(t) and t.device != dev:
                setattr(layer, attr, t.to(dev))
        for attr in ("conv_states", "recurrent_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                for k, t in d.items():
                    if torch.is_tensor(t) and t.device != dev:
                        d[k] = t.to(dev)
    for name in ("_k_meta", "_v_meta"):
        metas = getattr(cache, name, None)
        if metas:
            for k, tup in list(metas.items()):
                metas[k] = tuple(t.to(layer_device(k)) if torch.is_tensor(t) else t
                                 for t in tup)
    # self-check: any layer state left on CPU after the move? Diagnostic
    # for the intermittent "cpu vs cuda" cat failure — distinguishes
    # "setattr did not stick / wrong device resolved" (this list is
    # non-empty) from "the model swapped the cache object inside forward"
    # (this list is empty yet the forward still fails).
    left = []
    for idx, layer in enumerate(cache.layers):
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if torch.is_tensor(t) and t.device.type == "cpu":
                left.append((idx, attr, str(layer_device(idx))))
        for attr in ("conv_states", "recurrent_states"):
            d = getattr(layer, attr, None)
            if isinstance(d, dict):
                for k, t in d.items():
                    if torch.is_tensor(t) and t.device.type == "cpu":
                        left.append((idx, f"{attr}[{k}]",
                                     str(layer_device(idx))))
    if left:
        print(f"[place_cache] WARN {len(left)} tensors still on CPU: "
              f"{left[:6]}", flush=True)
    else:
        print("[place_cache] all layer state on accelerator devices",
              flush=True)


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
