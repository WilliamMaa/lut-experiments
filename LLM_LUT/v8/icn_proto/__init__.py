#!/usr/bin/env python3
"""ICN-defined addressing prototype: named, transferable KV objects.

See docs/icn-defined-addressing/05-request-lifecycle.md (v3) for the
current architecture spec: prefix block chain + fast/slow two-timescale
control. The heavy modules (kvname/kvcodec) import torch, so they are
loaded lazily — pure modules like blkchain stay importable without it.
"""

import importlib

__all__ = [
    "Repr", "KVName", "KVObject", "repr_bytes_per_token", "Geometry",
    "extract_object", "inject_object", "dumps", "loads", "place_cache",
]

_LAZY = {
    "Repr": ("kvname", "Repr"),
    "KVName": ("kvname", "KVName"),
    "KVObject": ("kvname", "KVObject"),
    "repr_bytes_per_token": ("kvname", "repr_bytes_per_token"),
    "Geometry": ("kvname", "Geometry"),
    "extract_object": ("kvcodec", "extract_object"),
    "inject_object": ("kvcodec", "inject_object"),
    "dumps": ("kvcodec", "dumps"),
    "loads": ("kvcodec", "loads"),
    "place_cache": ("kvcodec", "place_cache"),
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f".{_LAZY[name][0]}", __package__)
        return getattr(mod, _LAZY[name][1])
    raise AttributeError(name)
