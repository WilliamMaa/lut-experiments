#!/usr/bin/env python3
"""ICN-defined addressing prototype: named, transferable KV objects.

Step 1 scope: KV object layer — naming, representation definitions, and
serialization. See docs/icn-defined-addressing/02-real-prototype-plan.md.
"""

from .kvname import Repr, KVName, KVObject, repr_bytes_per_token, Geometry
from .kvcodec import extract_object, inject_object, dumps, loads, place_cache

__all__ = [
    "Repr", "KVName", "KVObject", "repr_bytes_per_token", "Geometry",
    "extract_object", "inject_object", "dumps", "loads", "place_cache",
]
