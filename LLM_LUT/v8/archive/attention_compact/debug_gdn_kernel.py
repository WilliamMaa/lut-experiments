#!/usr/bin/env python3
"""One-shot debug script to dump GDN kernel source and class attributes."""

import importlib
import inspect
import sys


def _dump():
    mods = [
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "transformers.models.qwen3.modeling_qwen3",
    ]
    for mod_name in mods:
        print("=" * 80)
        print(f"MODULE: {mod_name}")
        print("=" * 80)
        try:
            m = importlib.import_module(mod_name)
        except Exception as e:
            print(f"  import failed: {e}")
            continue

        names = [n for n in dir(m) if "delta" in n.lower() or "recurrent" in n.lower() or "chunk" in n.lower()]
        print(f"  relevant names: {names}")
        for n in names:
            obj = getattr(m, n)
            print(f"\n  --- {n} (id={id(obj)}) ---")
            try:
                src = inspect.getsource(obj)
                print(src)
            except Exception as e:
                print(f"    cannot get source: {e}")

    print("\n" + "=" * 80)
    print("Qwen3_5MoeGatedDeltaNet forward source")
    print("=" * 80)
    m = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    try:
        print(inspect.getsource(m.Qwen3_5MoeGatedDeltaNet.forward))
    except Exception as e:
        print(f"  cannot get source: {e}")


if __name__ == "__main__":
    _dump()
