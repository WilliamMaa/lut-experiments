#!/usr/bin/env python3
"""Inspect transformers DynamicCache internals."""

import transformers
from transformers.cache_utils import DynamicCache
import inspect


def main():
    print("=== DynamicCache.__init__ ===")
    print(inspect.getsource(DynamicCache.__init__))
    print()

    print("=== DynamicCache.update ===")
    print(inspect.getsource(DynamicCache.update))
    print()

    print("=== DynamicCache.get ===")
    print(inspect.getsource(DynamicCache.get) if hasattr(DynamicCache, "get") else "no get")
    print()

    print("=== DynamicCache.to ===")
    print(inspect.getsource(DynamicCache.to) if hasattr(DynamicCache, "to") else "no to")
    print()

    print("=== DynamicCache.layers type ===")
    print(type(DynamicCache().layers))
    print("len:", len(DynamicCache().layers))
    print()

    # Create a dummy cache with one layer to inspect internal state.
    import torch
    cache = DynamicCache()
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    cache.update(k, v, layer_idx=0)
    print("=== after one update, layers len ===")
    print(len(cache.layers))
    print()

    if len(cache.layers) > 0:
        layer = cache.layers[0]
        print("=== layer type ===")
        print(type(layer))
        print()
        print("=== layer attrs ===")
        for name in sorted(dir(layer)):
            print(name)
        print()
        print("=== layer.__init__ ===")
        print(inspect.getsource(layer.__init__) if hasattr(layer, "__init__") else "no init")
        print()
        print("=== layer.update ===")
        print(inspect.getsource(layer.update) if hasattr(layer, "update") else "no update")
        print()
        try:
            print("=== layer internal state ===")
            for k_attr, v_attr in layer.__dict__.items():
                print(f"{k_attr}: {type(v_attr)} {v_attr if not torch.is_tensor(v_attr) else v_attr.shape}")
        except Exception as e:
            print(e)
    print()

    # Inspect Qwen config-created cache layer types.
    print("=== Qwen config-created cache layer types ===")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("/home/u/downloads/models/Qwen3.6-35B-A3B")
        qwen_cache = DynamicCache(config=config)
        print(f"num layers: {len(qwen_cache.layers)}")
        for i, layer in enumerate(qwen_cache.layers):
            print(f"layer {i}: type={type(layer).__name__}, _layer_type={getattr(layer, '_layer_type', 'N/A')}")
    except Exception as e:
        print(e)
    print()

    print("=== public attrs ===")
    for name in sorted(dir(DynamicCache)):
        if not name.startswith("_"):
            print(name)
    print()

    print("=== all attrs ===")
    for name in sorted(dir(DynamicCache)):
        print(name)


if __name__ == "__main__":
    main()
