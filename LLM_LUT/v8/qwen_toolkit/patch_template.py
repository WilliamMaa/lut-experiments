#!/usr/bin/env python3
"""Reusable patch template for replacing a submodule in Qwen3.6-35B-A3B.

Usage example (replace layer20 GDN forward with a no-op):
  cd LLM_LUT/v8
  python -u qwen_toolkit/patch_template.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --layer_idx 20 \
    --submodule linear_attn \
    --device_map balanced_low_0
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer


class SubmodulePatch:
    """Base class for a submodule replacement.

    Subclasses must implement:
      - forward(self, module, *args, **kwargs): replacement forward logic.

    The original forward is available as self.orig_forward when inside forward().
    """

    def __init__(self):
        self.orig_forward = None
        self._patched_module = None

    def forward(self, module, *args, **kwargs):
        raise NotImplementedError

    def install(self, parent_module, submodule_name):
        submodule = getattr(parent_module, submodule_name)
        self._patched_module = submodule
        self.orig_forward = submodule.forward
        submodule.forward = lambda *args, **kwargs: self.forward(submodule, *args, **kwargs)
        return self

    def uninstall(self):
        if self._patched_module is not None and self.orig_forward is not None:
            self._patched_module.forward = self.orig_forward
        self._patched_module = None
        self.orig_forward = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--submodule", default="linear_attn")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="balanced_low_0")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path,
        args.torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    layer = model.model.layers[args.layer_idx]
    if not hasattr(layer, args.submodule):
        raise ValueError(f"Layer {args.layer_idx} has no submodule {args.submodule}")

    class DummyForward(SubmodulePatch):
        def forward(self, module, *args, **kwargs):
            # Replace with original to sanity-check the patch installs/uninstalls correctly.
            return self.orig_forward(*args, **kwargs)

    patch = DummyForward().install(layer, args.submodule)
    print(f"Patched {args.submodule} on layer {args.layer_idx}")
    patch.uninstall()
    print("Unpatched successfully")


if __name__ == "__main__":
    main()
