#!/usr/bin/env python3
"""Example of how to implement an EvalPatch for v8.

A patch must implement:
  - install(model): apply the modification
  - uninstall(model): revert the modification
  - name(): human-readable name
  - config(): dict of hyperparameters for logging
"""

from typing import Any, Dict
from common.evaluator import EvalPatch


class ExamplePatch(EvalPatch):
    """No-op example patch showing the required interface."""

    def __init__(self, layer_idx: int = 39):
        self.layer_idx = layer_idx
        self._original_module = None

    def install(self, model):
        """Replace a target module with a patched version."""
        # Example: locate the module
        # target = eval(f"model.model.layers[{self.layer_idx}].self_attn.o_proj")
        # self._original_module = target
        # model.model.layers[self.layer_idx].self_attn.o_proj = MyPatchedLinear(target)
        pass

    def uninstall(self, model):
        """Restore the original module."""
        # if self._original_module is not None:
        #     model.model.layers[self.layer_idx].self_attn.o_proj = self._original_module
        #     self._original_module = None
        pass

    def name(self) -> str:
        return f"example_patch_l{self.layer_idx}"

    def config(self) -> Dict[str, Any]:
        return {"layer_idx": self.layer_idx}
