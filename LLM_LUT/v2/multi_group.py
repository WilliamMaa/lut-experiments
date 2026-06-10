"""Multi-Group Replacement Engine for LLM-LUT v2.

Supports simultaneous replacement of multiple groups
(either same layer or different layers).
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

import torch
from config import get_hook_target
from r1_replacement import ReplacementEngine


class MultiGroupEngine:
    """
    Manage multiple ReplacementEngine instances.

    Usage:
        engine = MultiGroupEngine()
        for group_cfg in groups:
            re = ReplacementEngine(model, layer_id, group_id, ...)
            engine.add(re)
        engine.install_all()
        # ... evaluate ...
        engine.uninstall_all()
    """

    def __init__(self):
        self.engines = []

    def add(self, engine: ReplacementEngine):
        self.engines.append(engine)

    def install_all(self):
        for e in self.engines:
            e.install()
        print(f"[MultiGroup] {len(self.engines)} hooks installed")

    def uninstall_all(self):
        for e in self.engines:
            e.uninstall()
        print(f"[MultiGroup] {len(self.engines)} hooks removed")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "num_engines": len(self.engines),
            "engines": [{
                "layer_id": e.layer_id,
                "group_id": e.group_id,
                "group_size": e.group_size,
                "addr_idx": e.addr_idx.cpu(),
                "addr_mean": e.addr_mean.cpu(),
                "addr_std": e.addr_std.cpu(),
                "table": e.table.cpu(),
                "num_bins": e.num_bins,
                "addr_clip": e.addr_clip,
            } for e in self.engines]
        }, path)
        print(f"[MultiGroup] Saved to {path}")

    @classmethod
    def load(cls, model, path: str):
        ckpt = torch.load(path, map_location="cpu")
        engine = cls()
        for e_cfg in ckpt["engines"]:
            re = ReplacementEngine(
                model=model,
                layer_id=e_cfg["layer_id"],
                group_id=e_cfg["group_id"],
                group_size=e_cfg["group_size"],
                addr_idx=e_cfg["addr_idx"],
                addr_mean=e_cfg["addr_mean"],
                addr_std=e_cfg["addr_std"],
                table=e_cfg["table"],
                num_bins=e_cfg["num_bins"],
                addr_clip=e_cfg["addr_clip"],
            )
            engine.add(re)
        print(f"[MultiGroup] Loaded {len(engine.engines)} engines from {path}")
        return engine
