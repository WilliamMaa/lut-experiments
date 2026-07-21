"""
V6 LUT Replacement Engine for full model inference.

Loads LUT checkpoints built by build_lut_ffn_output.py and installs a forward
hook that replaces selected FFN output groups with table lookups.
"""

import os
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lut_ffn_output import AddressGreedyTree, Address2D, AddressHighOrderRandom, _TreeNode


class V6ReplacementEngine:
    """
    Hook-based functional replacement of FFN output groups using V6 LUTs.

    The hook is installed on the target module (typically the MLP block at a
    given layer). For every forward pass, it looks up the FFN output of the
    selected groups from the input activation and overwrites the corresponding
    output channels.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        checkpoint_dir: str,
        device: torch.device,
        hook_path: Optional[str] = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.group_specs: Dict[int, dict] = {}
        self._hook_handle = None

        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        for ckpt_path in sorted(ckpt_dir.glob("replacement_g*.pt")):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            gid = int(ckpt["group_id"])
            group_size = int(ckpt["group_size"])
            target_mode = ckpt.get("target_mode", "direct")
            addresses = ckpt["addresses"]
            tables = [t.to(device) for t in ckpt["lut_tables"]]
            gm = ckpt.get("group_mean")
            group_mean = gm.to(device) if gm is not None else None

            self.group_specs[gid] = {
                "group_size": group_size,
                "addresses": addresses,
                "tables": tables,
                "group_mean": group_mean,
                "target_mode": target_mode,
            }

        self.group_ids = sorted(self.group_specs.keys())
        if not self.group_ids:
            raise FileNotFoundError(f"No replacement_g*.pt found in {checkpoint_dir}")

        print(f"[V6Engine] loaded {len(self.group_ids)} groups from {ckpt_dir}: {self.group_ids}")

        if hook_path is None:
            try:
                self.hook_mod = model.model.layers[layer_idx].mlp
            except AttributeError as e:
                raise AttributeError(
                    f"Cannot auto-locate model.model.layers[{layer_idx}].mlp. "
                    f"Please pass --hook_path explicitly."
                ) from e
        else:
            try:
                self.hook_mod = eval(hook_path, {"model": model})
            except Exception as e:
                raise ValueError(f"Failed to evaluate hook_path={hook_path}: {e}") from e

    def _hook(self, module, inputs, output):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        out = output[0] if isinstance(output, tuple) else output

        B, S, hidden = out.shape
        for gid in self.group_ids:
            spec = self.group_specs[gid]
            gs = spec["group_size"]
            g_start = gid * gs
            g_end = g_start + gs
            if g_end > hidden:
                raise ValueError(
                    f"group {gid} range [{g_start}:{g_end}] exceeds hidden_size {hidden}"
                )

            pred = None
            for addr, table in zip(spec["addresses"], spec["tables"]):
                indices = addr.compute_indices(x).view(-1, addr.num_tables)  # [B*S, M]
                N = indices.shape[0]
                gathered = torch.zeros(
                    N, gs, device=self.device, dtype=table.dtype
                )
                for m in range(addr.num_tables):
                    idx_m = indices[:, m].clamp(0, table.shape[1] - 1)
                    gathered += table[m, idx_m]
                gathered = gathered.view(B, S, gs)
                pred = gathered if pred is None else pred + gathered

            if spec["target_mode"] == "residual_input":
                pred = pred + x[:, :, g_start:g_end]
            elif spec["target_mode"] == "residual_mean":
                gm = spec["group_mean"].view(1, 1, gs)
                pred = pred + gm.to(out.dtype)

            out[:, :, g_start:g_end] = pred.to(out.dtype)

        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    def install(self):
        if self._hook_handle is not None:
            return
        self._hook_handle = self.hook_mod.register_forward_hook(self._hook)
        print(f"[V6Engine] Hook installed on {self.hook_mod}")

    def uninstall(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            print("[V6Engine] Hook removed")

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()
        return False
