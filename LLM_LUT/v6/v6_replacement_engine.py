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

# Allow loading V6 checkpoints built by build_lut_ffn_output.py across different
# __main__ contexts. These classes are trusted because we built the checkpoints.
# NOTE: device_map here is a fixed, explicit map (e.g. balanced_low_0), never "auto".
torch.serialization.add_safe_globals([
    AddressGreedyTree, Address2D, AddressHighOrderRandom, _TreeNode,
])


def _load_v6_checkpoint(path: str):
    """Load a V6 checkpoint, allowing classes defined in build_lut_ffn_output.py.

    PyTorch 2.6+ defaults to weights_only=True; our classes are registered above.
    If that still fails (e.g. older PyTorch, or the checkpoint was pickled as
    __main__ classes from another script), inject the classes into the current
    __main__ module and fall back to full pickle.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        import __main__ as _main_mod
        for cls in (AddressGreedyTree, Address2D, AddressHighOrderRandom, _TreeNode):
            if not hasattr(_main_mod, cls.__name__):
                setattr(_main_mod, cls.__name__, cls)
        return torch.load(path, map_location="cpu", weights_only=False)


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
        device: Optional[torch.device] = None,
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
            ckpt = _load_v6_checkpoint(ckpt_path)
            gid = int(ckpt["group_id"])
            group_size = int(ckpt["group_size"])
            target_mode = ckpt.get("target_mode", "direct")
            addresses = ckpt["addresses"]
            if self.device is not None:
                tables = [t.to(self.device) for t in ckpt["lut_tables"]]
                gm = ckpt.get("group_mean")
                group_mean = gm.to(self.device) if gm is not None else None
            else:
                tables = [t for t in ckpt["lut_tables"]]
                gm = ckpt.get("group_mean")
                group_mean = gm if gm is not None else None

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
                indices = addr.compute_indices(x).view(-1, addr.num_tables)  # on x.device
                indices = indices.to(table.device)
                N = indices.shape[0]
                gathered = torch.zeros(
                    N, gs, device=table.device, dtype=table.dtype
                )
                for m in range(addr.num_tables):
                    idx_m = indices[:, m].clamp(0, table.shape[1] - 1)
                    gathered += table[m, idx_m]
                gathered = gathered.view(B, S, gs)
                pred = gathered if pred is None else pred + gathered

            if spec["target_mode"] == "residual_input":
                pred = pred + x[:, :, g_start:g_end]
            elif spec["target_mode"] == "residual_mean":
                gm = spec["group_mean"].view(1, 1, gs).to(out.dtype)
                pred = pred + gm

            out[:, :, g_start:g_end] = pred.to(device=out.device, dtype=out.dtype)

        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    def _relocate_tables_to_device(self, device: torch.device):
        """Move all LUT tables and group means to the target device."""
        for spec in self.group_specs.values():
            spec["tables"] = [t.to(device) for t in spec["tables"]]
            if spec["group_mean"] is not None:
                spec["group_mean"] = spec["group_mean"].to(device)
        self.device = device
        print(f"[V6Engine] Moved tables to {device}")

    def install(self):
        if self._hook_handle is not None:
            return

        if self.device is None:
            try:
                target_device = next(self.hook_mod.parameters()).device
            except StopIteration:
                target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            target_device = self.device

        self._relocate_tables_to_device(target_device)
        self._hook_handle = self.hook_mod.register_forward_hook(self._hook)
        print(f"[V6Engine] Hook installed on {self.hook_mod} (device={target_device})")

    def verify_replacement(self, hidden_size: int) -> bool:
        """
        Verify that the installed hook actually changes the MLP output.

        Runs a dummy input through the hook module with and without the hook,
        and checks whether the outputs differ. This proves the model sees the
        LUT output instead of the original MLP output for the replaced channels.
        """
        if self._hook_handle is None:
            raise RuntimeError("Hook is not installed. Call install() first.")

        try:
            param = next(self.hook_mod.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        dummy_x = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)

        # 1. Forward with hook active (LUT output)
        with torch.no_grad():
            out_with_hook = self.hook_mod(dummy_x)
        out_with_hook = out_with_hook[0] if isinstance(out_with_hook, tuple) else out_with_hook

        # 2. Temporarily remove hook and forward original MLP output
        self._hook_handle.remove()
        with torch.no_grad():
            out_without_hook = self.hook_mod(dummy_x)
        out_without_hook = out_without_hook[0] if isinstance(out_without_hook, tuple) else out_without_hook

        # 3. Reinstall hook
        self._hook_handle = self.hook_mod.register_forward_hook(self._hook)

        # 4. Compare on the same device
        out_with_hook = out_with_hook.to(device)
        out_without_hook = out_without_hook.to(device)
        diff = (out_with_hook - out_without_hook).abs().max().item()
        denom = out_without_hook.abs().max().item() + 1e-12
        rel_diff = diff / denom

        replaced_channels = max(gid * spec["group_size"] + spec["group_size"] for gid, spec in self.group_specs.items())
        print(f"[V6Engine] Replacement verification:")
        print(f"  output shape: {tuple(out_without_hook.shape)}")
        print(f"  replaced channels: {replaced_channels} / {hidden_size}")
        print(f"  max absolute diff: {diff:.6f}")
        print(f"  relative diff: {rel_diff:.2%}")

        if diff < 1e-4:
            print("[V6Engine] WARNING: Hook output is almost identical to original MLP output.")
            print("[V6Engine]          Replacement may not be active; check hook_path and group coverage.")
            return False
        else:
            print("[V6Engine] Replacement verified: LUT output differs from original MLP output.")
            return True

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
