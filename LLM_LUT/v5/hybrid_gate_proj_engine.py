"""
Hybrid gate_proj partial replacement engine for v5.

Replaces selected output-channel groups of mlp.gate_proj with O(1) LUT lookups.
The LUT predicts the pre-activation gate_proj output; SiLU and the element-wise
multiplication with up_proj happen in the original model forward.
"""

import os
import torch
import torch.nn.functional as F

from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup


class HybridGateProjEngine:
    """
    Partial gate_proj replacement with trainable ensemble LUT.

    Args:
        model: LLM already on target device
        layer_id: decoder layer to patch
        group_size: output channels per group (gate_proj output dimension)
    """

    def __init__(self, model, layer_id: int, group_size: int = 64):
        self.model = model
        self.layer_id = layer_id
        self.group_size = group_size

        self.group_configs = {}

        self.gate_proj = None
        self._original_forward = None
        self._hook_handle = None

        self._cached_x = None
        self._cached_indices = {}

        self._active_indices = None
        self._replaced_indices = None
        self._active_groups = None
        self._replaced_groups = None

    def add_group(self, group_id: int, address, lut_group: LUTGroup):
        """Add a group to be replaced."""
        self.group_configs[group_id] = (address, lut_group)

    def install(self):
        """Install hooks and patched forward."""
        if self._original_forward is not None:
            return

        layer = self.model.model.layers[self.layer_id]
        self.gate_proj = layer.mlp.gate_proj
        intermediate_size = self.gate_proj.weight.shape[0]
        num_groups = intermediate_size // self.group_size

        replaced_groups = sorted(self.group_configs.keys())
        active_groups = [g for g in range(num_groups) if g not in replaced_groups]

        active_channels = []
        replaced_channels = []
        for g in active_groups:
            active_channels.extend(range(g * self.group_size, (g + 1) * self.group_size))
        for g in replaced_groups:
            replaced_channels.extend(range(g * self.group_size, (g + 1) * self.group_size))

        self._active_groups = active_groups
        self._replaced_groups = replaced_groups
        self._active_indices = torch.tensor(
            active_channels, device=self.gate_proj.weight.device, dtype=torch.long
        )
        self._replaced_indices = torch.tensor(
            replaced_channels, device=self.gate_proj.weight.device, dtype=torch.long
        )

        self._hook_handle = self.gate_proj.register_forward_pre_hook(self._gate_proj_pre_hook)
        self._original_forward = self.gate_proj.forward
        engine = self

        def patched_forward(hidden):
            return engine._patched_forward(hidden)

        self.gate_proj.forward = patched_forward
        print(f"[v5] Hybrid gate_proj LUT installed: L{self.layer_id}, "
              f"groups={replaced_groups}, tables={len(replaced_groups)}")

    def uninstall(self):
        """Restore original forward and remove hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._original_forward is not None:
            self.gate_proj.forward = self._original_forward
            self._original_forward = None
        self._cached_x = None
        self._cached_indices = {}
        print(f"[v5] Hybrid gate_proj LUT removed: L{self.layer_id}")

    def _gate_proj_pre_hook(self, module, input):
        """Cache gate_proj input x and precompute bin indices."""
        x = input[0] if isinstance(input, tuple) else input
        self._cached_x = x
        self._cached_indices = {}
        for gid, (address, _) in self.group_configs.items():
            self._cached_indices[gid] = address.compute_indices(x)

    def _patched_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Partial matmul + LUT fill for gate_proj."""
        B, S, hidden_size = hidden.shape
        device = hidden.device
        dtype = hidden.dtype
        compute_dtype = dtype  # avoid full FP32 intermediate tensor

        with torch.autocast(device_type=device.type, enabled=False):
            # Active linear for non-replaced groups (cast weight to activation dtype)
            active_weight = self.gate_proj.weight[self._active_indices, :].to(compute_dtype)
            active_bias = None
            if self.gate_proj.bias is not None:
                active_bias = self.gate_proj.bias[self._active_indices].to(compute_dtype)
            active_out = F.linear(hidden, active_weight, active_bias)

            if not torch.isfinite(active_out).all():
                raise RuntimeError(f"[v5 L{self.layer_id} gate_proj] active_out has NaN/Inf")

            # LUT fill for replaced groups
            lut_outputs = []
            for gid in self._replaced_groups:
                address, lut_group = self.group_configs[gid]
                indices = self._cached_indices[gid]
                lut_out = lut_group(indices).to(compute_dtype)
                lut_outputs.append(lut_out)

            if lut_outputs:
                lut_outputs_cat = torch.cat(lut_outputs, dim=-1)
            else:
                lut_outputs_cat = torch.empty(B, S, 0, device=device, dtype=compute_dtype)

            if not torch.isfinite(lut_outputs_cat).all():
                raise RuntimeError(f"[v5 L{self.layer_id} gate_proj] lut_outputs has NaN/Inf")

            intermediate_size = self.gate_proj.weight.shape[0]
            full_out = torch.zeros(B, S, intermediate_size, device=device, dtype=compute_dtype)
            full_out = full_out.index_copy_(2, self._active_indices.to(device), active_out)
            if lut_outputs_cat.shape[-1] > 0:
                full_out = full_out.index_copy_(2, self._replaced_indices.to(device), lut_outputs_cat)

        return full_out.to(dtype)

    def trainable_parameters(self):
        """Return list of trainable parameters belonging to this engine."""
        params = []
        for _, lut_group in self.group_configs.values():
            params.append(lut_group.table)
        return params

    def save_group_checkpoints(self, output_dir: str):
        """Save per-group address + LUT table to v5 checkpoint format."""
        os.makedirs(output_dir, exist_ok=True)
        for gid, (address, lut_group) in self.group_configs.items():
            path = os.path.join(output_dir, f"replacement_l{self.layer_id}g{gid}.pt")
            state = {
                "proj_type": "gate_proj",
                "layer_id": self.layer_id,
                "group_id": gid,
                "group_size": self.group_size,
                "lut_table": lut_group.table.detach().cpu(),
            }
            if isinstance(address, Address2D):
                state["address_type"] = "2d"
                state["addr_idx"] = address.addr_idx
                state["addr_mean"] = address.addr_mean
                state["addr_std"] = address.addr_std
                state["num_bins"] = address.num_bins
                state["addr_clip"] = address.addr_clip
            elif isinstance(address, AddressHighOrderRandom):
                state["address_type"] = "high_order"
                state["input_dim"] = address.input_dim
                state["num_tables"] = address.num_tables
                state["num_bits"] = address.num_bits
                state["channels_per_bit"] = address.channels_per_bit
                state["channel_idx"] = address.channel_idx
                state["signs"] = address.signs
                state["addr_mean"] = address.addr_mean
                state["addr_std"] = address.addr_std
            elif isinstance(address, AddressGreedyTree):
                state["address_type"] = "tree"
                state["input_dim"] = address.input_dim
                state["num_bits"] = address.num_bits
                state["channels_per_bit"] = address.channels_per_bit
                state["tree_state"] = address.serialize()
            torch.save(state, path)
