"""
Hybrid partial replacement engine for v5.

Supports:
- 2D channel address or high-order random address
- Trainable ensemble LUT tables
- down_proj partial replacement (residual LUT delta)

The address generators are fixed/random (no trainable neural nets), while the
LUT table values are trainable nn.Parameters.
"""

import torch
import torch.nn.functional as F

from address import Address2D, AddressHighOrderRandom
from lut import LUTGroup


class HybridPartialEngine:
    """
    Partial down_proj replacement with trainable ensemble LUT.

    Args:
        model: LLM already on target device
        layer_id: decoder layer to patch
        group_size: output channels per group
        use_residual: if True, LUT stores (down_proj_output - residual) and
            reconstructs as ``residual_group + lut``. This matches v3/v4.
    """

    def __init__(self, model, layer_id: int, group_size: int = 64, use_residual: bool = True):
        self.model = model
        self.layer_id = layer_id
        self.group_size = group_size
        self.use_residual = use_residual

        self.group_configs = {}  # {group_id: (address_generator, LUTGroup)}

        self.down_proj = None
        self._original_forward = None
        self._hook_handle = None

        self._cached_normed_x = None
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
        mlp = layer.mlp
        self.down_proj = mlp.down_proj
        hidden_size = self.down_proj.weight.shape[0]
        num_groups = hidden_size // self.group_size

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
            active_channels, device=self.down_proj.weight.device, dtype=torch.long
        )
        self._replaced_indices = torch.tensor(
            replaced_channels, device=self.down_proj.weight.device, dtype=torch.long
        )

        self._hook_handle = mlp.register_forward_pre_hook(self._mlp_pre_hook)
        self._original_forward = self.down_proj.forward
        engine = self

        def patched_forward(hidden):
            return engine._patched_forward(hidden)

        self.down_proj.forward = patched_forward
        print(f"[v5] Hybrid LUT installed: L{self.layer_id}, "
              f"groups={replaced_groups}, tables={len(replaced_groups)}")

    def uninstall(self):
        """Restore original forward and remove hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._original_forward is not None:
            self.down_proj.forward = self._original_forward
            self._original_forward = None
        self._cached_normed_x = None
        self._cached_indices = {}
        print(f"[v5] Hybrid LUT removed: L{self.layer_id}")

    def _mlp_pre_hook(self, module, input):
        """Cache normed_x and precompute bin indices for all replaced groups."""
        normed_x = input[0] if isinstance(input, tuple) else input
        self._cached_normed_x = normed_x
        self._cached_indices = {}
        for gid, (address, _) in self.group_configs.items():
            self._cached_indices[gid] = address.compute_indices(normed_x)

    def _patched_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Partial matmul + LUT fill."""
        B, S, intermediate_size = hidden.shape
        device = hidden.device
        dtype = hidden.dtype
        compute_dtype = torch.float32

        with torch.autocast(device_type=device.type, enabled=False):
            hidden_f32 = hidden.to(compute_dtype)
            normed_x = self._cached_normed_x
            if normed_x is None:
                raise RuntimeError("normed_x not cached; MLP pre-hook was not called.")
            normed_x_f32 = normed_x.to(compute_dtype)
            hidden_size = normed_x_f32.shape[-1]

            # Active linear for non-replaced groups (sliced each forward for trainability)
            active_weight = self.down_proj.weight[self._active_indices, :].to(compute_dtype)
            active_bias = None
            if self.down_proj.bias is not None:
                active_bias = self.down_proj.bias[self._active_indices].to(compute_dtype)
            active_out = F.linear(hidden_f32, active_weight, active_bias)

            if not torch.isfinite(active_out).all():
                raise RuntimeError(f"[v5 L{self.layer_id}] active_out has NaN/Inf")

            # LUT fill for replaced groups
            lut_outputs = []
            for gid in self._replaced_groups:
                address, lut_group = self.group_configs[gid]
                indices = self._cached_indices[gid]  # [B, S, num_tables]
                lut_out = lut_group(indices)  # [B, S, group_size]

                if self.use_residual:
                    g_start = gid * self.group_size
                    lut_out = lut_out + normed_x_f32[:, :, g_start:g_start + self.group_size]

                lut_outputs.append(lut_out)

            if lut_outputs:
                lut_outputs_cat = torch.cat(lut_outputs, dim=-1)
            else:
                lut_outputs_cat = torch.empty(B, S, 0, device=device, dtype=compute_dtype)

            if not torch.isfinite(lut_outputs_cat).all():
                raise RuntimeError(f"[v5 L{self.layer_id}] lut_outputs has NaN/Inf")

            full_out = torch.zeros(B, S, hidden_size, device=device, dtype=compute_dtype)
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
        import os
        os.makedirs(output_dir, exist_ok=True)
        for gid, (address, lut_group) in self.group_configs.items():
            path = os.path.join(output_dir, f"replacement_l{self.layer_id}g{gid}.pt")
            state = {
                "layer_id": self.layer_id,
                "group_id": gid,
                "group_size": self.group_size,
                "use_residual": self.use_residual,
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
                state["num_tables"] = address.num_tables
                state["num_bits"] = address.num_bits
                state["channels_per_bit"] = address.channels_per_bit
                state["channel_idx"] = address.channel_idx
                state["signs"] = address.signs
                state["addr_mean"] = address.addr_mean
                state["addr_std"] = address.addr_std
            elif isinstance(address, AddressGreedyTree):
                state["address_type"] = "tree"
                state["num_bits"] = address.num_bits
                state["channels_per_bit"] = address.channels_per_bit
                state["tree_state"] = address.serialize()
            torch.save(state, path)


def load_group_checkpoint(model, path: str) -> "HybridPartialEngine":
    """Load a single group checkpoint and return a fresh engine with that group."""
    import os
    ckpt = torch.load(path, map_location="cpu")
    layer_id = ckpt["layer_id"]
    engine = HybridPartialEngine(model, layer_id, group_size=ckpt["group_size"],
                                 use_residual=ckpt.get("use_residual", True))
    if ckpt["address_type"] == "2d":
        address = Address2D(
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt["addr_clip"],
        )
    elif ckpt["address_type"] == "high_order":
        # input_dim is inferred from channel_idx; pass a placeholder here.
        address = AddressHighOrderRandom(
            input_dim=1,
            num_tables=ckpt["num_tables"],
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
        )
        address.channel_idx = ckpt["channel_idx"]
        address.signs = ckpt["signs"]
        address.input_dim = int(ckpt["channel_idx"].max().item()) + 1
    else:
        raise ValueError(f"Unknown address type: {ckpt['address_type']}")

    num_tables = ckpt["lut_table"].shape[0]
    num_entries = ckpt["lut_table"].shape[1]
    group_size = ckpt["lut_table"].shape[2]
    lut_group = LUTGroup(num_tables, num_entries, group_size, init_table=ckpt["lut_table"])
    engine.add_group(ckpt["group_id"], address, lut_group)
    return engine
