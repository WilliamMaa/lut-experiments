"""
Partial o_proj replacement engine for v4.

Replaces selected output-channel groups of an attention o_proj with a 2D LUT lookup,
while keeping the remaining groups as trainable linear weights.

The LUT stores a residual delta (output - input) by default, matching the down_proj
partial-LUT semantics.
"""

import torch
import torch.nn.functional as F


class TrainableOProjPartialEngine:
    """
    Partial o_proj replacement engine.

    Args:
        model: the LLM (already on target device)
        layer_id: target decoder layer
        group_size: channels per output group
        num_bins: LUT bins per address dimension
        addr_clip: address z-score clipping value
        use_residual: if True, the LUT stores (o_proj_output - o_proj_input)
            and reconstructs as ``input + lookup``. If False, the LUT stores
            the full output.
    """

    def __init__(self, model, layer_id: int, group_size: int = 64,
                 num_bins: int = 64, addr_clip: float = 3.0, use_residual: bool = True):
        self.model = model
        self.layer_id = layer_id
        self.group_size = group_size
        self.num_bins = num_bins
        self.addr_clip = addr_clip
        self.use_residual = use_residual

        # group_configs: {group_id: (addr_idx, addr_mean, addr_std, table)}
        self.group_configs = {}

        self.o_proj = None
        self._original_forward = None

        self._active_indices = None
        self._replaced_indices = None
        self._active_groups = None
        self._replaced_groups = None

    def add_group(self, group_id: int, addr_idx: torch.Tensor,
                  addr_mean: torch.Tensor, addr_std: torch.Tensor,
                  table: torch.Tensor):
        """Add a group to be replaced.

        Args:
            group_id: target output group within o_proj
            addr_idx: [2] address channel indices
            addr_mean: [2] calibration mean
            addr_std: [2] calibration std
            table: [num_bins, num_bins, group_size] LUT residual/full table
        """
        self.group_configs[group_id] = (
            addr_idx.cpu(),
            addr_mean.cpu(),
            addr_std.cpu(),
            table.cpu(),
        )

    def _compute_bin_indices(self, addr_source: torch.Tensor,
                             addr_idx: torch.Tensor,
                             addr_mean: torch.Tensor,
                             addr_std: torch.Tensor) -> torch.Tensor:
        """Compute per-token 2D bin indices [B, S, 2]."""
        addr_flat = addr_idx.to(addr_source.device).view(-1)
        addr_acts = addr_source.index_select(-1, addr_flat)

        mean = addr_mean.to(addr_source.device, addr_source.dtype).view(1, 1, -1)
        std = addr_std.to(addr_source.device, addr_source.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr_acts - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        return torch.round(qf).long().clamp(0, self.num_bins - 1)

    def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patched o_proj forward: partial matmul + LUT fill."""
        B, S, hidden_size = x.shape
        device = x.device
        dtype = x.dtype
        compute_dtype = torch.float32

        with torch.autocast(device_type=device.type, enabled=False):
            x_f32 = x.to(compute_dtype)

            # --- 1. Active linear projection for non-replaced groups ---
            active_weight = self.o_proj.weight[self._active_indices, :].to(compute_dtype)
            active_bias = None
            if self.o_proj.bias is not None:
                active_bias = self.o_proj.bias[self._active_indices].to(compute_dtype)
            active_out = F.linear(x_f32, active_weight, active_bias)

            if not torch.isfinite(active_out).all():
                raise RuntimeError(
                    f"[o_proj L{self.layer_id}] active_out has NaN/Inf"
                )

            # --- 2. LUT lookup for replaced groups ---
            replaced_groups = self._replaced_groups
            lut_outputs = []
            for gid in replaced_groups:
                addr_idx, addr_mean, addr_std, table = self.group_configs[gid]
                bin_idx = self._compute_bin_indices(x_f32, addr_idx, addr_mean, addr_std)
                b1 = bin_idx[:, :, 0].view(-1)
                b2 = bin_idx[:, :, 1].view(-1)
                tbl = table.to(device=device, dtype=compute_dtype)
                lut = tbl[b1, b2].view(B, S, self.group_size)

                if self.use_residual:
                    g_start = gid * self.group_size
                    lut = lut + x_f32[:, :, g_start:g_start + self.group_size]

                lut_outputs.append(lut)

            if lut_outputs:
                lut_outputs_cat = torch.cat(lut_outputs, dim=-1)
            else:
                lut_outputs_cat = torch.empty(B, S, 0, device=device, dtype=compute_dtype)

            if not torch.isfinite(lut_outputs_cat).all():
                raise RuntimeError(
                    f"[o_proj L{self.layer_id}] lut_outputs has NaN/Inf"
                )

            # --- 3. Assemble full output ---
            full_out = torch.zeros(B, S, hidden_size, device=device, dtype=compute_dtype)
            full_out = full_out.index_copy_(
                2, self._active_indices.to(device), active_out
            )
            if lut_outputs_cat.shape[-1] > 0:
                full_out = full_out.index_copy_(
                    2, self._replaced_indices.to(device), lut_outputs_cat
                )

        return full_out.to(dtype)

    def install(self):
        """Install partial o_proj replacement."""
        if self._original_forward is not None:
            return

        layer = self.model.model.layers[self.layer_id]
        self.o_proj = layer.self_attn.o_proj
        hidden_size = self.o_proj.weight.shape[0]
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
            active_channels, device=self.o_proj.weight.device, dtype=torch.long
        )
        self._replaced_indices = torch.tensor(
            replaced_channels, device=self.o_proj.weight.device, dtype=torch.long
        )

        self._original_forward = self.o_proj.forward
        engine = self

        def patched_forward(x):
            return engine._patched_forward(x)

        self.o_proj.forward = patched_forward
        print(f"[o_proj] Partial skip installed: L{self.layer_id}, groups={replaced_groups}")
        print(f"  Active channels: {len(active_channels)}, Replaced: {len(replaced_channels)}")

    def uninstall(self):
        """Remove partial o_proj replacement."""
        if self._original_forward is not None:
            self.o_proj.forward = self._original_forward
            self._original_forward = None
            print(f"[o_proj] Partial skip removed: L{self.layer_id}")

        self.o_proj = None
        self._active_indices = None
        self._replaced_indices = None
        self._active_groups = None
        self._replaced_groups = None

    def save(self, path: str):
        """Save engine state."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "layer_id": self.layer_id,
            "group_size": self.group_size,
            "num_bins": self.num_bins,
            "addr_clip": self.addr_clip,
            "use_residual": self.use_residual,
            "group_configs": {
                gid: {
                    "addr_idx": cfg[0],
                    "addr_mean": cfg[1],
                    "addr_std": cfg[2],
                    "table": cfg[3],
                }
                for gid, cfg in self.group_configs.items()
            },
        }, path)
        print(f"[o_proj] Saved to {path}")

    @classmethod
    def load(cls, model, path: str):
        """Load engine state and attach to model."""
        ckpt = torch.load(path, map_location="cpu")
        engine = cls(
            model=model,
            layer_id=ckpt["layer_id"],
            group_size=ckpt["group_size"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt["addr_clip"],
            use_residual=ckpt.get("use_residual", True),
        )
        for gid, cfg in ckpt["group_configs"].items():
            engine.add_group(
                group_id=gid,
                addr_idx=cfg["addr_idx"],
                addr_mean=cfg["addr_mean"],
                addr_std=cfg["addr_std"],
                table=cfg["table"],
            )
        print(f"[o_proj] Loaded from {path}")
        return engine
