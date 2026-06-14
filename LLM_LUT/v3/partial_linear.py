"""V3 Partial Linear Engine.

Replaces down_proj computation for selected groups with LUT lookup.
PyTorch-only implementation for Phase 1 numerical validation.

Usage:
    from v3.partial_linear import V3PartialEngine
    engine = V3PartialEngine(model, layer_id=21, group_size=64, num_bins=64)
    engine.add_group(gid, addr_idx, addr_mean, addr_std, table)
    engine.install()
    # ... run inference ...
    engine.uninstall()
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
V2_DIR = os.path.join(os.path.dirname(__file__), "..", "v2")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V2_DIR)

import torch
import torch.nn.functional as F

from triton_kernels import lut_fill, TRITON_AVAILABLE


class V3PartialEngine:
    """
    Partial down_proj replacement engine.

    Replaces selected output channel groups in down_proj with LUT lookup,
    skipping the corresponding matrix multiplication.

    Args:
        model: the LLM (already on target device)
        layer_id: target layer
        group_size: dimension of each group (default 64)
        num_bins: number of bins per head (default 64)
        addr_clip: address clipping value (default 3.0)
    """

    def __init__(self, model, layer_id: int, group_size: int = 64, num_bins: int = 64, addr_clip: float = 3.0):
        self.model = model
        self.layer_id = layer_id
        self.group_size = group_size
        self.num_bins = num_bins
        self.addr_clip = addr_clip

        # group_configs: {group_id: (addr_idx, addr_mean, addr_std, table)}
        self.group_configs = {}

        # Cache for inter-hook communication
        self._cached_normed_x = None
        self._cached_bin_idx = {}          # dict[gid -> [B,S,2]] (legacy, kept for compat)
        self._cached_bin_idx_tensor = None  # [B*S, num_groups, 2] (Triton path)

        # Pre-built batched tensors (populated in install)
        self._batched_tables = None        # [num_groups, 64, 64, 64]
        self._group_starts = None          # [num_groups] int32

        # Hook state
        self._mlp_hook_handle = None
        self._original_down_proj_forward = None
        self.down_proj = None

    def add_group(self, group_id: int, addr_idx: torch.Tensor, addr_mean: torch.Tensor,
                  addr_std: torch.Tensor, table: torch.Tensor):
        """Add a group to be replaced.

        Args:
            group_id: target group within layer output
            addr_idx: [2] — address channel indices
            addr_mean: [2] — per-channel mean from calibration
            addr_std: [2] — per-channel std from calibration
            table: [num_bins, num_bins, group_size] — 2D bucket table
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
        """Compute per-token 2D bin indices [B, seq, 2]."""
        addr_flat = addr_idx.to(addr_source.device).view(-1)
        addr_acts = addr_source.index_select(-1, addr_flat)

        mean = addr_mean.to(addr_source.device, addr_source.dtype).view(1, 1, -1)
        std = addr_std.to(addr_source.device, addr_source.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr_acts - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        bin_idx = torch.round(qf).long().clamp(0, self.num_bins - 1)
        return bin_idx

    def _mlp_pre_hook(self, module, input):
        """MLP forward pre-hook: cache normed_x and compute bin indices."""
        normed_x = input[0] if isinstance(input, tuple) else input
        self._cached_normed_x = normed_x

        replaced_groups = sorted(self.group_configs.keys())
        num_replaced = len(replaced_groups)
        B, S = normed_x.shape[:2]
        device = normed_x.device

        # Compute bin indices for all replaced groups
        # Legacy dict path (always populated)
        for gid, (addr_idx, addr_mean, addr_std, _table) in self.group_configs.items():
            bin_idx = self._compute_bin_indices(normed_x, addr_idx, addr_mean, addr_std)
            self._cached_bin_idx[gid] = bin_idx

        # Batched tensor path for Triton
        if num_replaced > 0 and self._batched_tables is not None:
            bin_idx_tensor = torch.empty(B, S, num_replaced, 2, device=device, dtype=torch.int64)
            for i, gid in enumerate(replaced_groups):
                bin_idx_tensor[:, :, i, :] = self._cached_bin_idx[gid]
            self._cached_bin_idx_tensor = bin_idx_tensor.view(B * S, num_replaced, 2)

    def _patched_down_proj_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Patched down_proj forward: partial matmul + LUT fill.

        Args:
            hidden: [B, S, intermediate_size] — SwiGLU output

        Returns:
            [B, S, hidden_size] — full down_proj output with replaced groups
        """
        B, S, intermediate_size = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Retrieve cached normed_x
        normed_x = self._cached_normed_x
        if normed_x is None:
            raise RuntimeError("normed_x not cached! MLP pre-hook was not called before down_proj.")

        hidden_size = normed_x.shape[-1]
        replaced_groups = sorted(self.group_configs.keys())

        # --- 1. Partial matmul for active channels ---
        active_out = F.linear(hidden, self._active_weight, self._active_bias)  # [B, S, active]

        # --- 2. LUT lookup for replaced channels ---
        if len(replaced_groups) == 0:
            lut_outputs = torch.empty(B, S, 0, device=device, dtype=dtype)
        elif self._batched_tables is not None and self._cached_bin_idx_tensor is not None:
            # Fast path: fused multi-group LUT fill (Triton or PyTorch batched)
            M = B * S
            normed_x_flat = normed_x.view(M, hidden_size)
            try:
                lut_outputs_flat = lut_fill(
                    self._cached_bin_idx_tensor,
                    self._batched_tables,
                    normed_x_flat,
                    self._group_starts,
                )
                lut_outputs = lut_outputs_flat.view(B, S, -1)
            except Exception as e:
                # Fallback to per-group loop on any error
                print(f"[V3] LUT fill error ({e}), falling back to per-group loop")
                lut_outputs = self._lut_fill_loop(B, S, normed_x, device, dtype)
        else:
            # Fallback: per-group Python loop
            lut_outputs = self._lut_fill_loop(B, S, normed_x, device, dtype)

        # --- 3. Assemble full output ---
        full_out = torch.zeros(B, S, hidden_size, device=device, dtype=dtype)
        full_out = full_out.index_copy_(2, self._active_indices, active_out)
        if lut_outputs.shape[-1] > 0:
            full_out = full_out.index_copy_(2, self._replaced_indices, lut_outputs)

        return full_out

    def _lut_fill_loop(self, B, S, normed_x, device, dtype):
        """PyTorch per-group LUT fill (fallback)."""
        replaced_groups = sorted(self.group_configs.keys())
        lut_outputs = []
        for gid in replaced_groups:
            table = self.group_configs[gid][3]
            bin_idx = self._cached_bin_idx[gid]  # [B, S, 2]

            b1 = bin_idx[:, :, 0].view(-1)
            b2 = bin_idx[:, :, 1].view(-1)
            t = table.to(device, dtype)
            lut_delta = t[b1, b2]
            lut_delta = lut_delta.view(B, S, self.group_size)

            g_start = gid * self.group_size
            normed_x_group = normed_x[:, :, g_start:g_start + self.group_size]
            lut_outputs.append(normed_x_group + lut_delta)

        return torch.cat(lut_outputs, dim=-1) if lut_outputs else torch.empty(B, S, 0, device=device, dtype=dtype)

    def install(self):
        """Install partial skip: MLP pre-hook + patched down_proj forward."""
        if self._original_down_proj_forward is not None:
            return

        # Get modules
        mlp = self.model.model.layers[self.layer_id].mlp
        self.down_proj = mlp.down_proj
        hidden_size = self.down_proj.weight.shape[0]
        num_groups = hidden_size // self.group_size
        replaced_groups = sorted(self.group_configs.keys())
        active_groups = [g for g in range(num_groups) if g not in replaced_groups]

        # Pre-compute channel indices
        self._active_channels = []
        self._replaced_channels = []
        for g in active_groups:
            self._active_channels.extend(range(g * self.group_size, (g + 1) * self.group_size))
        for g in replaced_groups:
            self._replaced_channels.extend(range(g * self.group_size, (g + 1) * self.group_size))
        self._active_indices = torch.tensor(self._active_channels, device=self.down_proj.weight.device, dtype=torch.long)
        self._replaced_indices = torch.tensor(self._replaced_channels, device=self.down_proj.weight.device, dtype=torch.long)

        # Pre-extract active weight/bias as contiguous tensors to avoid per-forward slice overhead
        self._active_weight = self.down_proj.weight[self._active_indices, :].clone().contiguous()
        self._active_bias = None
        if self.down_proj.bias is not None:
            self._active_bias = self.down_proj.bias[self._active_indices].clone().contiguous()

        # Pre-build batched tensors for fast LUT fill
        replaced_groups = sorted(self.group_configs.keys())
        num_replaced = len(replaced_groups)
        if num_replaced > 0:
            self._batched_tables = torch.stack([
                self.group_configs[g][3] for g in replaced_groups
            ], dim=0).to(self.down_proj.weight.device, self.down_proj.weight.dtype)
            self._group_starts = torch.tensor(
                [g * self.group_size for g in replaced_groups],
                device=self.down_proj.weight.device, dtype=torch.int32
            )

        # Install MLP pre-hook to cache normed_x and compute bins
        self._mlp_hook_handle = mlp.register_forward_pre_hook(self._mlp_pre_hook)

        # Patch down_proj forward
        self._original_down_proj_forward = self.down_proj.forward
        engine = self

        def patched_forward(hidden):
            return engine._patched_down_proj_forward(hidden)

        self.down_proj.forward = patched_forward
        kernel_name = "Triton" if TRITON_AVAILABLE else "PyTorch"
        print(f"[V3] Partial skip installed: L{self.layer_id}, groups={replaced_groups}")
        print(f"  Active channels: {len(self._active_channels)}, Replaced: {len(self._replaced_channels)}")
        print(f"  LUT fill backend: {kernel_name}")

    def uninstall(self):
        """Remove partial skip: restore original down_proj forward."""
        if self._mlp_hook_handle is not None:
            self._mlp_hook_handle.remove()
            self._mlp_hook_handle = None

        if self._original_down_proj_forward is not None:
            self.down_proj.forward = self._original_down_proj_forward
            self._original_down_proj_forward = None

        self._cached_normed_x = None
        self._cached_bin_idx = {}
        self._cached_bin_idx_tensor = None
        self._active_weight = None
        self._active_bias = None
        self._active_indices = None
        self._replaced_indices = None
        self._batched_tables = None
        self._batched_addr_mean = None
        self._batched_addr_std = None
        self._group_starts = None
        print(f"[V3] Partial skip removed")

    def save(self, path: str):
        """Save engine state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "layer_id": self.layer_id,
            "group_size": self.group_size,
            "num_bins": self.num_bins,
            "addr_clip": self.addr_clip,
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
        print(f"[V3] Saved to {path}")

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
        )
        for gid, cfg in ckpt["group_configs"].items():
            engine.add_group(
                group_id=gid,
                addr_idx=cfg["addr_idx"],
                addr_mean=cfg["addr_mean"],
                addr_std=cfg["addr_std"],
                table=cfg["table"],
            )
        print(f"[V3] Loaded from {path}")
        return engine
