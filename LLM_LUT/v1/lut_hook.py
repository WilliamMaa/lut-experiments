"""Trainable LUT Hook for LLM-LUT v1.

Replaces a specific group with LUT output during forward pass.
Model weights are frozen; only the LUT table is trainable.
"""

import torch


class TrainableLUTHook:
    """
    Hook that replaces target group with trainable LUT lookup.
    """

    def __init__(
        self,
        lut_table,  # TrainableLUTTable
        candidate_type: str,
        group_size: int,
        group_id: int,
        addr_idx: torch.Tensor,   # [heads]
        addr_mean: torch.Tensor,  # [heads]
        addr_std: torch.Tensor,   # [heads]
        num_bins: int = 64,
        addr_clip: float = 3.0,
    ):
        self.lut_table = lut_table
        self.candidate_type = candidate_type
        self.group_size = group_size
        self.group_id = group_id
        self.addr_idx = addr_idx
        self.addr_mean = addr_mean
        self.addr_std = addr_std
        self.num_bins = num_bins
        self.addr_clip = addr_clip

    def _compute_bin_indices(self, addr_source: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token bin indices from address activations.

        Args:
            addr_source: [B, seq_len, hidden_dim]
        Returns:
            bin_idx: [B, seq_len, heads] long
        """
        addr_flat = self.addr_idx.to(addr_source.device).view(-1)  # [heads]
        addr_acts = addr_source.index_select(-1, addr_flat)       # [B, seq, heads]

        mean = self.addr_mean.to(addr_source.device, addr_source.dtype).view(1, 1, -1)
        std = self.addr_std.to(addr_source.device, addr_source.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr_acts - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        bin_idx = torch.round(qf).long().clamp(0, self.num_bins - 1)
        return bin_idx

    def __call__(self, module, input, output):
        """
        Forward hook compatible with nn.Module.register_forward_hook.
        """
        addr_source = input[0] if isinstance(input, tuple) else input
        out_tensor = output[0] if isinstance(output, tuple) else output

        bin_idx = self._compute_bin_indices(addr_source)  # [B, seq, heads]
        B, seq, _ = bin_idx.shape

        # Flatten for LUT lookup: [B*seq, heads]
        bin_flat = bin_idx.view(-1, bin_idx.shape[-1])

        # Lookup (trainable)
        repl = self.lut_table(bin_flat)  # [B*seq, group_size]
        repl = repl.view(B, seq, self.group_size)

        if self.candidate_type == "mlp_delta":
            x = addr_source  # [B, seq, hidden]
            delta = out_tensor - x
            num_groups = delta.shape[-1] // self.group_size
            delta_g = delta.view(B, seq, num_groups, self.group_size)
            delta_g[:, :, self.group_id, :] = repl
            modified = x + delta_g.view(B, seq, -1)
        else:
            # down_proj, attn_out
            num_groups = out_tensor.shape[-1] // self.group_size
            out_g = out_tensor.view(B, seq, num_groups, self.group_size)
            out_g[:, :, self.group_id, :] = repl
            modified = out_g.view(B, seq, -1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class CaptureInputHook:
    """Capture module input."""
    def __init__(self):
        self.input = None

    def __call__(self, module, input):
        self.input = input[0].detach().clone() if isinstance(input, tuple) and len(input) > 0 else None


class CaptureOutputHook:
    """Capture module output (first element if tuple)."""
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach().clone()
        else:
            self.output = output.detach().clone()
