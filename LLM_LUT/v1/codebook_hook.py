"""Codebook Hook for LLM-LUT v1.1.

Replaces a specific group with codebook output during forward pass.
"""

import torch


class CodebookHook:
    """
    Hook that replaces target group with codebook lookup.
    """

    def __init__(
        self,
        codebook_table,
        candidate_type: str,
        group_size: int,
        group_id: int,
        addr_idx: torch.Tensor,   # [heads]
        addr_mean: torch.Tensor,  # [heads]
        addr_std: torch.Tensor,   # [heads]
        addr_clip: float = 3.0,
        hard: bool = False,       # True for inference eval
    ):
        self.codebook_table = codebook_table
        self.candidate_type = candidate_type
        self.group_size = group_size
        self.group_id = group_id
        self.addr_idx = addr_idx
        self.addr_mean = addr_mean
        self.addr_std = addr_std
        self.addr_clip = addr_clip
        self.hard = hard

    def _compute_address_vector(self, addr_source: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized 2D address vector from activation.

        Args:
            addr_source: [B, seq_len, hidden_dim]
        Returns:
            z: [B, seq_len, heads] float in [-clip, clip]
        """
        addr_flat = self.addr_idx.to(addr_source.device).view(-1)  # [heads]
        addr_acts = addr_source.index_select(-1, addr_flat)       # [B, seq, heads]

        mean = self.addr_mean.to(addr_source.device, addr_source.dtype).view(1, 1, -1)
        std = self.addr_std.to(addr_source.device, addr_source.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr_acts - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        return z  # [B, seq, heads]

    def __call__(self, module, input, output):
        """Forward hook."""
        addr_source = input[0] if isinstance(input, tuple) else input
        out_tensor = output[0] if isinstance(output, tuple) else output

        z = self._compute_address_vector(addr_source)  # [B, seq, heads]
        B, seq, heads = z.shape
        z_flat = z.view(-1, heads)  # [B*seq, heads]

        # Codebook lookup
        out = self.codebook_table(z_flat, hard=self.hard)
        if isinstance(out, tuple):
            repl = out[0]  # [B*seq, group_size]
        else:
            repl = out
        repl = repl.view(B, seq, self.group_size)

        if self.candidate_type == "mlp_delta":
            x = addr_source  # [B, seq, hidden]
            delta = out_tensor - x
            num_groups = delta.shape[-1] // self.group_size
            delta_g = delta.view(B, seq, num_groups, self.group_size)
            delta_g[:, :, self.group_id, :] = repl
            modified = x + delta_g.view(B, seq, -1)
        else:
            num_groups = out_tensor.shape[-1] // self.group_size
            out_g = out_tensor.view(B, seq, num_groups, self.group_size)
            out_g[:, :, self.group_id, :] = repl
            modified = out_g.view(B, seq, -1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
