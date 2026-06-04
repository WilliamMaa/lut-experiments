"""Hook-based perturbation engine for LLM-LUT v0.

No model structure modification; all perturbations are applied via forward hooks
during inference, and gradients are globally disabled.
"""

import torch
import torch.nn as nn


class PerturbationHook:
    """Apply zero / mean / noise / bucket perturbation to a specific group."""

    def __init__(
        self,
        candidate_type: str,
        group_size: int,
        group_id: int,
        mode: str,  # "zero", "mean", "noise", "bucket"
        mean_vec: torch.Tensor = None,       # [group_size]
        std_vec: torch.Tensor = None,        # [group_size]
        bucket_table: torch.Tensor = None,   # [num_bins, group_size]
        addr_idx: torch.Tensor = None,       # [heads]
        addr_mean: torch.Tensor = None,      # [heads]
        addr_std: torch.Tensor = None,       # [heads]
        num_bins: int = 64,
        addr_clip: float = 3.0,
        noise_sigma: float = 0.0,
    ):
        self.candidate_type = candidate_type
        self.group_size = group_size
        self.group_id = group_id
        self.mode = mode
        self.noise_sigma = noise_sigma
        self.num_bins = num_bins
        self.addr_clip = addr_clip

        # For mean/noise mode
        self.mean_vec = mean_vec
        self.std_vec = std_vec

        # For bucket mode
        self.bucket_table = bucket_table
        self.addr_idx = addr_idx
        self.addr_mean = addr_mean
        self.addr_std = addr_std

    def _apply_to_group(self, tensor, num_groups, replacer_fn):
        """Apply replacer_fn to the target group only."""
        B, seq_len, hidden = tensor.shape
        tensor_g = tensor.view(B, seq_len, num_groups, self.group_size)
        out_g = tensor_g.clone()
        out_g[:, :, self.group_id, :] = replacer_fn(tensor_g[:, :, self.group_id, :])
        return out_g.view(B, seq_len, -1)

    def _quantize_address(self, activation):
        """
        activation: [B, seq_len, heads] or [B, seq_len]
        Returns quantized float indices [B, seq_len, heads] in [0, num_bins-1].
        """
        if activation.dim() == 2:
            activation = activation.unsqueeze(-1)
        mean = self.addr_mean.to(activation.device, activation.dtype).view(1, 1, -1)
        std = self.addr_std.to(activation.device, activation.dtype).view(1, 1, -1).clamp_min(1e-6)
        z = (activation - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        return qf

    def _bucket_lookup(self, addr_acts):
        """ addr_acts: [B, seq_len, heads] -> replacement [B, seq_len, group_size] """
        qf = self._quantize_address(addr_acts)  # [B, seq_len, heads]
        qf_mean = qf.mean(dim=-1, keepdim=True)  # [B, seq_len, 1]
        bin_idx = torch.round(qf_mean).long().clamp(0, self.num_bins - 1)  # [B, seq_len, 1]
        table = self.bucket_table.to(addr_acts.device, addr_acts.dtype)  # [num_bins, group_size]
        B, seq_len, _ = bin_idx.shape
        bin_flat = bin_idx.view(-1)  # [B*seq_len]
        repl = table[bin_flat]       # [B*seq_len, group_size]
        return repl.view(B, seq_len, self.group_size)

    def _make_replacer(self, addr_source):
        """Return a function that replaces group activations based on mode.
        
        addr_source: tensor from which to gather address channels.
                     For down_proj this is input (intermediate dim).
                     For others this is usually hidden state.
        """
        if self.mode == "zero":
            return lambda g: torch.zeros_like(g)

        if self.mode == "mean":
            vec = self.mean_vec.to(addr_source.device, addr_source.dtype).view(1, 1, self.group_size)
            return lambda g: vec.expand_as(g)

        if self.mode == "noise":
            vec = self.mean_vec.to(addr_source.device, addr_source.dtype).view(1, 1, self.group_size)
            std = self.std_vec.to(addr_source.device, addr_source.dtype).view(1, 1, self.group_size)
            return lambda g: vec + self.noise_sigma * std * torch.randn_like(g)

        if self.mode == "bucket":
            addr_flat = self.addr_idx.to(addr_source.device).view(-1)  # [heads]
            addr_acts = addr_source.index_select(-1, addr_flat)        # [B, seq_len, heads]
            return lambda g: self._bucket_lookup(addr_acts)

        raise ValueError(f"Unknown mode: {self.mode}")

    def __call__(self, module, input, output):
        """
        Hook function compatible with nn.Module.register_forward_hook.
        
        For down_proj / attn_out: output is [B, seq_len, hidden_dim]
        For mlp_delta: output is [B, seq_len, hidden_dim] but we need delta = mlp(x) - x
        
        Address source:
          - down_proj: input[0] (intermediate activation, dim may differ from output)
          - mlp_delta: input[0] (hidden state)
          - attn_out: input[0] (hidden state)
        """
        addr_source = input[0] if isinstance(input, tuple) else input
        
        # Extract main tensor from tuple if needed (e.g., self_attn returns tuple)
        out_tensor = output[0] if isinstance(output, tuple) else output
        
        if self.candidate_type == "mlp_delta":
            x = addr_source  # [B, seq, hidden]
            delta = out_tensor - x
            num_groups = delta.shape[-1] // self.group_size
            replacer = self._make_replacer(addr_source)
            delta_perturbed = self._apply_to_group(delta, num_groups, replacer)
            modified = x + delta_perturbed
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        else:
            # down_proj, attn_out
            x = out_tensor
            num_groups = x.shape[-1] // self.group_size
            replacer = self._make_replacer(addr_source)
            modified = self._apply_to_group(x, num_groups, replacer)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified


class CaptureHook:
    """Simple hook to capture module output (first element if tuple)."""
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.output = output[0].detach().clone()
        else:
            self.output = output.detach().clone()


class CaptureInputHook:
    """Simple hook to capture module input."""
    def __init__(self):
        self.input = None

    def __call__(self, module, input):
        self.input = input[0].detach().clone() if isinstance(input, tuple) and len(input) > 0 else None
