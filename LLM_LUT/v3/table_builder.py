"""Standalone table building utilities for v3.

This module duplicates the necessary functionality from v0/hooks.py and
v1/train.py so that v3 does not depend on v1 source files.
v0/config.get_hook_target is still required because it encodes the model
structure mapping (layer -> module path).
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

import torch
from tqdm import tqdm
from config import get_hook_target


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


def collect_teacher_targets(
    model,
    calib_loader,
    layer_id: int,
    candidate_type: str,
    group_id: int,
    group_size: int,
    addr_idx: torch.Tensor,
    addr_mean: torch.Tensor,
    addr_std: torch.Tensor,
    num_bins: int = 64,
    addr_clip: float = 3.0,
    max_batches: int = 999999,
):
    """
    Run calibration set through frozen model, collect (bin_idx, target_delta).

    Returns:
        bin_indices: [N, heads] long
        targets:     [N, group_size]
        bucket_init: Tensor[num_bins^heads, group_size]  (bucket average for init)
    """
    model.eval()
    device = next(model.parameters()).device
    target_mod = get_hook_target(model, layer_id, candidate_type)

    cap_in = CaptureInputHook()
    cap_out = CaptureHook()
    handle_pre = target_mod.register_forward_pre_hook(cap_in)
    handle_post = target_mod.register_forward_hook(cap_out)

    all_bin_idx = []
    all_targets = []

    try:
        for bi, batch in enumerate(tqdm(calib_loader, desc="Collect targets", leave=False)):
            if bi >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            x = cap_in.input   # [B, seq, dim]
            y = cap_out.output  # [B, seq, dim]
            if x is None or y is None:
                continue

            B, seq, D = y.shape
            num_groups = D // group_size

            # Target: delta for group_id
            if candidate_type == "mlp_delta":
                delta = y - x
            else:
                delta = y  # for down_proj/attn_out, target is output itself

            delta_g = delta.view(B, seq, num_groups, group_size)[:, :, group_id, :]  # [B, seq, gs]

            # Address activations
            addr_flat = addr_idx.to(x.device).view(-1)  # [heads]
            addr_acts = x.index_select(-1, addr_flat)   # [B, seq, heads]

            mean = addr_mean.to(x.device, x.dtype).view(1, 1, -1)
            std = addr_std.to(x.device, x.dtype).view(1, 1, -1).clamp_min(1e-6)
            z = (addr_acts - mean) / std
            z = z.clamp(-addr_clip, addr_clip)
            qf = (z + addr_clip) / (2.0 * addr_clip) * (num_bins - 1)
            bin_idx = torch.round(qf).long().clamp(0, num_bins - 1)  # [B, seq, heads]

            # Mask padding
            if attention_mask is not None:
                mask = attention_mask.bool()  # [B, seq]
                bin_idx = bin_idx[mask]       # [N, heads]
                delta_g = delta_g[mask]       # [N, group_size]
            else:
                bin_idx = bin_idx.view(-1, bin_idx.shape[-1])
                delta_g = delta_g.view(-1, group_size)

            all_bin_idx.append(bin_idx.cpu())
            all_targets.append(delta_g.cpu())
    finally:
        handle_pre.remove()
        handle_post.remove()

    if len(all_bin_idx) == 0:
        raise RuntimeError("No teacher targets collected!")

    bin_indices = torch.cat(all_bin_idx, dim=0)  # [N, heads]
    targets = torch.cat(all_targets, dim=0)      # [N, group_size]

    # Build bucket average init from collected data
    heads = bin_indices.shape[1]
    if heads == 1:
        bucket_init = torch.zeros(num_bins, group_size)
        counts = torch.zeros(num_bins)
        for b in range(num_bins):
            mask = (bin_indices[:, 0] == b)
            cnt = mask.sum().item()
            counts[b] = cnt
            if cnt > 0:
                bucket_init[b] = targets[mask].mean(dim=0)
    else:
        bucket_init = torch.zeros(num_bins, num_bins, group_size)
        counts = torch.zeros(num_bins, num_bins)
        for b0 in range(num_bins):
            for b1 in range(num_bins):
                mask = (bin_indices[:, 0] == b0) & (bin_indices[:, 1] == b1)
                cnt = mask.sum().item()
                counts[b0, b1] = cnt
                if cnt > 0:
                    bucket_init[b0, b1] = targets[mask].mean(dim=0)

    coverage = (counts > 0).sum().item() / counts.numel()
    print(f"  [TARGETS] N={len(bin_indices)}, heads={heads}, coverage={coverage:.2%}")
    return bin_indices, targets, bucket_init


def build_joint_bucket_table(bin_indices: torch.Tensor, targets: torch.Tensor, num_bins: int, group_size: int):
    """
    Build full 2D bucket table from collected (bin_idx, target) pairs.

    Args:
        bin_indices: [N, 2]
        targets: [N, group_size]
    Returns:
        joint_table: [num_bins, num_bins, group_size]
    """
    joint = torch.zeros(num_bins, num_bins, group_size)
    counts = torch.zeros(num_bins, num_bins)
    for b0 in range(num_bins):
        for b1 in range(num_bins):
            mask = (bin_indices[:, 0] == b0) & (bin_indices[:, 1] == b1)
            cnt = mask.sum().item()
            counts[b0, b1] = cnt
            if cnt > 0:
                joint[b0, b1] = targets[mask].mean(dim=0)
    coverage = (counts > 0).sum().item() / counts.numel()
    print(f"  [JOINT BUCKET] coverage={coverage:.2%}")
    return joint
