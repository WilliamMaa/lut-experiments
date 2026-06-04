"""Bucket table builder for LLM-LUT v0.

Builds a non-trained lookup table:
    address scalar -> quantized bin -> average target group output
"""

import torch
from tqdm import tqdm

from config import get_hook_target
from hooks import CaptureInputHook, CaptureHook


def build_bucket_table(
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
    Args:
        addr_idx:  Tensor[heads]
        addr_mean: Tensor[heads]
        addr_std:  Tensor[heads]
    
    Returns:
        table: Tensor[num_bins, group_size]  (bucket averages)
        coverage: float
        per_bin_count: Tensor[num_bins]
        per_bin_var: Tensor[num_bins]
    """
    model.eval()
    device = next(model.parameters()).device
    target_mod = get_hook_target(model, layer_id, candidate_type)
    
    # Collect all (address, target) pairs
    all_addresses = []
    all_targets = []
    
    cap_in = CaptureInputHook()
    cap_out = CaptureHook()
    handle_pre = target_mod.register_forward_pre_hook(cap_in)
    handle_post = target_mod.register_forward_hook(cap_out)
    
    for bi, batch in enumerate(tqdm(calib_loader, desc=f"Bucket L{layer_id}.{candidate_type}.g{group_id}")):
        if bi >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        x = cap_in.input  # [B, seq, dim]
        y = cap_out.output  # [B, seq, dim]
        if x is None or y is None:
            continue
        
        B, seq, D = y.shape
        num_groups = D // group_size
        
        # Extract target group
        y_g = y.view(B, seq, num_groups, group_size)
        target = y_g[:, :, group_id, :]  # [B, seq, group_size]
        
        if candidate_type == "mlp_delta":
            x_g = x.view(B, seq, num_groups, group_size)
            target = target - x_g[:, :, group_id, :]
        
        # Extract address channels from x
        addr_flat = addr_idx.to(x.device).view(-1)  # [heads]
        addr_acts = x.index_select(-1, addr_flat)   # [B, seq, heads]
        
        # Quantize addresses: average heads into single scalar per token
        mean = addr_mean.to(x.device, x.dtype).view(1, 1, -1)
        std = addr_std.to(x.device, x.dtype).view(1, 1, -1).clamp_min(1e-6)
        z = (addr_acts - mean) / std
        z = z.clamp(-addr_clip, addr_clip)
        qf = (z + addr_clip) / (2.0 * addr_clip) * (num_bins - 1)
        addr_bin = torch.round(qf.mean(dim=-1)).long().clamp(0, num_bins - 1)  # [B, seq]
        
        # Mask out padding tokens if attention_mask provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            addr_bin = addr_bin[mask]
            target = target[mask]
        else:
            addr_bin = addr_bin.view(-1)
            target = target.view(-1, group_size)
        
        all_addresses.append(addr_bin.cpu())
        all_targets.append(target.cpu())
    
    handle_pre.remove()
    handle_post.remove()
    
    if len(all_addresses) == 0:
        # Fallback: empty table
        table = torch.zeros(num_bins, group_size)
        return table, 0.0, torch.zeros(num_bins), torch.zeros(num_bins)
    
    all_addresses = torch.cat(all_addresses, dim=0)  # [N]
    all_targets = torch.cat(all_targets, dim=0)      # [N, group_size]
    
    # Build table
    table = torch.zeros(num_bins, group_size)
    per_bin_count = torch.zeros(num_bins)
    per_bin_var = torch.zeros(num_bins)
    
    for b in range(num_bins):
        mask = (all_addresses == b)
        cnt = mask.sum().item()
        per_bin_count[b] = cnt
        if cnt > 0:
            vals = all_targets[mask]  # [cnt, group_size]
            table[b] = vals.mean(dim=0)
            if cnt > 1:
                per_bin_var[b] = vals.var(dim=0).mean()
    
    coverage = (per_bin_count > 0).sum().item() / num_bins
    print(f"  [BUCKET] L{layer_id}.{candidate_type}.g{group_id}: coverage={coverage:.2%}, total_tokens={len(all_addresses)}")
    return table, coverage, per_bin_count, per_bin_var
