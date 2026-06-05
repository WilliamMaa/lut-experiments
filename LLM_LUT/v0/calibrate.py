"""Address calibration for LLM-LUT v0.

Mirrors v10's calibrate_v10_addr but adapted to LLM token-level activations.
Key difference: down_proj has different input/output dims (4864 -> 896).
"""

import torch
from tqdm import tqdm

from config import get_hook_target
from hooks import CaptureInputHook, CaptureHook


def calibrate_llm_address(
    model,
    tokenizer,
    calib_loader,
    layer_ids,
    candidate_types,
    hidden_group_size: int = 64,
    intermediate_group_size: int = 128,
    heads: int = 2,
    max_batches: int = 999999,
):
    """
    Calibrate per-group/head address channels and quant stats.
    
    For each (layer, candidate_type):
      - Target groups are on the OUTPUT dimension.
      - Address channels are selected from the INPUT dimension.
      - For down_proj: input=intermediate_size, output=hidden_size.
      - For mlp_delta/attn_out: input=output=hidden_size.
    
    Returns:
        dict keyed by (layer_id, candidate_type) -> {
            "addr_idx":  Tensor[num_groups, heads],
            "addr_mean": Tensor[num_groups, heads],
            "addr_std":  Tensor[num_groups, heads],
            "group_means": Tensor[num_groups, group_size],
            "group_stds":  Tensor[num_groups, group_size],
            "num_groups": int,
            "group_size": int,
        }
    """
    model.eval()
    device = next(model.parameters()).device
    results = {}

    for layer_id in layer_ids:
        for cand_type in candidate_types:
            target_mod = get_hook_target(model, layer_id, cand_type)
            
            # Output dim for grouping
            if cand_type in ("down_proj", "mlp_delta", "attn_out"):
                dim_output = model.config.hidden_size
                group_size = hidden_group_size
            elif cand_type == "intermediate":
                dim_output = model.config.intermediate_size
                group_size = intermediate_group_size
            else:
                raise ValueError(cand_type)
            
            num_groups = dim_output // group_size
            
            # Input dim for address selection
            if cand_type == "down_proj":
                dim_input = model.config.intermediate_size
            else:
                dim_input = dim_output
            
            # Running statistics for INPUT (address selection)
            sum_in = torch.zeros(dim_input, device=device)
            sum_in2 = torch.zeros(dim_input, device=device)
            n_in = torch.zeros(dim_input, device=device)
            
            # Correlation proxy on INPUT per "input group" (for structured selection)
            # For down_proj we use input-group_size = 128; otherwise same as output group_size
            in_group_size = intermediate_group_size if cand_type == "down_proj" else group_size
            in_num_groups = dim_input // in_group_size
            corr_sum = torch.zeros(in_num_groups, in_group_size, device=device)
            corr_n = torch.zeros(in_num_groups, in_group_size, device=device)
            
            # Running statistics for OUTPUT (target group means/stds)
            sum_out = torch.zeros(dim_output, device=device)
            sum_out2 = torch.zeros(dim_output, device=device)
            n_out = torch.zeros(dim_output, device=device)
            
            cap_in = CaptureInputHook()
            cap_out = CaptureHook()
            handle_pre = target_mod.register_forward_pre_hook(cap_in)
            handle_post = target_mod.register_forward_hook(cap_out)
            
            try:
                for bi, batch in enumerate(tqdm(calib_loader, desc=f"Calibrate L{layer_id}.{cand_type}", leave=False)):
                    if bi >= max_batches:
                        break
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    with torch.no_grad():
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    x = cap_in.input  # [B, seq, dim_input]
                    y = cap_out.output  # [B, seq, dim_output]
                    if x is None or y is None:
                        continue
                    
                    B, seq = x.shape[0], x.shape[1]
                    
                    # Input stats for address selection
                    x_flat = x.float().permute(2, 0, 1).reshape(dim_input, -1)
                    sum_in += x_flat.sum(dim=1)
                    sum_in2 += (x_flat ** 2).sum(dim=1)
                    n_in += x_flat.shape[1]
                    
                    # Output stats for group replacement
                    y_flat = y.float().permute(2, 0, 1).reshape(dim_output, -1)
                    sum_out += y_flat.sum(dim=1)
                    sum_out2 += (y_flat ** 2).sum(dim=1)
                    n_out += y_flat.shape[1]
                    
                    # Correlation proxy on input (residual magnitude vs input activation)
                    if cand_type == "mlp_delta":
                        residual = y - x  # [B, seq, dim]  (same dim)
                    elif cand_type == "down_proj":
                        # For down_proj, residual proxy: output vs input (different dims)
                        # Use output y as proxy for "interestingness"
                        residual = y
                    else:
                        residual = y - x
                    
                    xg = x.float().view(B, seq, in_num_groups, in_group_size)
                    if residual.shape[-1] == dim_input:
                        rg = residual.float().view(B, seq, in_num_groups, in_group_size)
                        rmag = rg.abs().mean(dim=-1, keepdim=True)
                    else:
                        # Different dims: use per-token mean abs of output as proxy
                        rmag = residual.float().abs().mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, seq, 1, 1]
                    
                    proxy = (xg.abs() * rmag).mean(dim=(0, 1))  # [in_num_groups, in_group_size]
                    corr_sum += proxy
                    corr_n += 1
            finally:
                handle_pre.remove()
                handle_post.remove()
            
            # Input stats -> address selection
            mean_in = sum_in / n_in.clamp_min(1)
            var_in = sum_in2 / n_in.clamp_min(1) - mean_in ** 2
            std_in = var_in.clamp_min(1e-6).sqrt()
            corr = corr_sum / corr_n.clamp_min(1)
            
            addr_idx = torch.zeros(num_groups, heads, dtype=torch.long)
            addr_mean = torch.zeros(num_groups, heads, dtype=torch.float32)
            addr_std = torch.ones(num_groups, heads, dtype=torch.float32)
            
            for g in range(num_groups):
                # For address selection, iterate over all input groups/channels
                # Pick top channels globally from input dim, cycling through groups
                global_var_rank = torch.argsort(var_in, descending=True)
                global_corr_rank = torch.argsort(corr.view(-1), descending=True)
                
                chosen = []
                for h in range(heads):
                    rank = global_var_rank if h % 2 == 0 else global_corr_rank
                    pick = None
                    for cand in rank.tolist():
                        if cand not in chosen:
                            pick = cand
                            break
                    if pick is None:
                        pick = rank[0].item()
                    chosen.append(pick)
                    addr_idx[g, h] = pick
                    addr_mean[g, h] = mean_in[pick].cpu()
                    addr_std[g, h] = std_in[pick].cpu().clamp_min(1e-6)
            
            # Output stats -> group means/stds
            mean_out = sum_out / n_out.clamp_min(1)
            var_out = sum_out2 / n_out.clamp_min(1) - mean_out ** 2
            std_out = var_out.clamp_min(1e-6).sqrt()
            
            group_means = mean_out.view(num_groups, group_size).cpu()
            group_stds = std_out.view(num_groups, group_size).cpu()
            
            results[(layer_id, cand_type)] = {
                "addr_idx": addr_idx.cpu(),
                "addr_mean": addr_mean.cpu(),
                "addr_std": addr_std.cpu(),
                "group_means": group_means,
                "group_stds": group_stds,
                "num_groups": num_groups,
                "group_size": group_size,
            }
            print(f"  [CALIB] L{layer_id}.{cand_type}: out_groups={num_groups}, gs={group_size}, "
                  f"in_dim={dim_input}, out_dim={dim_output}, heads={heads}")
    
    return results
