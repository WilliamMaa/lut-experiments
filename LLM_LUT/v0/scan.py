"""Main sensitivity scan for LLM-LUT v0."""

import os
import json
import torch
from tqdm import tqdm

from config import V0Config, get_hook_target
from hooks import PerturbationHook
from bucket import build_bucket_table
from metrics import compute_model_metrics, compute_baseline_probs


def run_sensitivity_scan(
    model,
    tokenizer,
    calib_loader,
    eval_loader,
    addr_stats,
    config: V0Config,
    reference_probs=None,
    save_path: str = "results/scan_results.json",
):
    """
    Run zero / mean / noise / bucket scan for all candidates.
    
    Args:
        reference_probs: list of CPU tensors from compute_baseline_probs()
    
    Returns:
        list of result dicts.
    """
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # If no reference probs provided, compute baseline once
    if reference_probs is None:
        print("[SCAN] Computing baseline probabilities...")
        reference_probs = compute_baseline_probs(model, eval_loader)
        baseline_metrics = compute_model_metrics(model, eval_loader)
        print(f"[SCAN] Baseline: next_token_acc={baseline_metrics['next_token_acc']:.4f}, ppl={baseline_metrics['ppl']:.2f}")
    else:
        baseline_metrics = compute_model_metrics(model, eval_loader)
    
    all_results = []
    
    for layer_id in config.layer_ids:
        for cand_type in config.candidate_types:
            key = (layer_id, cand_type)
            stats = addr_stats[key]
            num_groups = stats["num_groups"]
            group_size = stats["group_size"]
            
            for group_id in range(num_groups):
                print(f"\n[SCAN] L{layer_id} {cand_type} group {group_id}/{num_groups}")
                group_result = {
                    "layer": layer_id,
                    "candidate_type": cand_type,
                    "group": group_id,
                    "group_size": group_size,
                }
                target_mod = get_hook_target(model, layer_id, cand_type)
                
                # --- ZERO ---
                hook_obj = PerturbationHook(
                    candidate_type=cand_type,
                    group_size=group_size,
                    group_id=group_id,
                    mode="zero",
                )
                handle = target_mod.register_forward_hook(hook_obj)
                try:
                    metrics_zero = compute_model_metrics(model, eval_loader, reference_probs)
                finally:
                    handle.remove()
                group_result.update({
                    "kl_zero": metrics_zero.get("avg_kl"),
                    "ppl_zero": metrics_zero["ppl"],
                    "acc_zero": metrics_zero["next_token_acc"],
                })
                print(f"  ZERO  -> KL={metrics_zero.get('avg_kl', 0):.6f}, PPL={metrics_zero['ppl']:.2f}, ACC={metrics_zero['next_token_acc']:.4f}")
                
                # --- MEAN ---
                mean_vec = stats["group_means"][group_id]
                hook_obj = PerturbationHook(
                    candidate_type=cand_type,
                    group_size=group_size,
                    group_id=group_id,
                    mode="mean",
                    mean_vec=mean_vec,
                )
                handle = target_mod.register_forward_hook(hook_obj)
                try:
                    metrics_mean = compute_model_metrics(model, eval_loader, reference_probs)
                finally:
                    handle.remove()
                group_result.update({
                    "kl_mean": metrics_mean.get("avg_kl"),
                    "ppl_mean": metrics_mean["ppl"],
                    "acc_mean": metrics_mean["next_token_acc"],
                })
                print(f"  MEAN  -> KL={metrics_mean.get('avg_kl', 0):.6f}, PPL={metrics_mean['ppl']:.2f}, ACC={metrics_mean['next_token_acc']:.4f}")
                
                # --- BUCKET ---
                addr_idx = stats["addr_idx"][group_id]
                addr_mean = stats["addr_mean"][group_id]
                addr_std = stats["addr_std"][group_id]
                
                bucket_table, coverage, per_bin_count, per_bin_var = build_bucket_table(
                    model=model,
                    calib_loader=calib_loader,
                    layer_id=layer_id,
                    candidate_type=cand_type,
                    group_id=group_id,
                    group_size=group_size,
                    addr_idx=addr_idx,
                    addr_mean=addr_mean,
                    addr_std=addr_std,
                    num_bins=config.num_bins,
                    addr_clip=config.addr_clip,
                )
                
                hook_obj = PerturbationHook(
                    candidate_type=cand_type,
                    group_size=group_size,
                    group_id=group_id,
                    mode="bucket",
                    bucket_table=bucket_table,
                    addr_idx=addr_idx,
                    addr_mean=addr_mean,
                    addr_std=addr_std,
                    num_bins=config.num_bins,
                    addr_clip=config.addr_clip,
                )
                handle = target_mod.register_forward_hook(hook_obj)
                try:
                    metrics_bucket = compute_model_metrics(model, eval_loader, reference_probs)
                finally:
                    handle.remove()
                group_result.update({
                    "kl_bucket": metrics_bucket.get("avg_kl"),
                    "ppl_bucket": metrics_bucket["ppl"],
                    "acc_bucket": metrics_bucket["next_token_acc"],
                    "bucket_coverage": coverage,
                    "bucket_bins_used": int((per_bin_count > 0).sum().item()),
                    "bucket_avg_var": per_bin_var[per_bin_count > 0].mean().item() if (per_bin_count > 0).any() else 0.0,
                })
                print(f"  BUCKET-> KL={metrics_bucket.get('avg_kl', 0):.6f}, PPL={metrics_bucket['ppl']:.2f}, ACC={metrics_bucket['next_token_acc']:.4f}, coverage={coverage:.2%}")
                
                all_results.append(group_result)
                
                # Save incrementally
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "baseline": baseline_metrics,
                        "results": all_results,
                        "config": {
                            "model_name": config.model_name,
                            "layer_ids": list(config.layer_ids),
                            "candidate_types": list(config.candidate_types),
                            "group_size": config.hidden_group_size,
                            "num_bins": config.num_bins,
                        }
                    }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SCAN] Complete. Results saved to {save_path}")
    return all_results, baseline_metrics
