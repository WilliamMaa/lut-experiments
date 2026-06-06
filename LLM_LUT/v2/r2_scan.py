"""Fast scan for LLM-LUT Scaling-R1.

Two-phase scan:
  1. Zero ablation all groups (fast filter)
  2. Mean + bucket on top-K per layer
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

import torch
from tqdm import tqdm

from hooks import PerturbationHook
from metrics import compute_model_metrics, compute_baseline_probs
from bucket import build_bucket_table
from config import get_hook_target


def eval_single_group(
    model, eval_loader, reference_probs,
    layer_id, cand_type, group_id, group_size,
    mode, mean_vec=None, bucket_table=None,
    addr_idx=None, addr_mean=None, addr_std=None,
    num_bins=64, addr_clip=3.0,
):
    """Evaluate a single group with given perturbation mode."""
    hook = PerturbationHook(
        candidate_type=cand_type,
        group_size=group_size,
        group_id=group_id,
        mode=mode,
        mean_vec=mean_vec,
        bucket_table=bucket_table,
        addr_idx=addr_idx,
        addr_mean=addr_mean,
        addr_std=addr_std,
        num_bins=num_bins,
        addr_clip=addr_clip,
    )
    target_mod = get_hook_target(model, layer_id, cand_type)
    handle = target_mod.register_forward_hook(hook)
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        handle.remove()
    return metrics


def fast_scan_groups(
    model,
    calib_loader,
    eval_loader,
    layer_ids,
    calib_results,
    group_size=64,
    num_bins=64,
    top_k=5,
):
    """
    Fast two-phase scan.

    Args:
        calib_results: dict from calibrate_llm_address, keyed by (layer_id, "mlp_delta")
    Returns:
        all_zero_results: list of dicts for all groups
        top_results: list of dicts for top-K groups per layer (with mean+bucket)
    """
    device = next(model.parameters()).device
    reference_probs = compute_baseline_probs(model, eval_loader)

    all_zero_results = []
    top_results = []

    for layer_id in layer_ids:
        calib = calib_results[(layer_id, "mlp_delta")]
        addr_idx_all = calib["addr_idx"]     # [num_groups, heads]
        addr_mean_all = calib["addr_mean"]   # [num_groups, heads]
        addr_std_all = calib["addr_std"]     # [num_groups, heads]
        group_means = calib["group_means"]   # [num_groups, group_size]
        num_groups = calib["num_groups"]

        print(f"\n[SCAN] Layer {layer_id}: {num_groups} groups")

        # Phase 1: Zero ablation for all groups
        layer_zero = []
        for group_id in tqdm(range(num_groups), desc=f"L{layer_id} zero", leave=False):
            metrics = eval_single_group(
                model, eval_loader, reference_probs,
                layer_id, "mlp_delta", group_id, group_size,
                mode="zero",
            )
            layer_zero.append({
                "layer": layer_id,
                "group": group_id,
                "kl_zero": metrics.get("avg_kl", 0.0),
                "ppl_zero": metrics["ppl"],
                "acc_zero": metrics["next_token_acc"],
            })

        # Rank by zero KL
        layer_zero.sort(key=lambda x: x["kl_zero"], reverse=True)
        top_groups = layer_zero[:top_k]

        # Phase 2: Mean + Bucket for top-K
        for item in tqdm(top_groups, desc=f"L{layer_id} mean+bucket", leave=False):
            gid = item["group"]
            mean_vec = group_means[gid]
            aidx = addr_idx_all[gid]
            amean = addr_mean_all[gid]
            astd = addr_std_all[gid]

            # Mean
            m_mean = eval_single_group(
                model, eval_loader, reference_probs,
                layer_id, "mlp_delta", gid, group_size,
                mode="mean", mean_vec=mean_vec,
            )
            item["kl_mean"] = m_mean.get("avg_kl", 0.0)
            item["ppl_mean"] = m_mean["ppl"]
            item["acc_mean"] = m_mean["next_token_acc"]

            # Bucket
            table, coverage, _, _, _ = build_bucket_table(
                model, calib_loader, layer_id, "mlp_delta", gid, group_size,
                aidx, amean, astd,
                num_bins=num_bins, binning_mode="uniform",
            )
            m_bucket = eval_single_group(
                model, eval_loader, reference_probs,
                layer_id, "mlp_delta", gid, group_size,
                mode="bucket", bucket_table=table,
                addr_idx=aidx, addr_mean=amean, addr_std=astd,
                num_bins=num_bins,
            )
            item["kl_bucket"] = m_bucket.get("avg_kl", 0.0)
            item["ppl_bucket"] = m_bucket["ppl"]
            item["acc_bucket"] = m_bucket["next_token_acc"]
            item["coverage"] = coverage
            item["recovery"] = (item["kl_zero"] - item["kl_bucket"]) / item["kl_zero"] if item["kl_zero"] > 0 else 0.0

        all_zero_results.extend(layer_zero)
        top_results.extend(top_groups)

    # Overall ranking by recovery
    top_results.sort(key=lambda x: x["recovery"], reverse=True)
    return all_zero_results, top_results
