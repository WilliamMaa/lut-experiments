"""LLM-LUT v0.5 Extended Experiment Entry.

Three experiments:
  A: Single-group × binning sweep (7 groups × 2 modes × 4 bin sizes)
  B: Multi-group combination test (8 configs, progressive stacking)
  C: Two-head vs single-head ablation (top 3 groups)

MANDATORY: Run gpu_sanity_check.py FIRST.
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import V0Config, get_hook_target
from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from bucket import build_bucket_table, compute_occupancy_entropy
from hooks import PerturbationHook
from metrics import compute_model_metrics, compute_baseline_probs
from rank import rank_candidates


def load_model_and_data(config: V0Config, device_str: str = "cuda:0"):
    """Load model to a single GPU and prepare data loaders."""
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, config.torch_dtype, torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as e:
        print(f"[WARN] Failed to load with {dtype}: {e}. Falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=config.trust_remote_code,
        )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Verify no other GPUs are touched
    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            print(f"[FATAL] GPU {i} has {torch.cuda.memory_allocated(i)/1024**2:.1f} MB allocated!")
            sys.exit(1)

    # Data
    prepare_data(tokenizer, config.calib_path, config.eval_path,
                 calib_size=config.calib_size, eval_size=config.eval_size,
                 max_seq_len=config.max_seq_len)
    calib_texts = load_jsonl(config.calib_path)
    eval_texts = load_jsonl(config.eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=config.max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=config.max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=config.calib_batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=config.eval_batch_size, shuffle=False)

    return model, tokenizer, calib_loader, eval_loader


def eval_single_group(model, eval_loader, reference_probs, layer_id, cand_type, group_id, group_size,
                       mode, mean_vec=None, bucket_table=None, addr_idx=None, addr_mean=None, addr_std=None,
                       num_bins=None, addr_clip=None):
    """Evaluate a single group with given perturbation mode."""
    hook_obj = PerturbationHook(
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
    handle = target_mod.register_forward_hook(hook_obj)
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs)
    finally:
        handle.remove()
    return metrics


def eval_multi_groups(model, eval_loader, reference_probs, layer_id, cand_type, group_configs):
    """
    Evaluate multiple groups simultaneously.
    group_configs: list of dicts, each with keys: group_id, group_size, mode, bucket_table, addr_idx, addr_mean, addr_std, num_bins, addr_clip
    """
    target_mod = get_hook_target(model, layer_id, cand_type)
    handles = []
    try:
        for cfg in group_configs:
            hook_obj = PerturbationHook(
                candidate_type=cand_type,
                group_size=cfg["group_size"],
                group_id=cfg["group_id"],
                mode=cfg["mode"],
                bucket_table=cfg.get("bucket_table"),
                addr_idx=cfg.get("addr_idx"),
                addr_mean=cfg.get("addr_mean"),
                addr_std=cfg.get("addr_std"),
                num_bins=cfg.get("num_bins"),
                addr_clip=cfg.get("addr_clip"),
            )
            handle = target_mod.register_forward_hook(hook_obj)
            handles.append(handle)
        metrics = compute_model_metrics(model, eval_loader, reference_probs)
    finally:
        for h in handles:
            h.remove()
    return metrics


def experiment_A(model, calib_loader, eval_loader, addr_stats, reference_probs, config):
    """Single-group × binning sweep."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Single-Group × Binning Sweep")
    print("=" * 70)

    layer_id = config.layer_ids[0]
    cand_type = config.candidate_types[0]
    stats = addr_stats[(layer_id, cand_type)]
    group_size = stats["group_size"]
    groups = config.groups_to_test
    binning_modes = config.binning_modes
    num_bins_list = config.num_bins_list

    # Pre-compute zero/mean baselines for each group (only once)
    baselines = {}
    for gid in groups:
        print(f"\n[EXP-A] Baseline for group {gid}")
        metrics_zero = eval_single_group(
            model, eval_loader, reference_probs, layer_id, cand_type, gid, group_size,
            mode="zero"
        )
        metrics_mean = eval_single_group(
            model, eval_loader, reference_probs, layer_id, cand_type, gid, group_size,
            mode="mean", mean_vec=stats["group_means"][gid]
        )
        baselines[gid] = {
            "kl_zero": metrics_zero.get("avg_kl", 0),
            "ppl_zero": metrics_zero["ppl"],
            "kl_mean": metrics_mean.get("avg_kl", 0),
            "ppl_mean": metrics_mean["ppl"],
        }
        print(f"  ZERO: KL={metrics_zero.get('avg_kl', 0):.6f}, PPL={metrics_zero['ppl']:.2f}")
        print(f"  MEAN: KL={metrics_mean.get('avg_kl', 0):.6f}, PPL={metrics_mean['ppl']:.2f}")

    results = []
    for gid in groups:
        for binning in binning_modes:
            for num_bins in num_bins_list:
                print(f"\n[EXP-A] Group {gid}, {binning}, bins={num_bins}")
                addr_idx = stats["addr_idx"][gid]
                addr_mean = stats["addr_mean"][gid]
                addr_std = stats["addr_std"][gid]

                bucket_table, coverage, per_bin_count, per_bin_var, q_bounds = build_bucket_table(
                    model=model,
                    calib_loader=calib_loader,
                    layer_id=layer_id,
                    candidate_type=cand_type,
                    group_id=gid,
                    group_size=group_size,
                    addr_idx=addr_idx,
                    addr_mean=addr_mean,
                    addr_std=addr_std,
                    num_bins=num_bins,
                    addr_clip=config.addr_clip,
                    binning_mode=binning,
                )

                metrics_bucket = eval_single_group(
                    model, eval_loader, reference_probs, layer_id, cand_type, gid, group_size,
                    mode="bucket",
                    bucket_table=bucket_table,
                    addr_idx=addr_idx,
                    addr_mean=addr_mean,
                    addr_std=addr_std,
                    num_bins=num_bins,
                    addr_clip=config.addr_clip,
                )

                kl_bucket = metrics_bucket.get("avg_kl", 0)
                kl_zero = baselines[gid]["kl_zero"]
                kl_mean = baselines[gid]["kl_mean"]
                recovery = (kl_zero - kl_bucket) / max(kl_zero, 1e-8) if kl_zero > 0 else 0.0
                bucket_advantage = kl_mean - kl_bucket
                entropy = compute_occupancy_entropy(per_bin_count)

                print(f"  BUCKET: KL={kl_bucket:.6f}, PPL={metrics_bucket['ppl']:.2f}, "
                      f"Recovery={recovery:.4f}, Adv={bucket_advantage:.6f}, "
                      f"Coverage={coverage:.2%}, Entropy={entropy:.4f}")

                results.append({
                    "layer": layer_id,
                    "candidate_type": cand_type,
                    "group": gid,
                    "binning_mode": binning,
                    "num_bins": num_bins,
                    "kl_zero": kl_zero,
                    "kl_mean": kl_mean,
                    "kl_bucket": kl_bucket,
                    "ppl_bucket": metrics_bucket["ppl"],
                    "recovery": recovery,
                    "bucket_advantage": bucket_advantage,
                    "bucket_coverage": coverage,
                    "bucket_entropy": entropy,
                    "bucket_bins_used": int((per_bin_count > 0).sum().item()),
                })

    return results


def experiment_B(model, calib_loader, eval_loader, addr_stats, reference_probs, config, best_config):
    """Multi-group combination test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Multi-Group Combination Test")
    print("=" * 70)

    layer_id = config.layer_ids[0]
    cand_type = config.candidate_types[0]
    stats = addr_stats[(layer_id, cand_type)]
    group_size = stats["group_size"]

    # Progressive stacking of top groups
    all_groups = config.groups_to_test
    combinations = [
        all_groups[:1],
        all_groups[:2],
        all_groups[:3],
        all_groups[:4],
        all_groups[:5],
        all_groups[:6],
        all_groups[:7],
    ]

    # Pre-build bucket tables for all groups using the best config
    binning = best_config["binning_mode"]
    num_bins = best_config["num_bins"]
    print(f"[EXP-B] Using best config: {binning}, bins={num_bins}")

    bucket_tables = {}
    for gid in all_groups:
        addr_idx = stats["addr_idx"][gid]
        addr_mean = stats["addr_mean"][gid]
        addr_std = stats["addr_std"][gid]
        table, coverage, per_bin_count, per_bin_var, _ = build_bucket_table(
            model=model, calib_loader=calib_loader, layer_id=layer_id,
            candidate_type=cand_type, group_id=gid, group_size=group_size,
            addr_idx=addr_idx, addr_mean=addr_mean, addr_std=addr_std,
            num_bins=num_bins, addr_clip=config.addr_clip, binning_mode=binning,
        )
        bucket_tables[gid] = {
            "table": table,
            "addr_idx": addr_idx,
            "addr_mean": addr_mean,
            "addr_std": addr_std,
            "coverage": coverage,
            "entropy": compute_occupancy_entropy(per_bin_count),
        }

    results = []
    for combo in combinations:
        print(f"\n[EXP-B] Testing groups: {combo}")
        group_configs = []
        for gid in combo:
            bt = bucket_tables[gid]
            group_configs.append({
                "group_id": gid,
                "group_size": group_size,
                "mode": "bucket",
                "bucket_table": bt["table"],
                "addr_idx": bt["addr_idx"],
                "addr_mean": bt["addr_mean"],
                "addr_std": bt["addr_std"],
                "num_bins": num_bins,
                "addr_clip": config.addr_clip,
            })

        metrics = eval_multi_groups(model, eval_loader, reference_probs, layer_id, cand_type, group_configs)
        kl = metrics.get("avg_kl", 0)
        print(f"  KL={kl:.6f}, PPL={metrics['ppl']:.2f}")
        results.append({
            "groups": list(combo),
            "num_groups": len(combo),
            "kl_bucket": kl,
            "ppl_bucket": metrics["ppl"],
        })

    return results


def experiment_C(model, calib_loader, eval_loader, addr_stats, reference_probs, config, best_config):
    """Two-head vs single-head ablation on top 3 groups."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Two-Head vs Single-Head Ablation")
    print("=" * 70)

    layer_id = config.layer_ids[0]
    cand_type = config.candidate_types[0]
    stats = addr_stats[(layer_id, cand_type)]
    group_size = stats["group_size"]
    top_groups = config.groups_to_test[:3]
    binning = best_config["binning_mode"]
    num_bins = best_config["num_bins"]

    # Need to re-calibrate with heads=1 to get single-head addr stats
    print("[EXP-C] Re-calibrating with heads=1 for single-head test...")
    addr_stats_1head = calibrate_llm_address(
        model=model,
        tokenizer=None,  # not used in calibrate
        calib_loader=calib_loader,
        layer_ids=[layer_id],
        candidate_types=[cand_type],
        hidden_group_size=group_size,
        heads=1,
    )
    stats_1head = addr_stats_1head[(layer_id, cand_type)]

    results = []
    for gid in top_groups:
        print(f"\n[EXP-C] Group {gid}")
        for heads_label, stats_src in [("1-head", stats_1head), ("2-head", stats)]:
            addr_idx = stats_src["addr_idx"][gid]
            addr_mean = stats_src["addr_mean"][gid]
            addr_std = stats_src["addr_std"][gid]

            table, coverage, per_bin_count, per_bin_var, _ = build_bucket_table(
                model=model, calib_loader=calib_loader, layer_id=layer_id,
                candidate_type=cand_type, group_id=gid, group_size=group_size,
                addr_idx=addr_idx, addr_mean=addr_mean, addr_std=addr_std,
                num_bins=num_bins, addr_clip=config.addr_clip, binning_mode=binning,
            )

            metrics = eval_single_group(
                model, eval_loader, reference_probs, layer_id, cand_type, gid, group_size,
                mode="bucket",
                bucket_table=table,
                addr_idx=addr_idx,
                addr_mean=addr_mean,
                addr_std=addr_std,
                num_bins=num_bins,
                addr_clip=config.addr_clip,
            )

            kl_bucket = metrics.get("avg_kl", 0)
            print(f"  {heads_label}: KL={kl_bucket:.6f}, Coverage={coverage:.2%}")
            results.append({
                "group": gid,
                "heads": heads_label,
                "kl_bucket": kl_bucket,
                "ppl_bucket": metrics["ppl"],
                "coverage": coverage,
                "entropy": compute_occupancy_entropy(per_bin_count),
            })

    return results


class V0_5Config(V0Config):
    """Extended config for v0.5."""
    layer_ids = (6,)
    candidate_types = ("mlp_delta",)
    groups_to_test = [4, 3, 8, 1, 13, 9, 0]
    binning_modes = ["uniform", "quantile"]
    num_bins_list = [32, 64, 128, 256]


def main():
    parser = argparse.ArgumentParser(description="LLM-LUT v0.5 Extended Experiment")
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_exp_a", action="store_true")
    parser.add_argument("--skip_exp_b", action="store_true")
    parser.add_argument("--skip_exp_c", action="store_true")
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()

    config = V0_5Config()
    config.calib_size = args.calib_size
    config.eval_size = args.eval_size
    config.max_seq_len = args.max_seq_len
    config.calib_batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.result_dir = args.result_dir

    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("LLM-LUT v0.5 Extended Experiment")
    print("=" * 70)
    print(f"Calib: {config.calib_size}, Eval: {config.eval_size}, MaxLen: {config.max_seq_len}")
    print(f"Groups: {config.groups_to_test}")
    print(f"Binning: {config.binning_modes}, Bins: {config.num_bins_list}")

    # Load model and data
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(config)

    # Calibration (heads=2, default)
    print("\n[CALIB] Running address calibration (heads=2)...")
    addr_stats = calibrate_llm_address(
        model=model, tokenizer=tokenizer, calib_loader=calib_loader,
        layer_ids=config.layer_ids, candidate_types=config.candidate_types,
        hidden_group_size=config.hidden_group_size, heads=config.heads,
    )

    # Compute baseline probabilities for KL
    print("\n[BASELINE] Computing baseline probabilities...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader)
    print(f"[BASELINE] PPL={baseline_metrics['ppl']:.2f}, ACC={baseline_metrics['next_token_acc']:.4f}")

    # Experiment A
    if not args.skip_exp_a:
        results_A = experiment_A(model, calib_loader, eval_loader, addr_stats, reference_probs, config)
        save_path_A = os.path.join(config.result_dir, "v0_5_experiment_A.json")
        with open(save_path_A, "w") as f:
            json.dump(results_A, f, indent=2, ensure_ascii=False)
        print(f"\n[EXP-A] Saved to {save_path_A}")

        # Rank and report
        rank_path = os.path.join(config.result_dir, "v0_5_report_A.md")
        ranked = rank_candidates(results_A, baseline_metrics, save_path=rank_path)

        # Pick best config for Experiment B and C
        best = ranked[0] if ranked else None
    else:
        best = None

    # Experiment B
    if not args.skip_exp_b and best is not None:
        results_B = experiment_B(model, calib_loader, eval_loader, addr_stats, reference_probs, config, best)
        save_path_B = os.path.join(config.result_dir, "v0_5_experiment_B.json")
        with open(save_path_B, "w") as f:
            json.dump(results_B, f, indent=2, ensure_ascii=False)
        print(f"\n[EXP-B] Saved to {save_path_B}")

    # Experiment C
    if not args.skip_exp_c and best is not None:
        results_C = experiment_C(model, calib_loader, eval_loader, addr_stats, reference_probs, config, best)
        save_path_C = os.path.join(config.result_dir, "v0_5_experiment_C.json")
        with open(save_path_C, "w") as f:
            json.dump(results_C, f, indent=2, ensure_ascii=False)
        print(f"\n[EXP-C] Saved to {save_path_C}")

    print("\n" + "=" * 70)
    print("LLM-LUT v0.5 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
