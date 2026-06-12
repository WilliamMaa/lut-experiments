"""LLM-LUT v1 Main Entry.

Trains trainable LUT for Layer 6 mlp_delta group 4,
and compares against zero/mean/bucket baselines.

MANDATORY: Run ../v0/gpu_sanity_check.py FIRST.
"""

# CRITICAL: Set BEFORE importing transformers / accelerate.
# accelerate reads env vars at import time; setting them later has no effect.
import os
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add v0 to path
V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

from config import get_hook_target
from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics

from v1_config import V1Config
from lut_table import TrainableLUTTable
from train import (
    collect_teacher_targets,
    train_lut_table,
    evaluate_lut,
    evaluate_baseline_modes,
)


def load_model_and_data(config: V1Config, device_str: str = "cuda:0"):
    """Load model to single GPU and prepare data loaders."""
    # Env vars already set at module top (before importing transformers).
    # Kept here as defense-in-depth only.
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
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

    # Verify single GPU
    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            raise RuntimeError(f"FATAL: GPU {i} has allocated memory!")

    # Data
    prepare_data(
        tokenizer,
        config.calib_path,
        config.eval_path,
        calib_size=config.calib_size,
        eval_size=config.eval_size,
        max_seq_len=config.max_seq_len,
    )
    calib_texts = load_jsonl(config.calib_path)
    eval_texts = load_jsonl(config.eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=config.max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=config.max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=config.calib_batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=config.eval_batch_size, shuffle=False)

    return model, tokenizer, calib_loader, eval_loader


def run_v1_experiment(config: V1Config, device_str: str = "cuda:0"):
    """Run full v1.0 experiment."""
    device = torch.device(device_str)

    print("=" * 60)
    print("LLM-LUT v1.0 Experiment")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(config, device_str=device_str)

    # 2. Calibrate address
    print("\n[2/6] Calibrating address channels...")
    calib_results = calibrate_llm_address(
        model,
        tokenizer,
        calib_loader,
        layer_ids=(config.layer_id,),
        candidate_types=(config.candidate_type,),
        hidden_group_size=config.hidden_group_size,
        intermediate_group_size=128,
        heads=config.heads,
    )
    calib = calib_results[(config.layer_id, config.candidate_type)]

    addr_idx_all = calib["addr_idx"]    # [num_groups, heads]
    addr_mean_all = calib["addr_mean"]  # [num_groups, heads]
    addr_std_all = calib["addr_std"]    # [num_groups, heads]
    group_means = calib["group_means"]  # [num_groups, group_size]

    addr_idx_g = addr_idx_all[config.target_group].cpu()    # [heads]
    addr_mean_g = addr_mean_all[config.target_group].cpu()  # [heads]
    addr_std_g = addr_std_all[config.target_group].cpu()    # [heads]
    mean_vec = group_means[config.target_group].cpu()       # [group_size]

    print(f"  Group {config.target_group}: addr_idx={addr_idx_g.tolist()}")

    # 3. Compute baseline reference probs (original model)
    print("\n[3/6] Computing baseline reference probabilities...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Baseline PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    # 4. Collect teacher targets
    print("\n[4/6] Collecting teacher targets for group 4...")
    bin_idx_2h, targets, bucket_init_2h = collect_teacher_targets(
        model,
        calib_loader,
        layer_id=config.layer_id,
        candidate_type=config.candidate_type,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        num_bins=config.num_bins,
        addr_clip=config.addr_clip,
    )

    # Also collect 1-head version (use only first head)
    bin_idx_1h = bin_idx_2h[:, 0:1]  # [N, 1]
    bucket_init_1h = torch.zeros(config.num_bins, config.group_size)
    counts_1h = torch.zeros(config.num_bins)
    for b in range(config.num_bins):
        mask = (bin_idx_1h[:, 0] == b)
        cnt = mask.sum().item()
        counts_1h[b] = cnt
        if cnt > 0:
            bucket_init_1h[b] = targets[mask].mean(dim=0)

    # 5. Train and evaluate for each head config
    print("\n[5/6] Training LUT tables...")
    all_results = {}

    for num_heads, bin_idx, bucket_init, label in [
        (1, bin_idx_1h, bucket_init_1h, "1-head"),
        (2, bin_idx_2h, bucket_init_2h, "2-head"),
    ]:
        print(f"\n--- {label} ---")

        # Slice address for num_heads
        addr_idx_h = addr_idx_g[:num_heads]
        addr_mean_h = addr_mean_g[:num_heads]
        addr_std_h = addr_std_g[:num_heads]

        # Build table
        lut_table = TrainableLUTTable(
            num_bins=config.num_bins,
            group_size=config.group_size,
            num_heads=num_heads,
            init_table=bucket_init,
        )
        lut_table = lut_table.to(device)

        # Train
        history = train_lut_table(
            lut_table,
            bin_indices=bin_idx,
            targets=targets,
            num_epochs=config.num_epochs,
            lr=config.lr,
            alpha_cosine=config.alpha_cosine,
            batch_size=256,
            device=device,
        )

        # Evaluate baselines (zero, mean, bucket)
        print(f"  Evaluating baselines for {label}...")
        baseline_results = evaluate_baseline_modes(
            model,
            eval_loader,
            reference_probs,
            layer_id=config.layer_id,
            candidate_type=config.candidate_type,
            group_id=config.target_group,
            group_size=config.group_size,
            addr_idx=addr_idx_h,
            addr_mean=addr_mean_h,
            addr_std=addr_std_h,
            num_bins=config.num_bins,
            addr_clip=config.addr_clip,
            bucket_table=bucket_init if num_heads == 1 else None,
            mean_vec=mean_vec,
        )

        # For 2-head, also evaluate bucket with frozen 2D table
        if num_heads == 2:
            bucket_table_2d = TrainableLUTTable(
                num_bins=config.num_bins,
                group_size=config.group_size,
                num_heads=2,
                init_table=bucket_init,
            )
            bucket_table_2d = bucket_table_2d.to(device)
            for p in bucket_table_2d.parameters():
                p.requires_grad_(False)
            bucket_metrics_2d = evaluate_lut(
                model, eval_loader, reference_probs,
                bucket_table_2d,
                layer_id=config.layer_id,
                candidate_type=config.candidate_type,
                group_id=config.target_group,
                group_size=config.group_size,
                addr_idx=addr_idx_h,
                addr_mean=addr_mean_h,
                addr_std=addr_std_h,
                num_bins=config.num_bins,
                addr_clip=config.addr_clip,
            )
            baseline_results["bucket_2d"] = bucket_metrics_2d

        # Evaluate trainable LUT
        print(f"  Evaluating trainable LUT for {label}...")
        trainable_metrics = evaluate_lut(
            model,
            eval_loader,
            reference_probs,
            lut_table,
            layer_id=config.layer_id,
            candidate_type=config.candidate_type,
            group_id=config.target_group,
            group_size=config.group_size,
            addr_idx=addr_idx_h,
            addr_mean=addr_mean_h,
            addr_std=addr_std_h,
            num_bins=config.num_bins,
            addr_clip=config.addr_clip,
        )

        all_results[label] = {
            "history": history,
            "baselines": baseline_results,
            "trainable": trainable_metrics,
        }

    # 6. Report
    print("\n[6/6] Generating report...")
    os.makedirs(config.result_dir, exist_ok=True)

    report_lines = []
    report_lines.append("# LLM-LUT v1.0 Experiment Report\n")
    report_lines.append(f"- Layer: {config.layer_id}, Candidate: {config.candidate_type}, Group: {config.target_group}\n")
    report_lines.append(f"- Baseline PPL: {baseline_metrics['ppl']:.2f}, Acc: {baseline_metrics['next_token_acc']:.4f}\n")
    report_lines.append("\n")

    for label in ["1-head", "2-head"]:
        res = all_results[label]
        report_lines.append(f"\n## {label}\n\n")

        report_lines.append("### Baselines\n\n")
        report_lines.append("| Mode | KL | PPL | Acc |\n")
        report_lines.append("|------|-----|-----|-----|\n")

        # Zero / Mean / Bucket
        kl_zero = None
        kl_bucket = None
        kl_bucket_2d = None
        for mode in ["zero", "mean", "bucket", "bucket_2d"]:
            if mode in res["baselines"]:
                m = res["baselines"][mode]
                kl = m.get("avg_kl", "N/A")
                if isinstance(kl, float):
                    if mode == "zero":
                        kl_zero = kl
                    if mode == "bucket":
                        kl_bucket = kl
                    if mode == "bucket_2d":
                        kl_bucket_2d = kl
                    kl_str = f"{kl:.4f}"
                else:
                    kl_str = str(kl)
                display_mode = mode if mode != "bucket_2d" else "bucket (2D)"
                report_lines.append(f"| {display_mode} | {kl_str} | {m['ppl']:.2f} | {m['next_token_acc']:.4f} |\n")

        # Trainable
        m = res["trainable"]
        kl_train = m.get("avg_kl", None)
        kl_str = f"{kl_train:.4f}" if isinstance(kl_train, float) else str(kl_train)
        report_lines.append(f"| **trainable** | **{kl_str}** | **{m['ppl']:.2f}** | **{m['next_token_acc']:.4f}** |\n")

        # Recovery
        if kl_zero is not None and kl_train is not None:
            if kl_bucket is not None:
                recovery_bucket = (kl_zero - kl_bucket) / kl_zero
                report_lines.append(f"\n- Bucket (1D) Recovery: {recovery_bucket:.2%}\n")
            if kl_bucket_2d is not None:
                recovery_bucket_2d = (kl_zero - kl_bucket_2d) / kl_zero
                report_lines.append(f"- Bucket (2D) Recovery: {recovery_bucket_2d:.2%}\n")
            recovery_train = (kl_zero - kl_train) / kl_zero
            report_lines.append(f"- Trainable Recovery: {recovery_train:.2%}\n")
            # Compare trainable vs best bucket
            best_bucket_kl = kl_bucket_2d if kl_bucket_2d is not None else kl_bucket
            if best_bucket_kl is not None:
                if recovery_train > (kl_zero - best_bucket_kl) / kl_zero:
                    diff = recovery_train - (kl_zero - best_bucket_kl) / kl_zero
                    report_lines.append(f"- **Trainable beats best bucket by {diff:.2%}** ✅\n")
                else:
                    diff = (kl_zero - best_bucket_kl) / kl_zero - recovery_train
                    report_lines.append(f"- Trainable falls behind best bucket by {diff:.2%}\n")

        # Training history
        report_lines.append("\n### Training History\n\n")
        report_lines.append("| Epoch | Loss | MSE | Cos Loss |\n")
        report_lines.append("|-------|------|-----|----------|\n")
        for h in res["history"]:
            report_lines.append(f"| {h['epoch']} | {h['loss']:.6f} | {h['mse']:.6f} | {h['cos_loss']:.6f} |\n")

    report_path = os.path.join(config.result_dir, "v1.0_report.md")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"  Report saved to {report_path}")

    # JSON results
    json_path = os.path.join(config.result_dir, "v1.0_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "config": {
                "layer_id": config.layer_id,
                "candidate_type": config.candidate_type,
                "target_group": config.target_group,
                "num_bins": config.num_bins,
                "lr": config.lr,
                "num_epochs": config.num_epochs,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    print("\n" + "=" * 60)
    print("v1.0 Experiment Complete")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-LUT v1.0 Experiment")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()

    config = V1Config()
    run_v1_experiment(config, device_str=args.device)
