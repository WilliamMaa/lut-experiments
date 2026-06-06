"""LLM-LUT v1.1 Main Entry — Learned Codebook.

MANDATORY: Run ../v0/gpu_sanity_check.py FIRST.
"""

import os
import sys
import json
import torch

# Critical: before any import that might trigger accelerate
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

from config import get_hook_target
from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics

from v1_config import V1Config
from codebook_table import LearnableCodebookTable
from train import (
    collect_codebook_targets,
    init_codebook_from_targets,
    train_codebook_table,
    evaluate_codebook,
    evaluate_baseline_modes,
)


def load_model_and_data(config: V1Config, device_str: str = "cuda:0"):
    """Load model to single GPU and prepare data loaders."""
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

    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            raise RuntimeError(f"FATAL: GPU {i} has allocated memory!")

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


def run_v1_1_experiment(config: V1Config):
    """Run v1.1 learned codebook experiment."""
    device = torch.device("cuda:0")
    num_centroids = config.num_bins  # Re-use num_bins as K

    print("=" * 60)
    print("LLM-LUT v1.1 Experiment — Learned Codebook")
    print(f"K={num_centroids}, group={config.target_group}, layer={config.layer_id}")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(config)

    # 2. Calibrate
    print("\n[2/6] Calibrating address channels...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=(config.layer_id,),
        candidate_types=(config.candidate_type,),
        hidden_group_size=config.hidden_group_size,
        intermediate_group_size=128,
        heads=config.heads,
    )
    calib = calib_results[(config.layer_id, config.candidate_type)]
    addr_idx_all = calib["addr_idx"]
    addr_mean_all = calib["addr_mean"]
    addr_std_all = calib["addr_std"]
    group_means = calib["group_means"]

    addr_idx_g = addr_idx_all[config.target_group].cpu()
    addr_mean_g = addr_mean_all[config.target_group].cpu()
    addr_std_g = addr_std_all[config.target_group].cpu()
    mean_vec = group_means[config.target_group].cpu()

    # 3. Baseline
    print("\n[3/6] Computing baseline reference probabilities...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Baseline PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    # 4. Collect targets
    print("\n[4/6] Collecting codebook targets...")
    address_vectors, targets = collect_codebook_targets(
        model, calib_loader,
        layer_id=config.layer_id,
        candidate_type=config.candidate_type,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        addr_clip=config.addr_clip,
    )

    # 5. K-means init + train
    print("\n[5/6] Training learned codebook...")
    centroids_init, table_init = init_codebook_from_targets(
        address_vectors, targets, num_centroids=num_centroids, seed=config.seed
    )

    codebook = LearnableCodebookTable(
        num_centroids=num_centroids,
        group_size=config.group_size,
        address_dim=config.heads,
        init_centroids=centroids_init,
        init_table=table_init,
        init_temperature=1.0,
        min_temperature=0.05,
    )
    codebook = codebook.to(device)

    history = train_codebook_table(
        codebook,
        address_vectors=address_vectors,
        targets=targets,
        num_epochs=60,
        lr=1e-2,
        alpha_cosine=0.1,
        beta_usage=0.5,
        batch_size=512,
        device=device,
    )

    # 6. Evaluate
    print("\n[6/6] Evaluating codebook and baselines...")

    # Baselines: zero, mean, bucket (1D/2D via v0 hook)
    baseline_results = evaluate_baseline_modes(
        model, eval_loader, reference_probs,
        layer_id=config.layer_id,
        candidate_type=config.candidate_type,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        num_bins=config.num_bins,
        addr_clip=config.addr_clip,
        bucket_table=None,  # v1.1 doesn't use fixed bucket; we'll use k-means init as proxy
        mean_vec=mean_vec,
    )

    # Evaluate codebook (hard assignment, simulating inference)
    print("  Evaluating codebook (hard assignment)...")
    codebook_hard_metrics = evaluate_codebook(
        model, eval_loader, reference_probs, codebook,
        layer_id=config.layer_id,
        candidate_type=config.candidate_type,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        addr_clip=config.addr_clip,
        hard=True,
    )

    # Also evaluate soft assignment for comparison
    print("  Evaluating codebook (soft assignment)...")
    codebook_soft_metrics = evaluate_codebook(
        model, eval_loader, reference_probs, codebook,
        layer_id=config.layer_id,
        candidate_type=config.candidate_type,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        addr_clip=config.addr_clip,
        hard=False,
    )

    # Report
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nBaseline PPL={baseline_metrics['ppl']:.2f} Acc={baseline_metrics['next_token_acc']:.4f}")

    kl_zero = baseline_results.get("zero", {}).get("avg_kl", 0)
    
    # v1.0 2D bucket reference for direct comparison
    v10_bucket_kl = 0.6066
    v10_bucket_ppl = 41.94
    v10_bucket_acc = 0.4883
    
    all_results = [
        ("zero", baseline_results.get("zero")),
        ("mean", baseline_results.get("mean")),
        ("v1.0_bucket_2d", {"avg_kl": v10_bucket_kl, "ppl": v10_bucket_ppl, "next_token_acc": v10_bucket_acc}),
        ("codebook_hard", codebook_hard_metrics),
        ("codebook_soft", codebook_soft_metrics),
    ]
    
    for name, metrics in all_results:
        if metrics is None:
            continue
        kl = metrics.get("avg_kl", 0)
        recovery = (kl_zero - kl) / kl_zero if kl_zero > 0 else 0
        marker = ""
        if name == "codebook_hard":
            if kl < v10_bucket_kl:
                marker = " ✅ BEATS v1.0 BUCKET"
            else:
                marker = ""
        print(f"  {name:15s}: KL={kl:.4f} PPL={metrics['ppl']:.2f} Acc={metrics['next_token_acc']:.4f} Recovery={recovery:.2%}{marker}")

    # Save report
    os.makedirs(config.result_dir, exist_ok=True)
    report_path = os.path.join(config.result_dir, "v1.1_report.md")
    with open(report_path, "w") as f:
        f.write("# LLM-LUT v1.1 Experiment Report\n\n")
        f.write(f"- Layer: {config.layer_id}, Candidate: {config.candidate_type}, Group: {config.target_group}\n")
        f.write(f"- K (centroids): {num_centroids}\n")
        f.write(f"- Baseline PPL: {baseline_metrics['ppl']:.2f}, Acc: {baseline_metrics['next_token_acc']:.4f}\n\n")
        f.write("## Results\n\n")
        f.write("| Mode | KL | PPL | Acc | Recovery |\n")
        f.write("|------|-----|-----|-----|----------|\n")
        for name, metrics in all_results:
            if metrics is None:
                continue
            kl = metrics.get("avg_kl", 0)
            recovery = (kl_zero - kl) / kl_zero if kl_zero > 0 else 0
            f.write(f"| {name} | {kl:.4f} | {metrics['ppl']:.2f} | {metrics['next_token_acc']:.4f} | {recovery:.2%} |\n")
        f.write("\n## Training History\n\n")
        f.write("| Epoch | Loss | MSE | Cos | Usage | Temp | Coverage | Entropy |\n")
        f.write("|-------|------|-----|-----|-------|------|----------|---------|\n")
        for h in history:
            f.write(f"| {h['epoch']} | {h['loss']:.6f} | {h['mse']:.6f} | {h['cos_loss']:.6f} | "
                    f"{h['usage_loss']:.6f} | {h['temperature']:.4f} | {h['hard_coverage']:.2%} | {h['hard_entropy']:.4f} |\n")
    print(f"\n  Report saved to {report_path}")

    json_path = os.path.join(config.result_dir, "v1.1_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "config": {"layer_id": config.layer_id, "candidate_type": config.candidate_type,
                       "target_group": config.target_group, "num_centroids": num_centroids},
            "baselines": baseline_results,
            "codebook_hard": codebook_hard_metrics,
            "codebook_soft": codebook_soft_metrics,
            "history": history,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    print("\n" + "=" * 60)
    print("v1.1 Experiment Complete")
    print("=" * 60)


if __name__ == "__main__":
    config = V1Config()
    run_v1_1_experiment(config)
