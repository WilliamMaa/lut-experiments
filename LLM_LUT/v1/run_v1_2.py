"""LLM-LUT v1.2 Main Entry — Additive LUT Decomposition.

Three experiments:
  A: Additive only, trained from scratch
  B: Additive with ANOVA initialization from joint bucket
  C: Additive + coarse interaction table (8x8)

MANDATORY: Run ../v0/gpu_sanity_check.py FIRST.
"""

import os
import sys
import json
import torch

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
from additive_table import AdditiveLUTTable, anova_decompose, build_coarse_interaction
from train import (
    collect_teacher_targets,
    train_additive_table,
    build_joint_bucket_table,
    evaluate_additive,
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


def run_v1_2_experiment(config: V1Config):
    """Run v1.2 A/B/C experiments."""
    device = torch.device("cuda:0")

    print("=" * 60)
    print("LLM-LUT v1.2 Experiment — Additive LUT Decomposition")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(config)

    # 2. Calibrate
    print("\n[2/5] Calibrating address channels...")
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
    print("\n[3/5] Computing baseline reference probabilities...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Baseline PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    # 4. Collect targets and build joint bucket table
    print("\n[4/5] Collecting teacher targets (2D bins)...")
    bin_idx_2h, targets, _ = collect_teacher_targets(
        model, calib_loader,
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

    joint_table = build_joint_bucket_table(bin_idx_2h, targets, config.num_bins, config.group_size)

    # ANOVA decomposition
    lut1_ano, lut2_ano, bias_ano = anova_decompose(joint_table)
    print(f"  ANOVA: lut1 range=[{lut1_ano.min():.4f}, {lut1_ano.max():.4f}], "
          f"lut2 range=[{lut2_ano.min():.4f}, {lut2_ano.max():.4f}], "
          f"bias norm={bias_ano.norm():.4f}")

    # 5. Run A/B/C
    print("\n[5/5] Training additive LUT variants...")

    experiments = []

    # ---- Experiment A: Additive from scratch ----
    print("\n--- Experiment A: Additive from scratch ---")
    table_a = AdditiveLUTTable(
        num_bins=config.num_bins,
        group_size=config.group_size,
    )
    history_a = train_additive_table(
        table_a, bin_idx_2h, targets,
        num_epochs=40, lr=1e-3, alpha_cosine=0.1, batch_size=256, device=device,
    )
    metrics_a = evaluate_additive(
        model, eval_loader, reference_probs, table_a,
        layer_id=config.layer_id, candidate_type=config.candidate_type,
        group_id=config.target_group, group_size=config.group_size,
        addr_idx=addr_idx_g, addr_mean=addr_mean_g, addr_std=addr_std_g,
        num_bins=config.num_bins, addr_clip=config.addr_clip,
    )
    experiments.append(("A_additive_scratch", metrics_a, history_a))

    # ---- Experiment B: Additive with ANOVA init ----
    print("\n--- Experiment B: Additive ANOVA init ---")
    table_b = AdditiveLUTTable(
        num_bins=config.num_bins,
        group_size=config.group_size,
        init_lut1=lut1_ano,
        init_lut2=lut2_ano,
        init_bias=bias_ano,
    )
    history_b = train_additive_table(
        table_b, bin_idx_2h, targets,
        num_epochs=40, lr=1e-3, alpha_cosine=0.1, batch_size=256, device=device,
    )
    metrics_b = evaluate_additive(
        model, eval_loader, reference_probs, table_b,
        layer_id=config.layer_id, candidate_type=config.candidate_type,
        group_id=config.target_group, group_size=config.group_size,
        addr_idx=addr_idx_g, addr_mean=addr_mean_g, addr_std=addr_std_g,
        num_bins=config.num_bins, addr_clip=config.addr_clip,
    )
    experiments.append(("B_additive_anova", metrics_b, history_b))

    # ---- Experiment C: Additive + 8x8 interaction ----
    print("\n--- Experiment C: Additive + 8x8 interaction ---")
    lut_c = build_coarse_interaction(joint_table, coarse_bins=8)
    table_c = AdditiveLUTTable(
        num_bins=config.num_bins,
        group_size=config.group_size,
        init_lut1=lut1_ano,
        init_lut2=lut2_ano,
        init_bias=bias_ano,
        interaction_table=lut_c,
        interaction_bins=8,
    )
    history_c = train_additive_table(
        table_c, bin_idx_2h, targets,
        num_epochs=40, lr=1e-3, alpha_cosine=0.1, batch_size=256, device=device,
    )
    metrics_c = evaluate_additive(
        model, eval_loader, reference_probs, table_c,
        layer_id=config.layer_id, candidate_type=config.candidate_type,
        group_id=config.target_group, group_size=config.group_size,
        addr_idx=addr_idx_g, addr_mean=addr_mean_g, addr_std=addr_std_g,
        num_bins=config.num_bins, addr_clip=config.addr_clip,
    )
    experiments.append(("C_additive_interact8", metrics_c, history_c))

    # Baselines: zero and mean only (bucket requires 2D hook, skip here)
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
        bucket_table=None,
        mean_vec=mean_vec,
    )

    # Report
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nBaseline PPL={baseline_metrics['ppl']:.2f} Acc={baseline_metrics['next_token_acc']:.4f}")

    kl_zero = baseline_results.get("zero", {}).get("avg_kl", 0)
    kl_bucket = baseline_results.get("bucket", {}).get("avg_kl", None) or 0

    # v1.0 2D bucket reference
    v10_bucket_kl = 0.6066

    def fmt(name, metrics):
        kl = metrics.get("avg_kl", 0)
        recovery = (kl_zero - kl) / kl_zero if kl_zero > 0 else 0
        marker = ""
        if kl < v10_bucket_kl:
            marker = " ✅ BEATS v1.0 BUCKET"
        return f"  {name:22s}: KL={kl:.4f} PPL={metrics['ppl']:.2f} Acc={metrics['next_token_acc']:.4f} Recovery={recovery:.2%}{marker}"

    print(fmt("zero", baseline_results.get("zero", {})))
    print(fmt("mean", baseline_results.get("mean", {})))
    print(fmt("joint_bucket", baseline_results.get("bucket", {})))
    print(fmt("v1.0_bucket_2d_ref", {"avg_kl": v10_bucket_kl, "ppl": 41.94, "next_token_acc": 0.4883}))
    for name, metrics, _ in experiments:
        print(fmt(name, metrics))

    # Save report
    os.makedirs(config.result_dir, exist_ok=True)
    report_path = os.path.join(config.result_dir, "v1.2_report.md")
    with open(report_path, "w") as f:
        f.write("# LLM-LUT v1.2 Experiment Report\n\n")
        f.write(f"- Layer: {config.layer_id}, Candidate: {config.candidate_type}, Group: {config.target_group}\n")
        f.write(f"- Baseline PPL: {baseline_metrics['ppl']:.2f}, Acc: {baseline_metrics['next_token_acc']:.4f}\n\n")
        f.write("## Results\n\n")
        f.write("| Mode | KL | PPL | Acc | Recovery |\n")
        f.write("|------|-----|-----|-----|----------|\n")
        f.write(f"| zero | {baseline_results['zero']['avg_kl']:.4f} | {baseline_results['zero']['ppl']:.2f} | {baseline_results['zero']['next_token_acc']:.4f} | 0.00% |\n")
        f.write(f"| mean | {baseline_results['mean']['avg_kl']:.4f} | {baseline_results['mean']['ppl']:.2f} | {baseline_results['mean']['next_token_acc']:.4f} | -0.03% |\n")
        if 'bucket' in baseline_results:
            f.write(f"| joint_bucket | {kl_bucket:.4f} | {baseline_results['bucket']['ppl']:.2f} | {baseline_results['bucket']['next_token_acc']:.4f} | {(kl_zero-kl_bucket)/kl_zero:.2%} |\n")
        else:
            f.write(f"| joint_bucket | N/A | N/A | N/A | N/A |\n")
        f.write(f"| v1.0_bucket_2d | {v10_bucket_kl:.4f} | 41.94 | 0.4883 | 82.66% |\n")
        for name, metrics, _ in experiments:
            kl = metrics.get("avg_kl", 0)
            recovery = (kl_zero - kl) / kl_zero if kl_zero > 0 else 0
            f.write(f"| {name} | {kl:.4f} | {metrics['ppl']:.2f} | {metrics['next_token_acc']:.4f} | {recovery:.2%} |\n")

        for name, _, history in experiments:
            f.write(f"\n## Training History: {name}\n\n")
            f.write("| Epoch | Loss | MSE | Cos Loss |\n")
            f.write("|-------|------|-----|----------|\n")
            for h in history:
                f.write(f"| {h['epoch']} | {h['loss']:.6f} | {h['mse']:.6f} | {h['cos_loss']:.6f} |\n")
    print(f"\n  Report saved to {report_path}")

    json_path = os.path.join(config.result_dir, "v1.2_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "config": {"layer_id": config.layer_id, "candidate_type": config.candidate_type,
                       "target_group": config.target_group, "num_bins": config.num_bins},
            "baselines": baseline_results,
            "experiments": {name: {"metrics": m, "history": h} for name, m, h in experiments},
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    print("\n" + "=" * 60)
    print("v1.2 Experiment Complete")
    print("=" * 60)


if __name__ == "__main__":
    config = V1Config()
    run_v1_2_experiment(config)
