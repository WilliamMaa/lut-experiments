"""LLM-LUT R1: 0.5B Functional Replacement Model.

Builds, evaluates, and saves a 2D bucket replacement for Layer 6 mlp_delta group 4.
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
V1_DIR = os.path.join(os.path.dirname(__file__), "..", "v1")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V1_DIR)

from config import get_hook_target
from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics
from train import collect_teacher_targets, build_joint_bucket_table
from v1_config import V1Config

from r1_replacement import ReplacementEngine, run_generation_eval, GENERATION_PROMPTS


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


def run_r1_experiment(config: V1Config):
    """Run full R1 experiment."""
    device = torch.device("cuda:0")
    result_dir = config.result_dir
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("LLM-LUT R1: 0.5B Functional Replacement")
    print("=" * 60)

    # 1. Load model and data
    print("\n[1/7] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(config)

    # 2. Address calibration
    print("\n[2/7] Calibrating address channels...")
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

    addr_idx_g = addr_idx_all[config.target_group].cpu()
    addr_mean_g = addr_mean_all[config.target_group].cpu()
    addr_std_g = addr_std_all[config.target_group].cpu()

    # 3. Build 2D bucket table
    print("\n[3/7] Building 2D bucket table...")
    bin_idx, targets, _ = collect_teacher_targets(
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
    joint_table = build_joint_bucket_table(bin_idx, targets, config.num_bins, config.group_size)

    # 4. Create and install replacement engine
    print("\n[4/7] Installing replacement engine...")
    engine = ReplacementEngine(
        model=model,
        layer_id=config.layer_id,
        group_id=config.target_group,
        group_size=config.group_size,
        addr_idx=addr_idx_g,
        addr_mean=addr_mean_g,
        addr_std=addr_std_g,
        table=joint_table,
        num_bins=config.num_bins,
        addr_clip=config.addr_clip,
    )

    # 5. Evaluate original model (baseline)
    print("\n[5/7] Evaluating original model...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    orig_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Original: PPL={orig_metrics['ppl']:.2f} Acc={orig_metrics['next_token_acc']:.4f}")

    # 6. Evaluate with replacement
    print("\n[6/7] Evaluating replacement model...")
    engine.install()
    repl_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Replacement: KL={repl_metrics['avg_kl']:.4f} PPL={repl_metrics['ppl']:.2f} "
          f"Acc={repl_metrics['next_token_acc']:.4f}")

    print(f"  Replacement KL vs original: {repl_metrics['avg_kl']:.4f}")

    # Generation sanity
    print("\n  Running generation sanity check...")
    gen_results = run_generation_eval(model, tokenizer, GENERATION_PROMPTS, max_new_tokens=128, device=device)
    for item in gen_results:
        print(f"\n  Prompt: {item['prompt']}")
        print(f"  Output: {item['output'][:200]}{'...' if len(item['output']) > 200 else ''}")

    # Save checkpoint
    ckpt_path = os.path.join(result_dir, "r1_replacement.pt")
    engine.save(ckpt_path)

    # Uninstall and verify clean state
    print("\n[7/7] Verifying clean uninstall...")
    engine.uninstall()
    clean_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Clean (hook removed): KL={clean_metrics['avg_kl']:.4f} PPL={clean_metrics['ppl']:.2f} "
          f"Acc={clean_metrics['next_token_acc']:.4f}")

    # Report
    print("\n" + "=" * 60)
    print("R1 Summary")
    print("=" * 60)
    print(f"  Original model:     PPL={orig_metrics['ppl']:.2f} Acc={orig_metrics['next_token_acc']:.4f}")
    print(f"  Replacement model:  KL={repl_metrics['avg_kl']:.4f} PPL={repl_metrics['ppl']:.2f} Acc={repl_metrics['next_token_acc']:.4f}")
    print(f"  Clean uninstall:    KL={clean_metrics['avg_kl']:.4f} PPL={clean_metrics['ppl']:.2f} Acc={clean_metrics['next_token_acc']:.4f}")

    # Save report
    report_path = os.path.join(result_dir, "r1_report.md")
    with open(report_path, "w") as f:
        f.write("# LLM-LUT R1 Report\n\n")
        f.write(f"- Model: {config.model_name}\n")
        f.write(f"- Layer: {config.layer_id}, Group: {config.target_group}, Type: {config.candidate_type}\n")
        f.write(f"- Method: 2-head uniform joint bucket, {config.num_bins}×{config.num_bins} bins\n\n")
        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Original | 0.0000 | {orig_metrics['ppl']:.2f} | {orig_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Replacement | {repl_metrics['avg_kl']:.4f} | {repl_metrics['ppl']:.2f} | {repl_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Clean uninstall | {clean_metrics['avg_kl']:.4f} | {clean_metrics['ppl']:.2f} | {clean_metrics['next_token_acc']:.4f} |\n\n")
        f.write("## Generation Samples\n\n")
        for item in gen_results:
            f.write(f"**Prompt:** {item['prompt']}\n\n")
            f.write(f"**Output:** {item['output']}\n\n---\n\n")
    print(f"\n  Report saved to {report_path}")

    # Save JSON
    json_path = os.path.join(result_dir, "r1_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {"model": config.model_name, "layer": config.layer_id, "group": config.target_group},
            "original": orig_metrics,
            "replacement": repl_metrics,
            "clean": clean_metrics,
            "generation": gen_results,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    print("\n" + "=" * 60)
    print("R1 Experiment Complete")
    print("=" * 60)


if __name__ == "__main__":
    config = V1Config()
    config.result_dir = "results"  # relative to v2/
    run_r1_experiment(config)
