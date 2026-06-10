"""LLM-LUT Scaling-R1: Multi-Scale Scan + Replacement.

Usage:
    python run_r2.py --model Qwen/Qwen2.5-1.5B-Instruct --output_dir results/r2_1.5b
    python run_r2.py --model Qwen/Qwen2.5-3B-Instruct --output_dir results/r2_3b
    python run_r2.py --model Qwen/Qwen2.5-7B-Instruct --output_dir results/r2_7b

MANDATORY: Run ../v0/gpu_sanity_check.py FIRST.
"""

import os
import sys
import json
import argparse
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

from r1_replacement import ReplacementEngine
from r2_scan import fast_scan_groups
from r2_auto_eval import generate_outputs, AUTO_PROMPTS


def load_model_and_data(model_name, calib_size, eval_size, max_seq_len, batch_size, device_str="cuda:0"):
    """Load model to single GPU and prepare data loaders."""
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, "bfloat16", torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load with {dtype}: {e}. Falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            raise RuntimeError(f"FATAL: GPU {i} has allocated memory!")

    calib_path = "../v0/data/calib.jsonl"
    eval_path = "../v0/data/eval.jsonl"
    prepare_data(tokenizer, calib_path, eval_path, calib_size=calib_size, eval_size=eval_size, max_seq_len=max_seq_len)
    calib_texts = load_jsonl(calib_path)
    eval_texts = load_jsonl(eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=batch_size, shuffle=False)

    return model, tokenizer, calib_loader, eval_loader


def run_scaling_experiment(args):
    """Run full Scaling-R1 experiment for a given model."""
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"LLM-LUT Scaling-R1: {args.model_name}")
    print(f"Layers: {args.layer_percentiles} | Group size: {args.group_size} | Bins: {args.num_bins}")
    print("=" * 70)

    # 1. Load
    print("\n[1/6] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model_name, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size
    )
    num_layers = model.config.num_hidden_layers
    print(f"  Model: {num_layers} layers, hidden={model.config.hidden_size}, intermediate={model.config.intermediate_size}")

    # Compute target layers from depth percentiles
    layer_percentiles = [float(x) for x in args.layer_percentiles.split(",")]
    layer_ids = [int(p * num_layers) for p in layer_percentiles]
    layer_ids = list(dict.fromkeys(layer_ids))  # dedup
    print(f"  Target layers: {layer_ids}")

    # 2. Calibrate
    print("\n[2/6] Calibrating address channels...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=tuple(layer_ids),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,  # heuristic
        heads=2,
    )

    # 3. Fast scan
    print("\n[3/6] Running fast scan (zero all + mean/bucket top-K)...")
    all_zero, top_results = fast_scan_groups(
        model, calib_loader, eval_loader, layer_ids, calib_results,
        group_size=args.group_size, num_bins=args.num_bins, top_k=args.top_k,
    )

    if not top_results:
        print("[FATAL] No candidates found!")
        return

    best = top_results[0]
    print(f"\n  Best candidate: Layer {best['layer']}, Group {best['group']}")
    print(f"    KL Zero={best['kl_zero']:.4f} Mean={best['kl_mean']:.4f} Bucket={best['kl_bucket']:.4f}")
    print(f"    PPL Bucket={best['ppl_bucket']:.2f} Acc={best['acc_bucket']:.4f} Recovery={best['recovery']:.2%}")

    # 4. Build replacement for best group
    print("\n[4/6] Building replacement engine...")
    calib_best = calib_results[(best["layer"], "mlp_delta")]
    addr_idx_g = calib_best["addr_idx"][best["group"]]
    addr_mean_g = calib_best["addr_mean"][best["group"]]
    addr_std_g = calib_best["addr_std"][best["group"]]

    from train import collect_teacher_targets, build_joint_bucket_table
    bin_idx, targets, _ = collect_teacher_targets(
        model, calib_loader, best["layer"], "mlp_delta", best["group"], args.group_size,
        addr_idx_g, addr_mean_g, addr_std_g, num_bins=args.num_bins,
    )
    joint_table = build_joint_bucket_table(bin_idx, targets, args.num_bins, args.group_size)

    engine = ReplacementEngine(
        model=model, layer_id=best["layer"], group_id=best["group"],
        group_size=args.group_size, addr_idx=addr_idx_g, addr_mean=addr_mean_g,
        addr_std=addr_std_g, table=joint_table, num_bins=args.num_bins,
    )

    # 5. Evaluate
    print("\n[5/6] Evaluating original vs replacement...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    orig_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Original: PPL={orig_metrics['ppl']:.2f} Acc={orig_metrics['next_token_acc']:.4f}")

    engine.install()
    repl_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Replacement: KL={repl_metrics['avg_kl']:.4f} PPL={repl_metrics['ppl']:.2f} Acc={repl_metrics['next_token_acc']:.4f}")

    # 6. Generation samples
    print("\n[6/6] Collecting generation samples...")
    repl_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    engine.uninstall()
    clean_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    # Re-install for checkpoint save
    engine.install()

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "replacement.pt")
    engine.save(ckpt_path)

    # Save generation samples to separate file
    gen_path = os.path.join(args.output_dir, "generations.md")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(f"# Generation Samples: {args.model_name}\n\n")
        f.write(f"- Selected: Layer {best['layer']}, Group {best['group']}\n")
        f.write(f"- {args.gen_samples} samples per prompt\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"## Prompt {idx+1}: {item['prompt']}\n\n")
            f.write("### Original (no hook)\n\n")
            for i, text in enumerate(clean_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("### Replacement (with hook)\n\n")
            for i, text in enumerate(repl_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")
    print(f"  Generation samples saved to {gen_path}")

    # Save report
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# LLM-LUT Scaling-R1: {args.model_name}\n\n")
        f.write(f"- Layers scanned: {layer_ids}\n")
        f.write(f"- Group size: {args.group_size}, Bins: {args.num_bins}×{args.num_bins}\n")
        f.write(f"- Selected: Layer {best['layer']}, Group {best['group']}\n\n")
        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Original | 0.0000 | {orig_metrics['ppl']:.2f} | {orig_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Replacement | {repl_metrics['avg_kl']:.4f} | {repl_metrics['ppl']:.2f} | {repl_metrics['next_token_acc']:.4f} |\n\n")
        f.write("## Top Scan Results\n\n")
        f.write("| Layer | Group | KL Zero | KL Mean | KL Bucket | PPL Bucket | Acc Bucket | Recovery |\n")
        f.write("|-------|-------|---------|---------|-----------|------------|------------|----------|\n")
        for r in top_results[:10]:
            f.write(f"| {r['layer']} | {r['group']} | {r['kl_zero']:.4f} | {r.get('kl_mean', 0):.4f} | "
                    f"{r.get('kl_bucket', 0):.4f} | {r.get('ppl_bucket', 0):.2f} | {r.get('acc_bucket', 0):.4f} | "
                    f"{r.get('recovery', 0):.2%} |\n")
        f.write(f"\nGeneration samples saved in [{os.path.basename(gen_path)}]({os.path.basename(gen_path)})\n")
    print(f"  Report saved to {report_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "config": {"layer_ids": layer_ids, "group_size": args.group_size, "num_bins": args.num_bins,
                       "selected_layer": best["layer"], "selected_group": best["group"]},
            "original": orig_metrics,
            "replacement": repl_metrics,
            "generations": {
                "prompts": [p["prompt"] for p in AUTO_PROMPTS],
                "original": clean_gen,
                "replacement": repl_gen,
            },
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    engine.uninstall()
    print("\n" + "=" * 70)
    print("Scaling-R1 Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-LUT Scaling-R1")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name or path")
    parser.add_argument("--layer_percentiles", default="0.25,0.5,0.75",
                        help="Comma-separated depth percentiles for layer selection")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K groups per layer for mean+bucket phase")
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=10,
                        help="Number of generation samples per prompt")
    parser.add_argument("--output_dir", default="results/r2")
    args = parser.parse_args()
    run_scaling_experiment(args)
