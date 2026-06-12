"""LLM-LUT Multi-Group Replacement.

Builds and evaluates simultaneous replacement of multiple groups.
Can be same layer or different layers.

Usage:
    # 1.5B multi-group (top-3 from scan)
    python run_multi_group.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --groups "21,16;21,20;14,8" --output_dir results/multi_1.5b

    # 3B multi-group
    python run_multi_group.py --model Qwen/Qwen2.5-3B-Instruct \
        --groups "TBD" --output_dir results/multi_3b
"""

import os
import sys
import json
import argparse
import torch

os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

from transformers import AutoModelForCausalLM, AutoTokenizer

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
V1_DIR = os.path.join(os.path.dirname(__file__), "..", "v1")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V1_DIR)

from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics
from train import collect_teacher_targets, build_joint_bucket_table
from r2_auto_eval import generate_outputs, AUTO_PROMPTS
from multi_group import MultiGroupEngine
from r1_replacement import ReplacementEngine


def load_model_and_data(model_name, calib_size, eval_size, max_seq_len, batch_size, device_str="cuda:0"):
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, "bfloat16", torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True, device_map=device_str)
    except Exception as e:
        print(f"[WARN] Failed to load with {dtype}: {e}. Falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map=device_str)

    # model already on device via device_map
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


def parse_groups(groups_str):
    """Parse 'layer,group;layer,group;...' into list of tuples."""
    result = []
    for part in groups_str.split(";"):
        part = part.strip()
        if not part:
            continue
        layer, group = part.split(",")
        result.append((int(layer.strip()), int(group.strip())))
    return result


def run_multi_group_experiment(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    group_configs = parse_groups(args.groups)

    print("=" * 70)
    print(f"LLM-LUT Multi-Group Replacement: {args.model_name}")
    print(f"Groups: {group_configs}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model_name, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )

    # 2. Calibrate all needed layers
    layer_ids = sorted(set(l for l, _ in group_configs))
    print(f"\n[2/4] Calibrating layers: {layer_ids}...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=tuple(layer_ids),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,
        heads=2,
    )

    # 3. Build replacement for each group
    print(f"\n[3/4] Building replacement for {len(group_configs)} groups...")
    multi_engine = MultiGroupEngine()
    for layer_id, group_id in group_configs:
        print(f"  Building L{layer_id} G{group_id}...")
        calib = calib_results[(layer_id, "mlp_delta")]
        addr_idx = calib["addr_idx"][group_id]
        addr_mean = calib["addr_mean"][group_id]
        addr_std = calib["addr_std"][group_id]

        bin_idx, targets, _ = collect_teacher_targets(
            model, calib_loader, layer_id, "mlp_delta", group_id, args.group_size,
            addr_idx, addr_mean, addr_std, num_bins=args.num_bins,
        )
        joint_table = build_joint_bucket_table(bin_idx, targets, args.num_bins, args.group_size)

        engine = ReplacementEngine(
            model=model, layer_id=layer_id, group_id=group_id,
            group_size=args.group_size, addr_idx=addr_idx, addr_mean=addr_mean,
            addr_std=addr_std, table=joint_table, num_bins=args.num_bins,
        )
        multi_engine.add(engine)

    # 4. Evaluate
    print("\n[4/4] Evaluating...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    orig_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Original: PPL={orig_metrics['ppl']:.2f} Acc={orig_metrics['next_token_acc']:.4f}")

    # Single-group baseline (first group only)
    multi_engine.engines[0].install()
    single_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Single-group ({group_configs[0]}): KL={single_metrics['avg_kl']:.4f} PPL={single_metrics['ppl']:.2f} Acc={single_metrics['next_token_acc']:.4f}")
    multi_engine.engines[0].uninstall()

    # Multi-group
    multi_engine.install_all()
    multi_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Multi-group: KL={multi_metrics['avg_kl']:.4f} PPL={multi_metrics['ppl']:.2f} Acc={multi_metrics['next_token_acc']:.4f}")

    # Generation samples
    multi_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    multi_engine.uninstall_all()
    clean_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    # Save generations
    gen_path = os.path.join(args.output_dir, "generations.md")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Group Generation Samples: {args.model_name}\n\n")
        f.write(f"- Groups: {group_configs}\n")
        f.write(f"- {args.gen_samples} samples per prompt\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"## Prompt {idx+1}: {item['prompt']}\n\n")
            f.write("### Original (no hook)\n\n")
            for i, text in enumerate(clean_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("### Multi-group replacement\n\n")
            for i, text in enumerate(multi_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")
    print(f"  Generation samples saved to {gen_path}")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "multi_group.pt")
    multi_engine.save(ckpt_path)

    # Report
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# LLM-LUT Multi-Group: {args.model_name}\n\n")
        f.write(f"- Groups: {group_configs}\n")
        f.write(f"- Group size: {args.group_size}, Bins: {args.num_bins}×{args.num_bins}\n\n")
        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Original | 0.0000 | {orig_metrics['ppl']:.2f} | {orig_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Single-group | {single_metrics['avg_kl']:.4f} | {single_metrics['ppl']:.2f} | {single_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Multi-group | {multi_metrics['avg_kl']:.4f} | {multi_metrics['ppl']:.2f} | {multi_metrics['next_token_acc']:.4f} |\n")
        f.write(f"\nGeneration samples saved in [{os.path.basename(gen_path)}]({os.path.basename(gen_path)})\n")
    print(f"  Report saved to {report_path}")

    # JSON
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "groups": group_configs,
            "original": orig_metrics,
            "single": single_metrics,
            "multi": multi_metrics,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    print("\n" + "=" * 70)
    print("Multi-Group Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-LUT Multi-Group Replacement")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--groups", default="21,16;21,20;14,8",
                        help="Semicolon-separated list of layer,group pairs (e.g. '21,16;21,20;14,8')")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="results/multi_group")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()
    run_multi_group_experiment(args)
