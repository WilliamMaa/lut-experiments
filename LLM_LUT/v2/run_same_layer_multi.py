"""Same-Layer Multi-Group Generation Evaluation.

Evaluates cumulative replacement of multiple groups within the SAME layer.
Useful for verifying that low-KL same-layer groups compose safely.

Usage:
    python run_same_layer_multi.py --model Qwen/Qwen2.5-3B-Instruct \
        --layer 27 --groups "29,4,15" --output_dir results/3B_l27_multi
"""

import os
import sys
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
from train import collect_teacher_targets, build_joint_bucket_table
from metrics import compute_baseline_probs, compute_model_metrics
from r2_auto_eval import generate_outputs, AUTO_PROMPTS
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


def run_same_layer_experiment(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    group_ids = [int(g.strip()) for g in args.groups.split(",")]
    layer_id = args.layer

    print("=" * 70)
    print(f"Same-Layer Multi-Group Generation: {args.model_name}")
    print(f"Layer: {layer_id}, Groups: {group_ids}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model_name, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )

    # 2. Calibrate target layer
    print(f"\n[2/3] Calibrating layer {layer_id}...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=(layer_id,),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,
        heads=2,
    )

    # 3. Build all replacements
    print(f"\n[3/4] Building {len(group_ids)} replacements...")
    engines = []
    for gid in group_ids:
        print(f"  L{layer_id} G{gid}...")
        calib = calib_results[(layer_id, "mlp_delta")]
        addr_idx = calib["addr_idx"][gid]
        addr_mean = calib["addr_mean"][gid]
        addr_std = calib["addr_std"][gid]

        bin_idx, targets, _ = collect_teacher_targets(
            model, calib_loader, layer_id, "mlp_delta", gid, args.group_size,
            addr_idx, addr_mean, addr_std, num_bins=args.num_bins,
        )
        joint_table = build_joint_bucket_table(bin_idx, targets, args.num_bins, args.group_size)

        engine = ReplacementEngine(
            model=model, layer_id=layer_id, group_id=gid,
            group_size=args.group_size, addr_idx=addr_idx, addr_mean=addr_mean,
            addr_std=addr_std, table=joint_table, num_bins=args.num_bins,
        )
        engines.append(engine)

    # 4. Metrics: Original
    print("\n[4/4] Evaluating metrics...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    orig_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Original:      PPL={orig_metrics['ppl']:.2f}  Acc={orig_metrics['next_token_acc']:.4f}")

    # 5. Metrics: Single-group
    engines[0].install()
    single_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Single G{group_ids[0]}:  KL={single_metrics['avg_kl']:.4f}  PPL={single_metrics['ppl']:.2f}  Acc={single_metrics['next_token_acc']:.4f}")
    engines[0].uninstall()

    # 6. Metrics: Multi-group
    for e in engines:
        e.install()
    multi_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Multi {group_ids}: KL={multi_metrics['avg_kl']:.4f}  PPL={multi_metrics['ppl']:.2f}  Acc={multi_metrics['next_token_acc']:.4f}")

    # 7. Generate: Original (no hooks)
    print("\n[Gen] Original (no hooks)...")
    orig_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    # 8. Generate: First group only
    print(f"[Gen] Single-group L{layer_id} G{group_ids[0]}...")
    engines[0].install()
    single_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    engines[0].uninstall()

    # 9. Generate: All groups
    print(f"[Gen] Multi-group L{layer_id} {group_ids}...")
    for e in engines:
        e.install()
    multi_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    for e in engines:
        e.uninstall()

    # 10. Save report with metrics + generations
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Same-Layer Multi-Group: {args.model_name}\n\n")
        f.write(f"- Layer: {layer_id}, Groups: {group_ids}\n")
        f.write(f"- Group size: {args.group_size}, Bins: {args.num_bins}×{args.num_bins}\n\n")

        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Original | 0.0000 | {orig_metrics['ppl']:.2f} | {orig_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Single G{group_ids[0]} | {single_metrics['avg_kl']:.4f} | {single_metrics['ppl']:.2f} | {single_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| Multi {group_ids} | {multi_metrics['avg_kl']:.4f} | {multi_metrics['ppl']:.2f} | {multi_metrics['next_token_acc']:.4f} |\n\n")

        f.write("## Generation Samples\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"### Prompt {idx+1}: {item['prompt']}\n\n")

            f.write("**Original (no hook)**\n\n")
            for i, text in enumerate(orig_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")

            f.write(f"**Single-group L{layer_id} G{group_ids[0]}**\n\n")
            for i, text in enumerate(single_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")

            f.write(f"**Multi-group L{layer_id} {group_ids}**\n\n")
            for i, text in enumerate(multi_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")
    print(f"  Report saved to {report_path}")

    # 11. Save JSON
    json_path = os.path.join(args.output_dir, "results.json")
    import json
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "layer": layer_id,
            "groups": group_ids,
            "original": orig_metrics,
            "single": single_metrics,
            "multi": multi_metrics,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    # 12. Save checkpoints
    for i, gid in enumerate(group_ids):
        ckpt_path = os.path.join(args.output_dir, f"replacement_l{layer_id}g{gid}.pt")
        engines[i].save(ckpt_path)

    print("\n" + "=" * 70)
    print("Same-Layer Multi-Group Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Same-Layer Multi-Group Generation")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--groups", default="29,4,15",
                        help="Comma-separated group IDs within the same layer (e.g. '29,4,15')")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="results/same_layer_multi")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()
    run_same_layer_experiment(args)
