"""Multi-Layer Multi-Group Replacement.

Supports arbitrary cross-layer combinations. Each layer can have multiple groups.

Usage:
    # L27 (3 groups) + L21 (2 groups)
    python run_multi_layer.py --model Qwen/Qwen2.5-3B-Instruct \
        --config "27:29,4,15;21:16,20" --output_dir results/3B_l27_l21

    # L27 only (same as run_same_layer_multi.py)
    python run_multi_layer.py --model Qwen/Qwen2.5-3B-Instruct \
        --config "27:29,4,15" --output_dir results/3B_l27_only
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

from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics
from train import collect_teacher_targets, build_joint_bucket_table
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


def parse_config(config_str):
    """Parse 'layer:g1,g2;layer:g1,g2' into dict.
    Returns: {layer_id: [group_ids]}
    """
    result = {}
    for part in config_str.split(";"):
        part = part.strip()
        if not part:
            continue
        layer_part, groups_part = part.split(":")
        layer_id = int(layer_part.strip())
        group_ids = [int(g.strip()) for g in groups_part.split(",")]
        result[layer_id] = group_ids
    return result


def run_multi_layer_experiment(args):
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)
    config = parse_config(args.config)

    print("=" * 70)
    print(f"Multi-Layer Multi-Group: {args.model_name}")
    for lid, gids in config.items():
        print(f"  Layer {lid}: Groups {gids}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model_name, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size
    )

    # 2. Calibrate all layers
    layer_ids = sorted(config.keys())
    print(f"\n[2/4] Calibrating layers: {layer_ids}...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=tuple(layer_ids),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,
        heads=2,
    )

    # 3. Build all replacements
    print(f"\n[3/4] Building replacements...")
    all_engines = []  # list of (layer_id, group_id, engine)
    for layer_id, group_ids in config.items():
        calib = calib_results[(layer_id, "mlp_delta")]
        for gid in group_ids:
            print(f"  L{layer_id} G{gid}...")
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
            all_engines.append((layer_id, gid, engine))

    # 4. Evaluate metrics
    print("\n[4/4] Evaluating metrics...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    orig_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Original:  PPL={orig_metrics['ppl']:.2f}  Acc={orig_metrics['next_token_acc']:.4f}")

    # Single-layer baselines: first group of each layer
    single_metrics = {}
    for layer_id, group_ids in config.items():
        first_gid = group_ids[0]
        # find engine
        for lid, gid, engine in all_engines:
            if lid == layer_id and gid == first_gid:
                engine.install()
                m = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
                engine.uninstall()
                single_metrics[layer_id] = m
                print(f"  Single L{layer_id} G{first_gid}: KL={m['avg_kl']:.4f}  PPL={m['ppl']:.2f}  Acc={m['next_token_acc']:.4f}")
                break

    # Per-layer multi-group (all groups in one layer)
    per_layer_multi = {}
    for layer_id, group_ids in config.items():
        for lid, gid, engine in all_engines:
            if lid == layer_id:
                engine.install()
        m = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
        for lid, gid, engine in all_engines:
            if lid == layer_id:
                engine.uninstall()
        per_layer_multi[layer_id] = m
        print(f"  Multi  L{layer_id} {group_ids}: KL={m['avg_kl']:.4f}  PPL={m['ppl']:.2f}  Acc={m['next_token_acc']:.4f}")

    # All layers combined
    for _, _, engine in all_engines:
        engine.install()
    all_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    for _, _, engine in all_engines:
        engine.uninstall()
    print(f"  All layers combined: KL={all_metrics['avg_kl']:.4f}  PPL={all_metrics['ppl']:.2f}  Acc={all_metrics['next_token_acc']:.4f}")

    # 5. Generations
    print("\n[Gen] Original...")
    orig_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    # Per-layer generation (first group only as baseline)
    layer_gen = {}
    for layer_id, group_ids in config.items():
        first_gid = group_ids[0]
        for lid, gid, engine in all_engines:
            if lid == layer_id and gid == first_gid:
                engine.install()
                layer_gen[layer_id] = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
                engine.uninstall()
                break

    # All combined
    print("[Gen] All layers combined...")
    for _, _, engine in all_engines:
        engine.install()
    all_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    for _, _, engine in all_engines:
        engine.uninstall()

    # 6. Save report
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Layer Multi-Group: {args.model_name}\n\n")
        f.write(f"- Config: {config}\n")
        f.write(f"- Group size: {args.group_size}, Bins: {args.num_bins}×{args.num_bins}\n\n")

        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Original | 0.0000 | {orig_metrics['ppl']:.2f} | {orig_metrics['next_token_acc']:.4f} |\n")
        for layer_id in layer_ids:
            m = single_metrics[layer_id]
            f.write(f"| Single L{layer_id} | {m['avg_kl']:.4f} | {m['ppl']:.2f} | {m['next_token_acc']:.4f} |\n")
        for layer_id in layer_ids:
            m = per_layer_multi[layer_id]
            f.write(f"| Multi L{layer_id} | {m['avg_kl']:.4f} | {m['ppl']:.2f} | {m['next_token_acc']:.4f} |\n")
        f.write(f"| All Combined | {all_metrics['avg_kl']:.4f} | {all_metrics['ppl']:.2f} | {all_metrics['next_token_acc']:.4f} |\n\n")

        f.write("## Generation Samples\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"### Prompt {idx+1}: {item['prompt']}\n\n")

            f.write("**Original (no hook)**\n\n")
            for i, text in enumerate(orig_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")

            for layer_id in layer_ids:
                f.write(f"**Single L{layer_id} G{config[layer_id][0]}**\n\n")
                for i, text in enumerate(layer_gen[layer_id][idx]):
                    f.write(f"{i+1}. {text}\n\n")

            f.write(f"**All layers combined**\n\n")
            for i, text in enumerate(all_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")
    print(f"  Report saved to {report_path}")

    # 7. Save JSON
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "config": config,
            "original": orig_metrics,
            "single_per_layer": {str(k): v for k, v in single_metrics.items()},
            "multi_per_layer": {str(k): v for k, v in per_layer_multi.items()},
            "all_combined": all_metrics,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    # 8. Save checkpoints
    for layer_id, gid, engine in all_engines:
        ckpt_path = os.path.join(args.output_dir, f"replacement_l{layer_id}g{gid}.pt")
        engine.save(ckpt_path)

    print("\n" + "=" * 70)
    print("Multi-Layer Experiment Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Layer Multi-Group Replacement")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--config", required=True,
                        help="Layer:group config, e.g. '27:29,4,15;21:16,20'")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="results/multi_layer")
    args = parser.parse_args()
    run_multi_layer_experiment(args)
