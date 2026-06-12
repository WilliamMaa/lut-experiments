"""V3 Phase 1 Validation: Numerical equivalence of partial linear vs functional hook.

Loads existing v2 replacement checkpoints, builds a V3PartialEngine,
and verifies that metrics (PPL/Acc/KL) and generation match the v2 results.

Usage:
    cd v3
    python run_v3_validation.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint_dir ../v2/results/7B_l21_6group \
        --output_dir results/v3_7B_l21
"""

import os
import sys
import json
import argparse
import glob
import torch

os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
V2_DIR = os.path.join(os.path.dirname(__file__), "..", "v2")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V2_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from data import prepare_data, load_jsonl, TextDataset
from metrics import compute_baseline_probs, compute_model_metrics
from r2_auto_eval import generate_outputs, AUTO_PROMPTS

from partial_linear import V3PartialEngine


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
            print(f"[WARN] GPU {i} has allocated memory; proceeding because device_map={device_str} is explicit single-GPU.")

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


def load_checkpoints_into_engine(checkpoint_dir, engine):
    """Load all replacement_*.pt checkpoints from directory into engine."""
    pattern = os.path.join(checkpoint_dir, "replacement_l*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No replacement_*.pt found in {checkpoint_dir}")

    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        gid = ckpt["group_id"]
        engine.add_group(
            group_id=gid,
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            table=ckpt["table"],
        )
        print(f"  Loaded L{ckpt['layer_id']} G{gid} from {os.path.basename(path)}")


def run_v3_validation(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"V3 Phase 1 Validation: {args.model_name}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model_name, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )

    # 2. Build V3 engine from checkpoints
    print("\n[2/4] Loading replacement checkpoints...")
    sample_ckpt = torch.load(sorted(glob.glob(os.path.join(args.checkpoint_dir, "replacement_l*.pt")))[0], map_location="cpu")
    layer_id = sample_ckpt["layer_id"]
    group_size = sample_ckpt["group_size"]
    num_bins = sample_ckpt["num_bins"]

    engine = V3PartialEngine(model, layer_id=layer_id, group_size=group_size, num_bins=num_bins)
    load_checkpoints_into_engine(args.checkpoint_dir, engine)

    # 3. Evaluate baseline
    print("\n[3/4] Evaluating baseline (no replacement)...")
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Baseline: PPL={baseline_metrics['ppl']:.2f} Acc={baseline_metrics['next_token_acc']:.4f}")

    # 4. Evaluate V3 partial (with engine safely wrapped)
    print("\n[4/4] Evaluating V3 partial skip...")
    engine.install()
    try:
        v3_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        engine.uninstall()
    print(f"  V3 Partial: KL={v3_metrics['avg_kl']:.4f} PPL={v3_metrics['ppl']:.2f} Acc={v3_metrics['next_token_acc']:.4f}")

    # 5. Generation samples (baseline first, then v3 with engine installed)
    print("\n[Gen] Collecting generation samples...")
    baseline_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    engine.install()
    try:
        v3_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    finally:
        engine.uninstall()

    # 6. Save report
    report_path = os.path.join(args.output_dir, "v3_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# V3 Phase 1 Validation: {args.model_name}\n\n")
        f.write(f"- Layer: {layer_id}, Groups: {sorted(engine.group_configs.keys())}\n")
        f.write(f"- Group size: {group_size}, Bins: {num_bins}×{num_bins}\n\n")

        f.write("## Metrics\n\n")
        f.write("| Version | KL | PPL | Acc |\n")
        f.write("|---------|-----|-----|-----|\n")
        f.write(f"| Baseline | 0.0000 | {baseline_metrics['ppl']:.2f} | {baseline_metrics['next_token_acc']:.4f} |\n")
        f.write(f"| V3 Partial | {v3_metrics['avg_kl']:.4f} | {v3_metrics['ppl']:.2f} | {v3_metrics['next_token_acc']:.4f} |\n\n")

        delta_ppl = v3_metrics['ppl'] - baseline_metrics['ppl']
        delta_acc = v3_metrics['next_token_acc'] - baseline_metrics['next_token_acc']
        f.write(f"**Δ PPL**: {delta_ppl:+.2f}  ")
        f.write(f"**Δ Acc**: {delta_acc:+.4f}  ")
        f.write(f"**KL**: {v3_metrics['avg_kl']:.4f}\n\n")

        f.write("## Generation Comparison\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"### Prompt {idx+1}: {item['prompt']}\n\n")
            f.write("**Baseline**\n\n")
            for i, text in enumerate(baseline_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("**V3 Partial**\n\n")
            for i, text in enumerate(v3_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")

    print(f"  Report saved to {report_path}")

    # 7. Save JSON
    json_path = os.path.join(args.output_dir, "v3_validation_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "layer": layer_id,
            "groups": sorted(engine.group_configs.keys()),
            "baseline": baseline_metrics,
            "v3_partial": v3_metrics,
        }, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")

    # 8. Numerical sanity check
    print("\n" + "=" * 70)
    print("Numerical Sanity Check")
    print("=" * 70)
    kl = v3_metrics['avg_kl']
    ppl_delta = abs(v3_metrics['ppl'] - baseline_metrics['ppl'])
    acc_delta = abs(v3_metrics['next_token_acc'] - baseline_metrics['next_token_acc'])

    PASS = True
    if kl > 0.5:
        print(f"  [FAIL] KL too high: {kl:.4f} (threshold: 0.5)")
        PASS = False
    else:
        print(f"  [PASS] KL = {kl:.4f}")

    if ppl_delta > 2.0:
        print(f"  [FAIL] PPL delta too large: {ppl_delta:.2f} (threshold: 2.0)")
        PASS = False
    else:
        print(f"  [PASS] PPL delta = {ppl_delta:.2f}")

    if acc_delta > 0.02:
        print(f"  [FAIL] Acc delta too large: {acc_delta:.4f} (threshold: 0.02)")
        PASS = False
    else:
        print(f"  [PASS] Acc delta = {acc_delta:.4f}")

    if PASS:
        print("\n  >>> V3 Phase 1 PASSED: Numerical equivalence confirmed.")
        print("  >>> Proceed to Triton kernel (Phase 2).")
    else:
        print("\n  >>> V3 Phase 1 FAILED: Check partial linear implementation.")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Phase 1 Validation")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing replacement_*.pt checkpoints from v2")
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128,
                        help="Eval samples for metrics (default 128 for speed)")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=5,
                        help="Generation samples per prompt (default 5 for speed)")
    parser.add_argument("--output_dir", default="results/v3_validation")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()
    run_v3_validation(args)
