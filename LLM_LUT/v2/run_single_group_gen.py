"""Single-Group Generation Evaluation.

Run generation comparison for ONE specific (layer, group) pair.
Useful for diagnostic checks like L27 G29 language-drift baseline.

Usage:
    python run_single_group_gen.py --model Qwen/Qwen2.5-3B-Instruct \
        --layer 27 --group 29 --output_dir results/3B_l27g29
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


def load_model_and_data(model_name, calib_size, max_seq_len, batch_size, device_str="cuda:0"):
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
    prepare_data(tokenizer, calib_path, eval_path, calib_size=calib_size, eval_size=1, max_seq_len=max_seq_len)
    calib_texts = load_jsonl(calib_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=batch_size, shuffle=False)

    return model, tokenizer, calib_loader


def run_single_group_generation(args):
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Single-Group Generation: {args.model_name}")
    print(f"Target: Layer {args.layer}, Group {args.group}")
    print("=" * 70)

    # 1. Load
    print("\n[1/3] Loading model and data...")
    model, tokenizer, calib_loader = load_model_and_data(
        args.model_name, args.calib_size, args.max_seq_len, args.batch_size
    )

    # 2. Calibrate target layer
    print(f"\n[2/3] Calibrating layer {args.layer}...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=(args.layer,),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,
        heads=2,
    )

    # 3. Build replacement
    print(f"\n[3/3] Building replacement for L{args.layer} G{args.group}...")
    calib = calib_results[(args.layer, "mlp_delta")]
    addr_idx = calib["addr_idx"][args.group]
    addr_mean = calib["addr_mean"][args.group]
    addr_std = calib["addr_std"][args.group]

    bin_idx, targets, _ = collect_teacher_targets(
        model, calib_loader, args.layer, "mlp_delta", args.group, args.group_size,
        addr_idx, addr_mean, addr_std, num_bins=args.num_bins,
    )
    joint_table = build_joint_bucket_table(bin_idx, targets, args.num_bins, args.group_size)

    engine = ReplacementEngine(
        model=model, layer_id=args.layer, group_id=args.group,
        group_size=args.group_size, addr_idx=addr_idx, addr_mean=addr_mean,
        addr_std=addr_std, table=joint_table, num_bins=args.num_bins,
    )

    # 4. Generate: Original
    print("\n[Eval] Generating ORIGINAL outputs...")
    orig_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)

    # 5. Generate: Replacement
    print("[Eval] Generating REPLACEMENT outputs...")
    engine.install()
    repl_gen = generate_outputs(model, tokenizer, AUTO_PROMPTS, num_samples=args.gen_samples, max_new_tokens=128, device=device)
    engine.uninstall()

    # 6. Save generations
    gen_path = os.path.join(args.output_dir, "generations.md")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(f"# Single-Group Generation Samples: {args.model_name}\n\n")
        f.write(f"- Layer: {args.layer}, Group: {args.group}\n")
        f.write(f"- {args.gen_samples} samples per prompt\n\n")
        for idx, item in enumerate(AUTO_PROMPTS):
            f.write(f"## Prompt {idx+1}: {item['prompt']}\n\n")
            f.write("### Original (no hook)\n\n")
            for i, text in enumerate(orig_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("### Replacement (with hook)\n\n")
            for i, text in enumerate(repl_gen[idx]):
                f.write(f"{i+1}. {text}\n\n")
            f.write("---\n\n")
    print(f"  Generation samples saved to {gen_path}")

    # 7. Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "replacement.pt")
    engine.save(ckpt_path)

    print("\n" + "=" * 70)
    print("Single-Group Generation Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Group Generation Eval")
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="results/single_group")
    args = parser.parse_args()
    run_single_group_generation(args)
