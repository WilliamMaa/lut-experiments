"""
Joint evaluation for down_proj + o_proj hybrid LUT replacement (no fine-tuning).

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval_joint.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --down_configs "21:8,22:8,23:8" \
        --down_checkpoint_root ../v5/outputs_tree_21_23 \
        --o_configs "17:8" \
        --o_checkpoint_root ../v5/outputs_o_proj_l17 \
        --eval_size 128 --max_seq_len 512 \
        --output_dir results/eval_joint_l21_23_l17
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from engine import HybridPartialEngine, HybridOProjEngine
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup
from utils import load_model_and_data
from metrics import compute_model_metrics, compute_baseline_probs, format_bytes


def parse_configs(arg_str: str) -> List[Tuple[int, int]]:
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        layer_str, count_str = part.split(":")
        configs.append((int(layer_str), int(count_str)))
    return configs


def build_address(ckpt: dict):
    """Build an address generator from a checkpoint dict."""
    if ckpt["address_type"] == "2d":
        address = Address2D(
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt.get("addr_clip", 3.0),
        )
    elif ckpt["address_type"] == "high_order":
        address = AddressHighOrderRandom(
            input_dim=1,
            num_tables=ckpt["num_tables"],
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
        )
        address.channel_idx = ckpt["channel_idx"]
        address.signs = ckpt["signs"]
        address.input_dim = int(ckpt.get("input_dim", ckpt["channel_idx"].max().item() + 1))
    elif ckpt["address_type"] == "tree":
        address = AddressGreedyTree(
            input_dim=1,
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            tree_state=ckpt["tree_state"],
        )

        def max_ch(node):
            if "leaf_index" in node:
                return 0
            return max(max(node["channel_idx"]) + 1, max_ch(node["left"]), max_ch(node["right"]))

        address.input_dim = int(ckpt.get("input_dim", max_ch(ckpt["tree_state"]["tree"])))
    else:
        raise ValueError(f"Unknown address type: {ckpt['address_type']}")
    return address


def build_down_engine_for_layer(model, layer_id: int, group_count: int,
                                checkpoint_root: str, group_size: int) -> HybridPartialEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No down_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    engine = HybridPartialEngine(model, layer_id, group_size=group_size)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def build_o_proj_engine_for_layer(model, layer_id: int, group_count: int,
                                  checkpoint_root: str, group_size: int) -> HybridOProjEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No o_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    first_ckpt = torch.load(paths[0], map_location="cpu")
    mode = first_ckpt.get("mode", "direct")
    engine = HybridOProjEngine(model, layer_id, group_size=group_size, mode=mode)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--down_configs", required=True,
                        help="Comma-separated layer:count for down_proj, e.g. '21:8,22:8,23:8'")
    parser.add_argument("--down_checkpoint_root", default="../v5/outputs_tree_21_23")
    parser.add_argument("--o_configs", required=True,
                        help="Comma-separated layer:count for o_proj, e.g. '17:8'")
    parser.add_argument("--o_checkpoint_root", default="../v5/outputs_o_proj_l17")
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--output_dir", default="results/eval_joint_down_o")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolate_gpu", action="store_true")
    args = parser.parse_args()

    if args.isolate_gpu and args.device.startswith("cuda:"):
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    down_configs = parse_configs(args.down_configs)
    o_configs = parse_configs(args.o_configs)

    print("=" * 70)
    print("v5 Joint down_proj + o_proj Evaluation (no fine-tuning)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"down_proj configs: {down_configs}")
    print(f"o_proj configs: {o_configs}")
    print("=" * 70)

    print("\n[1/3] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, eval_size=args.eval_size, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, device_str=args.device, calib_size=0,
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    print("\n[2/3] Building hybrid LUT engines...")
    down_engines = []
    for layer_id, group_count in down_configs:
        engine = build_down_engine_for_layer(
            model, layer_id, group_count, args.down_checkpoint_root, args.group_size
        )
        down_engines.append(engine)

    o_engines = []
    for layer_id, group_count in o_configs:
        engine = build_o_proj_engine_for_layer(
            model, layer_id, group_count, args.o_checkpoint_root, args.group_size
        )
        o_engines.append(engine)

    print("\n[3/3] Evaluating...")

    # Original model baseline
    print("  Original model (no replacement)...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)
        original_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
    print(f"    KL={original_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={original_metrics['ppl']:.2f}, Acc={original_metrics['next_token_acc']:.4f}")

    # Joint replacement
    print("  Joint replacement (down_proj + o_proj installed)...")
    for engine in down_engines + o_engines:
        engine.install()

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            joint_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
    print(f"    KL={joint_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={joint_metrics['ppl']:.2f}, Acc={joint_metrics['next_token_acc']:.4f}")

    for engine in down_engines + o_engines:
        engine.uninstall()

    # MAC reduction
    per_layer_total = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    full_model_total = num_layers * per_layer_total
    down_eliminated = sum(count * args.group_size * intermediate_size for _, count in down_configs)
    o_eliminated = sum(count * args.group_size * hidden_size for _, count in o_configs)
    mac_reduction = (down_eliminated + o_eliminated) / full_model_total

    # LUT storage
    total_bytes = 0
    for layer_id, group_count in down_configs:
        ckpt_dir = os.path.join(args.down_checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2
    for layer_id, group_count in o_configs:
        ckpt_dir = os.path.join(args.o_checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2

    summary = {
        "model": args.model,
        "down_configs": down_configs,
        "o_configs": o_configs,
        "mac_reduction_ratio": mac_reduction,
        "lut_storage_bytes": total_bytes,
        "lut_storage_human": format_bytes(total_bytes),
        "original": {
            "kl": original_metrics.get("avg_kl", 0.0),
            "ppl": original_metrics["ppl"],
            "acc": original_metrics["next_token_acc"],
        },
        "joint_replacement": {
            "kl": joint_metrics.get("avg_kl", 0.0),
            "ppl": joint_metrics["ppl"],
            "acc": joint_metrics["next_token_acc"],
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("v5 JOINT EVALUATION COMPLETE")
    print("=" * 70)
    print(f"MAC reduction ratio: {mac_reduction*100:.3f}%")
    print(f"LUT storage: {format_bytes(total_bytes)}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
