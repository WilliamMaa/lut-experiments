"""
Measure layerwise hidden-state drift between original model and a partially
replaced model (down_proj / o_proj LUT).

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python measure_layerwise_drift.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --down_configs "15:12,...,27:12" \
        --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
        --o_configs "15:8,16:8,17:8,27:8" \
        --o_checkpoint_root ../v5/outputs_o_proj_exp \
        --eval_size 128 --max_seq_len 512 \
        --output_json results/drift_exp_v1.json
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine import HybridPartialEngine, HybridOProjEngine
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup
from utils import load_model_and_data


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


def build_down_engine(model, layer_id, group_count, checkpoint_root, group_size):
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No down_proj checkpoints for L{layer_id} G{group_count}")
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


def build_o_engine(model, layer_id, group_count, checkpoint_root, group_size):
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No o_proj checkpoints for L{layer_id} G{group_count}")
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


def measure_pair(teacher_model, student_model, eval_loader, output_json):
    """Measure per-layer hidden-state drift between teacher and student."""
    device = teacher_model.device
    num_layers = len(teacher_model.model.layers)

    sum_err = [0.0] * num_layers
    sum_sq_err = [0.0] * num_layers
    max_err = [0.0] * num_layers
    total_tokens = [0] * num_layers

    teacher_hooks = []
    student_hooks = []
    teacher_hidden = {}
    student_hidden = {}

    def make_hook(store, layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            store[layer_idx] = h.detach()
        return hook

    for i in range(num_layers):
        teacher_hooks.append(teacher_model.model.layers[i].register_forward_hook(make_hook(teacher_hidden, i)))
        student_hooks.append(student_model.model.layers[i].register_forward_hook(make_hook(student_hidden, i)))

    teacher_model.eval()
    student_model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Measure drift", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            teacher_hidden.clear()
            student_hidden.clear()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _ = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                _ = student_model(input_ids=input_ids, attention_mask=attention_mask)

            for i in range(num_layers):
                h_t = teacher_hidden[i].float()
                h_s = student_hidden[i].float()
                mask = attention_mask[:, 1:].bool() if attention_mask is not None else torch.ones(
                    h_t.shape[0], h_t.shape[1], dtype=torch.bool, device=device
                )
                # h shape [B, S, D]; mask [B, S]
                h_t_flat = h_t[mask]
                h_s_flat = h_s[mask]

                diff_norm = torch.norm(h_s_flat - h_t_flat, p=2, dim=-1)
                teacher_norm = torch.norm(h_t_flat, p=2, dim=-1).clamp_min(1e-8)
                rel_err = (diff_norm / teacher_norm).cpu()

                n = rel_err.numel()
                if n == 0:
                    continue
                sum_err[i] += rel_err.sum().item()
                sum_sq_err[i] += (rel_err ** 2).sum().item()
                max_err[i] = max(max_err[i], rel_err.max().item())
                total_tokens[i] += n

    for h in teacher_hooks:
        h.remove()
    for h in student_hooks:
        h.remove()

    result = []
    for i in range(num_layers):
        if total_tokens[i] > 0:
            mean = sum_err[i] / total_tokens[i]
            var = sum_sq_err[i] / total_tokens[i] - mean ** 2
            std = var ** 0.5 if var > 0 else 0.0
        else:
            mean = std = max_err[i] = 0.0
        result.append({
            "layer": i,
            "mean_relative_error": round(mean, 6),
            "std_relative_error": round(std, 6),
            "max_relative_error": round(max_err[i], 6),
            "total_tokens": total_tokens[i],
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\nLayerwise drift:")
    for r in result:
        print(f"  L{r['layer']:2d}: mean={r['mean_relative_error']:.4f}, "
              f"std={r['std_relative_error']:.4f}, max={r['max_relative_error']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--down_configs", default="",
                        help="Comma-separated layer:count for down_proj, e.g. '21:8,22:8,23:8'")
    parser.add_argument("--down_checkpoint_root", default="../v5/outputs_tree_21_23")
    parser.add_argument("--o_configs", default="",
                        help="Comma-separated layer:count for o_proj, e.g. '17:8'")
    parser.add_argument("--o_checkpoint_root", default="../v5/outputs_o_proj_l17")
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--output_json", default="results/drift.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolate_gpu", action="store_true")
    args = parser.parse_args()

    if args.isolate_gpu and args.device.startswith("cuda:"):
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    down_configs = parse_configs(args.down_configs) if args.down_configs else []
    o_configs = parse_configs(args.o_configs) if args.o_configs else []

    print("Loading teacher model...")
    teacher, _, _, eval_loader = load_model_and_data(
        args.model, eval_size=args.eval_size, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, device_str=args.device, calib_size=0,
    )

    print("Loading student model...")
    student, _, _, _ = load_model_and_data(
        args.model, eval_size=args.eval_size, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, device_str=args.device, calib_size=0,
    )

    print("Building engines for student...")
    for layer_id, group_count in down_configs:
        engine = build_down_engine(student, layer_id, group_count, args.down_checkpoint_root, args.group_size)
        engine.install()
    for layer_id, group_count in o_configs:
        engine = build_o_engine(student, layer_id, group_count, args.o_checkpoint_root, args.group_size)
        engine.install()

    measure_pair(teacher, student, eval_loader, args.output_json)

    for layer_id, group_count in down_configs:
        # uninstall is handled per engine; we didn't keep refs, but can uninstall all by finding hooks?
        pass


if __name__ == "__main__":
    main()
