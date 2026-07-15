"""
Sequential deployment-aware build for down_proj + o_proj hybrid LUT.

For each layer, in order:
    1. Build o_proj LUT on the current student distribution (if configured).
    2. Install o_proj engine.
    3. Build down_proj LUT on the current student distribution (if configured).
    4. Install down_proj engine.

This guarantees every LUT is built after all upstream replacements that affect
its input have already been deployed.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_lut_sequential.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --down_configs "18:8,19:8,20:8,21:8,22:8,23:8" \
        --o_configs "15:8,16:8,17:8" \
        --address_mode tree --num_bits 10 --tree_candidates 32 --tree_min_samples 32 \
        --calib_size 512 --eval_size 128 \
        --output_root ../v5/outputs_sequential_small
"""

import os
import json
import argparse
import gc
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data import load_jsonl
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup
from engine import HybridPartialEngine, HybridOProjEngine

# Import capture/build helpers from existing scripts
from build_lut import (
    tokenize_texts,
    capture_mlp_residual,
    select_2d_address,
    evaluate_group as evaluate_down_group,
)
from build_lut_o_proj import (
    capture_o_proj_residual,
    evaluate_group as evaluate_o_group,
)


V0_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0", "data")


def parse_configs(arg_str: str) -> List[Tuple[int, int, List[int]]]:
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        layer_id = int(tokens[0])
        # Support both layer:count and layer:count;id1;id2;... (and legacy layer:count:id1;id2)
        rest_after_layer = tokens[1].split(";")
        count = int(rest_after_layer[0])
        if len(rest_after_layer) >= 2:
            group_ids = [int(x) for x in rest_after_layer[1:]]
        elif len(tokens) >= 3:
            group_ids = [int(x) for x in tokens[2].split(";")]
        else:
            group_ids = list(range(count))
        if len(group_ids) != count:
            raise ValueError(f"Config {part}: group id count mismatch")
        configs.append((layer_id, count, group_ids))
    return configs


def parse_o_modes(arg_str: str) -> Dict[int, str]:
    """Parse per-layer o_proj modes, e.g. '15:direct,16:direct,27:delta'."""
    modes = {}
    if not arg_str:
        return modes
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        layer_str, mode = part.split(":")
        if mode not in ("direct", "delta"):
            raise ValueError(f"Unknown o_proj mode: {mode}")
        modes[int(layer_str)] = mode
    return modes


def build_address_from_ckpt(ckpt: dict):
    """Build an address generator from a checkpoint dict."""
    addr_type = ckpt.get("address_type", "tree")
    if addr_type == "2d":
        return Address2D(
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt.get("addr_clip", 3.0),
        )
    if addr_type == "high_order":
        address = AddressHighOrderRandom(
            input_dim=ckpt.get("input_dim", 1),
            num_tables=ckpt["num_tables"],
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
        )
        address.channel_idx = ckpt["channel_idx"]
        address.signs = ckpt["signs"]
        address.input_dim = int(ckpt.get("input_dim", address.channel_idx.max().item() + 1))
        return address
    # tree
    address = AddressGreedyTree(
        input_dim=ckpt.get("input_dim", 1),
        num_bits=ckpt["num_bits"],
        channels_per_bit=ckpt["channels_per_bit"],
        tree_state=ckpt["tree_state"],
    )
    return address


def install_o_proj_engine(model, layer_id, group_ids, save_dir, mode, group_size, device):
    """Install a previously built o_proj engine from checkpoint files."""
    engine = HybridOProjEngine(model, layer_id, group_size=group_size, mode=mode)
    for gid in group_ids:
        ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        address = build_address_from_ckpt(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(device)
        engine.add_group(gid, address, lut_group)
    engine.install()
    return engine


def install_down_proj_engine(model, layer_id, group_ids, save_dir, group_size, device):
    """Install a previously built down_proj engine from checkpoint files."""
    engine = HybridPartialEngine(model, layer_id, group_size=group_size)
    for gid in group_ids:
        ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        address = build_address_from_ckpt(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(device)
        engine.add_group(gid, address, lut_group)
    engine.install()
    return engine


def is_layer_complete(layer_id, group_ids, output_root):
    """Check whether all group checkpoints exist for a layer."""
    if not group_ids:
        return True
    save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", f"g{len(group_ids)}")
    return all(os.path.exists(os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt"))
               for gid in group_ids)


def build_tree_address(calib_x, group_target, num_bits, channels_per_bit, seed,
                       num_candidates, min_samples, max_samples):
    address = AddressGreedyTree(
        input_dim=calib_x.shape[-1],
        num_bits=num_bits,
        channels_per_bit=channels_per_bit,
        seed=seed,
    )
    address.build(
        calib_x,
        group_target,
        num_candidates=num_candidates,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    return address


def build_down_proj_layer(model, tokenizer, layer_id, group_ids, group_size,
                          address_mode, num_bins, num_bits, channels_per_bit,
                          tree_candidates, tree_min_samples, tree_max_samples,
                          calib_texts, eval_texts, max_seq_len, device, output_root):
    """Build down_proj LUTs for a single layer on the current student model."""
    data = capture_mlp_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                                max_seq_len, device)
    calib_x = data["calib_x"]
    calib_down = data["calib_down"]
    eval_x = data["eval_x"]
    eval_down = data["eval_down"]
    hidden_size = data["hidden_size"]

    group_results = []
    save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", f"g{len(group_ids)}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gid in tqdm(group_ids, desc=f"L{layer_id} down_proj groups", leave=False):
        g_start = gid * group_size
        g_end = g_start + group_size
        group_target = calib_down[:, g_start:g_end]

        seed = layer_id * 1000 + gid
        if address_mode == "2d":
            addr_idx, _ = select_2d_address(calib_x, group_target, group_size, num_bins)
            address = Address2D(
                addr_idx=addr_idx,
                addr_mean=calib_x[:, addr_idx].mean(dim=0),
                addr_std=calib_x[:, addr_idx].std(dim=0),
                num_bins=num_bins,
            )
        elif address_mode == "high_order":
            address = AddressHighOrderRandom(
                input_dim=hidden_size,
                num_tables=1,
                num_bits=num_bits,
                channels_per_bit=channels_per_bit,
                seed=seed,
            )
            address.fit_calibration(calib_x.unsqueeze(0))
        else:  # tree
            address = build_tree_address(
                calib_x, group_target, num_bits, channels_per_bit, seed,
                tree_candidates, tree_min_samples, tree_max_samples,
            )

        indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_group = LUTGroup(address.num_tables, address.num_entries, group_size, device=calib_x.device)
        lut_group.initialize_from_calibration(indices, group_target)

        eval_metrics = evaluate_down_group(
            eval_x, eval_down, address, lut_group, gid, group_size, use_residual=True
        )
        print(f"  [L{layer_id} down g{gid:2d}] rel_mse={eval_metrics['relative_mse']:.4f}, "
              f"rmse={eval_metrics['rmse']:.4f}")

        ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
        state = {
            "layer_id": layer_id,
            "group_id": gid,
            "group_size": group_size,
            "use_residual": True,
            "lut_table": lut_group.table.detach().cpu(),
        }
        if isinstance(address, Address2D):
            state.update({
                "address_type": "2d",
                "addr_idx": address.addr_idx,
                "addr_mean": address.addr_mean,
                "addr_std": address.addr_std,
                "num_bins": address.num_bins,
                "addr_clip": address.addr_clip,
            })
        elif isinstance(address, AddressHighOrderRandom):
            state.update({
                "address_type": "high_order",
                "input_dim": address.input_dim,
                "num_tables": address.num_tables,
                "num_bits": address.num_bits,
                "channels_per_bit": address.channels_per_bit,
                "channel_idx": address.channel_idx,
                "signs": address.signs,
                "addr_mean": address.addr_mean,
                "addr_std": address.addr_std,
            })
        elif isinstance(address, AddressGreedyTree):
            state.update({
                "address_type": "tree",
                "input_dim": address.input_dim,
                "num_bits": address.num_bits,
                "channels_per_bit": address.channels_per_bit,
                "tree_state": address.serialize(),
            })
        torch.save(state, ckpt_path)

        group_results.append({"group_id": gid, **eval_metrics})

    return group_results, save_dir


def build_o_proj_layer(model, tokenizer, layer_id, group_ids, group_size,
                       address_mode, num_bins, num_bits, channels_per_bit,
                       tree_candidates, tree_min_samples, tree_max_samples,
                       mode, calib_texts, eval_texts, max_seq_len, device, output_root):
    """Build o_proj LUTs for a single layer on the current student model."""
    data = capture_o_proj_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                                   max_seq_len, device)
    calib_x = data["calib_x"]
    calib_out = data["calib_out"]
    eval_x = data["eval_x"]
    eval_out = data["eval_out"]
    hidden_size = data["hidden_size"]

    if mode == "delta":
        target = calib_out - calib_x
    else:
        target = calib_out

    group_results = []
    save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", f"g{len(group_ids)}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gid in tqdm(group_ids, desc=f"L{layer_id} o_proj groups", leave=False):
        g_start = gid * group_size
        g_end = g_start + group_size
        group_target = target[:, g_start:g_end]

        seed = layer_id * 1000 + gid
        if address_mode == "tree":
            address = build_tree_address(
                calib_x, group_target, num_bits, channels_per_bit, seed,
                tree_candidates, tree_min_samples, tree_max_samples,
            )
        else:
            raise NotImplementedError(f"o_proj address_mode {address_mode} not implemented")

        indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_group = LUTGroup(address.num_tables, address.num_entries, group_size, device=calib_x.device)
        lut_group.initialize_from_calibration(indices, group_target)

        eval_metrics = evaluate_o_group(
            eval_x, eval_out, address, lut_group, gid, group_size, mode
        )
        print(f"  [L{layer_id} o   g{gid:2d}] rel_mse={eval_metrics['relative_mse']:.4f}, "
              f"rmse={eval_metrics['rmse']:.4f}")

        ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
        state = {
            "proj_type": "o_proj",
            "layer_id": layer_id,
            "group_id": gid,
            "group_size": group_size,
            "mode": mode,
            "address_type": "tree",
            "input_dim": hidden_size,
            "num_bits": address.num_bits,
            "channels_per_bit": address.channels_per_bit,
            "tree_state": address.serialize(),
            "lut_table": lut_group.table.detach().cpu(),
        }
        torch.save(state, ckpt_path)

        group_results.append({"group_id": gid, **eval_metrics})

    return group_results, save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--down_configs", default="",
                        help="layer:count[:group_ids] for down_proj")
    parser.add_argument("--o_configs", default="",
                        help="layer:count[:group_ids] for o_proj")
    parser.add_argument("--address_mode", default="tree", choices=["2d", "high_order", "tree"])
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--num_bits", type=int, default=10)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="../v5/outputs_sequential")
    parser.add_argument("--tree_candidates", type=int, default=32)
    parser.add_argument("--tree_min_samples", type=int, default=32)
    parser.add_argument("--tree_max_samples", type=int, default=16384)
    parser.add_argument("--o_mode", default="direct", choices=["direct", "delta"],
                        help="Default o_proj reconstruction mode")
    parser.add_argument("--o_modes", default="",
                        help="Per-layer o_proj modes, e.g. '15:direct,16:direct,27:delta'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints; skip completed layers")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    down_configs = parse_configs(args.down_configs) if args.down_configs else []
    o_configs = parse_configs(args.o_configs) if args.o_configs else []

    # Determine layer order
    down_by_layer = {lid: gids for lid, _, gids in down_configs}
    o_by_layer = {lid: gids for lid, _, gids in o_configs}
    o_modes = parse_o_modes(args.o_modes)
    all_layers = sorted(set(down_by_layer.keys()) | set(o_by_layer.keys()))

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_texts = load_jsonl(os.path.join(V0_DATA_DIR, "calib.jsonl"))[:args.calib_size]
    eval_texts = load_jsonl(os.path.join(V0_DATA_DIR, "eval.jsonl"))[:args.eval_size]

    output_root = args.output_root
    Path(output_root).mkdir(parents=True, exist_ok=True)

    summary_path = os.path.join(output_root, "summary.json")
    installed_engines = []
    down_results = []
    o_results = []

    if args.resume and os.path.exists(summary_path):
        print(f"Resuming from {summary_path}")
        with open(summary_path, "r") as f:
            prev = json.load(f)
        down_results = prev.get("down_results", [])
        o_results = prev.get("o_results", [])

    for layer_id in all_layers:
        print(f"\n{'='*60}")
        print(f"[Layer {layer_id}] Sequential build")
        print(f"{'='*60}")

        # Within a layer: o_proj first, then down_proj
        if layer_id in o_by_layer:
            group_ids = o_by_layer[layer_id]
            mode = o_modes.get(layer_id, args.o_mode)
            save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", f"g{len(group_ids)}")

            if args.resume and is_layer_complete(layer_id, group_ids, output_root):
                print(f"  Resuming o_proj groups {group_ids} ({mode}) from {save_dir}")
                engine = install_o_proj_engine(model, layer_id, group_ids, save_dir, mode, args.group_size, device)
                installed_engines.append(engine)
                print(f"  Installed o_proj engine for L{layer_id}")
            else:
                print(f"  Building o_proj groups {group_ids} ({args.address_mode}, {mode})")
                layer_results, save_dir = build_o_proj_layer(
                    model, tokenizer, layer_id, group_ids, args.group_size,
                    args.address_mode, args.num_bins, args.num_bits, args.channels_per_bit,
                    args.tree_candidates, args.tree_min_samples, args.tree_max_samples,
                    mode, calib_texts, eval_texts, args.max_seq_len, device,
                    output_root,
                )
                # Remove any previous result for this layer to keep summary clean on re-run
                o_results = [r for r in o_results if r["layer_id"] != layer_id]
                o_results.append({
                    "layer_id": layer_id,
                    "group_ids": group_ids,
                    "mode": mode,
                    "save_dir": save_dir,
                    "groups": layer_results,
                })

                engine = install_o_proj_engine(model, layer_id, group_ids, save_dir, mode, args.group_size, device)
                installed_engines.append(engine)
                print(f"  Installed o_proj engine for L{layer_id}")
            gc.collect()
            torch.cuda.empty_cache()

        if layer_id in down_by_layer:
            group_ids = down_by_layer[layer_id]
            save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", f"g{len(group_ids)}")

            if args.resume and is_layer_complete(layer_id, group_ids, output_root):
                print(f"  Resuming down_proj groups {group_ids} from {save_dir}")
                engine = install_down_proj_engine(model, layer_id, group_ids, save_dir, args.group_size, device)
                installed_engines.append(engine)
                print(f"  Installed down_proj engine for L{layer_id}")
            else:
                print(f"  Building down_proj groups {group_ids} ({args.address_mode})")
                layer_results, save_dir = build_down_proj_layer(
                    model, tokenizer, layer_id, group_ids, args.group_size,
                    args.address_mode, args.num_bins, args.num_bits, args.channels_per_bit,
                    args.tree_candidates, args.tree_min_samples, args.tree_max_samples,
                    calib_texts, eval_texts, args.max_seq_len, device,
                    output_root,
                )
                down_results = [r for r in down_results if r["layer_id"] != layer_id]
                down_results.append({
                    "layer_id": layer_id,
                    "group_ids": group_ids,
                    "save_dir": save_dir,
                    "groups": layer_results,
                })

                engine = install_down_proj_engine(model, layer_id, group_ids, save_dir, args.group_size, device)
                installed_engines.append(engine)
                print(f"  Installed down_proj engine for L{layer_id}")
            gc.collect()
            torch.cuda.empty_cache()

    # Uninstall all engines after build (checkpoints already saved)
    for engine in installed_engines:
        engine.uninstall()

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "address_mode": args.address_mode,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "group_size": args.group_size,
            "down_configs": down_configs,
            "o_configs": o_configs,
            "o_mode": args.o_mode,
            "down_results": down_results,
            "o_results": o_results,
        }, f, indent=2)

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
