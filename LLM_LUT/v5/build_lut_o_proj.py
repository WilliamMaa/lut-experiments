"""
Build v5 hybrid LUT checkpoints for o_proj partial replacement.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_lut_o_proj.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --configs "27:8" \
        --address_mode tree --num_bits 10 --tree_candidates 128 --tree_min_samples 32 \
        --mode delta \
        --output_root ../v5/outputs_o_proj

--mode can be:
    direct: LUT predicts o_proj(x) directly
    delta:  LUT predicts (o_proj(x) - x), reconstruction is x + lut
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data import load_jsonl
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup


V0_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0", "data")


def tokenize_texts(tokenizer, texts, max_seq_len):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )


def _run_forward_batches(model, tokens, batch_size, device):
    """Run model forward in small batches to reduce peak GPU memory."""
    n = tokens["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokens.items()}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(**batch)


def capture_o_proj_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                            max_seq_len, device, batch_size=32):
    """Capture o_proj input (attn_output) and o_proj output for a layer."""
    layer = model.model.layers[layer_id]
    o_proj = layer.self_attn.o_proj
    hidden_size = o_proj.weight.shape[0]

    captured = {"calib_x": [], "calib_out": [], "eval_x": [], "eval_out": []}

    def make_hook(key_x, key_out):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            captured[key_x].append(x.detach())
            captured[key_out].append(output.detach())
        return hook

    calib_texts_plain = [t["text"] if isinstance(t, dict) else t for t in calib_texts]
    eval_texts_plain = [t["text"] if isinstance(t, dict) else t for t in eval_texts]
    calib_tok = tokenize_texts(tokenizer, calib_texts_plain, max_seq_len)
    eval_tok = tokenize_texts(tokenizer, eval_texts_plain, max_seq_len)

    handle = o_proj.register_forward_hook(make_hook("calib_x", "calib_out"))
    try:
        model.eval()
        _run_forward_batches(model, calib_tok, batch_size, device)
        handle.remove()
        handle = o_proj.register_forward_hook(make_hook("eval_x", "eval_out"))
        _run_forward_batches(model, eval_tok, batch_size, device)
    finally:
        handle.remove()

    def flatten(seq):
        return torch.cat([t.view(-1, t.shape[-1]) for t in seq], dim=0)

    return {
        "calib_x": flatten(captured["calib_x"]),
        "calib_out": flatten(captured["calib_out"]),
        "eval_x": flatten(captured["eval_x"]),
        "eval_out": flatten(captured["eval_out"]),
        "hidden_size": hidden_size,
    }


def evaluate_group(eval_x, eval_out, address, lut_group, gid, group_size, mode):
    with torch.no_grad():
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_out = lut_group(indices)
        g_start = gid * group_size
        if mode == "delta":
            lut_out = lut_out + eval_x[:, g_start:g_start + group_size]
        true = eval_out[:, g_start:g_start + group_size]
        mse = F.mse_loss(lut_out, true.float(), reduction="mean").item()
        var = true.var().item()
        return {
            "mse": mse,
            "rmse": mse ** 0.5,
            "relative_mse": mse / (var + 1e-8),
        }


def parse_configs(arg_str: str) -> List[Tuple[int, int, List[int]]]:
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        layer_id = int(tokens[0])
        count = int(tokens[1])
        if len(tokens) >= 3:
            group_ids = [int(x) for x in tokens[2].split(";")]
        else:
            group_ids = list(range(count))
        if len(group_ids) != count:
            raise ValueError(f"Config {part}: group id count mismatch")
        configs.append((layer_id, count, group_ids))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--configs", required=True,
                        help="layer:count[:group_ids] e.g. '27:8' or '27:4:0;1;2;3'")
    parser.add_argument("--address_mode", default="tree", choices=["2d", "high_order", "tree"])
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--num_tables", type=int, default=1)
    parser.add_argument("--num_bits", type=int, default=10)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="../v5/outputs_o_proj")
    parser.add_argument("--mode", default="delta", choices=["direct", "delta"],
                        help="direct: LUT predicts o_proj(x); delta: LUT predicts o_proj(x) - x")
    parser.add_argument("--tree_candidates", type=int, default=128,
                        help="Random projection candidates per split (tree mode)")
    parser.add_argument("--tree_min_samples", type=int, default=32,
                        help="Min samples to split a node (tree mode)")
    parser.add_argument("--tree_max_samples", type=int, default=65536,
                        help="Subsample calibration data for tree building")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    configs = parse_configs(args.configs)

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

    all_results = []
    for layer_id, count, group_ids in configs:
        print(f"\n[Layer {layer_id}] Building o_proj LUTs for groups {group_ids} ({args.address_mode}, {args.mode})")
        data = capture_o_proj_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                                       args.max_seq_len, device)
        calib_x = data["calib_x"]
        calib_out = data["calib_out"]
        eval_x = data["eval_x"]
        eval_out = data["eval_out"]
        hidden_size = data["hidden_size"]

        if args.mode == "delta":
            target = calib_out - calib_x
        else:
            target = calib_out

        layer_results = {"layer_id": layer_id, "mode": args.mode, "groups": []}
        save_dir = os.path.join(args.output_root, "checkpoints", f"l{layer_id}", f"g{count}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for gid in tqdm(group_ids, desc=f"L{layer_id} groups"):
            g_start = gid * args.group_size
            g_end = g_start + args.group_size
            group_target = target[:, g_start:g_end]

            if args.address_mode == "2d":
                raise NotImplementedError("2D address for o_proj not yet implemented")
            elif args.address_mode == "high_order":
                raise NotImplementedError("High-order address for o_proj not yet implemented")
            else:  # tree
                seed = layer_id * 1000 + gid
                address = AddressGreedyTree(
                    input_dim=hidden_size,
                    num_bits=args.num_bits,
                    channels_per_bit=args.channels_per_bit,
                    seed=seed,
                )
                address.build(calib_x, group_target,
                              num_candidates=args.tree_candidates,
                              min_samples=args.tree_min_samples,
                              max_samples=args.tree_max_samples)

            indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
            lut_group = LUTGroup(address.num_tables, address.num_entries, args.group_size, device=calib_x.device)
            lut_group.initialize_from_calibration(indices, group_target)

            eval_metrics = evaluate_group(
                eval_x, eval_out, address, lut_group, gid, args.group_size, args.mode
            )
            print(f"  group {gid:2d}: rel_mse={eval_metrics['relative_mse']:.4f}, "
                  f"rmse={eval_metrics['rmse']:.4f}")

            ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
            state = {
                "proj_type": "o_proj",
                "layer_id": layer_id,
                "group_id": gid,
                "group_size": args.group_size,
                "mode": args.mode,
                "address_type": "tree",
                "input_dim": hidden_size,
                "num_bits": address.num_bits,
                "channels_per_bit": address.channels_per_bit,
                "tree_state": address.serialize(),
                "lut_table": lut_group.table.detach().cpu(),
            }
            torch.save(state, ckpt_path)

            layer_results["groups"].append({"group_id": gid, **eval_metrics})
        all_results.append(layer_results)

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "proj_type": "o_proj",
            "address_mode": args.address_mode,
            "mode": args.mode,
            "group_size": args.group_size,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "configs": configs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
