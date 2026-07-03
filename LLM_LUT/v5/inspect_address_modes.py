"""
Compare 2D channel address vs high-order random address for down_proj LUT reconstruction.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inspect_address_modes.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layers "21,22,23" --num_groups 8 \
        --max_seq_len 512 --calib_size 256 --eval_size 128
"""

import os
import json
import argparse
from typing import List

import torch

from build_lut import capture_mlp_residual, select_2d_address, evaluate_group, parse_configs
from address import Address2D, AddressHighOrderRandom
from lut import LUTGroup
from transformers import AutoModelForCausalLM, AutoTokenizer


V0_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0", "data")


def build_and_evaluate(calib_x, calib_down, eval_x, eval_down, group_ids, group_size,
                       address_mode, num_bins, num_tables, num_bits, channels_per_bit,
                       use_residual):
    target = calib_down - calib_x if use_residual else calib_down
    results = []
    for gid in group_ids:
        g_start = gid * group_size
        group_target = target[:, g_start:g_start + group_size]

        if address_mode == "2d":
            if gid == group_ids[0]:
                addr_idx, _ = select_2d_address(calib_x, target, group_size, num_bins)
                layer_addr_idx = addr_idx
            else:
                addr_idx = layer_addr_idx
            address = Address2D(
                addr_idx,
                calib_x[:, addr_idx].mean(dim=0),
                calib_x[:, addr_idx].std(dim=0),
                num_bins=num_bins,
            )
        else:
            seed = 1000 + gid
            address = AddressHighOrderRandom(
                input_dim=calib_x.shape[-1],
                num_tables=num_tables,
                num_bits=num_bits,
                channels_per_bit=channels_per_bit,
                seed=seed,
            )
            address.fit_calibration(calib_x.unsqueeze(0))

        indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_group = LUTGroup(address.num_tables, address.num_entries, group_size, device=calib_x.device)
        lut_group.initialize_from_calibration(indices, group_target)

        metrics = evaluate_group(eval_x, eval_down, address, lut_group, gid, group_size, use_residual)
        results.append({"group_id": gid, **metrics})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", default="21,22,23",
                        help="Comma-separated layer IDs to inspect")
    parser.add_argument("--num_groups", type=int, default=8,
                        help="Replace first N groups in each layer")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins_2d", type=int, default=64)
    parser.add_argument("--num_tables", type=int, default=4)
    parser.add_argument("--num_bits", type=int, default=10)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default="results/address_mode_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    layers = [int(x.strip()) for x in args.layers.split(",")]
    group_ids = list(range(args.num_groups))

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from data import load_jsonl
    calib_texts = load_jsonl(os.path.join(V0_DATA_DIR, "calib.jsonl"))[:args.calib_size]
    eval_texts = load_jsonl(os.path.join(V0_DATA_DIR, "eval.jsonl"))[:args.eval_size]

    output = {"model": args.model, "layers": []}

    for layer_id in layers:
        print(f"\n[Layer {layer_id}] Comparing address modes...")
        data = capture_mlp_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                                    args.max_seq_len, device)
        calib_x = data["calib_x"]
        calib_down = data["calib_down"]
        eval_x = data["eval_x"]
        eval_down = data["eval_down"]

        res_2d = build_and_evaluate(
            calib_x, calib_down, eval_x, eval_down, group_ids, args.group_size,
            "2d", args.num_bins_2d, 1, 0, 0, use_residual=True
        )
        res_ho = build_and_evaluate(
            calib_x, calib_down, eval_x, eval_down, group_ids, args.group_size,
            "high_order", 0, args.num_tables, args.num_bits, args.channels_per_bit,
            use_residual=True
        )

        avg_2d = sum(r["relative_mse"] for r in res_2d) / len(res_2d)
        avg_ho = sum(r["relative_mse"] for r in res_ho) / len(res_ho)
        print(f"  2D        avg rel_mse={avg_2d:.4f}")
        print(f"  high_order avg rel_mse={avg_ho:.4f} (M={args.num_tables}, B={args.num_bits})")

        output["layers"].append({
            "layer_id": layer_id,
            "2d": {"avg_relative_mse": avg_2d, "groups": res_2d},
            "high_order": {"avg_relative_mse": avg_ho, "groups": res_ho},
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved comparison to {args.output_path}")


if __name__ == "__main__":
    main()
