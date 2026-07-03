"""
Build v5 hybrid LUT checkpoints for down_proj partial replacement.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_lut.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --configs "21:8,22:8" \
        --address_mode high_order --num_tables 4 --num_bits 10 \
        --output_root ../v5/outputs

Or use the classic 2D address:
    python build_lut.py --configs "21:8" --address_mode 2d --num_bins 64
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import itertools

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data import load_jsonl
from address import Address2D, AddressHighOrderRandom
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


def capture_mlp_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                         max_seq_len, device):
    """Capture MLP input (residual) and down_proj output for a layer."""
    layer = model.model.layers[layer_id]
    mlp = layer.mlp
    down_proj = mlp.down_proj
    hidden_size = down_proj.weight.shape[0]
    intermediate_size = down_proj.weight.shape[1]

    captured = {"x": [], "down_out": []}

    def mlp_hook(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        captured["x"].append(x.detach())
        captured["down_out"].append(output.detach())

    handle = mlp.register_forward_hook(mlp_hook)
    calib_tok = tokenize_texts(tokenizer, calib_texts, max_seq_len)
    eval_tok = tokenize_texts(tokenizer, eval_texts, max_seq_len)

    try:
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(**{k: v.to(device) for k, v in calib_tok.items()})
                _ = model(**{k: v.to(device) for k, v in eval_tok.items()})
    finally:
        handle.remove()

    def flatten(seq):
        return torch.cat([t.view(-1, t.shape[-1]) for t in seq], dim=0)

    return {
        "calib_x": flatten(captured["x"][0::2]),
        "calib_down": flatten(captured["down_out"][0::2]),
        "eval_x": flatten(captured["x"][1::2]),
        "eval_down": flatten(captured["down_out"][1::2]),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def select_2d_address(calib_x, target, group_size, num_bins, num_candidates=8):
    """Pick a good 2-channel address pair using the first group target."""
    hidden_size = calib_x.shape[-1]
    channel_var = calib_x.var(dim=0)
    top_channels = torch.topk(channel_var, k=min(num_candidates * 2, hidden_size)).indices.tolist()
    pairs = list(itertools.combinations(top_channels[:num_candidates], 2))[:20]

    out_group = target[:, :group_size]
    best_rmse = float("inf")
    best_pair = None
    for c1, c2 in pairs:
        addr_idx = torch.tensor([c1, c2], device=calib_x.device)
        addr = Address2D(addr_idx,
                         calib_x[:, addr_idx].mean(dim=0),
                         calib_x[:, addr_idx].std(dim=0),
                         num_bins=num_bins)
        indices = addr.compute_indices(calib_x.unsqueeze(0)).view(-1, 1)  # [N,1]
        lut = LUTGroup(1, addr.num_entries, group_size)
        lut.initialize_from_calibration(indices, out_group)
        rec = lut(indices).squeeze(1)
        rmse = F.mse_loss(rec, out_group.float(), reduction="mean").item() ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_pair = (c1, c2)
    if best_pair is None:
        best_pair = (0, 1)
    return torch.tensor(best_pair, device=calib_x.device), best_rmse


def evaluate_group(eval_x, eval_down, address, lut_group, group_id, group_size, use_residual):
    with torch.no_grad():
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
        rec = lut_group(indices)
        g_start = group_id * group_size
        if use_residual:
            rec = rec + eval_x[:, g_start:g_start + group_size]
        true = eval_down[:, g_start:g_start + group_size]
        mse = F.mse_loss(rec, true.float(), reduction="mean").item()
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
                        help="layer:count[:group_ids] e.g. '21:8' or '21:4:0;1;2;3'")
    parser.add_argument("--address_mode", default="high_order", choices=["2d", "high_order"])
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--num_tables", type=int, default=4)
    parser.add_argument("--num_bits", type=int, default=10)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="../v5/outputs")
    parser.add_argument("--use_residual", action="store_true", default=True,
                        help="LUT stores down_proj_output - residual (default True)")
    parser.add_argument("--no_residual", dest="use_residual", action="store_false")
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
        print(f"\n[Layer {layer_id}] Building LUTs for groups {group_ids} ({args.address_mode})")
        data = capture_mlp_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                                    args.max_seq_len, device)
        calib_x = data["calib_x"]
        calib_down = data["calib_down"]
        eval_x = data["eval_x"]
        eval_down = data["eval_down"]
        hidden_size = data["hidden_size"]

        target = calib_down - calib_x if args.use_residual else calib_down

        layer_results = {"layer_id": layer_id, "groups": []}
        save_dir = os.path.join(args.output_root, "checkpoints", f"l{layer_id}", f"g{count}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for gid in group_ids:
            g_start = gid * args.group_size
            g_end = g_start + args.group_size
            group_target = target[:, g_start:g_end]

            if args.address_mode == "2d":
                if gid == group_ids[0]:
                    addr_idx, _ = select_2d_address(
                        calib_x, target, args.group_size, args.num_bins
                    )
                    layer_addr_idx = addr_idx
                else:
                    addr_idx = layer_addr_idx
                address = Address2D(
                    addr_idx,
                    calib_x[:, addr_idx].mean(dim=0),
                    calib_x[:, addr_idx].std(dim=0),
                    num_bins=args.num_bins,
                )
            else:
                seed = layer_id * 1000 + gid
                address = AddressHighOrderRandom(
                    input_dim=hidden_size,
                    num_tables=args.num_tables,
                    num_bits=args.num_bits,
                    channels_per_bit=args.channels_per_bit,
                    seed=seed,
                )
                address.fit_calibration(calib_x.unsqueeze(0))

            indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
            lut_group = LUTGroup(address.num_tables, address.num_entries, args.group_size)
            lut_group.initialize_from_calibration(indices, group_target)

            eval_metrics = evaluate_group(
                eval_x, eval_down, address, lut_group, gid, args.group_size, args.use_residual
            )
            print(f"  group {gid:2d}: rel_mse={eval_metrics['relative_mse']:.4f}, "
                  f"rmse={eval_metrics['rmse']:.4f}")

            ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
            state = {
                "layer_id": layer_id,
                "group_id": gid,
                "group_size": args.group_size,
                "use_residual": args.use_residual,
                "lut_table": lut_group.table.detach().cpu(),
            }
            if isinstance(address, Address2D):
                state["address_type"] = "2d"
                state["addr_idx"] = address.addr_idx
                state["addr_mean"] = address.addr_mean
                state["addr_std"] = address.addr_std
                state["num_bins"] = address.num_bins
                state["addr_clip"] = address.addr_clip
            else:
                state["address_type"] = "high_order"
                state["num_tables"] = address.num_tables
                state["num_bits"] = address.num_bits
                state["channels_per_bit"] = address.channels_per_bit
                state["channel_idx"] = address.channel_idx
                state["signs"] = address.signs
                state["addr_mean"] = address.addr_mean
                state["addr_std"] = address.addr_std
            torch.save(state, ckpt_path)

            layer_results["groups"].append({"group_id": gid, **eval_metrics})
        all_results.append(layer_results)

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "address_mode": args.address_mode,
            "group_size": args.group_size,
            "num_bins": args.num_bins,
            "num_tables": args.num_tables,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "configs": configs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
