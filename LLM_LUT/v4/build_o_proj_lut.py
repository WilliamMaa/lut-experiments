"""
Build residual LUT checkpoints for partial o_proj replacement.

Usage:
    cd LLM_LUT/v4
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_o_proj_lut.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --configs "17:4,15:4,16:4,20:4" \
        --calib_size 256 --eval_size 128 \
        --output_root ../v3/o_proj_outputs

Configs can also explicitly select group ids:
    --configs "17:4:0,1,2,3"

If group ids are omitted, the first `count` groups (0..count-1) are used.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


V0_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0", "data")


def load_jsonl(path: str) -> List[Dict]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if "text" in obj else obj["prompt"])
    return texts


def tokenize_texts(tokenizer, texts: List[str], max_seq_len: int):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )


def compute_address_bins(x: torch.Tensor, addr_idx: torch.Tensor,
                         num_bins: int = 64, addr_clip: float = 3.0) -> torch.Tensor:
    """
    Args:
        x: [B, S, hidden_size]
        addr_idx: [2] channel indices
    Returns:
        bin_idx: [B, S, 2]
    """
    addr = x.index_select(-1, addr_idx.to(x.device))
    mean = addr.mean(dim=(0, 1), keepdim=True)
    std = addr.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    z = (addr - mean) / std
    z = z.clamp(-addr_clip, addr_clip)
    qf = (z + addr_clip) / (2.0 * addr_clip) * (num_bins - 1)
    return torch.round(qf).long().clamp(0, num_bins - 1)


def select_address_channels(inputs: torch.Tensor, targets: torch.Tensor,
                            group_size: int, num_bins: int,
                            num_candidates: int = 8) -> Tuple[torch.Tensor, float]:
    """Pick a good address channel pair using the first group of targets."""
    hidden_size = inputs.shape[-1]
    channel_var = inputs.var(dim=0)
    top_channels = torch.topk(channel_var, k=min(num_candidates * 2, hidden_size)).indices.tolist()

    import itertools
    pairs = list(itertools.combinations(top_channels[:num_candidates], 2))[:20]

    best_rmse = float("inf")
    best_pair = None

    out_group = targets[:, :group_size]
    for c1, c2 in pairs:
        addr_idx = torch.tensor([c1, c2], device=inputs.device)
        try:
            bins = compute_address_bins(inputs, addr_idx, num_bins)
            flat = bins[:, 0] * num_bins + bins[:, 1]
            cells = num_bins * num_bins
            table = torch.zeros(cells, group_size, device=inputs.device, dtype=torch.float32)
            counts = torch.zeros(cells, device=inputs.device, dtype=torch.float32)
            flat_exp = flat.unsqueeze(1).expand(-1, group_size)
            table.scatter_add_(0, flat_exp, out_group.float())
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
            table = table / counts.clamp_min(1.0).unsqueeze(1)
            lut = table[flat]
            rmse = F.mse_loss(lut, out_group.float(), reduction="mean").item() ** 0.5
            if rmse < best_rmse:
                best_rmse = rmse
                best_pair = (c1, c2)
        except Exception:
            continue
    if best_pair is None:
        best_pair = (0, 1)
    return torch.tensor(best_pair, device=inputs.device), best_rmse


def build_group_table(inputs: torch.Tensor, targets: torch.Tensor,
                      group_id: int, group_size: int, num_bins: int,
                      addr_idx: torch.Tensor) -> torch.Tensor:
    """Build [num_bins, num_bins, group_size] table for one group."""
    g_start = group_id * group_size
    g_end = g_start + group_size
    out_group = targets[:, g_start:g_end]

    bins = compute_address_bins(inputs, addr_idx, num_bins)
    flat = bins[:, 0] * num_bins + bins[:, 1]
    cells = num_bins * num_bins

    table = torch.zeros(cells, group_size, device=inputs.device, dtype=torch.float32)
    counts = torch.zeros(cells, device=inputs.device, dtype=torch.float32)
    flat_exp = flat.unsqueeze(1).expand(-1, group_size)
    table.scatter_add_(0, flat_exp, out_group.float())
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    table = table / counts.clamp_min(1.0).unsqueeze(1)
    return table.view(num_bins, num_bins, group_size)


def evaluate_group_reconstruction(inputs: torch.Tensor, outputs: torch.Tensor,
                                  table: torch.Tensor, group_id: int, group_size: int,
                                  addr_idx: torch.Tensor, use_residual: bool) -> Dict:
    """Return MSE/RMSE for one group on eval data."""
    g_start = group_id * group_size
    g_end = g_start + group_size
    out_group = outputs[:, g_start:g_end]
    in_group = inputs[:, g_start:g_end]

    bins = compute_address_bins(inputs, addr_idx, table.shape[0])
    flat = bins[:, 0] * table.shape[0] + bins[:, 1]
    lut = table.view(-1, group_size)[flat]
    if use_residual:
        rec = in_group.float() + lut.float()
    else:
        rec = lut
    mse = F.mse_loss(rec, out_group.float(), reduction="mean").item()
    var = out_group.var().item()
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "relative_mse": mse / (var + 1e-8),
    }


def capture_o_proj_data(model, tokenizer, layer_id: int, calib_texts: List[str],
                        eval_texts: List[str], max_seq_len: int, device) -> Dict:
    """Capture o_proj inputs/outputs for calib and eval."""
    layer = model.model.layers[layer_id]
    o_proj = layer.self_attn.o_proj
    hidden_size = o_proj.weight.shape[1]

    captured = {"input": [], "output": []}

    def hook(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        captured["input"].append(inp.detach())
        captured["output"].append(output.detach())

    handle = o_proj.register_forward_hook(hook)
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
        return torch.cat([x.view(-1, hidden_size) for x in seq], dim=0)

    return {
        "calib_inputs": flatten(captured["input"][0::2]),
        "calib_outputs": flatten(captured["output"][0::2]),
        "eval_inputs": flatten(captured["input"][1::2]),
        "eval_outputs": flatten(captured["output"][1::2]),
    }


def parse_o_proj_configs(arg_str: str) -> List[Tuple[int, int, List[int]]]:
    """Parse '17:4,15:4:0,1,2,3' -> [(layer, count, [group_ids]), ...]."""
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        layer_id = int(tokens[0].strip())
        count = int(tokens[1].strip())
        if len(tokens) >= 3:
            group_ids = [int(x.strip()) for x in tokens[2].split(";")]
        else:
            group_ids = list(range(count))
        if len(group_ids) != count:
            raise ValueError(f"Config {part}: group id count {len(group_ids)} != {count}")
        configs.append((layer_id, count, group_ids))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--configs", required=True,
                        help="Comma-separated layer:count[:group_ids], e.g. '17:4,15:4:0,1,2,3'")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="../v3/o_proj_outputs")
    parser.add_argument("--addr_candidates", type=int, default=8,
                        help="Number of high-variance channel candidates to try for address selection.")
    parser.add_argument("--per_group_addr", action="store_true",
                        help="Select a different address channel pair for each group (slower but potentially better).")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    configs = parse_o_proj_configs(args.configs)

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
        print(f"\n[Layer {layer_id}] Building o_proj LUTs for groups {group_ids}")
        data = capture_o_proj_data(
            model, tokenizer, layer_id, calib_texts, eval_texts,
            args.max_seq_len, device
        )
        calib_inputs = data["calib_inputs"]
        calib_outputs = data["calib_outputs"]
        eval_inputs = data["eval_inputs"]
        eval_outputs = data["eval_outputs"]

        # Residual target: what the LUT has to add to the input.
        calib_targets = calib_outputs - calib_inputs

        layer_results = {"layer_id": layer_id, "groups": []}
        save_dir = os.path.join(args.output_root, "checkpoints", f"l{layer_id}", f"g{count}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for gid in group_ids:
            if args.per_group_addr:
                g_start = gid * args.group_size
                g_end = g_start + args.group_size
                addr_idx, _ = select_address_channels(
                    calib_inputs, calib_targets[:, g_start:g_end],
                    args.group_size, args.num_bins, num_candidates=args.addr_candidates
                )
            else:
                # Use first selected group to pick a single address pair for the layer.
                if gid == group_ids[0]:
                    addr_idx, _ = select_address_channels(
                        calib_inputs, calib_targets,
                        args.group_size, args.num_bins, num_candidates=args.addr_candidates
                    )
                    layer_addr_idx = addr_idx
                else:
                    addr_idx = layer_addr_idx

            table = build_group_table(
                calib_inputs, calib_targets, gid, args.group_size, args.num_bins, addr_idx
            )

            eval_metrics = evaluate_group_reconstruction(
                eval_inputs, eval_outputs, table, gid, args.group_size, addr_idx, use_residual=True
            )
            print(f"  group {gid:2d}: rel_mse={eval_metrics['relative_mse']:.4f}, "
                  f"rmse={eval_metrics['rmse']:.4f}")

            ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
            torch.save({
                "addr_idx": addr_idx.cpu(),
                "addr_mean": calib_inputs[:, addr_idx].mean(dim=0).cpu(),
                "addr_std": calib_inputs[:, addr_idx].std(dim=0).cpu(),
                "table": table.cpu(),
                "eval_metrics": eval_metrics,
            }, ckpt_path)

            layer_results["groups"].append({"group_id": gid, **eval_metrics})
        all_results.append(layer_results)

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "group_size": args.group_size,
            "num_bins": args.num_bins,
            "configs": configs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
