"""
Inspect whether o_proj is a viable LUT replacement target.

This is a standalone probe: for each candidate layer, we collect o_proj
inputs/outputs on a small calibration set, build a simple 2D LUT per output
group (using 2 input channels as address), and measure the reconstruction
error on an eval set.

Unlike down_proj, o_proj output is not close to its input, so we store the
full output value in the LUT table instead of a residual delta.

Usage:
    cd LLM_LUT/v3
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inspect_o_proj_lut.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layers "15,16,17,18,19,20,21,22,23,24,25,26,27" \
        --group_size 64 --num_bins 64 \
        --calib_size 256 --eval_size 128 \
        --output_path results/o_proj_lut_inspection.json
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


# Reuse v0 data location.
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


def prepare_texts(tokenizer, max_seq_len: int, calib_size: int, eval_size: int):
    calib_path = os.path.join(V0_DATA_DIR, "calib.jsonl")
    eval_path = os.path.join(V0_DATA_DIR, "eval.jsonl")

    if not os.path.exists(calib_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"{calib_path} or {eval_path} not found. Run v0 data preparation first."
        )

    calib_texts = load_jsonl(calib_path)[:calib_size]
    eval_texts = load_jsonl(eval_path)[:eval_size]

    def tokenize(texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
        )

    return tokenize(calib_texts), tokenize(eval_texts)


def compute_address_bins(x: torch.Tensor, addr_idx: torch.Tensor,
                         num_bins: int = 64, addr_clip: float = 3.0) -> torch.Tensor:
    """
    Args:
        x: [B, S, hidden_size]
        addr_idx: [2] channel indices
    Returns:
        bin_idx: [B, S, 2]
    """
    addr = x.index_select(-1, addr_idx.to(x.device))  # [B, S, 2]
    mean = addr.mean(dim=(0, 1), keepdim=True)
    std = addr.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    z = (addr - mean) / std
    z = z.clamp(-addr_clip, addr_clip)
    qf = (z + addr_clip) / (2.0 * addr_clip) * (num_bins - 1)
    bin_idx = torch.round(qf).long().clamp(0, num_bins - 1)
    return bin_idx


def build_lut_for_layer(inputs: torch.Tensor, outputs: torch.Tensor,
                        group_size: int, num_bins: int, addr_idx: torch.Tensor) -> torch.Tensor:
    """
    Build a 2D LUT table for one layer's o_proj.

    Args:
        inputs:  [N, hidden_size]  (flattened tokens)
        outputs: [N, hidden_size]
        group_size: output channels per group
        num_bins: LUT bins per address dim
        addr_idx: [2] input channel indices used as address
    Returns:
        table: [num_bins, num_bins, hidden_size]
    """
    hidden_size = outputs.shape[-1]
    bin_idx = compute_address_bins(inputs, addr_idx, num_bins)  # [N, 2]
    flat_idx = bin_idx[:, 0] * num_bins + bin_idx[:, 1]  # [N]

    num_cells = num_bins * num_bins
    table = torch.zeros(num_cells, hidden_size, device=outputs.device, dtype=outputs.dtype)
    counts = torch.zeros(num_cells, device=outputs.device, dtype=torch.float32)

    # Vectorized accumulation: scatter_add over all tokens.
    flat_idx_exp = flat_idx.unsqueeze(1).expand(-1, hidden_size)
    table.scatter_add_(0, flat_idx_exp, outputs)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

    counts = counts.clamp_min(1.0).unsqueeze(1)
    table = table / counts
    return table.view(num_bins, num_bins, hidden_size)


def evaluate_lut_reconstruction(inputs: torch.Tensor, outputs: torch.Tensor,
                                table: torch.Tensor, group_size: int,
                                addr_idx: torch.Tensor) -> Dict:
    """
    Evaluate how well the LUT reconstructs the true o_proj output.
    Returns relative MSE and per-group relative MSE.
    """
    hidden_size = outputs.shape[-1]
    num_groups = hidden_size // group_size
    num_bins = table.shape[0]
    bin_idx = compute_address_bins(inputs, addr_idx, num_bins)
    flat_idx = bin_idx[:, 0] * num_bins + bin_idx[:, 1]  # [N]

    table_flat = table.view(num_bins * num_bins, hidden_size)
    reconstructed = table_flat[flat_idx]  # [N, hidden_size]

    mse = F.mse_loss(reconstructed, outputs, reduction="mean").item()
    output_var = outputs.var().item()
    relative_mse = mse / (output_var + 1e-8)

    # Per-group RMSE.
    group_rmse = []
    for g in range(num_groups):
        g_start = g * group_size
        g_end = g_start + group_size
        rec_group = reconstructed[:, g_start:g_end]
        out_group = outputs[:, g_start:g_end]
        g_mse = F.mse_loss(rec_group, out_group, reduction="mean").item()
        group_rmse.append(g_mse ** 0.5)

    return {
        "mse": mse,
        "relative_mse": relative_mse,
        "rmse": mse ** 0.5,
        "group_rmse": group_rmse,
    }


def select_address_channels(inputs: torch.Tensor, outputs: torch.Tensor,
                            group_size: int, num_bins: int, num_candidates: int = 8) -> Tuple[torch.Tensor, float]:
    """
    Try several candidate address channel pairs and pick the one with lowest reconstruction error.
    Returns (best_addr_idx, best_relative_mse).
    """
    hidden_size = inputs.shape[-1]
    num_groups = hidden_size // group_size

    # Candidate channel pairs: high-variance channels might be informative.
    channel_var = inputs.var(dim=0)  # [hidden_size]
    top_channels = torch.topk(channel_var, k=min(num_candidates * 2, hidden_size)).indices.tolist()

    best_rmse = float("inf")
    best_pair = None

    # Try a few pairs from high-variance channels.
    import itertools
    pairs = list(itertools.combinations(top_channels[:num_candidates], 2))[:20]

    # Use first group to quickly score address pairs.
    g_start = 0
    g_end = group_size
    out_group = outputs[:, g_start:g_end]

    for c1, c2 in pairs:
        addr_idx = torch.tensor([c1, c2], device=inputs.device)
        try:
            table = build_lut_for_layer(inputs, out_group, group_size, num_bins, addr_idx)
            metrics = evaluate_lut_reconstruction(inputs, out_group, table, group_size, addr_idx)
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_pair = (c1, c2)
        except Exception:
            continue

    if best_pair is None:
        best_pair = (0, 1)

    return torch.tensor(best_pair, device=inputs.device), best_rmse


def inspect_layer(model, tokenizer, layer_id: int, calib_batch, eval_batch,
                  group_size: int, num_bins: int, device) -> Dict:
    """Inspect o_proj LUT viability for one layer."""
    layer = model.model.layers[layer_id]
    o_proj = layer.self_attn.o_proj
    hidden_size = o_proj.weight.shape[1]

    captured = {"input": [], "output": []}

    def hook(module, input, output):
        # input is tuple, output is tensor
        inp = input[0] if isinstance(input, tuple) else input
        captured["input"].append(inp.detach())
        captured["output"].append(output.detach())

    handle = o_proj.register_forward_hook(hook)

    try:
        model.eval()
        with torch.no_grad():
            calib_inputs = {k: v.to(device) for k, v in calib_batch.items()}
            eval_inputs = {k: v.to(device) for k, v in eval_batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(**calib_inputs)
                _ = model(**eval_inputs)
    finally:
        handle.remove()

    # We called model(**calib_batch) then model(**eval_batch); the hook fired twice.
    if len(captured["input"]) < 2:
        raise RuntimeError(f"Expected 2 hook captures (calib + eval), got {len(captured['input'])}")

    calib_inputs = captured["input"][0].view(-1, hidden_size)
    calib_outputs = captured["output"][0].view(-1, hidden_size)
    eval_inputs = captured["input"][1].view(-1, hidden_size)
    eval_outputs = captured["output"][1].view(-1, hidden_size)

    # Select good address channels using calib data (first group only for speed).
    addr_idx, _ = select_address_channels(calib_inputs, calib_outputs, group_size, num_bins)

    # Build full LUT from calib data.
    table = build_lut_for_layer(calib_inputs, calib_outputs, group_size, num_bins, addr_idx)

    # Evaluate on eval data.
    metrics = evaluate_lut_reconstruction(eval_inputs, eval_outputs, table, group_size, addr_idx)

    metrics["group_rmse_mean"] = sum(metrics["group_rmse"]) / len(metrics["group_rmse"])
    metrics["group_rmse_max"] = max(metrics["group_rmse"])
    metrics["addr_channels"] = addr_idx.tolist()
    metrics["num_groups"] = hidden_size // group_size

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", default="15,16,17,18,19,20,21,22,23,24,25,26,27")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default="results/o_proj_lut_inspection.json")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=str(device), low_cpu_mem_usage=False
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    calib_batch, eval_batch = prepare_texts(tokenizer, args.max_seq_len, args.calib_size, args.eval_size)

    print(f"\nInspecting o_proj LUT viability for layers: {layers}")
    print("=" * 70)

    results = []
    for layer_id in tqdm(layers, desc="Layers"):
        metrics = inspect_layer(
            model, tokenizer, layer_id, calib_batch, eval_batch,
            args.group_size, args.num_bins, device
        )
        results.append({
            "layer_id": layer_id,
            **metrics,
        })
        print(f"L{layer_id:2d}: rel_mse={metrics['relative_mse']:.4f}, "
              f"rmse={metrics['rmse']:.4f}, group_rmse_mean={metrics['group_rmse_mean']:.4f}, "
              f"group_rmse_max={metrics['group_rmse_max']:.4f}, "
              f"addr={metrics['addr_channels']}")

    # Rank by relative MSE (lower = better).
    results_sorted = sorted(results, key=lambda r: r["relative_mse"])

    print("\n" + "=" * 70)
    print("RANKING: best -> worst o_proj LUT candidates")
    print("=" * 70)
    for r in results_sorted:
        print(f"L{r['layer_id']:2d}: rel_mse={r['relative_mse']:.4f}")

    output = {
        "model": args.model,
        "group_size": args.group_size,
        "num_bins": args.num_bins,
        "layers": results_sorted,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
