"""
v3 latency benchmark.

Measures:
1. Baseline down_proj latency (no hook)
2. v2 functional hook overhead (full matmul + overwrite)
3. v3 partial skip latency (partial matmul + LUT fill)

Decomposes v3 into:
- partial matmul
- LUT lookup (per-group PyTorch vs fused Triton)
- output assembly (index_copy)

Usage (with real checkpoints):
    python benchmark_latency.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layer 21 --groups "26,50,51,4,7,40" \
        --checkpoint_dir ../v2/results/7B_l21_6group_ckpt

Usage (dummy mode, no model load):
    python benchmark_latency.py --dummy --hidden_size 3584 --intermediate_size 14336 \
        --batch_size 1 --seq_len 128 --num_groups 6
"""

import os
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"

import argparse
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, os.path.dirname(__file__))

from triton_kernels import lut_fill, TRITON_AVAILABLE, pytorch_lut_fill

WARMUP = 10
REPEATS = 50


def benchmark_fn(fn, *args, warmup=WARMUP, repeats=REPEATS):
    """Benchmark a callable with CUDA sync. Returns median ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


def parse_groups(groups_str):
    return [int(g.strip()) for g in groups_str.split(",")]


def create_dummy_data(batch_size, seq_len, intermediate_size, hidden_size, num_groups, device, dtype):
    """Create dummy tensors for benchmarking without real model/checkpoints."""
    group_size = 64
    num_bins = 64
    M = batch_size * seq_len

    hidden = torch.randn(batch_size, seq_len, intermediate_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)
    normed_x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    # Random replaced groups
    all_groups = list(range(hidden_size // group_size))
    replaced_groups = all_groups[:num_groups]
    active_groups = all_groups[num_groups:]

    active_channels = []
    for g in active_groups:
        active_channels.extend(range(g * group_size, (g + 1) * group_size))
    replaced_channels = []
    for g in replaced_groups:
        replaced_channels.extend(range(g * group_size, (g + 1) * group_size))

    active_indices = torch.tensor(active_channels, device=device, dtype=torch.long)
    replaced_indices = torch.tensor(replaced_channels, device=device, dtype=torch.long)

    active_weight = weight[active_indices, :].clone().contiguous()
    active_bias = bias[active_indices].clone().contiguous()

    # Dummy LUT tables and bin indices
    bin_idx = torch.randint(0, num_bins, (M, num_groups, 2), device=device, dtype=torch.int64)
    tables = torch.randn(num_groups, num_bins, num_bins, group_size, device=device, dtype=dtype)
    group_starts = torch.tensor([g * group_size for g in replaced_groups], device=device, dtype=torch.int32)
    addr_mean = torch.randn(num_groups, group_size, device=device, dtype=dtype)
    addr_std = torch.ones(num_groups, group_size, device=device, dtype=dtype) * 0.5

    return {
        "hidden": hidden,
        "weight": weight,
        "bias": bias,
        "normed_x": normed_x,
        "active_weight": active_weight,
        "active_bias": active_bias,
        "active_indices": active_indices,
        "replaced_indices": replaced_indices,
        "replaced_groups": replaced_groups,
        "active_groups": active_groups,
        "bin_idx": bin_idx,
        "tables": tables,
        "group_starts": group_starts,
        "addr_mean": addr_mean,
        "addr_std": addr_std,
        "M": M,
        "num_groups": num_groups,
        "group_size": group_size,
    }


def benchmark_real(args):
    """Benchmark with real model and checkpoints."""
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            print(f"[WARN] GPU {i} has allocated memory; proceeding because device_map={args.device} is explicit single-GPU.")

    device = model.device
    layer_id = args.layer
    group_list = parse_groups(args.groups)

    mlp = model.model.layers[layer_id].mlp
    hidden_size = mlp.down_proj.weight.shape[0]
    intermediate_size = mlp.down_proj.weight.shape[1]
    group_size = 64
    num_groups_total = hidden_size // group_size

    # Load fine-tuned down_proj weight if provided
    if getattr(args, "finetuned_weight", None):
        print(f"[Benchmark] Loading fine-tuned weight from {args.finetuned_weight}")
        ft_weight = torch.load(args.finetuned_weight, map_location="cpu")
        mlp.down_proj.weight.data.copy_(
            ft_weight.to(mlp.down_proj.weight.device, mlp.down_proj.weight.dtype)
        )

    hidden = torch.randn(args.batch_size, args.seq_len, intermediate_size, device=device, dtype=torch.float16)
    normed_x = torch.randn(args.batch_size, args.seq_len, hidden_size, device=device, dtype=torch.float16)

    # Pre-extract active weight
    replaced_groups_set = set(group_list)
    active_groups = [g for g in range(num_groups_total) if g not in replaced_groups_set]
    active_channels = []
    replaced_channels = []
    for g in active_groups:
        active_channels.extend(range(g * group_size, (g + 1) * group_size))
    for g in sorted(replaced_groups_set):
        replaced_channels.extend(range(g * group_size, (g + 1) * group_size))
    active_indices = torch.tensor(active_channels, device=device, dtype=torch.long)
    replaced_indices = torch.tensor(replaced_channels, device=device, dtype=torch.long)
    active_weight = mlp.down_proj.weight[active_indices, :].clone().contiguous()
    active_bias = None
    if mlp.down_proj.bias is not None:
        active_bias = mlp.down_proj.bias[active_indices].clone().contiguous()

    # Load checkpoints
    group_configs = {}
    for gid in group_list:
        ckpt_path = os.path.join(args.checkpoint_dir, f"replacement_l{layer_id}g{gid}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        group_configs[gid] = {
            "table": ckpt["table"].to(device, torch.float16),
            "addr_mean": ckpt["addr_mean"].to(device, torch.float16),
            "addr_std": ckpt["addr_std"].to(device, torch.float16),
        }

    M = args.batch_size * args.seq_len
    bin_idx = torch.randint(0, 64, (M, len(group_list), 2), device=device, dtype=torch.int64)
    tables = torch.stack([group_configs[g]["table"] for g in group_list], dim=0)
    group_starts = torch.tensor([g * group_size for g in group_list], device=device, dtype=torch.int32)
    addr_mean = torch.stack([group_configs[g]["addr_mean"] for g in group_list], dim=0)
    addr_std = torch.stack([group_configs[g]["addr_std"] for g in group_list], dim=0)

    return {
        "model": model,
        "hidden": hidden,
        "normed_x": normed_x,
        "weight": mlp.down_proj.weight,
        "bias": mlp.down_proj.bias,
        "active_weight": active_weight,
        "active_bias": active_bias,
        "active_indices": active_indices,
        "replaced_indices": replaced_indices,
        "replaced_groups": group_list,
        "bin_idx": bin_idx,
        "tables": tables,
        "group_starts": group_starts,
        "addr_mean": addr_mean,
        "addr_std": addr_std,
        "M": M,
        "num_groups": len(group_list),
        "group_size": group_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def run_benchmarks(data, args):
    """Run all benchmarks and return results dict."""
    device = data["hidden"].device
    dtype = data["hidden"].dtype
    B, S = args.batch_size, args.seq_len
    M = data["M"]
    hidden_size = data["hidden_size"]
    num_groups = data["num_groups"]
    group_size = data["group_size"]

    # Unpack data
    hidden = data["hidden"]
    weight = data["weight"]
    bias = data["bias"]
    normed_x = data["normed_x"]
    active_weight = data["active_weight"]
    active_bias = data["active_bias"]
    active_indices = data["active_indices"]
    replaced_indices = data["replaced_indices"]
    bin_idx = data["bin_idx"]
    tables = data["tables"]
    group_starts = data["group_starts"]
    addr_mean = data["addr_mean"]
    addr_std = data["addr_std"]
    replaced_groups = data["replaced_groups"]

    results = {}

    # ============================================================
    # 1. Baseline full down_proj
    # ============================================================
    baseline_fn = lambda: F.linear(hidden, weight, bias)
    results["baseline_ms"] = round(benchmark_fn(baseline_fn), 4)

    # ============================================================
    # 2. v2 functional hook (full matmul + overwrite)
    # ============================================================
    def v2_fn():
        out = F.linear(hidden, weight, bias)
        for i, gid in enumerate(replaced_groups):
            b1 = bin_idx[:, i, 0].flatten()
            b2 = bin_idx[:, i, 1].flatten()
            lut_delta = tables[i, b1, b2]
            lut_delta = lut_delta.view(B, S, group_size)
            g_start = gid * group_size
            out[:, :, g_start:g_start + group_size] = normed_x[:, :, g_start:g_start + group_size] + lut_delta
        return out
    results["v2_functional_ms"] = round(benchmark_fn(v2_fn), 4)

    # ============================================================
    # 3. v3 partial skip (PyTorch loop)
    # ============================================================
    def v3_pytorch_fn():
        active_out = F.linear(hidden, active_weight, active_bias)
        out = torch.zeros(B, S, hidden_size, device=device, dtype=dtype)
        out.index_copy_(2, active_indices, active_out)
        # PyTorch per-group LUT fill
        for i, gid in enumerate(replaced_groups):
            b1 = bin_idx[:, i, 0].flatten()
            b2 = bin_idx[:, i, 1].flatten()
            lut_delta = tables[i, b1, b2]
            lut_delta = lut_delta.view(B, S, group_size)
            g_start = gid * group_size
            out[:, :, g_start:g_start + group_size] = normed_x[:, :, g_start:g_start + group_size] + lut_delta
        return out
    results["v3_pytorch_ms"] = round(benchmark_fn(v3_pytorch_fn), 4)

    # ============================================================
    # 4. v3 partial skip (Triton fused LUT fill)
    # ============================================================
    if TRITON_AVAILABLE:
        def v3_triton_fn():
            active_out = F.linear(hidden, active_weight, active_bias)
            out = torch.zeros(B, S, hidden_size, device=device, dtype=dtype)
            out.index_copy_(2, active_indices, active_out)
            # Triton fused LUT fill
            normed_x_flat = normed_x.view(M, hidden_size)
            lut_out = lut_fill(bin_idx, tables, normed_x_flat, group_starts)
            lut_out = lut_out.view(B, S, -1)
            out.index_copy_(2, replaced_indices, lut_out)
            return out
        try:
            results["v3_triton_ms"] = round(benchmark_fn(v3_triton_fn), 4)
        except Exception as e:
            print(f"[Benchmark] Triton path failed: {e}")
            results["v3_triton_ms"] = None
    else:
        results["v3_triton_ms"] = None

    # ============================================================
    # 5. Decomposition
    # ============================================================
    results["partial_matmul_ms"] = round(benchmark_fn(
        lambda: F.linear(hidden, active_weight, active_bias)
    ), 4)

    def index_copy_fn():
        a = F.linear(hidden, active_weight, active_bias)
        out = torch.zeros(B, S, hidden_size, device=device, dtype=dtype)
        out.index_copy_(2, active_indices, a)
        return out
    results["index_copy_ms"] = round(
        benchmark_fn(index_copy_fn) - results["partial_matmul_ms"], 4
    )

    # PyTorch LUT only
    def pytorch_lut_only_fn():
        out = torch.zeros(B, S, num_groups * group_size, device=device, dtype=dtype)
        for i, gid in enumerate(replaced_groups):
            b1 = bin_idx[:, i, 0].flatten()
            b2 = bin_idx[:, i, 1].flatten()
            lut_delta = tables[i, b1, b2]
            lut_delta = lut_delta.view(B, S, group_size)
            g_start = gid * group_size
            out[:, :, i * group_size:(i + 1) * group_size] = normed_x[:, :, g_start:g_start + group_size] + lut_delta
        return out
    results["pytorch_lut_only_ms"] = round(benchmark_fn(pytorch_lut_only_fn), 4)

    # Triton LUT only
    if TRITON_AVAILABLE:
        def triton_lut_only_fn():
            normed_x_flat = normed_x.view(M, hidden_size)
            return lut_fill(bin_idx, tables, normed_x_flat, group_starts)
        try:
            results["triton_lut_only_ms"] = round(benchmark_fn(triton_lut_only_fn), 4)
        except Exception as e:
            results["triton_lut_only_ms"] = None
    else:
        results["triton_lut_only_ms"] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--groups", default="26,50,51,4,7,40")
    parser.add_argument("--checkpoint_dir", default="../v2/results/7B_l21_6group_ckpt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--output", default="outputs/latency_breakdown.json")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    parser.add_argument("--finetuned_weight", default=None, help="Path to fine-tuned down_proj weight (e.g. epoch3_down_proj.pt)")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data (no model load)")
    # Dummy mode params
    parser.add_argument("--hidden_size", type=int, default=3584)
    parser.add_argument("--intermediate_size", type=int, default=14336)
    parser.add_argument("--num_groups", type=int, default=6)
    args = parser.parse_args()

    if args.dummy:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"[Dummy mode] Device: {device}, dtype: {dtype}")
        data = create_dummy_data(
            args.batch_size, args.seq_len, args.intermediate_size,
            args.hidden_size, args.num_groups, device, dtype
        )
        data["hidden_size"] = args.hidden_size
        data["intermediate_size"] = args.intermediate_size
    else:
        print(f"[Real model] Loading {args.model}...")
        data = benchmark_real(args)

    print(f"[Benchmark] Running with B={args.batch_size}, S={args.seq_len}")
    print(f"  Hidden: {data['hidden_size']}, Intermediate: {data['intermediate_size']}")
    print(f"  Replaced groups: {data['replaced_groups']} ({len(data['replaced_groups'])} groups)")
    print(f"  Triton available: {TRITON_AVAILABLE}")
    print()

    results = run_benchmarks(data, args)

    # Metadata
    results.update({
        "model": args.model if not args.dummy else "dummy",
        "layer": args.layer,
        "groups": data["replaced_groups"],
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hidden_size": data["hidden_size"],
        "intermediate_size": data["intermediate_size"],
        "replaced_ratio": (len(data["replaced_groups"]) * 64) / data["hidden_size"],
        "triton_available": TRITON_AVAILABLE,
    })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print report
    print("=" * 70)
    print("LATENCY BENCHMARK REPORT")
    print("=" * 70)
    print(f"Model:      {results['model']}")
    print(f"Layer:      {results['layer']}, Groups: {results['groups']}")
    print(f"Batch:      {results['batch_size']}, Seq: {results['seq_len']}")
    print(f"Replaced:   {results['replaced_ratio']*100:.1f}% of channels")
    print(f"Triton:     {results['triton_available']}")
    print()

    baseline = results["baseline_ms"]
    print(f"1. Baseline (full matmul)          : {baseline:.3f} ms")
    print(f"2. v2 Functional (matmul+OW)       : {results['v2_functional_ms']:.3f} ms  ({results['v2_functional_ms']/baseline:.2f}x)")
    print(f"3. v3 Partial (PyTorch loop)       : {results['v3_pytorch_ms']:.3f} ms  ({results['v3_pytorch_ms']/baseline:.2f}x)")
    if results["v3_triton_ms"] is not None:
        print(f"4. v3 Partial (Triton LUT)         : {results['v3_triton_ms']:.3f} ms  ({results['v3_triton_ms']/baseline:.2f}x)")
        triton_speedup = results["v3_pytorch_ms"] / results["v3_triton_ms"]
        print(f"   Triton vs PyTorch LUT speedup   : {triton_speedup:.2f}x")
    else:
        print(f"4. v3 Partial (Triton LUT)         : N/A")
    print()
    print("--- Decomposition ---")
    print(f"   Partial matmul only             : {results['partial_matmul_ms']:.3f} ms")
    print(f"   index_copy (active)             : {results['index_copy_ms']:.3f} ms")
    print(f"   PyTorch LUT only (per-group)    : {results['pytorch_lut_only_ms']:.3f} ms")
    if results["triton_lut_only_ms"] is not None:
        print(f"   Triton LUT only (fused)         : {results['triton_lut_only_ms']:.3f} ms")
        print(f"   LUT speedup (Triton/PyTorch)    : {results['pytorch_lut_only_ms']/results['triton_lut_only_ms']:.2f}x")
    print()

    if results.get("v3_triton_ms") is not None and results["v3_triton_ms"] < baseline:
        speedup = baseline / results["v3_triton_ms"]
        print(f"✅ v3+Triton is FASTER than baseline by {speedup:.2f}x")
    else:
        print(f"⚠️  v3 is not yet faster than baseline.")
        print("   To achieve real speedup:")
        print("   - Need more replaced groups (higher ratio)")
        print("   - Or a full fused partial matmul + LUT Triton kernel")
    print("=" * 70)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
