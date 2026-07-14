"""
Per-group/module sensitivity scanner for LLM_LUT v5.

For each candidate group in each target module, replace that group's output with
its calibration mean and evaluate the model. The resulting ranking can guide
sensitivity-aware MAC allocation across down_proj, o_proj, gate_proj, q_proj,
up_proj, k_proj, v_proj, etc.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python scan_module_sensitivity.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --modules down_proj,o_proj,gate_proj,q_proj \
        --layers 15-27 \
        --group_size 64 \
        --calib_size 256 --eval_size 64 \
        --output_json results/sensitivity_scan.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_model_and_data
from metrics import compute_model_metrics, compute_baseline_probs


def parse_modules(arg_str: str) -> List[str]:
    return [m.strip() for m in arg_str.split(",") if m.strip()]


def parse_layers(arg_str: str) -> List[int]:
    """Parse '0-27' or '15,16,17'."""
    layers = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def get_module(model, layer_id: int, module_name: str):
    layer = model.model.layers[layer_id]
    if module_name == "down_proj":
        return layer.mlp.down_proj
    elif module_name == "gate_proj":
        return layer.mlp.gate_proj
    elif module_name == "up_proj":
        return layer.mlp.up_proj
    elif module_name == "o_proj":
        return layer.self_attn.o_proj
    elif module_name == "q_proj":
        return layer.self_attn.q_proj
    elif module_name == "k_proj":
        return layer.self_attn.k_proj
    elif module_name == "v_proj":
        return layer.self_attn.v_proj
    else:
        raise ValueError(f"Unknown module: {module_name}")


def capture_group_means(model, module, calib_loader, group_size, device):
    """Capture module output on calibration data and compute per-group mean."""
    outputs = []

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        outputs.append(h.detach().float().cpu())

    handle = module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for batch in calib_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()

    all_out = torch.cat([o.view(-1, o.shape[-1]) for o in outputs], dim=0)
    output_dim = all_out.shape[-1]
    num_groups = (output_dim + group_size - 1) // group_size
    means = []
    for gid in range(num_groups):
        g_start = gid * group_size
        g_end = min(g_start + group_size, output_dim)
        means.append(all_out[:, g_start:g_end].mean(dim=0))
    return means, output_dim


def evaluate_with_group_replaced(model, eval_loader, baseline_eval_probs, module, group_id,
                                 group_size, group_mean, device):
    """Install a hook that replaces one group output with its mean, run eval."""

    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        g_start = group_id * group_size
        g_end = min(g_start + group_size, h.shape[-1])
        # In-place replacement; no grad needed
        h[..., g_start:g_end] = group_mean.to(h.device, h.dtype)
        return out

    handle = module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
    handle.remove()
    return metrics


def mac_saved_per_token(module_name: str, group_size: int, weight) -> int:
    """Compute MACs saved per token by replacing this output group."""
    # weight shape: [out_dim, in_dim]
    in_dim = weight.shape[1]
    return group_size * in_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--modules", required=True,
                        help="Comma-separated module names, e.g. down_proj,o_proj,gate_proj,q_proj")
    parser.add_argument("--layers", default="0-27",
                        help="Layers to scan, e.g. '15-27' or '15,16,17'")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", default="results/sensitivity_scan.json")
    args = parser.parse_args()

    modules = parse_modules(args.modules)
    layers = parse_layers(args.layers)

    print("Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model,
        eval_size=args.eval_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device_str=args.device,
        calib_size=args.calib_size,
    )
    device = model.device

    print("Computing baseline metrics...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)
        baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
    print(f"Baseline: PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}, "
          f"KL={baseline_metrics.get('avg_kl', 0):.4f}")

    results = {
        "model": args.model,
        "modules": modules,
        "layers": layers,
        "group_size": args.group_size,
        "calib_size": args.calib_size,
        "eval_size": args.eval_size,
        "baseline": {
            "ppl": baseline_metrics["ppl"],
            "acc": baseline_metrics["next_token_acc"],
            "kl": baseline_metrics.get("avg_kl", 0.0),
        },
        "scans": {},
    }

    for module_name in modules:
        print(f"\n{'='*60}")
        print(f"Scanning module: {module_name}")
        print(f"{'='*60}")
        module_results = []

        for layer_id in layers:
            if layer_id >= len(model.model.layers):
                continue
            module = get_module(model, layer_id, module_name)
            output_dim = module.weight.shape[0]
            num_groups = (output_dim + args.group_size - 1) // args.group_size

            print(f"\n[Layer {layer_id}] {module_name}: output_dim={output_dim}, groups={num_groups}")
            print("  Capturing calibration means...")
            group_means, _ = capture_group_means(
                model, module, calib_loader, args.group_size, device
            )

            for gid in tqdm(range(num_groups), desc=f"L{layer_id} groups", leave=False):
                metrics = evaluate_with_group_replaced(
                    model, eval_loader, baseline_eval_probs, module, gid,
                    args.group_size, group_means[gid], device,
                )
                mac = mac_saved_per_token(module_name, args.group_size, module.weight)
                module_results.append({
                    "layer": layer_id,
                    "module": module_name,
                    "group_id": gid,
                    "group_size": min(args.group_size, output_dim - gid * args.group_size),
                    "output_dim": output_dim,
                    "mac_saved_per_token": mac,
                    "ppl": metrics["ppl"],
                    "acc": metrics["next_token_acc"],
                    "kl": metrics.get("avg_kl", 0.0),
                    "delta_ppl": metrics["ppl"] - baseline_metrics["ppl"],
                    "delta_kl": metrics.get("avg_kl", 0.0) - baseline_metrics.get("avg_kl", 0.0),
                    "delta_acc": baseline_metrics["next_token_acc"] - metrics["next_token_acc"],
                })

        results["scans"][module_name] = module_results

    # Global ranking: best replacement score = -delta_ppl / mac_saved
    all_candidates = []
    for module_name, cand_list in results["scans"].items():
        for c in cand_list:
            c["score_ppl_per_mac"] = -c["delta_ppl"] / max(c["mac_saved_per_token"], 1)
            c["score_acc_per_mac"] = -c["delta_acc"] / max(c["mac_saved_per_token"], 1)
            all_candidates.append(c)

    all_candidates.sort(key=lambda x: x["score_ppl_per_mac"], reverse=True)
    results["global_ranking_ppl"] = all_candidates[:200]

    all_candidates.sort(key=lambda x: x["score_acc_per_mac"], reverse=True)
    results["global_ranking_acc"] = all_candidates[:200]

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output_json}")
    print("\nTop 10 by PPL-per-MAC:")
    for i, c in enumerate(results["global_ranking_ppl"][:10], 1):
        print(f"  {i:2d}. {c['module']} L{c['layer']}g{c['group_id']}: "
              f"ΔPPL={c['delta_ppl']:.3f}, MAC/token={c['mac_saved_per_token']}, "
              f"score={c['score_ppl_per_mac']:.6f}")


if __name__ == "__main__":
    main()
