"""
非均匀层 group 分配搜索。

基于 v3 expand_ratio 生成的单层敏感度 summary，组合出候选配置，
用 v4 自带的 V4PartialEngine 评估，输出 Pareto 表。

用法:
    cd LLM_LUT/v4
    python search_layer_configs.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint_root ../v3/outputs \
        --summary_root ../v3/results/summaries \
        --output_path results/layer_search_pareto.json
"""

import os

# Parse --device before importing torch, so we can hide all other GPUs
# from the process via CUDA_VISIBLE_DEVICES. This avoids multi-GPU bugs.
import argparse as _ap
_earliest_parser = _ap.ArgumentParser(add_help=False)
_earliest_parser.add_argument("--device", default="cuda:0")
_earliest_args, _ = _earliest_parser.parse_known_args()

if _earliest_args.device.startswith("cuda:"):
    _gpu_id = _earliest_args.device.split(":", 1)[1]
    # Only set CUDA_VISIBLE_DEVICES if the user has not already set it.
    # This respects explicit external isolation like CUDA_VISIBLE_DEVICES=1.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
    _canonical_device = "cuda:0"
else:
    _canonical_device = _earliest_args.device

import sys
import json
import argparse
from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict

import torch

from trainable_engine import load_model_and_data
from metrics import compute_baseline_probs, compute_model_metrics, compute_mac_reduction, format_bytes
from partial_linear_quantized import V4PartialEngine


def get_group_ids_for_count(summary: Dict, count: int) -> List[int]:
    """Return the group_ids used for a specific num_groups in the summary."""
    progressive = summary.get("progressive", [])
    for item in progressive:
        if isinstance(item, dict) and item.get("num_groups") == count and "group_ids" in item:
            return [int(g) for g in item["group_ids"]]
    return []


def load_groups_for_layer(checkpoint_root: str, layer_id: int, group_count: int,
                          group_ids: List[int] = None):
    """Return list of (group_id, checkpoint_path) for a layer config.

    If group_ids is provided, only load those groups; otherwise load all
    checkpoints in the directory.
    """
    import glob
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    prefix = f"replacement_l{layer_id}g"
    suffix = ".pt"
    pattern = os.path.join(ckpt_dir, f"{prefix}*{suffix}")
    paths = sorted(glob.glob(pattern))
    groups = []
    for p in paths:
        name = os.path.basename(p)
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        gid = int(name[len(prefix):-len(suffix)])
        if group_ids is not None and gid not in group_ids:
            continue
        groups.append((gid, p))
    return groups


def compute_lut_storage(configs: List[Tuple[int, int]], checkpoint_root: str,
                        lut_dtype: str = "fp32") -> int:
    """Total LUT table bytes for a multi-layer config."""
    bytes_per_el = {"fp32": 4, "fp16": 2, "int8": 1}.get(lut_dtype, 4)
    total = 0
    for layer_id, group_count in configs:
        groups = load_groups_for_layer(checkpoint_root, layer_id, group_count)
        for _, path in groups:
            ckpt = torch.load(path, map_location="cpu")
            table = ckpt["table"]
            total += table.numel() * bytes_per_el
    return total


def build_engine_for_layer(model, layer_id: int, group_count: int, checkpoint_root: str,
                           lut_dtype: str = "fp32", summary: Dict = None) -> V4PartialEngine:
    """Build a V4PartialEngine for one layer, supporting quantized checkpoints."""
    group_ids = get_group_ids_for_count(summary, group_count) if summary else None
    groups = load_groups_for_layer(checkpoint_root, layer_id, group_count, group_ids=group_ids)
    if not groups:
        raise ValueError(f"No checkpoints found for L{layer_id} G{group_count} in {checkpoint_root}")
    if len(groups) != group_count:
        print(f"[WARN] L{layer_id} G{group_count}: expected {group_count} checkpoints, found {len(groups)}")

    engine = V4PartialEngine(model, layer_id, group_size=64, num_bins=64)
    for gid, path in groups:
        ckpt = torch.load(path, map_location="cpu")
        engine.add_group(
            group_id=gid,
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            table=ckpt["table"],
            scale=ckpt.get("scale"),
            zero_point=ckpt.get("zero_point", 0.0),
            quantization=ckpt.get("quantization", "fp32"),
        )
    return engine


def evaluate_multi_layer(model, eval_loader, reference_probs, configs, checkpoint_root,
                         lut_dtype: str = "fp32", summaries: Dict[int, Dict] = None):
    """Evaluate a list of (layer_id, group_count) configs installed simultaneously.

    Uses V4PartialEngine so INT8/FP16 checkpoints are handled correctly.
    """
    engines = []
    try:
        for layer_id, group_count in configs:
            summary = summaries.get(layer_id) if summaries else None
            engine = build_engine_for_layer(
                model, layer_id, group_count, checkpoint_root,
                lut_dtype=lut_dtype, summary=summary,
            )
            engine.install()
            engines.append(engine)
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        for engine in reversed(engines):
            engine.uninstall()
    return metrics


def load_layer_summary(summary_root: str, layer_id: int) -> Dict:
    """Load v3 expand_ratio summary JSON for a layer."""
    path = os.path.join(summary_root, f"expand_ratio_l{layer_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary not found: {path}. Run v3/expand_ratio.py first.")
    with open(path, "r") as f:
        return json.load(f)


def get_available_group_counts(summary: Dict) -> List[int]:
    """Extract available group counts from summary's progressive results.

    v3 expand_ratio.py writes progressive as a list of dicts, each with
    'num_groups' field.
    """
    progressive = summary.get("progressive", [])
    counts = []
    for item in progressive:
        if isinstance(item, dict) and "num_groups" in item:
            counts.append(int(item["num_groups"]))
    return sorted(counts)


def get_ppl_for_count(summary: Dict, count: int) -> float:
    """Return PPL for a given group count, or a large value if missing."""
    progressive = summary.get("progressive", [])
    for item in progressive:
        if isinstance(item, dict) and item.get("num_groups") == count:
            return item.get("ppl", float("inf"))
    return float("inf")


def generate_non_uniform_configs(layers: List[int],
                                 summary_root: str,
                                 default_candidates: Dict[int, List[int]] = None,
                                 max_configs: int = 50) -> List[List[Tuple[int, int]]]:
    """
    Generate candidate non-uniform configurations.

    If default_candidates is provided, it maps layer_id -> list of candidate group counts.
    Otherwise infer from summary files (use all available counts).
    """
    if default_candidates is None:
        default_candidates = {}
        for lid in layers:
            summary = load_layer_summary(summary_root, lid)
            counts = get_available_group_counts(summary)
            if not counts:
                raise ValueError(f"No progressive counts found for L{lid}")
            default_candidates[lid] = counts

    candidate_lists = [default_candidates[lid] for lid in layers]
    all_combos = list(product(*candidate_lists))

    # Sort by total groups (conservative first) and trim to max_configs.
    all_combos = sorted(all_combos, key=lambda combo: sum(combo))
    all_combos = all_combos[:max_configs]

    return [[(lid, cnt) for lid, cnt in zip(layers, combo)] for combo in all_combos]


def is_pareto_dominated(row: Dict, others: List[Dict]) -> bool:
    """Return True if another config is no worse on all objectives and strictly better on at least one."""
    for other in others:
        if other is row:
            continue
        # Objectives: lower MAC reduction? No, we want higher MAC reduction and lower PPL/KL.
        # A config dominates another if it has >= MAC reduction and <= PPL and <= KL, with at least one strict.
        better_or_equal = (
            other["mac_reduction_ratio"] >= row["mac_reduction_ratio"] and
            other["ppl"] <= row["ppl"] and
            other["kl"] <= row["kl"]
        )
        strictly_better = (
            other["mac_reduction_ratio"] > row["mac_reduction_ratio"] or
            other["ppl"] < row["ppl"] or
            other["kl"] < row["kl"]
        )
        if better_or_equal and strictly_better:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", default="19,20,21,22,23", help="Comma-separated candidate layer IDs")
    parser.add_argument("--checkpoint_root", default="../v3/outputs",
                        help="Root directory containing checkpoints/l{layer}/g{count}")
    parser.add_argument("--summary_root", default="../v3/outputs/summaries",
                        help="Directory containing expand_ratio_l*.json summary files")
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to expose to this process (e.g. cuda:1). Other GPUs are hidden via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--output_path", default="results/layer_search_pareto.json")
    parser.add_argument("--max_configs", type=int, default=50,
                        help="Maximum number of candidate configurations to evaluate")
    parser.add_argument("--lut_dtype", default="fp32", choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()

    # Use the canonical device derived before torch was imported.
    args.device = _canonical_device

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x.strip()) for x in args.layers.split(",")]

    # Load layer summaries and build default candidate sets based on sensitivity.
    print("=" * 70)
    print("Non-Uniform Layer Group Search")
    print("=" * 70)
    print(f"Layers: {layers}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Summary root: {args.summary_root}")
    print("=" * 70)

    # Default non-uniform candidate ranges based on MULTI_LAYER_SCAN_ANALYSIS.md sensitivity.
    sensitivity_candidates = {
        19: [4, 8],      # most sensitive
        20: [8, 12],
        21: [12, 16],
        22: [16],        # tolerant
        23: [16],        # most tolerant
    }
    default_candidates = {}
    for lid in layers:
        summary = load_layer_summary(args.summary_root, lid)
        available = set(get_available_group_counts(summary))
        if lid in sensitivity_candidates:
            default_candidates[lid] = [c for c in sensitivity_candidates[lid] if c in available]
        if not default_candidates.get(lid):
            default_candidates[lid] = sorted(available)
        print(f"  L{lid}: candidate groups {default_candidates[lid]}, "
              f"available summary counts {sorted(available)}")

    # Generate candidate configs.
    candidate_configs = generate_non_uniform_configs(
        layers, args.summary_root, default_candidates=default_candidates, max_configs=args.max_configs
    )
    print(f"\nGenerated {len(candidate_configs)} candidate configurations")

    # Load model and data once.
    print("\n[1/2] Loading model and data...")
    model, tokenizer, _calib_loader, eval_loader = load_model_and_data(
        args.model, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    # Compute baseline reference probabilities.
    print("\n[2/2] Computing baseline reference probabilities...")
    model.eval()
    with torch.no_grad():
        reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=None)
    print(f"  Baseline: PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    # Pre-load summaries for all layers.
    summaries = {lid: load_layer_summary(args.summary_root, lid) for lid in layers}

    # Evaluate each candidate.
    print("\n[Eval] Evaluating candidate configurations...")
    results = []
    for idx, configs in enumerate(candidate_configs, 1):
        print(f"  [{idx}/{len(candidate_configs)}] {configs} ...", end=" ")
        try:
            metrics = evaluate_multi_layer(
                model, eval_loader, reference_probs, configs, args.checkpoint_root,
                lut_dtype=args.lut_dtype, summaries=summaries,
            )
            mac_ratio = compute_mac_reduction(configs, hidden_size, intermediate_size, num_layers)
            storage_bytes = compute_lut_storage(configs, args.checkpoint_root)
            entry = {
                "configs": configs,
                "kl": metrics.get("avg_kl", 0.0),
                "ppl": metrics["ppl"],
                "acc": metrics["next_token_acc"],
                "mac_reduction_ratio": mac_ratio,
                "lut_storage_bytes": storage_bytes,
                "lut_storage_human": format_bytes(storage_bytes),
            }
            results.append(entry)
            print(f"KL={entry['kl']:.4f}, PPL={entry['ppl']:.2f}, Acc={entry['acc']:.4f}, "
                  f"MAC↓={mac_ratio*100:.2f}%, LUT={format_bytes(storage_bytes)}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"configs": configs, "error": str(e)})

    # Compute Pareto frontier (max MAC reduction, min PPL, min KL).
    successful = [r for r in results if "error" not in r]
    pareto = [r for r in successful if not is_pareto_dominated(r, successful)]
    pareto = sorted(pareto, key=lambda r: r["mac_reduction_ratio"])

    summary = {
        "model": args.model,
        "layers": layers,
        "baseline": {
            "ppl": baseline_metrics["ppl"],
            "acc": baseline_metrics["next_token_acc"],
        },
        "candidates": results,
        "pareto_frontier": pareto,
    }

    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {args.output_path}")

    # Print summary tables.
    print("\n" + "=" * 70)
    print("PARETO FRONTIER")
    print("=" * 70)
    print(f"{'Config':>28} | {'MAC↓':>7} | {'KL':>6} | {'PPL':>6} | {'Acc':>6} | {'LUT':>10}")
    print("-" * 70)
    for r in pareto:
        cfg_str = ",".join(f"L{l}:{c}" for l, c in r["configs"])
        print(f"{cfg_str:>28} | {r['mac_reduction_ratio']*100:>6.2f}% | {r['kl']:>6.4f} | "
              f"{r['ppl']:>6.2f} | {r['acc']:>6.4f} | {r['lut_storage_human']:>10}")

    print("\n" + "=" * 70)
    print("ALL SUCCESSFUL CONFIGURATIONS")
    print("=" * 70)
    for r in successful:
        cfg_str = ",".join(f"L{l}:{c}" for l, c in r["configs"])
        print(f"{cfg_str:>28} | {r['mac_reduction_ratio']*100:>6.2f}% | {r['kl']:>6.4f} | "
              f"{r['ppl']:>6.2f} | {r['acc']:>6.4f} | {r['lut_storage_human']:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
