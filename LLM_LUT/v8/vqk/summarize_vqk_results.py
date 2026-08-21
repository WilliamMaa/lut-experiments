#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Aggregate multiple VQK eval JSONs into a summary table.

Usage:
  cd LLM_LUT/v8
  python -u vqk/summarize_vqk_results.py \
    --result_dir results \
    --pattern 'vqk_*.json' \
    --output_json results/vqk_summary_l39_o_proj.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_result(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Aggregate VQK eval JSONs")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--pattern", default="vqk_*.json")
    parser.add_argument("--output_json", default="results/vqk_summary.json")
    parser.add_argument("--sort_by", default="patched_ppl", choices=["patched_ppl", "ppl_delta", "top1_agreement"])
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    paths = sorted(result_dir.glob(args.pattern))
    if not paths:
        print(f"No files matched {result_dir}/{args.pattern}")
        return

    rows: List[Dict[str, Any]] = []
    for path in paths:
        data = load_result(path)
        cfg = data.get("patch", {}).get("config", {})
        baseline = data.get("baseline", {})
        patched = data.get("patched", {})
        delta = data.get("delta", {})
        logit = data.get("logit_metrics", {})

        rows.append({
            "file": path.name,
            "method": data.get("patch", {}).get("name", "").split("_")[0],
            "layer_idx": cfg.get("layer_idx"),
            "module_path": cfg.get("module_path"),
            "bits": cfg.get("bits"),
            "block_size": cfg.get("block_size"),
            "baseline_ppl": baseline.get("ppl"),
            "patched_ppl": patched.get("ppl"),
            "ppl_delta": delta.get("ppl"),
            "ppl_relative": delta.get("ppl_relative"),
            "top1_agreement": logit.get("top1_agreement"),
            "top5_agreement": logit.get("top5_agreement"),
            "avg_kl": logit.get("avg_kl"),
            "eos_success_rate": patched.get("generation_metrics", {}).get("eos_success_rate"),
            "repetition_rate": patched.get("generation_metrics", {}).get("repetition_rate"),
            "avg_output_length": patched.get("generation_metrics", {}).get("avg_output_length"),
        })

    rows.sort(key=lambda r: r.get(args.sort_by, float("inf")))

    print(f"\n{'='*100}")
    print(f"VQK summary: {len(rows)} results")
    print(f"{'='*100}\n")
    header = (
        f"{'ID':<20} {'Method':<6} {'Bits':>5} {'Block':>6} "
        f"{'Base PPL':>10} {'Patch PPL':>10} {'ΔPPL':>10} "
        f"{'Top-1':>8} {'Top-5':>8} {'Avg KL':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        method = r.get("method", "-")
        bits = r.get("bits", "-")
        block = r.get("block_size", "-")
        base_ppl = r.get("baseline_ppl")
        patch_ppl = r.get("patched_ppl")
        delta = r.get("ppl_delta")
        top1 = r.get("top1_agreement")
        top5 = r.get("top5_agreement")
        avg_kl = r.get("avg_kl")
        print(
            f"{r['file']:<20} {method:<6} {bits:>5} {block:>6} "
            f"{base_ppl:>10.4f} {patch_ppl:>10.4f} {delta:>+10.4f} "
            f"{top1:>8.4f} {top5:>8.4f} {avg_kl:>10.4f}"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
