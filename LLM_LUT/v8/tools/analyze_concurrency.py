#!/usr/bin/env python3
"""Tabulate v8 concurrency-serving cell results (kv_cache/concurrent_serve.py).

Reads every {config}_n{N}.json under --input-dir and prints a
configs x concurrency table for the requested metric, with sustainability
markers. Default: one table per key metric (TTFT, TPOT, output tok/s,
peak HBM, EOS, fact accuracy) in markdown, plus a final
max-sustainable-concurrency summary per config.

Usage:
    python tools/analyze_concurrency.py results/concurrency
    python tools/analyze_concurrency.py results/concurrency --metric tpot_mean_s
"""

import argparse
import json
from pathlib import Path

LADDER_ORDER = ["full", "hh", "hh_merge", "hh_merge_m4", "m4_k8v8"]

METRICS = {
    "ttft": ("ttft_mean_s", "{:8.2f}s", "TTFT (mean)"),
    "tpot": ("tpot_mean_s", "{:8.1f}ms", "TPOT (mean)"),
    "tput": ("output_tokens_per_s", "{:8.1f}", "Output tok/s (wall-clock)"),
    "hbm": ("peak_hbm_mb", "{:8.1f}", "Peak HBM (GB)"),
    "eos": ("eos_success_rate", "{:8.3f}", "EOS success"),
    "fact": ("fact_accuracy", None, "Fact accuracy"),
    "wall": ("wall_time_s", "{:8.0f}s", "Cell wall time"),
}


def fmt(metric_key, record):
    field, spec, _ = METRICS[metric_key]
    if record is None:
        return " " * (len(spec.format(0)) if spec else 8)
    if record.get("status") == "oom":
        return "     OOM"
    v = record.get(field)
    if v is None:
        return "     n/a"
    if metric_key == "tpot":
        v *= 1000.0
    elif metric_key == "hbm":
        v /= 1024.0
    elif metric_key == "fact":
        v = v.get("overall") if isinstance(v, dict) else v
        return f"{v:8.3f}" if v is not None else "     n/a"
    return spec.format(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="directory with cell JSONs")
    parser.add_argument("--metric", choices=list(METRICS.keys()), default=None,
                        help="print a single-metric table instead of all")
    parser.add_argument("--markdown", action="store_true", help="pipe-friendly plain markdown")
    args = parser.parse_args()

    cells = {}
    for p in sorted(Path(args.input_dir).glob("*_n*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cells[(rec["config"], rec["concurrency"])] = rec

    configs = [c for c in LADDER_ORDER if any(k[0] == c for k in cells)]
    configs += sorted({k[0] for k in cells} - set(configs))
    concurrencies = sorted({k[1] for k in cells})

    def marker(rec):
        if rec is None or rec.get("status") != "ok":
            return ""
        return "*" if rec.get("sustainable") else ""

    metric_keys = [args.metric] if args.metric else list(METRICS.keys())
    for mk in metric_keys:
        _, _, title = METRICS[mk]
        header = f"{title:24s}" + "".join(f"{('N=' + str(n)):>10s}" for n in concurrencies)
        print(f"\n### {title}" + (" (markdown)" if args.markdown else ""))
        print(header)
        print("-" * len(header))
        for c in configs:
            row = f"{c:24s}"
            for n in concurrencies:
                row += f"{fmt(mk, cells.get((c, n))) + marker(cells.get((c, n))):>10s}"
            print(row)

    print("\n### Max sustainable concurrency (EOS within tolerance of full, HBM within budget)")
    print(f"{'config':24s}{'max N':>8s}")
    print("-" * 32)
    for c in configs:
        ok = [n for n in concurrencies
              if (c, n) in cells and cells[(c, n)].get("sustainable")]
        print(f"{c:24s}{(str(max(ok)) if ok else '-'):>8s}")

    print("\n* = sustainable under the recorded budget/EOS rule; OOM = cell exceeded HBM.")


if __name__ == "__main__":
    main()
