#!/usr/bin/env python3
"""Compare cluster-run summaries across --policy settings.

Scans results/icn_proto/blkcluster_*.json (or the paths given on the
command line), groups runs by config.policy, and prints the metrics the
05 v3 §6 step-3 ladder is judged on: hit_rate, resumed turns,
new_tokens_processed (system-level GPU compute), transfers, latency.

Usage:
    python -m icn_proto.compare_policies                 # all runs
    python -m icn_proto.compare_policies run1.json run2.json ...
"""

import glob
import json
import os
import sys

METRICS = ("wall_s", "throughput_rps", "hit_rate", "resumed",
           "new_tokens_processed", "published_blocks", "transfers",
           "transfer_bytes", "avg_latency_s", "failed", "prefill_rate",
           "xfer_rate")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_paths(argv):
    if argv:
        return argv
    files = glob.glob(os.path.join(HERE, "results", "icn_proto",
                                   "blkcluster_*.json"))
    if not files:
        sys.exit("no blkcluster_*.json under results/icn_proto")
    return sorted(files, key=os.path.getmtime)


def main():
    runs = []
    for p in pick_paths(sys.argv[1:]):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {p}: {e}")
            continue
        cfg = d.get("config", {})
        row = {"policy": cfg.get("policy", "?"),
               "stamp": os.path.basename(p)}
        for k in METRICS:
            row[k] = d.get(k)
        runs.append(row)

    hdr = f"{'policy':<7}{'wall_s':>8}{'rps':>8}{'hit':>7}{'resumed':>9}" \
          f"{'new_tok':>9}{'xfer':>6}{'xfer_MB':>9}{'lat_s':>8}{'fail':>6}"
    print(hdr)
    for r in runs:
        print(f"{r['policy']:<7}{r['wall_s']:>8}{r['throughput_rps']:>8}"
              f"{r['hit_rate']:>7}{r['resumed']:>9}{r['new_tokens_processed']:>9}"
              f"{r['transfers']:>6}{r['transfer_bytes'] / 1e6:>9.1f}"
              f"{r['avg_latency_s']:>8}{r['failed']:>6}  {r['stamp']}")

    # latest run per policy: the b0-vs-ours compute saving headline
    latest = {}
    for r in runs:
        latest[r["policy"]] = r
    if "b0" in latest and "ours" in latest:
        b0, ours = latest["b0"], latest["ours"]
        saved = b0["new_tokens_processed"] - ours["new_tokens_processed"]
        pct = 100.0 * saved / b0["new_tokens_processed"] \
            if b0["new_tokens_processed"] else 0.0
        print(f"\ncompute saving (b0 -> ours): {saved} fewer prefill tokens "
              f"({pct:.1f}% of b0); wall {b0['wall_s']}s -> {ours['wall_s']}s")


if __name__ == "__main__":
    main()
