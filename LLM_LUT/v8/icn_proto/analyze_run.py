#!/usr/bin/env python3
"""Analyze one cluster run JSON: break latency down by turn class.

Turn classes:
  fresh    - no resume (first turns): full prefill of cum_tokens
  local    - resumed on the worker already holding the object
  transfer - resumed after a cross-worker fetch/deliver (latency includes
             the scheduler-mediated transfer on the critical path)

Usage:
    python -m icn_proto.analyze_run results/icn_proto/cluster_p1_....json
    python -m icn_proto.analyze_run            # newest cluster_*.json
"""

import glob
import json
import os
import sys


def pick(path):
    if path:
        return path
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "icn_proto")
    files = sorted(glob.glob(os.path.join(here, "cluster_*.json")),
                   key=os.path.getmtime)
    if not files:
        sys.exit("no cluster_*.json under results/icn_proto")
    return files[-1]


def avg(xs):
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def main():
    path = pick(sys.argv[1] if len(sys.argv) > 1 else None)
    print("file:", path)
    d = json.load(open(path))
    recs = d["records"]

    tr = [r for r in recs if r.get("transfer_bytes", 0) > 0]
    loc = [r for r in recs if r.get("transfer_bytes", 0) == 0 and r.get("resumed")]
    fr = [r for r in recs if not r.get("resumed")]

    print(f"turns with transfer : {len(tr):3d}  avg latency {avg([r['latency_s'] for r in tr])}  avg queue {avg([r.get('queue_s', 0) for r in tr])}  avg xfer {avg([r.get('xfer_s', 0) for r in tr])}")
    print(f"local resumed turns : {len(loc):3d}  avg latency {avg([r['latency_s'] for r in loc])}  avg queue {avg([r.get('queue_s', 0) for r in loc])}")
    print(f"fresh turns         : {len(fr):3d}  avg latency {avg([r['latency_s'] for r in fr])}  avg queue {avg([r.get('queue_s', 0) for r in fr])}")

    if tr:
        mb = sum(r["transfer_bytes"] for r in tr) / len(tr) / 1e6
        print(f"avg transfer size   : {mb:.1f} MB  total {sum(r['transfer_bytes'] for r in tr) / 1e9:.2f} GB")
    dec = [r["decision"]["chosen"] for r in recs
           if r.get("decision") and r["decision"].get("chosen")]
    if dec:
        import collections
        print("p2 decision modes   :", dict(collections.Counter(dec)))
    if d.get("nrs_reuse"):
        print(f"nrs reuse names     : {len(d['nrs_reuse'])} names, "
              f"total {sum(d['nrs_reuse'].values())} cross-session reuses")
    if loc:
        print(f"resumed prefill_s   : {avg([r.get('prefill_s', 0) for r in loc])} (avg)")
    if fr:
        toks = sum(r["cum_tokens"] for r in fr)
        sec = sum(r.get("prefill_s", 0) for r in fr)
        print(f"fresh prefill_s     : {[r.get('prefill_s') for r in fr]}")
        print(f"fresh cum_tokens    : {[r['cum_tokens'] for r in fr]}")
        if sec > 0:
            print(f"implied prefill rate: {toks / sec:,.0f} tok/s")

    print("summary:", {k: d[k] for k in
                       ("wall_s", "hit_rate", "recompute_tokens", "transfers",
                        "transfer_bytes", "avg_latency_s", "failed")})


if __name__ == "__main__":
    main()
