#!/usr/bin/env python3
"""E1 run report: print the residency-opportunity verdict block of a
run-cluster summary JSON (07-regime-study §4 metrics + repl_reject
histogram). Auto-picks the newest blkcluster JSON when no path is given.

Usage:
    python -m icn_proto.e1_report                       # newest json
    python -m icn_proto.e1_report results/icn_proto/blkcluster_s4t40_XXX.json
"""

import glob
import json
import os
import sys

KEYS = ("wall_s", "throughput_rps", "hit_rate", "failed",
        "session_turn_migration_rate", "remote_resume_opportunities",
        "remote_resume_served_local_due_to_replication",
        "repl_planned", "replications", "replicated_bytes",
        "new_tokens_processed", "transfers", "transfer_bytes",
        "rederivation_tokens", "degrade_rederiv_tokens",
        "evictions", "evicted_blocks")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cands = glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "icn_proto", "blkcluster_*.json"))
        if not cands:
            sys.exit("no blkcluster_*.json under results/icn_proto/")
        path = max(cands, key=os.path.getmtime)
    d = json.load(open(path, encoding="utf-8"))
    print(f"== {path}")
    for k in KEYS:
        print(f"  {k}: {d.get(k)}")
    print(f"  repl_reject: {json.dumps(d.get('repl_reject'))}")
    cr = d.get("c_recompute") or []
    if cr:
        print("  c_recompute: " + ", ".join(
            f"<={b['bucket_max']}tok: {b['tok_per_s']}t/s(n={b['n']})"
            for b in cr))


if __name__ == "__main__":
    main()
