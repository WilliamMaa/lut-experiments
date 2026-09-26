#!/usr/bin/env python3
"""Print the failing records of recent blkcluster summary JSONs.

Usage:
    python -m icn_proto.failed_report            # scan newest 4 JSONs
    python -m icn_proto.failed_report N          # scan newest N JSONs
    python -m icn_proto.failed_report path.json  # one specific JSON

Only JSONs with failed > 0 print anything. For each failed record:
request_id, chosen worker, decision mode, and the error (truncated).
"""

import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "icn_proto")


def report(path):
    d = json.load(open(path, encoding="utf-8"))
    n = d.get("failed") or 0
    if not n:
        return 0
    print(f"== {path} failed {n}")
    for r in d.get("records") or []:
        if r.get("ok"):
            continue
        dec = (r.get("decision") or {}).get("mode")
        print(f"  {r.get('request_id')} -> {r.get('worker')}"
              f" mode={dec} {repr(r.get('error'))[:200]}")
    return n


def main():
    args = sys.argv[1:]
    if args and args[-1].endswith(".json"):
        sys.exit(0 if report(args[-1]) else 1)
    n = int(args[0]) if args else 4
    cands = sorted(glob.glob(os.path.join(RESULTS, "blkcluster_*.json")),
                   key=os.path.getmtime)[-n:]
    if not cands:
        sys.exit(f"no blkcluster_*.json under {RESULTS}")
    total = sum(report(p) for p in cands)
    if not total:
        print(f"no failures in newest {len(cands)} JSON(s)")


if __name__ == "__main__":
    main()
