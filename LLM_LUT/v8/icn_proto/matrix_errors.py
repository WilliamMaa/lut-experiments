#!/usr/bin/env python3
"""Dump diagnostics for bad matrix cells from the step-5 manifest.

For every row that crashed, hung, or lacks a JSON, print the cell key,
wall time, exit code, and the tail of the captured stderr (scheduler or
worker traceback lives there).

Usage:
    python -m icn_proto.matrix_errors
    python -m icn_proto.matrix_errors --manifest path/to/manifest.json
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        HERE, "results", "icn_proto", "matrix_step5_manifest.json"))
    ap.add_argument("--tail", type=int, default=2500,
                    help="stderr tail characters per bad row")
    args = ap.parse_args()

    rows = json.load(open(args.manifest, encoding="utf-8"))
    seen = {}
    for r in rows:
        if r.get("json"):
            seen[r["json"]] = seen.get(r["json"], 0) + 1
    bad = [r for r in rows
           if not r.get("json") or r.get("rc") not in (0, None)
           or r.get("failed") not in (0, None)
           or (r.get("json") and seen[r["json"]] > 1)]
    if not bad:
        print("manifest clean: no bad cells")
        return
    for r in bad:
        dup = r.get("json") and seen[r["json"]] > 1
        print(f"=== share={r['share']} policy={r['policy']} rep={r['rep']} "
              f"rc={r.get('rc')} cell_s={r.get('cell_s')} "
              f"failed={r.get('failed')} dup_json={dup} ===")
        err = (r.get("error") or "").strip()
        if err:
            print(err[-args.tail:])
        else:
            print("(no stderr captured)")
        print()


if __name__ == "__main__":
    main()
