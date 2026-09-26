#!/usr/bin/env python3
"""Scan a matrix manifest and report stale_resume_retries per cell.

The 2026-09-27 snapshot-clobber fix (12-e2-diag incident 2) changed the
scheduler; cells that ran on the old code are only trustworthy if the
bug never fired there. Its footprint is recorded per cell in the
summary JSON's spill.stale_resume_retries: 0 = the clobber never
affected a placement decision in that cell, keep the cell; >0 = drop
and re-run.

Usage:
    python -m icn_proto.matrix_report                 # report only
    python -m icn_proto.matrix_report --drop-stale    # report + rewrite
                                                      # manifest without
                                                      # affected rows

After --drop-stale, re-run the runbook §5/§5a commands unchanged: the
manifest resume skips surviving cells and re-runs the dropped ones.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def retries_of(row):
    p = row.get("json")
    if not p or not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (d.get("spill") or {}).get("stale_resume_retries", 0)


def main():
    drop = "--drop-stale" in sys.argv
    manifest = None
    args = [a for a in sys.argv[1:] if a != "--drop-stale"]
    if args:
        manifest = args[0]
    if manifest is None:
        manifest = os.path.join(ROOT, "results", "icn_proto",
                                "matrix_e2.json")
    rows = json.load(open(manifest, encoding="utf-8"))
    print(f"manifest: {manifest}  ({len(rows)} cells)")
    keep, drop_rows = [], []
    for r in rows:
        n = retries_of(r)
        tag = "?" if n is None else str(n)
        print(f"  {r.get('wl', '?'):<34} {r.get('policy', '?'):<5} "
              f"rep={r.get('rep')}  stale_resume_retries={tag}")
        if n is None or n > 0:
            drop_rows.append(r)
        else:
            keep.append(r)
    print(f"\n{len(drop_rows)} cell(s) affected or unreadable "
          f"(retries>0), {len(keep)} clean")
    if drop and drop_rows:
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(keep, f, indent=1)
        print(f"manifest rewritten without the {len(drop_rows)} affected "
              f"row(s) — re-run the runbook §5/§5a commands to refill")


if __name__ == "__main__":
    main()
