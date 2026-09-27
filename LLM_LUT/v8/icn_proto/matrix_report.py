#!/usr/bin/env python3
"""Scan a matrix manifest and report / drop cells.

The 2026-09-27 snapshot-clobber fix (12-e2-diag incident 2) changed the
scheduler. Two drop modes:

  --drop-stale      drop cells whose JSON records stale_resume_retries>0
                    (the bug demonstrably fired there) or is unreadable
  --drop-before STAMP   drop every cell whose JSON was written before
                    STAMP (YYYYMMDD), for a fully clean post-fix matrix.
                    STAMP is parsed from the filename
                    blkcluster_s8t40_<stamp>.json

Usage:
    python -m icn_proto.matrix_report                     # report only
    python -m icn_proto.matrix_report --drop-stale
    python -m icn_proto.matrix_report --drop-before 20260927

The table prints each cell's JSON stamp so old vs new rows are visible
at a glance. After a drop, re-run the runbook §5/§5a commands
unchanged: the manifest resume skips surviving cells and re-runs the
dropped ones.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP_RE = re.compile(r"blkcluster_s8t40_(\d{8})_\d{6}\.json$")


def retries_of(row):
    p = row.get("json")
    if not p or not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (d.get("spill") or {}).get("stale_resume_retries", 0)


def stamp_of(row):
    m = STAMP_RE.search(row.get("json") or "")
    return m.group(1) if m else "????????"


def main():
    drop_stale = "--drop-stale" in sys.argv
    before = None
    args = []
    it = iter(a for a in sys.argv[1:] if a != "--drop-stale")
    for a in it:
        if a == "--drop-before":
            before = next(it)
        else:
            args.append(a)
    manifest = args[0] if args else os.path.join(
        ROOT, "results", "icn_proto", "matrix_e2.json")
    rows = json.load(open(manifest, encoding="utf-8"))
    print(f"manifest: {manifest}  ({len(rows)} cells)")
    keep, drop_rows = [], []
    for r in rows:
        n = retries_of(r)
        stamp = stamp_of(r)
        tag = "?" if n is None else str(n)
        print(f"  {stamp}  {r.get('wl', '?'):<34} "
              f"{r.get('policy', '?'):<5} rep={r.get('rep')}  "
              f"stale_resume_retries={tag}")
        stale = n is None or n > 0
        old = before is not None and stamp < before
        if stale or old:
            drop_rows.append(r)
        else:
            keep.append(r)
    why = []
    if drop_stale:
        why.append("stale/unreadable")
    if before:
        why.append(f"older than {before}")
    print(f"\n{len(drop_rows)} cell(s) dropped ({' or '.join(why)}), "
          f"{len(keep)} kept")
    if (drop_stale or before) and drop_rows:
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(keep, f, indent=1)
        print(f"manifest rewritten — re-run the runbook §5/§5a commands "
              f"to refill the {len(drop_rows)} dropped cell(s)")


if __name__ == "__main__":
    main()
