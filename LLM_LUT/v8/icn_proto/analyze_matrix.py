#!/usr/bin/env python3
"""Analyze a boundary/mixed matrix run: decision modes per cell + object
composition (E3/E4 primary evidence, docs 03-icn-kv-principle §5).

Usage:
    python -m icn_proto.analyze_matrix                  # newest matrix_*.json
    python -m icn_proto.analyze_matrix path/to/matrix_boundary_xxx.json
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "icn_proto")


def pick(path):
    if path:
        return path
    files = sorted(
        glob.glob(os.path.join(OUT, "matrix_*.json")),
        key=os.path.getmtime)
    if not files:
        sys.exit("no matrix_*.json under results/icn_proto")
    return files[-1]


def mb(n):
    return round(n / 1e6, 1) if n else 0.0


def cell_object_stats(summary_path):
    """Per turn-class object byte averages from a cell's run JSON."""
    if not summary_path:
        return None
    try:
        with open(summary_path) as f:
            d = json.load(f)
    except OSError:
        return None
    rows = {"doc": [], "q": []}
    for r in d.get("records", []):
        b = r.get("obj_bytes")
        if not b:
            continue
        rows["doc" if r.get("turn") == -1 else "q"].append(
            (b, r.get("obj_attn_bytes", 0), r.get("obj_linear_bytes", 0)))
    out = {}
    for k, v in rows.items():
        if not v:
            continue
        n = len(v)
        out[k] = {
            "obj_mb": round(sum(x[0] for x in v) / n / 1e6, 1),
            "attn_mb": round(sum(x[1] for x in v) / n / 1e6, 1),
            "lin_mb": round(sum(x[2] for x in v) / n / 1e6, 1),
        }
    return out


def main():
    path = pick(sys.argv[1] if len(sys.argv) > 1 else None)
    print("file:", path)
    d = json.load(open(path))
    print(f"{'cell':<30} {'decisions':<38} {'xfr':>4} "
          f"{'obj(doc)':>22} {'obj(q)':>22}")
    for c in d.get("cells", []):
        chosen = (c.get("p2_decisions") or {}).get("mode") or {}
        dec = ",".join(f"{k}:{v}" for k, v in sorted(chosen.items())) or "-"
        stats = cell_object_stats(c.get("summary")) or {}
        doc = stats.get("doc")
        q = stats.get("q")
        doc_s = (f"{doc['obj_mb']} ({doc['attn_mb']}+{doc['lin_mb']})"
                 if doc else "-")
        q_s = (f"{q['obj_mb']} ({q['attn_mb']}+{q['lin_mb']})"
               if q else "-")
        print(f"{c['name']:<30} {dec:<38} {c.get('transfers', '?'):>4} "
              f"{doc_s:>22} {q_s:>22}")
    agree = d.get("token_agreement")
    if agree:
        print("\ntoken agreement (E4 quality leg):")
        for k, v in agree.items():
            print(f"  {k}: {v['identical']}/{v['turns']} = {v['agreement']}")


if __name__ == "__main__":
    main()
