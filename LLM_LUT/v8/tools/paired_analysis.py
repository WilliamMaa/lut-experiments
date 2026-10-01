#!/usr/bin/env python3
"""Paired per-question analysis across configs (docs/23 §10).

Aggregate fact accuracy hides which component fixes or breaks which question.
This tool joins the per-question `fact_details` (session, turn, qtype,
correct) stored in each cell JSON and reports, per concurrency N and config
pair, the 2x2 transition counts:

            B correct   B wrong
  A correct     a           b      (b = A fixes what B breaks)
  A wrong       c           d      (c = B fixes what A breaks)

b/c are the McNemar discordant pairs; with n_disc = b + c the exact binomial
two-sided p-value tests whether the pair differs. |b - c| / (b + c) gives the
direction and size of the paired difference — what aggregate accuracy cannot.

Usage:
    python tools/paired_analysis.py results/concurrency            # all pairs, all N
    python tools/paired_analysis.py results/concurrency -n 8 -a hh_merge_m4 -b m4_k8v8
    python tools/paired_analysis.py results/concurrency --matrix   # full config x config matrix at each N
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path


def binom_two_sided(k, n):
    """Exact two-sided binomial p-value for P(X >= max(k, n-k)) * 2 tails."""
    if n == 0:
        return 1.0
    from math import comb
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_details(input_dir):
    """-> {(config, N): {(session, turn): correct}}"""
    out = {}
    for p in sorted(Path(input_dir).glob("*_n*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        details = rec.get("fact_details")
        if not details:
            continue  # cells from before the field existed
        key = {(d["session"], d["turn"]): bool(d["correct"]) for d in details}
        out[(rec["config"], rec["concurrency"])] = key
    return out


def paired_counts(deta, detb):
    a = b = c = d = 0
    for k, va in deta.items():
        if k not in detb:
            continue
        vb = detb[k]
        if va and vb:
            a += 1
        elif va and not vb:
            b += 1
        elif not va and vb:
            c += 1
        else:
            d += 1
    return a, b, c, d


def fmt_pair(name_a, name_b, deta, detb):
    a, b, c, d = paired_counts(deta, detb)
    n = a + b + c + d
    if n == 0:
        return None
    p = binom_two_sided(min(b, c), b + c)
    acc_a = (a + b) / n
    acc_b = (a + c) / n
    direction = f"{name_a}>{name_b}" if b > c else (f"{name_b}>{name_a}" if c > b else "tie")
    return (f"  {name_a:12s} acc={acc_a:.3f}  {name_b:12s} acc={acc_b:.3f}  "
            f"n={n}  [a={a} b(a✓b✗)={b} c(a✗b✓)={c} d={d}]  "
            f"discordant={b + c}  dir={direction}  McNemar p={p:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("-n", "--concurrency", type=int, default=None,
                    help="only this N (default: every N present)")
    ap.add_argument("-a", "--config-a", default=None, help="restrict to this pair")
    ap.add_argument("-b", "--config-b", default=None)
    ap.add_argument("--matrix", action="store_true",
                    help="full config x config matrix of paired deltas at each N")
    args = ap.parse_args()

    data = load_details(args.input_dir)
    if not data:
        raise SystemExit("no fact_details found (cells predate the field — "
                         "rerun needed to populate per-question records)")
    ns = sorted({n for (_, n) in data} if args.concurrency is None else [args.concurrency])
    configs = sorted({c for (c, _) in data})

    for n in ns:
        avail = [c for c in configs if (c, n) in data]
        print(f"\n=== N={n} ===")
        for ca, cb in combinations(avail, 2):
            if args.config_a and {ca, cb} != {args.config_a, args.config_b}:
                continue
            line = fmt_pair(ca, cb, data[(ca, n)], data[(cb, n)])
            if line:
                print(line)
        if args.matrix:
            print(f"  paired delta acc(row) - acc(col), N={n}:")
            header = " " * 16 + "".join(f"{c:>12s}" for c in avail)
            print(header)
            for ca in avail:
                row = f"  {ca:14s}"
                for cb in avail:
                    if ca == cb:
                        row += " " * 12
                        continue
                    _, b, c, dd = paired_counts(data[(ca, n)], data[(cb, n)])
                    tot = b + c + dd + paired_counts(data[(ca, n)], data[(cb, n)])[0]
                    if tot == 0:
                        row += " " * 12
                        continue
                    row += f"{(b - c) / tot:>+12.3f}"
                print(row)


if __name__ == "__main__":
    main()
