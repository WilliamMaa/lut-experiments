#!/usr/bin/env python3
"""Step-0 opportunity estimate for interest aggregation (PIT, matrix
row 7, docs/icn-defined-addressing/15 §3).

Reads existing matrix manifests / cell JSONs and quantifies how much
duplicate work PIT could suppress, BEFORE any implementation:

  A. compute merges: turns with identical prefix fingerprint (fp)
     whose compute windows [t_assign, t_assign + latency] overlap and
     that both recomputed the same prefix on different workers.
  B. fetch merges: concurrent demand fetches (overlapping windows,
     different target workers) of identical block-name sets.

records carry fp, E, xfer_blocks, t_arrive, queue_s, latency_s, so
the estimate needs zero new experiments.

Usage:
    python -m icn_proto.pit_opportunity results/icn_proto/matrix_e2.json
    python -m icn_proto.pit_opportunity results/icn_proto/<cell>.json
"""

import json
import os
import sys


def load_cells(path):
    if os.path.isdir(path):
        out = []
        for f in sorted(os.listdir(path)):
            if f.endswith(".json") and f.startswith("blkcluster_"):
                out.append(os.path.join(path, f))
        return out
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):            # matrix manifest rows
        return [r["json"] for r in d if r.get("json")]
    return [path]                       # a single cell summary JSON


def windows(rec):
    """(start, end) of the turn's compute window, or None."""
    lat, q = rec.get("latency_s"), rec.get("queue_s")
    ta = rec.get("t_arrive")
    if lat is None or q is None or ta is None:
        return None
    start = ta + q
    return start, start + lat


def cell_opportunity(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    recs = [r for r in d.get("records", []) if r.get("ok")]
    # A: same fp, overlapping windows, different workers, both fresh
    by_fp = {}
    for r in recs:
        fp = r.get("fp")
        w = windows(r)
        if fp and w and (r.get("E") or 0) == 0:
            by_fp.setdefault(fp, []).append((w, r))
    a_pairs = 0
    a_tok = 0
    for fp, items in by_fp.items():
        items.sort()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (s1, e1), r1 = items[i]
                (s2, e2), r2 = items[j]
                if s2 >= e1:
                    break
                if r1.get("worker") != r2.get("worker"):
                    a_pairs += 1
                    a_tok += r2.get("prefill_tokens") or 0
    # B: same xfer set, overlapping windows, different targets
    by_x = {}
    for r in recs:
        names = r.get("xfer_blocks") or []
        w = windows(r)
        if len(names) > 1 and w:
            by_x.setdefault(tuple(names), []).append((w, r))
    b_pairs = 0
    for names, items in by_x.items():
        items.sort()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (s1, e1), r1 = items[i]
                (s2, e2), r2 = items[j]
                if s2 >= e1:
                    break
                if r1.get("worker") != r2.get("worker"):
                    b_pairs += 1
    return {"turns": len(recs), "a_compute_pairs": a_pairs,
            "a_recompute_tokens": a_tok, "b_fetch_pairs": b_pairs}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cells = load_cells(sys.argv[1])
    if not cells:
        sys.exit("no cell JSONs found")
    print(f"{'cell':<72} {'turns':>6} {'A_pairs':>8} "
          f"{'A_re_tok':>9} {'B_pairs':>8}")
    tot = {"turns": 0, "a_compute_pairs": 0, "a_recompute_tokens": 0,
           "b_fetch_pairs": 0}
    for c in cells:
        try:
            o = cell_opportunity(c)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{os.path.basename(c):<72}  unreadable: {exc}")
            continue
        for k in tot:
            tot[k] += o[k]
        print(f"{os.path.basename(c):<72} {o['turns']:>6} "
              f"{o['a_compute_pairs']:>8} {o['a_recompute_tokens']:>9} "
              f"{o['b_fetch_pairs']:>8}")
    print(f"{'TOTAL':<72} {tot['turns']:>6} {tot['a_compute_pairs']:>8} "
          f"{tot['a_recompute_tokens']:>9} {tot['b_fetch_pairs']:>8}")
    print("\nA_pairs = 同前缀并发重算对数（PIT 可抑制的重复计算）")
    print("A_re_tok = 被抑制的话可省的 prefill tokens（ waiter 侧）")
    print("B_pairs = 同块集并发 fetch 对数（PIT 可合并的重复搬运）")


if __name__ == "__main__":
    main()
