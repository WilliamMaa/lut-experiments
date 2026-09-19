#!/usr/bin/env python3
"""Step 4 decision-boundary experiment driver
(docs/icn-defined-addressing/02-real-prototype-plan.md §Step 4).

This script IS the experiment definition. Hypotheses, encoded as suites:

  boundary  H1: for m_sp4 chains P1 beats P0 iff context L > L* ~= 22-45K
                tokens (S_obj * R_prefill / R_xfer, coefficients measured
                in Step 2.5); for bf16 chains P0 wins through L = 96K
                (L*_bf16 ~= 233K, unreachable here). Maps two crossing
                curves over L in {8K, 32K, 96K} tokens.
            Cells: 3 lengths x {bf16, m_sp4} x {p0, p1} = 12 runs.
  mixed     H2: with long+short sessions in one workload, no single extreme
                is globally right, so P2 (per-turn argmin of the calibrated
                cost model) should match max(P0, P1) pointwise and beat
                both in wall time. H3: P2's logged per-turn decisions
                (records[].decision) follow the predicted boundary.
            Cells: mixed workload x {p0, p1, p2(m_sp4), p2(k8v8)} = 4 runs.
            The k8v8 cell doubles as the quality leg: token agreement of
            compressed chains vs the p0 reference is the cluster-level
            quality measurement (EOS deltas are the v8-eval anchor).

Every cell is one run_cluster invocation (own worker spawn/teardown);
cells run sequentially and each is snapshotted, so partial results are
inspectable while running. Timing varies on the shared box, so wall-time
conclusions use the per-turn decision logs (H3) as primary evidence and
wall as secondary.

Usage (from v8 root, on the GPU box):
    python -m icn_proto.run_matrix --suite boundary --dry-run
    nohup python -m icn_proto.run_matrix --suite all > logs/matrix.log 2>&1 &
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "icn_proto")

KEYS = ("wall_s", "throughput_rps", "hit_rate", "resumed",
        "recompute_tokens", "transfers", "transfer_bytes",
        "avg_latency_s", "failed", "prefill_rate", "xfer_rate")

# context lengths in ~tokens and the doc_chars that produce them
# (Chinese trace: tokens ~= doc_chars * 0.6, tiled docs truncated at
# doc_chars; doc_repeat is set large enough that truncation always bites).
LENGTHS = [("8K", 14000), ("32K", 55000), ("96K", 160000)]


def suite_cells(suite):
    """Return [{'name', 'tag', kwargs-for-run_cluster}, ...]."""
    cells = []
    if suite in ("boundary", "all"):
        for lname, lchars in LENGTHS:
            for repr_name in ("bf16", "m_sp4"):
                for policy in ("p0", "p1"):
                    cells.append({
                        "name": f"boundary-{lname}-{repr_name}-{policy}",
                        "tag": f"{lname}-{repr_name}-{policy}",
                        "kwargs": dict(policy=policy, repr=repr_name,
                                       sessions=24,
                                       turns_per_session=6,
                                       doc_repeat=200, doc_repeat_alt=200,
                                       doc_chars=lchars,
                                       mem_budget_gb=8.0)})
        # bf16 @ 96K: object ~= 2 GB; shrink the workload so resident KV
        # (12 sessions' latest objects / 2 workers ~= 12 GB) + 35 GB
        # weights stay under the free HBM.
        for i, c in enumerate(cells):
            if c["tag"].startswith("96K-bf16"):
                c["kwargs"]["sessions"] = 12
                c["name"] += "-s12"
                c["tag"] += "-s12"
    if suite in ("mixed", "all"):
        for policy, repr_name in (("p0", "m_sp4"), ("p1", "m_sp4"),
                                  ("p2", "m_sp4"), ("p2", "k8v8")):
            cells.append({
                "name": f"mixed-{repr_name}-{policy}",
                "tag": f"mixed-{repr_name}-{policy}",
                "kwargs": dict(policy=policy, repr=repr_name,
                               sessions=24, turns_per_session=6,
                               doc_repeat=48, doc_repeat_alt=6,
                               doc_chars=60000, mem_budget_gb=8.0)})
    return cells


def build_cmd(args, kw, port):
    return [sys.executable, "-m", "icn_proto.run_cluster",
            "--policy", kw["policy"], "--repr", kw["repr"],
            "--sessions", str(kw["sessions"]),
            "--turns-per-session", str(kw["turns_per_session"]),
            "--doc-repeat", str(kw["doc_repeat"]),
            "--doc-repeat-alt", str(kw["doc_repeat_alt"]),
            "--doc-chars", str(kw["doc_chars"]),
            "--mem-budget-gb", str(kw["mem_budget_gb"]),
            "--port", str(port),
            "--gpu-pool", args.gpu_pool,
            "--model-path", args.model_path]


def snapshot():
    return {f: os.path.getmtime(f)
            for f in glob.glob(os.path.join(OUT, "cluster_*.json"))}


def new_summary(before):
    after = snapshot()
    fresh = [f for f in after if f not in before]
    if not fresh:
        files = sorted(after, key=after.get)
        return files[-1] if files else None
    return max(fresh, key=after.get)


def token_agreement(cells):
    """Quality leg: for pairs of cells, per (session,turn) decoded-id
    agreement. Only pairs whose workloads share session/turn ids (same
    suite) are compared; cross-repr divergence is the measurable effect,
    cross-policy same-repr agreement is the resume-correctness check."""
    loaded = {}
    for c in cells:
        p = c.get("summary")
        if not p:
            continue
        with open(p) as f:
            loaded[c["name"]] = json.load(f)
    report = {}
    names = list(loaded)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = {r["request_id"]: r.get("decoded_ids") for r in
                  loaded[a]["records"] if r.get("decoded_ids")}
            db = {r["request_id"]: r.get("decoded_ids") for r in
                  loaded[b]["records"] if r.get("decoded_ids")}
            common = [k for k in da if k in db]
            if not common:
                continue
            same = sum(1 for k in common if da[k] == db[k])
            report[f"{a} vs {b}"] = {
                "turns": len(common), "identical": same,
                "agreement": round(same / len(common), 4)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all",
                    choices=["boundary", "mixed", "all"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeats per cell (shared-box timing noise)")
    ap.add_argument("--port", type=int, default=5700)
    ap.add_argument("--gpu-pool", default="2,3,5,6")
    ap.add_argument("--model-path",
                    default="/home/u/downloads/models/Qwen3.6-35B-A3B")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    base = suite_cells(args.suite)
    cells = []
    for seed in range(args.seeds):
        for c in base:
            cc = dict(c)
            cc["name"] = f"{c['name']}-seed{seed}" if args.seeds > 1 else c["name"]
            cc["seed"] = seed
            cells.append(cc)
    print(f"[matrix] suite={args.suite}: {len(cells)} runs")
    port = args.port
    for c in cells:
        port += 1
        c["port"] = port
        c["cmd"] = build_cmd(args, c["kwargs"], port)
        print(f"[matrix] {c['name']}: {' '.join(c['cmd'][2:])}")
    if args.dry_run:
        return

    results = []
    for i, c in enumerate(cells):
        print(f"\n[matrix] === [{i + 1}/{len(cells)}] {c['name']} ===",
              flush=True)
        t0 = time.time()
        before = snapshot()
        proc = subprocess.run(c["cmd"], cwd=ROOT)
        summary_path = new_summary(before)
        cell = {"name": c["name"], "tag": c["tag"], "rc": proc.returncode,
                "run_s": round(time.time() - t0, 1)}
        if summary_path:
            with open(summary_path) as f:
                s = json.load(f)
            cell["summary"] = summary_path
            for k in KEYS:
                cell[k] = s.get(k)
            if s.get("repr_of"):
                import collections
                cell["repr_assign"] = dict(collections.Counter(
                    s["repr_of"].values()))
            if s.get("policy") == "p2":
                dec = [r["decision"] for r in s["records"] if r.get("decision")]
                cell["p2_decisions"] = {
                    "n": len(dec),
                    "chosen": dict(collections.Counter(
                        d["chosen"] for d in dec))} if dec else None
        else:
            cell["error"] = "no summary produced"
        cells[i].update(cell)
        results.append(cell)
        partial = os.path.join(args.out,
                               f"matrix_{args.suite}_partial_{i + 1}.json")
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"suite": args.suite, "config": vars(args),
                       "cells": results}, f, indent=2, ensure_ascii=False)

    agree = token_agreement(cells)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"matrix_{args.suite}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"suite": args.suite, "config": vars(args),
                   "cells": results, "token_agreement": agree},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[matrix] -> {path}")
    print(f"{'cell':<28} {'wall_s':>7} {'thru':>7} {'hit':>6} "
          f"{'recomp':>9} {'xMB':>7} {'xn':>4} {'lat':>6} {'fail':>4}")
    for c in results:
        print(f"{c['name']:<28} {c.get('wall_s', '?'):>7} "
              f"{c.get('throughput_rps', '?'):>7} {c.get('hit_rate', '?'):>6} "
              f"{c.get('recompute_tokens', '?'):>9} "
              f"{round(c.get('transfer_bytes', 0) / 1e6):>7} "
              f"{c.get('transfers', '?'):>4} {c.get('avg_latency_s', '?'):>6} "
              f"{c.get('failed', '?'):>4}")
    if agree:
        print("\ntoken agreement (quality/correctness leg):")
        for k, v in agree.items():
            print(f"  {k}: {v['identical']}/{v['turns']} = {v['agreement']}")


if __name__ == "__main__":
    main()
