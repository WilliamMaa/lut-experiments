#!/usr/bin/env python3
"""Experiment matrix driver: run policies x scales x seeds sequentially and
collect comparison tables. Each cell is one run_cluster invocation (it
spawns and tears down its own workers), so cells are independent and a
failure in one does not kill the matrix.

The question this matrix answers (Step 4 of
docs/icn-defined-addressing/02-real-prototype-plan.md):

    Under compute saturation, does the P2 cost-model allocator beat the two
    extremes (P0 = always recompute, P1 = always migrate)?

Usage (from v8 root, on the GPU box):
    python -m icn_proto.run_matrix --seeds 2 --scales 24,48 \
        --policies p0,p1,p2 -- repr m_sp4 ...
    python -m icn_proto.run_matrix --dry-run    # print the run list only

Long runs: nohup python -m icn_proto.run_matrix ... > logs/matrix.log 2>&1 &
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


def build_cmds(args):
    """Expand the matrix into run_cluster command lines."""
    cmds = []
    port = args.port
    for scale in args.scales:
        for policy in args.policies:
            for seed in range(args.seeds):
                port += 1
                tag = f"s{scale}-{policy}-seed{seed}"
                out_name = f"cluster_{policy}_{args.repr}_x{scale}_seed{seed}"
                cmd = [sys.executable, "-m", "icn_proto.run_cluster",
                       "--policy", policy, "--repr", args.repr,
                       "--sessions", str(scale),
                       "--turns-per-session", str(args.turns_per_session),
                       "--doc-repeat", str(args.doc_repeat),
                       "--doc-repeat-alt", str(args.doc_repeat_alt),
                       "--doc-chars", str(args.doc_chars),
                       "--mem-budget-gb", str(args.mem_budget_gb),
                       "--port", str(port),
                       "--gpu-pool", args.gpu_pool,
                       "--model-path", args.model_path]
                cmds.append((tag, out_name, cmd))
    return cmds


def snapshot():
    return {f: os.path.getmtime(f)
            for f in glob.glob(os.path.join(OUT, "cluster_*.json"))}


def new_summary(before):
    after = snapshot()
    fresh = [f for f in after if f not in before]
    if not fresh:
        # no new file: fall back to the most recently touched one
        files = sorted(after, key=after.get)
        return files[-1] if files else None
    return max(fresh, key=after.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="24,48",
                    help="comma-separated session counts (workload intensity)")
    ap.add_argument("--policies", default="p0,p1,p2")
    ap.add_argument("--seeds", type=int, default=2,
                    help="repeats per cell (timing on the shared box varies)")
    ap.add_argument("--port", type=int, default=5700)
    ap.add_argument("--repr", default="m_sp4")
    ap.add_argument("--turns-per-session", type=int, default=6)
    ap.add_argument("--doc-repeat", type=int, default=48)
    ap.add_argument("--doc-repeat-alt", type=int, default=6)
    ap.add_argument("--doc-chars", type=int, default=60000)
    ap.add_argument("--mem-budget-gb", type=float, default=8.0)
    ap.add_argument("--gpu-pool", default="2,3,5,6")
    ap.add_argument("--model-path",
                    default="/home/u/downloads/models/Qwen3.6-35B-A3B")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cmds = build_cmds(args)
    print(f"[matrix] {len(cmds)} runs: "
          f"{len(args.scales.split(','))} scales x "
          f"{len(args.policies.split(','))} policies x {args.seeds} seeds")
    for tag, _, cmd in cmds:
        print(f"[matrix] {tag}: {' '.join(cmd[2:])}")
    if args.dry_run:
        return

    results = []
    for i, (tag, out_name, cmd) in enumerate(cmds):
        print(f"\n[matrix] === [{i + 1}/{len(cmds)}] {tag} ===", flush=True)
        t0 = time.time()
        before = snapshot()
        proc = subprocess.run(cmd, cwd=ROOT)
        summary_path = new_summary(before)
        cell = {"tag": tag, "rc": proc.returncode,
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
        else:
            cell["error"] = "no summary produced"
        results.append(cell)
        # incremental save: the matrix is inspectable while still running
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp = os.path.join(args.out, f"matrix_partial_{stamp}.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "cells": results}, f,
                      indent=2, ensure_ascii=False)

    # ---- aggregate table --------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"matrix_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "cells": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n[matrix] -> {path}")
    hdr = ("scale policy seed  wall_s  thru_rps  hit_rate  recompute  "
           "xfer_MB  xfer_n  avg_lat  failed")
    print(hdr)
    for c in results:
        p = c["tag"].split("-")
        print(f"{p[0]:>6} {p[1]:<7} {p[2][-1]:>4}  {c.get('wall_s', '?'):>6} "
              f"{c.get('throughput_rps', '?'):>8}  {c.get('hit_rate', '?'):>8} "
              f"{c.get('recompute_tokens', '?'):>9}  "
              f"{round(c.get('transfer_bytes', 0) / 1e6):>7}  "
              f"{c.get('transfers', '?'):>6}  {c.get('avg_latency_s', '?'):>7} "
              f"{c.get('failed', '?'):>7}")


if __name__ == "__main__":
    main()
