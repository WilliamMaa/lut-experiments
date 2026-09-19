#!/usr/bin/env python3
"""THE benchmark: ICN sharing on vs off, same request stream.

One number answers "does the ICN idea work here": the ratio of total
prefill tokens (GPU compute accounting) between the no-share baseline
(every session prefills its own document) and the ICN system (content-
addressed objects shared across sessions). Everything else (E1/E2) is
mechanism precondition; this is the system-level payoff.

Runs two cells sequentially through run_cluster (same workload, same
cards), then prints the delta. Usage:

    python -m icn_proto.run_sharing --sessions 16 --turns-per-session 4 \
        --doc-chars 8000 --doc-repeat 8 --doc-repeat-alt 8 \
        --gpu-pool 2,3,5,6 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "icn_proto")

CELLS = [("noshare", ["--no-share"]), ("icn", [])]


def snapshot():
    import glob
    return {f: os.path.getmtime(f)
            for f in glob.glob(os.path.join(OUT, "cluster_*.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns-per-session", type=int, default=4)
    ap.add_argument("--doc-chars", type=int, default=8000)
    ap.add_argument("--doc-repeat", type=int, default=8)
    ap.add_argument("--doc-repeat-alt", type=int, default=8)
    ap.add_argument("--port", type=int, default=5620)
    ap.add_argument("--gpu-pool", default="2,3,5,6")
    ap.add_argument("--model-path",
                    default="/home/u/downloads/models/Qwen3.6-35B-A3B")
    args = ap.parse_args()

    results = {}
    for i, (label, extra) in enumerate(CELLS):
        port = args.port + i
        cmd = [sys.executable, "-m", "icn_proto.run_cluster",
               "--policy", "p2", "--repr", "bf16",
               "--sessions", str(args.sessions),
               "--turns-per-session", str(args.turns_per_session),
               "--doc-repeat", str(args.doc_repeat),
               "--doc-repeat-alt", str(args.doc_repeat_alt),
               "--doc-chars", str(args.doc_chars),
               "--port", str(port),
               "--gpu-pool", args.gpu_pool,
               "--model-path", args.model_path] + extra
        print(f"\n[sharing] === {label}: {' '.join(extra) or '(sharing on)'} ===",
              flush=True)
        before = snapshot()
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        after = snapshot()
        fresh = [f for f in after if f not in before]
        if not fresh:
            print(f"[sharing] {label}: no summary produced (rc={rc})")
            continue
        with open(max(fresh, key=lambda f: after[f])) as f:
            s = json.load(f)
        results[label] = s
        print(f"[sharing] {label}: prefill_tokens={s['new_tokens_processed']} "
              f"wall_s={s['wall_s']} transfers={s['transfers']} "
              f"run_s={round(time.time() - t0, 1)}", flush=True)

    if len(results) == 2:
        base = results["noshare"]["new_tokens_processed"]
        icn = results["icn"]["new_tokens_processed"]
        print("\n=== THE number ===")
        print(f"prefill tokens  no-share: {base:>9,}   ICN: {icn:>9,}")
        print(f"compute saved by ICN sharing: {base - icn:,} tokens "
              f"({(1 - icn / base) * 100:.1f}% of total prefill)")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT, f"sharing_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cells": {k: {kk: v[kk] for kk in
                                ("new_tokens_processed", "wall_s", "transfers",
                                 "recompute_tokens", "nrs_reuse")}
                             for k, v in results.items()}},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[sharing] -> {path}")


if __name__ == "__main__":
    main()
