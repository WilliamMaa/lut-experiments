#!/usr/bin/env python3
"""Step-5 experiment matrix: baseline ladder x content skew x reps.

Each cell spawns a FRESH cluster via run_cluster.py (new workers, new
scheduler), runs the closed-loop workload, and parses the summary JSON
it leaves in results/icn_proto/. The manifest is rewritten after every
cell so an interrupted matrix keeps its results.

Differentiation rationale (4 workers): with doc-share=2 and 16
sessions, each hot document lands on ~2 of 4 workers. B3 pays a
critical-path fetch every time load spills onto a cold worker; ours
pays one off-path replication (G_rep > 0) and resumes locally after.
At 2 workers both docs end up everywhere after the first arrivals and
the controller is correctly idle — do not read that as a bug.

Usage:
    python -m icn_proto.matrix_step5 \
        --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
        --gpu-pool 0,1,2,3,4,5,6,7 --sessions 16 --reps 3
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results", "icn_proto")

METRICS = ("wall_s", "throughput_rps", "hit_rate", "resumed",
           "new_tokens_processed", "published_blocks", "transfers",
           "transfer_bytes", "replications", "replicated_bytes",
           "evictions", "avg_latency_s", "failed", "prefill_rate",
           "xfer_rate")


def pctl(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def newest_json(after, stdout):
    """The scheduler prints 'summary -> <path>'; trust that, fall back
    to mtime only if the line is missing."""
    for line in reversed((stdout or "").splitlines()):
        if "summary ->" in line:
            p = line.split("summary ->", 1)[1].strip()
            if os.path.exists(p):
                return p
    files = [f for f in glob.glob(os.path.join(RESULTS, "blkcluster_*.json"))
             if os.path.getmtime(f) > after]
    return max(files, key=os.path.getmtime) if files else None


def run_cell(args, share, pol, rep, port):
    subprocess.run(["pkill", "-f", "icn_proto.worker"],
                   check=False, capture_output=True)
    time.sleep(3)
    cmd = [sys.executable, "-m", "icn_proto.run_cluster",
           "--policy", pol,
           "--sessions", str(args.sessions),
           "--turns-per-session", str(args.turns_per_session),
           "--doc-chars", str(args.doc_chars),
           "--doc-repeat", str(args.doc_repeat),
           "--doc-repeat-alt", str(args.doc_repeat_alt),
           "--doc-share", str(share),
           "--port", str(port),
           "--gpu-pool", args.gpu_pool,
           "--gpus-per-worker", str(args.gpus_per_worker),
           "--model-path", args.model_path]
    if args.budget_mb > 0:
        cmd += ["--worker-mem-budget-mb", str(args.budget_mb)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    path = newest_json(t0 - 5, proc.stdout)
    row = {"share": share, "policy": pol, "rep": rep,
           "budget_mb": args.budget_mb,
           "rc": proc.returncode, "cell_s": round(time.time() - t0, 1),
           "json": path,
           # keep the tail on every cell: crashes that still match an
           # mtime-old JSON used to lose their traceback
           "stderr_tail": ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-3000:]}
    if path:
        d = json.load(open(path, encoding="utf-8"))
        for k in METRICS:
            row[k] = d.get(k)
        lat = [r["latency_s"] for r in d["records"] if r.get("latency_s")]
        row["p50_s"] = round(pctl(lat, 0.5), 4)
        row["p95_s"] = round(pctl(lat, 0.95), 4)
        row["slo_att"] = round(sum(1 for x in lat if x <= args.slo_s)
                               / max(1, len(lat)), 4)
        row["n"] = len(lat)
    else:
        row["error"] = (proc.stderr or "")[-2000:]
    return row


def aggregate(rows, slo_s):
    groups = {}
    for r in rows:
        if r.get("json") and r.get("failed") == 0:
            groups.setdefault((r.get("budget_mb", 0.0), r["share"],
                               r["policy"]), []).append(r)
    print(f"\n{'budget':>7} {'share':>6} {'policy':<7}{'runs':>5}{'rps':>8}"
          f"{'hit':>7}"
          f"{'new_tok':>9}{'xfer':>6}{'repl':>6}{'evict':>7}{'p50':>8}"
          f"{'p95':>8}"
          f"{'SLO@' + str(slo_s) + 's':>9}{'wall':>8}")
    for (budget, share, pol), rs in sorted(groups.items()):
        def m(k):
            return statistics.mean(r[k] for r in rs)
        print(f"{budget:>7.0f} {share:>6} {pol:<7}{len(rs):>5}"
              f"{m('throughput_rps'):>8.4f}"
              f"{m('hit_rate'):>7.3f}{m('new_tokens_processed'):>9.0f}"
              f"{m('transfers'):>6.1f}{m('replications'):>6.1f}"
              f"{m('evictions'):>7.1f}"
              f"{m('p50_s'):>8.3f}{m('p95_s'):>8.3f}{m('slo_att'):>9.3f}"
              f"{m('wall_s'):>8.1f}")
    bad = [r for r in rows if not r.get("json") or r.get("failed")]
    if bad:
        print(f"\nWARNING: {len(bad)} cell(s) failed or produced no JSON:")
        for r in bad:
            print(f"  share={r['share']} policy={r['policy']} rep={r['rep']} "
                  f"rc={r.get('rc')} failed={r.get('failed')} json={r.get('json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpu-pool", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--gpus-per-worker", type=int, default=2)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns-per-session", type=int, default=3)
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--doc-repeat", type=int, default=2)
    ap.add_argument("--doc-repeat-alt", type=int, default=2)
    ap.add_argument("--shares", default="2,8",
                    help="doc-share values: 2 = global hot docs reused by "
                         "half the sessions; 8 = mild reuse")
    ap.add_argument("--policies", default="b0,b1,b2,b3,ours")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port-base", type=int, default=5700)
    ap.add_argument("--slo-s", type=float, default=2.0,
                    help="per-turn latency SLO for attainment")
    ap.add_argument("--budget-mb", type=float, default=0.0,
                    help="per-worker residency budget passed through to "
                         "run_cluster --worker-mem-budget-mb (0 = off)")
    ap.add_argument("--manifest", default=os.path.join(
        RESULTS, "matrix_step5_manifest.json"))
    ap.add_argument("--drop-bad", action="store_true",
                    help="remove corrupt rows from the manifest (rc!=0, "
                         "no JSON, failed>0, or a JSON path shared by "
                         "multiple rows) and exit — then rerun to re-do "
                         "just those cells")
    args = ap.parse_args()

    shares = [int(x) for x in args.shares.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    if args.drop_bad:
        if not os.path.exists(args.manifest):
            sys.exit("no manifest to clean")
        rows = json.load(open(args.manifest, encoding="utf-8"))
        seen = {}
        for r in rows:
            if r.get("json"):
                seen[r["json"]] = seen.get(r["json"], 0) + 1
        keep, drop = [], []
        for r in rows:
            dup = r.get("json") and seen[r["json"]] > 1
            if dup:
                # the earliest row owning this JSON is the real run;
                # later rows matched it mtime-only after crashing
                if r["json"] not in [k.get("json") for k in keep]:
                    keep.append(r)
                    continue
                drop.append(r)
                continue
            if (not r.get("json") or r.get("rc") not in (0, None)
                    or r.get("failed") not in (0, None)):
                drop.append(r)
                continue
            keep.append(r)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(keep, f, indent=1)
        print(f"dropped {len(drop)} bad row(s):")
        for r in drop:
            print(f"  share={r['share']} policy={r['policy']} rep={r['rep']} "
                  f"rc={r.get('rc')} json={os.path.basename(r.get('json') or '?')}")
        return

    cells = [(sh, pol, rep) for sh in shares for pol in policies
             for rep in range(args.reps)]
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    if os.path.exists(args.manifest):
        try:
            rows = json.load(open(args.manifest, encoding="utf-8"))
            print(f"resuming: {len(rows)} cell(s) already in manifest")
        except json.JSONDecodeError:
            pass
    # budget is part of the cell key: a v2 (budget>0) run must not
    # inherit v1 (budget=0) cells from the manifest
    done = {(r.get("budget_mb", 0.0), r["share"], r["policy"], r["rep"])
            for r in rows}

    for i, (sh, pol, rep) in enumerate(cells):
        if (args.budget_mb, sh, pol, rep) in done:
            continue
        print(f"=== cell share={sh} policy={pol} rep={rep} "
              f"({len(done) + 1}/{len(cells)}) ===", flush=True)
        row = run_cell(args, sh, pol, rep, args.port_base + i)
        rows.append(row)
        done.add((sh, pol, rep))
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        tag = "OK" if row.get("json") and row.get("failed") == 0 else "BAD"
        print(f"    {tag} rc={row.get('rc')} cell_s={row.get('cell_s')} "
              f"json={os.path.basename(row['json'] or '?')}", flush=True)
    aggregate(rows, args.slo_s)


if __name__ == "__main__":
    main()
