#!/usr/bin/env python3
"""Cluster launcher: spawn workers + run the scheduler in this process.

GPU allocation is fully explicit: `--gpu-pool "2,3,5,6"` is dealt into
`--gpus-per-worker 2` consecutive cards per worker, each worker pinned via
CUDA_VISIBLE_DEVICES and loading the model with the fixed `balanced_low_0`
split across its two cards. No auto device_map anywhere (project red line).

Example:
    python -m icn_proto.run_cluster --policy p1 --repr m_sp4 --sessions 4 \
        --turns-per-session 4 --gpu-pool 2,3,5,6 \
        --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from icn_proto import scheduler as sched_mod


def spawn_workers(pool, gpus_per_worker, scheduler_addr, args):
    procs = []
    for wi in range(len(pool) // gpus_per_worker):
        cards = pool[wi * gpus_per_worker:(wi + 1) * gpus_per_worker]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, cards)))
        cmd = [sys.executable, "-m", "icn_proto.worker",
               "--scheduler", scheduler_addr,
               "--model-path", args.model_path,
               "--device", args.device,
               "--dtype", args.dtype,
               "--repr", args.repr,
               "--worker-id", f"w{wi}",
               "--spill-mb", str(getattr(args, "spill_mb", 0.0))]
        print(f"[launcher] w{wi} <- CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
    return procs


def watchdog(procs, stop):
    """If any worker dies unexpectedly, kill the whole experiment."""
    while not stop.is_set():
        for i, p in enumerate(procs):
            if p.poll() is not None:
                print(f"[launcher] worker pid={p.pid} exited rc={p.returncode} "
                      f"unexpectedly, aborting", file=sys.stderr)
                os._exit(2)
        time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-pool", default="2,3,5,6",
                    help="comma-separated free GPU ids, dealt round-robin")
    ap.add_argument("--gpus-per-worker", type=int, default=2)
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--device", default="explicit_even",
                    help="worker-side model placement: explicit_even "
                         "(default) = deterministic half/half layer split "
                         "over the worker's visible cards — the intent of "
                         "balanced_low_0 without tenant-dependent "
                         "reshuffling. Never 'auto' (project red line).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-share", action="store_true",
                    help="baseline shortcut for the sharing A/B (run_sharing): "
                         "forces --policy b0 (load-only, no content awareness, "
                         "every turn prefills its full prefix). Kept so old "
                         "matrix cells keep working; new experiments should "
                         "pass --policy directly.")
    sched_mod.add_args(ap)
    args = ap.parse_args()

    if args.no_share:
        if args.policy != "b0":
            print(f"[launcher] --no-share: overriding --policy {args.policy} "
                  f"-> b0 (load-only baseline)")
        args.policy = "b0"

    pool = [int(x) for x in args.gpu_pool.split(",") if x.strip() != ""]
    if len(pool) % args.gpus_per_worker != 0:
        ap.error(f"gpu-pool ({len(pool)} cards) not divisible by "
                 f"--gpus-per-worker {args.gpus_per_worker}")
    n_workers = len(pool) // args.gpus_per_worker
    args.bind = f"tcp://127.0.0.1:{args.port}"
    scheduler_addr = args.bind
    worker_ids = [f"w{i}" for i in range(n_workers)]

    procs = spawn_workers(pool, args.gpus_per_worker, scheduler_addr, args)
    stop = threading.Event()
    wd = threading.Thread(target=watchdog, args=(procs, stop), daemon=True)
    wd.start()

    def kill_all(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(3)
        for p in procs:
            if p.poll() is None:
                p.kill()
        os._exit(130)

    signal.signal(signal.SIGINT, kill_all)
    signal.signal(signal.SIGTERM, kill_all)

    try:
        out = sched_mod.Scheduler(args, worker_ids).run()
    finally:
        stop.set()
        for p in procs:
            try:
                rc = p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
            print(f"[launcher] worker pid={p.pid} rc={rc}")
    sys.exit(0 if out.get("failed", 1) == 0 else 1)


if __name__ == "__main__":
    main()
