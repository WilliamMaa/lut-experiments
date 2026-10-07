#!/usr/bin/env bash
# Capacity sweep (docs/34): v8 vs full-KV under IDENTICAL conditions and a
# real session backlog. This is the missing baseline that answers "v8
# increases supported concurrency from A to B" — the per-concurrency v8-only
# table (docs/33) cannot attribute anything to KV compression.
#
# Workload: 64 sessions x ~64k tokens x 8-turn QA (one session per doc).
# Backends: full (plain vllm serve, plugin absent), v8-1024, v8-4096.
# Grid: N in N_LIST (offered concurrency == server --max-num-seqs, so
# admission is never the hidden limiter; max-num-seqs is fixed at server
# start, hence one server restart per (backend, N)).
#
# Identical for every config: GPUs, TP2, --enforce-eager, prefix caching
# off, chunked prefill default (8192), --gpu-memory-utilization 0.88,
# --max-model-len 131072, same prompts/max-tokens.
#
# Recorded per config: fact_acc, errors, wall, sessions/h, P50/P95 session
# latency, peak GPU HBM (polled during bench), server KV cache size (log),
# OOM/admission/preemption counts (log). Results:
#   results/capacity_<backend>_c<N>.json      (bench summary + records)
#   results/capacity_<backend>_c<N>.hbm       (peak MiB per GPU)
#   logs/capacity_<backend>_c<N>.log          (server log)
#
# Prereq (once): generate the 64-doc workload
#   python tools/gen_longctx_multiturn.py --target-tokens 65536 \
#       --num-docs 64 --tokenizer-path "$MODEL" \
#       --out data/longctx_multi_turn_65536_64docs.jsonl
#
# Usage (remote, from LLM_LUT/v8). Staged — full first (~3-4h), v8 after:
#   BACKENDS="full" bash tools/run_capacity_sweep.sh
#   BACKENDS="v8-1024" bash tools/run_capacity_sweep.sh
#   BACKENDS="v8-4096" bash tools/run_capacity_sweep.sh
# Env overrides: N_LIST="1 2 4 8 16 32" DATA=... GPUS=6,7 PORT=18002
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/longctx_multi_turn_65536_64docs.jsonl}"
BACKENDS="${BACKENDS:-full v8-1024 v8-4096}"
N_LIST="${N_LIST:-1 2 4 8 16 32}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"
MAX_LEN=131072
STARTUP_SLEEP="${STARTUP_SLEEP:-300}"
UTIL=0.88

cd "$(dirname "$0")/.." || exit 1
mkdir -p results logs

if [ ! -f "$DATA" ]; then
    echo "missing workload: $DATA"
    echo "generate it (see header), then rerun"
    exit 1
fi

cleanup() {
    pkill -f "vllm_plugin.serve" 2>/dev/null
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "vllm.entrypoints" 2>/dev/null
    sleep 3
    local me gpu pid owner
    me=$(id -u)
    for gpu in ${GPUS//,/ }; do
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu" 2>/dev/null); do
            owner=$(stat -c %u "/proc/$pid" 2>/dev/null)
            if [ "$owner" = "$me" ]; then
                echo "  killing leftover pid=$pid on GPU $gpu"
                kill "$pid" 2>/dev/null
            fi
        done
    done
    sleep 5
}

wait_gpus_free() {
    local gpu used_mib retries=0 ok
    while [ "$retries" -lt 30 ]; do
        ok=1
        for gpu in ${GPUS//,/ }; do
            used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" 2>/dev/null | head -1)
            if [ "${used_mib:-99999}" -gt 10000 ]; then
                echo "  GPU $gpu still used: ${used_mib} MiB, waiting..."
                ok=0
            fi
        done
        [ "$ok" = "1" ] && return 0
        sleep 10
        retries=$((retries + 1))
    done
    echo "GPUs did not free up in 300s; check nvidia-smi"
    return 1
}

hbm_poll() {
    # $1 = output file; samples both GPUs every 5s until killed
    local out="$1" gpu used
    : > "$out"
    while true; do
        for gpu in ${GPUS//,/ }; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" 2>/dev/null | head -1)
            echo "${gpu} ${used:-0}" >> "$out"
        done
        sleep 5
    done
}

start_server() {
    # $1 = backend (full | v8-1024 | v8-4096), $2 = N, $3 = log file
    local backend="$1" n="$2" log="$3"
    local -a flags=(--enforce-eager --max-model-len "$MAX_LEN"
                    --tensor-parallel-size 2 --max-num-seqs "$n"
                    --gpu-memory-utilization "$UTIL"
                    --disable-prefix-caching --port "$PORT")
    if [ "$backend" = "full" ]; then
        CUDA_VISIBLE_DEVICES="$GPUS" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            vllm serve "$MODEL" "${flags[@]}" > "$log" 2>&1 &
    else
        local slots="${backend#v8-}"
        CUDA_VISIBLE_DEVICES="$GPUS" V8_COMPRESS_SLOTS="$slots" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -m vllm_plugin.serve "$MODEL" "${flags[@]}" \
            > "$log" 2>&1 &
    fi
    local up=0 i
    for i in $(seq 1 $((STARTUP_SLEEP / 10))); do
        if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "UP $backend N=$n (after ~$((i * 10))s)"
            grep -E "GPU KV cache size|Maximum concurrency" "$log" | tail -2
            up=1
            break
        fi
        if grep -q "CUDA out of memory\|torch.OutOfMemoryError" "$log" 2>/dev/null; then
            echo "OOM during startup of $backend N=$n"
            return 1
        fi
        sleep 10
    done
    [ "$up" = "1" ] && return 0
    echo "FAILED to start $backend N=$n (Traceback):"
    grep -n -A30 "Traceback" "$log" | head -80
    return 1
}

for BACKEND in $BACKENDS; do
    for N in $N_LIST; do
        tag="${BACKEND}_c${N}"
        echo "=== $tag ==="
        cleanup
        if ! wait_gpus_free; then
            continue
        fi
        log="logs/capacity_${tag}.log"
        if ! start_server "$BACKEND" "$N" "$log"; then
            continue
        fi

        hbm="results/capacity_${tag}.hbm"
        hbm_poll "$hbm" &
        local_poller=$!
        python tools/bench_concurrency.py \
            --base-url "http://localhost:${PORT}" \
            --model "$MODEL" --data "$DATA" \
            --concurrency "$N" \
            --out "results/capacity_${tag}.json"
        kill "$local_poller" 2>/dev/null
        wait "$local_poller" 2>/dev/null

        # post-run health: server alive? OOM / preemption in log?
        if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "server alive after bench"
        else
            echo "SERVER DIED during bench (check $log)"
        fi
        grep -c "CUDA out of memory\|torch.OutOfMemoryError" "$log" 2>/dev/null \
            | sed 's/^/  OOM lines: /'
        grep -c "rewind detected" "$log" 2>/dev/null | sed 's/^/  rewind resets: /'
    done
done

cleanup

echo "=== capacity sweep done ==="
python - <<'PYEOF'
import glob, json, statistics
for f in sorted(glob.glob("results/capacity_*.json")):
    with open(f, encoding="utf-8") as fh:
        s = json.load(fh)
    durs = sorted(s.get("session_seconds", []))
    p50 = durs[len(durs) // 2] if durs else float("nan")
    p95 = durs[min(len(durs) - 1, int(0.95 * len(durs)))] if durs else float("nan")
    hbm_f = f.replace(".json", ".hbm")
    peak = {}
    try:
        with open(hbm_f) as hh:
            for line in hh:
                gpu, mib = line.split()
                peak[gpu] = max(peak.get(gpu, 0), int(mib))
    except OSError:
        pass
    wall = s["wall_seconds"]
    print(f"{f}: acc={s['fact_acc']:.4f} errs={s['n_errors']} "
          f"wall={wall:.0f}s sess/h={s['n_sessions'] / wall * 3600:.1f} "
          f"P50={p50:.0f}s P95={p95:.0f}s peakHBM={peak}")
PYEOF
