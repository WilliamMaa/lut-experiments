#!/usr/bin/env bash
# Concurrency sweep, docs/28 §8 step 4. For each V8_COMPRESS_SLOTS value:
# start the compressed server ONCE on GPUs 6,7 (port 18002), then run
# tools/bench_concurrency.py at each concurrency level N against it.
# Results land in results/bench_c<N>_slots<S>.json.
#
# Usage (remote, from LLM_LUT/v8):
#     bash tools/run_concurrency_sweep.sh
# Env overrides:
#     MODEL=/path SLOTS_LIST="1024 4096" N_LIST="1 8 16" PORT=18002 \
#         bash tools/run_concurrency_sweep.sh
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/longctx_multi_turn_65536.jsonl}"
SLOTS_LIST="${SLOTS_LIST:-1024 4096}"
N_LIST="${N_LIST:-1 8 16}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"
MAX_LEN=131072
STARTUP_SLEEP="${STARTUP_SLEEP:-240}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p results logs

# Same orphan-cleanup lesson as run_slots_sweep.sh: pkill only kills the
# APIServer; spawned EngineCore/Worker children must be killed via the GPU
# occupancy table, own uid only (shared machine).
cleanup() {
    pkill -f "vllm_plugin.serve" 2>/dev/null
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
    echo "GPUs did not free up in 300s; check nvidia-smi and kill leftovers manually"
    return 1
}

for S in $SLOTS_LIST; do
    echo "=== slots=$S ==="
    cleanup
    if ! wait_gpus_free; then
        continue
    fi

    # gpu-memory-utilization 0.88: same plugin prefill-transient headroom
    # decision as run_slots_sweep.sh (v2026-10-04o), keep the two scripts
    # identical on server flags so their results are comparable.
    CUDA_VISIBLE_DEVICES="$GPUS" V8_COMPRESS_SLOTS="$S" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --max-model-len "$MAX_LEN" \
        --tensor-parallel-size 2 --max-num-seqs 4 \
        --gpu-memory-utilization 0.88 \
        --port "$PORT" > "logs/vllm_ccy_s${S}.log" 2>&1 &

    up=0
    for i in $(seq 1 $((STARTUP_SLEEP / 10))); do
        if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "UP slots=$S (after ~$((i * 10))s)"
            up=1
            break
        fi
        sleep 10
    done
    if [ "$up" != "1" ]; then
        echo "FAILED to start slots=$S, root cause (Traceback sections):"
        grep -n -A30 "Traceback" "logs/vllm_ccy_s${S}.log" | head -120
        echo "--- (log: logs/vllm_ccy_s${S}.log) ---"
        continue
    fi

    for N in $N_LIST; do
        echo "--- slots=$S concurrency=$N ---"
        python tools/bench_concurrency.py \
            --base-url "http://localhost:${PORT}" \
            --model "$MODEL" --data "$DATA" \
            --concurrency "$N" \
            --out "results/bench_c${N}_slots${S}.json"
    done
done

echo "=== sweep done ==="
for S in $SLOTS_LIST; do
    for N in $N_LIST; do
        f="results/bench_c${N}_slots${S}.json"
        if [ -f "$f" ]; then
            python - "$f" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    s = json.load(fh)
print(f"{sys.argv[1]}: acc={s['fact_acc']:.4f} "
      f"wall={s['wall_seconds']:.0f}s errs={s['n_errors']}")
PYEOF
        fi
    done
done
