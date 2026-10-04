#!/usr/bin/env bash
# 64k slots sweep, docs/28 §8 step 3c.
# Restarts the compressed server at V8_COMPRESS_SLOTS in {1024, 2048, 4096}
# on GPUs 0,1 (port 18002) and runs eval_longctx_server.py per slot.
# Results land in results/eval_64k_slots<S>.json.
#
# Usage (remote, from LLM_LUT/v8):
#     bash tools/run_slots_sweep.sh
# Env overrides:
#     MODEL=/path SLOTS_LIST="1024 2048 4096" PORT=18002 bash tools/run_slots_sweep.sh
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/longctx_multi_turn_65536.jsonl}"
SLOTS_LIST="${SLOTS_LIST:-1024 2048 4096}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"
MAX_LEN=131072
STARTUP_SLEEP="${STARTUP_SLEEP:-240}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p results logs

# pkill -f "vllm_plugin.serve" only kills the APIServer main process; the
# spawned EngineCore/Worker children have `spawn_main` cmdlines that don't
# match, turn into orphans, and keep holding GPU memory (实测堆到 73GB 后
# 新服务 ValueError: Free memory ... less than desired)。清理必须走 GPU
# 占用表，且只杀自己 uid 的进程（共享机，不能误伤别人）。
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

# refuse to start until every target GPU has < 10 GiB residue (fresh server
# needs ~40 GiB per GPU for this model)
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

    CUDA_VISIBLE_DEVICES="$GPUS" V8_COMPRESS_SLOTS="$S" \
        python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --max-model-len "$MAX_LEN" \
        --tensor-parallel-size 2 --max-num-seqs 4 \
        --port "$PORT" > "logs/vllm_64k_s${S}.log" 2>&1 &

    # wait for health instead of a blind fixed sleep
    for i in $(seq 1 $((STARTUP_SLEEP / 10))); do
        if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "UP slots=$S (after ~$((i * 10))s)"
            break
        fi
        sleep 10
        if [ "$i" -eq $((STARTUP_SLEEP / 10)) ]; then
            echo "FAILED to start slots=$S, root cause (Traceback sections):"
            grep -n -A30 "Traceback" "logs/vllm_64k_s${S}.log" | head -120
            echo "--- (log: logs/vllm_64k_s${S}.log) ---"
            continue 2
        fi
    done

    python tools/eval_longctx_server.py \
        --base-url "http://localhost:${PORT}" \
        --model "$MODEL" --data "$DATA" \
        --out "results/eval_64k_slots${S}.json"
done

echo "=== sweep done ==="
for S in $SLOTS_LIST; do
    python tools/dump_eval_answers.py --results "results/eval_64k_slots${S}.json" 2>/dev/null | head -4
done
