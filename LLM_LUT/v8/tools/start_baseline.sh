#!/usr/bin/env bash
# Start the full-KV baseline server (NO plugin) on GPUs 0,1, port 18003.
# Waits for /health, prints version/memory lines. Uses util 0.88 to match
# the plugin server's headroom decision (v2026-10-04o); stock kernels are
# used (no --enforce-eager) — this is the reference point, not the plugin.
#
# Usage (remote, from LLM_LUT/v8):
#     bash tools/start_baseline.sh
set -u

MODEL="/home/u/downloads/models/Qwen3.6-35B-A3B"
PORT="${PORT:-18003}"
GPUS="0,1"
MAX_LEN=131072
STARTUP_SLEEP=300

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

cleanup() {
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

echo "=== cleanup GPUs $GPUS ==="
cleanup

echo "=== starting baseline (port $PORT) ==="
CUDA_VISIBLE_DEVICES="$GPUS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    vllm serve "$MODEL" \
        --max-model-len "$MAX_LEN" \
        --tensor-parallel-size 2 --max-num-seqs 4 \
        --gpu-memory-utilization 0.88 \
        --port "$PORT" > logs/vllm_baseline.log 2>&1 &

for i in $(seq 1 $((STARTUP_SLEEP / 10))); do
    if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "UP (after ~$((i * 10))s)"
        grep "GPU KV cache size" logs/vllm_baseline.log | tail -1
        exit 0
    fi
    sleep 10
done
echo "FAILED to start, root cause (Traceback sections):"
grep -n -A30 "Traceback" logs/vllm_baseline.log | head -120
echo "--- (log: logs/vllm_baseline.log) ---"
exit 1
