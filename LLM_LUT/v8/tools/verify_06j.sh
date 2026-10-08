#!/usr/bin/env bash
# 06j verification runbook (docs/37 gates). Run from LLM_LUT/v8 on the
# remote box (vllm_py310 env). Steps: Gate A (integration regression, incl.
# new pressure-eviction case) then Gate D (server startup capacity numbers).
# Usage: bash tools/verify_06j.sh
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"

cd "$(dirname "$0")/.." || exit 1

echo "=== Gate A: integration regression ==="
python vllm_plugin/tests/test_integration.py
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "GATE A FAILED (rc=$rc); stop here"
    exit "$rc"
fi
echo "GATE A PASS"

echo "=== Gate D: startup capacity numbers ==="
pkill -f vllm_plugin.serve 2>/dev/null
sleep 3

CUDA_VISIBLE_DEVICES="$GPUS" \
    python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --no-enable-prefix-caching \
        --max-model-len 131072 --tensor-parallel-size 2 \
        --max-num-seqs 8 --gpu-memory-utilization 0.88 \
        --port "$PORT" > logs/verify_06j_gated.log 2>&1 &
SRV=$!

up=0
for i in $(seq 1 60); do
    if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "server died during startup; last log lines:"
        tail -30 logs/verify_06j_gated.log
        exit 1
    fi
    sleep 10
done

if [ "$up" != "1" ]; then
    echo "server did not come up in 600s; last log lines:"
    tail -30 logs/verify_06j_gated.log
    kill "$SRV" 2>/dev/null
    exit 1
fi

echo "--- capacity numbers (06j: Maximum concurrency should be far above 27.08x) ---"
grep -E "KV cache size|Maximum concurrency" logs/verify_06j_gated.log | tail -4

kill "$SRV" 2>/dev/null
sleep 5
pkill -f vllm_plugin.serve 2>/dev/null
echo "=== verify_06j done; now run: bash tools/repro_concurrency.sh ==="
