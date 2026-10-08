#!/usr/bin/env bash
# 06j phase-2 verification (docs/37 gates B + quality check under pressure).
# Gate B: with fixed-budget accounting, 16 concurrent 64k sessions must keep
# kv_cache_usage_perc FLAT and small (vs full's token-proportional growth),
# i.e. physical residency per request ~= B_target blocks regardless of the
# 64k logical length. Quality: 64k prompts exceed SLOTS+STAGING (17408), so
# pressure eviction is active -- fact_acc here is the quality gate for it.
# Usage (remote, vllm_py310, from LLM_LUT/v8): bash tools/verify_06j_phase2.sh
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"
DATA="${DATA:-data/longctx_multi_turn_65536_64docs.jsonl}"

cd "$(dirname "$0")/.." || exit 1

pkill -f vllm_plugin.serve 2>/dev/null
sleep 3

CUDA_VISIBLE_DEVICES="$GPUS" \
    python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --no-enable-prefix-caching \
        --max-model-len 131072 --tensor-parallel-size 2 \
        --max-num-seqs 16 --gpu-memory-utilization 0.88 \
        --port "$PORT" > logs/verify_06j_phase2.log 2>&1 &
SRV=$!

up=0
for i in $(seq 1 60); do
    if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "server died during startup; last log lines:"
        tail -30 logs/verify_06j_phase2.log
        exit 1
    fi
    sleep 10
done
if [ "$up" != "1" ]; then
    echo "server did not come up in 600s"
    kill "$SRV" 2>/dev/null
    exit 1
fi

echo "=== capacity numbers ==="
grep -E "KV cache size|Maximum concurrency" logs/verify_06j_phase2.log | tail -2

echo "=== Gate B + quality: 16 concurrent x 64k sessions (pressure eviction active) ==="
PORT="$PORT" CONCURRENCY=16 MAX_DOCS=16 DATA="$DATA" \
    OUT=results/verify_06j_phase2_c16.json \
    bash tools/telemetry_probe.sh

kill "$SRV" 2>/dev/null
sleep 5
pkill -f vllm_plugin.serve 2>/dev/null
echo "=== phase2 done ==="
