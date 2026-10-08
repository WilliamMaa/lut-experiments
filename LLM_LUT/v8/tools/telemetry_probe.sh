#!/usr/bin/env bash
# Telemetry probe (docs/36 step A): run a shortened bench against a LIVE
# server and sample vLLM metrics while it runs. Answers: was full-KV under
# real KV pressure (peak kv_cache_usage, preemption count)?
#
# Prereq: full server already up on PORT (started WITHOUT --enable-metrics;
# 0.19.x exposes /metrics by default).
#
# Usage: bash tools/telemetry_probe.sh
#   Env: PORT=18002 CONCURRENCY=16 MAX_DOCS=16 DATA=... OUT=...
set -u

PORT="${PORT:-18002}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_DOCS="${MAX_DOCS:-16}"
MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/longctx_multi_turn_65536_64docs.jsonl}"
OUT="${OUT:-results/telemetry_full_c32.json}"

if ! curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "no live server on port $PORT; start it first"
    exit 1
fi

curl -s "localhost:${PORT}/metrics" | grep -E "^vllm:num_preemptions_total" \
    | sed 's/^/preemptions BEFORE: /'

python tools/bench_concurrency.py \
    --base-url "http://localhost:${PORT}" \
    --model "$MODEL" --data "$DATA" \
    --concurrency "$CONCURRENCY" --max-docs "$MAX_DOCS" \
    --out "$OUT" > logs/telemetry_bench.out 2>&1 &
BP=$!

MAXU=0
while kill -0 "$BP" 2>/dev/null; do
    U=$(curl -s "localhost:${PORT}/metrics" \
        | awk '/^vllm:kv_cache_usage_perc/{print $2}')
    if [ -n "$U" ]; then
        MAXU=$(python -c "print(max(${MAXU}, ${U}))")
    fi
    sleep 10
done
wait "$BP"

echo "peak_kv_usage=$MAXU"
curl -s "localhost:${PORT}/metrics" | grep -E "^vllm:num_preemptions_total" \
    | sed 's/^/preemptions AFTER:  /'
tail -5 logs/telemetry_bench.out
