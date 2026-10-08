#!/usr/bin/env bash
# 06j phase-3: Gate B shape proof. One v8 server; three probe points:
#   P1: N=1 x 64k   -> usage1_64k
#   P2: N=1 x 32k   -> usage1_32k  (flat-vs-length: P2 ~= P1)
#   P3: N=16 x 64k  -> usage16     (linear-in-N:    P3 ~= 16 x P1)
# Pass criteria: |P2/P1 - 1| < 0.35 and |P3/P1 - 16| / 16 < 0.35
# Usage: bash tools/verify_06j_phase3.sh
set -u

MODEL="${MODEL:-/home/u/downloads/models/Qwen3.6-35B-A3B}"
PORT="${PORT:-18002}"
GPUS="${GPUS:-6,7}"
DATA64="${DATA64:-data/longctx_multi_turn_65536_64docs.jsonl}"
DATA32="${DATA32:-data/longctx_multi_turn_32768.jsonl}"

cd "$(dirname "$0")/.." || exit 1

pkill -f vllm_plugin.serve 2>/dev/null
sleep 3

CUDA_VISIBLE_DEVICES="$GPUS" \
    python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --no-enable-prefix-caching \
        --max-model-len 131072 --tensor-parallel-size 2 \
        --max-num-seqs 16 --gpu-memory-utilization 0.88 \
        --port "$PORT" > logs/verify_06j_phase3.log 2>&1 &
SRV=$!

up=0
for i in $(seq 1 60); do
    if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "server died during startup"; tail -30 logs/verify_06j_phase3.log; exit 1
    fi
    sleep 10
done
[ "$up" = "1" ] || { echo "server did not come up"; kill "$SRV" 2>/dev/null; exit 1; }

run_probe() { # $1 tag, $2 concurrency, $3 max-docs, $4 data
    echo "--- probe $1 (N=$2 docs=$3 data=$4) ---"
    PORT="$PORT" CONCURRENCY="$2" MAX_DOCS="$3" DATA="$4" \
        OUT="results/verify_06j_phase3_$1.json" \
        bash tools/telemetry_probe.sh 2>&1 | grep -E "peak_kv_usage|preemptions AFTER|fact_acc|errors="
}

P1=$(run_probe n1_64k 1 2 "$DATA64" | awk -F= '/peak/{print $2}')
P2=$(run_probe n1_32k 1 2 "$DATA32" | awk -F= '/peak/{print $2}')
P3=$(run_probe n16_64k 16 16 "$DATA64" | awk -F= '/peak/{print $2}')

kill "$SRV" 2>/dev/null
sleep 5
pkill -f vllm_plugin.serve 2>/dev/null

echo "=== Gate B shape ==="
echo "usage N=1 64k : $P1"
echo "usage N=1 32k : $P2   (flat-vs-length: want ~= P1)"
echo "usage N=16 64k: $P3   (linear-in-N:    want ~= 16*P1)"
python - "$P1" "$P2" "$P3" <<'EOF'
import sys
p1, p2, p3 = map(float, sys.argv[1:4])
flat = abs(p2 / p1 - 1)
lin = abs(p3 / p1 - 16) / 16
print(f"flat deviation={flat:.2%}  linear deviation={lin:.2%}")
print("GATE B " + ("PASS" if flat < 0.35 and lin < 0.35 else "FAIL"))
EOF
