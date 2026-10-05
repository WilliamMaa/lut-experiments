#!/usr/bin/env bash
# Quick concurrency repro, docs/28 §8 step 4b.
# Starts the compressed server ONCE on GPUs 6,7 (all flags baked in here —
# do NOT copy the serve line into a terminal by hand; this script exists
# because pasting long single-line commands into this terminal eats
# random characters and silently corrupts env vars) and runs a cheap
# concurrency-4 bench on the 32k data (2 docs) to verify the v2026-10-04p
# request-identity fix.
#
# Usage (remote, from LLM_LUT/v8):
#     bash tools/repro_concurrency.sh
# Env overrides:
#     SLOTS=4096 N=8 DOCS=4 DATA=data/longctx_multi_turn_65536.jsonl \
#         bash tools/repro_concurrency.sh
set -u

MODEL="/home/u/downloads/models/Qwen3.6-35B-A3B"
DATA="${DATA:-data/longctx_multi_turn_32768.jsonl}"
SLOTS="${SLOTS:-1024}"
N="${N:-4}"
DOCS="${DOCS:-2}"
PORT="${PORT:-18002}"
GPUS="6,7"
MAX_LEN=131072
STARTUP_SLEEP=240

cd "$(dirname "$0")/.." || exit 1
mkdir -p results logs

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

echo "=== cleanup ==="
cleanup

echo "=== starting server (slots=$SLOTS gpus=$GPUS port=$PORT) ==="
# All env vars are set INSIDE this script on purpose: pasting env-var
# prefixes into the terminal has repeatedly eaten characters
# (e.g. "V8_" vanished, merging into CUDA_VISIBLE_DEVICES).
CUDA_VISIBLE_DEVICES="$GPUS" \
V8_COMPRESS_SLOTS="$SLOTS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m vllm_plugin.serve "$MODEL" \
        --enforce-eager --max-model-len "$MAX_LEN" \
        --tensor-parallel-size 2 --max-num-seqs 4 \
        --gpu-memory-utilization 0.88 \
        --port "$PORT" > logs/vllm_repro.log 2>&1 &

up=0
for i in $(seq 1 $((STARTUP_SLEEP / 10))); do
    if curl -s "localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "UP (after ~$((i * 10))s)"
        up=1
        break
    fi
    sleep 10
done
if [ "$up" != "1" ]; then
    echo "FAILED to start, root cause (Traceback sections):"
    grep -n -A30 "Traceback" logs/vllm_repro.log | head -120
    echo "--- (log: logs/vllm_repro.log) ---"
    exit 1
fi

echo "=== version check (must say 2026-10-04p) ==="
grep "PLUGIN_VERSION" logs/vllm_repro.log | head -2

echo "=== bench: N=$N docs=$DOCS data=$DATA ==="
python tools/bench_concurrency.py \
    --base-url "http://localhost:${PORT}" \
    --model "$MODEL" --data "$DATA" \
    --concurrency "$N" --max-docs "$DOCS" \
    --out results/bench_repro_c${N}.json

echo "=== verification ==="
echo -n "state reset count (must be 0): "
grep -c "state reset" logs/vllm_repro.log
echo -n "stale metadata healed count (large is OK): "
grep -c "stale metadata healed" logs/vllm_repro.log
echo "=== sample answers ==="
python - "results/bench_repro_c${N}.json" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
for r in d["records"]:
    for a in r["answers"][:2]:
        print(r["sid"], r["correct"], "|", a["answer"][:100].replace("\n", " "))
PYEOF
