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

for S in $SLOTS_LIST; do
    echo "=== slots=$S ==="
    pkill -f "vllm_plugin.serve" 2>/dev/null ; sleep 3

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
