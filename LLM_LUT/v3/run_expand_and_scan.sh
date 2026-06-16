#!/usr/bin/env bash
# Run expand_ratio.py for layers 19-23 (independent ranking, empty used_groups)
# then run multi_layer_scan.py.
#
# Usage (from v3/ directory on the GPU server):
#   bash run_expand_and_scan.sh
#
# GPU allocation:
#   - Batch 1: L19 -> cuda:0, L20 -> cuda:1, L21 -> cuda:2
#   - Batch 2: L22 -> cuda:0, L23 -> cuda:1
# Assumes cuda:3 is occupied by another VLLM process.

set -euo pipefail

cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_ROOT="outputs"
TARGET_COUNTS="4,8,12,16"
USED_GROUPS=""

mkdir -p "${OUTPUT_ROOT}/logs"

run_layer() {
  local layer=$1
  local device=$2
  echo "[START] L${layer} on ${device}"
  python expand_ratio.py \
    --model "${MODEL}" \
    --layer "${layer}" \
    --used_groups "${USED_GROUPS}" \
    --target_counts "${TARGET_COUNTS}" \
    --output_root "${OUTPUT_ROOT}" \
    --device "${device}" \
    > "${OUTPUT_ROOT}/logs/expand_ratio_l${layer}.log" 2>&1
  echo "[DONE]  L${layer} on ${device}"
}

# Batch 1: three layers on three free GPUs
echo "=== Batch 1: L19/L20/L21 ==="
run_layer 19 cuda:0 &
run_layer 20 cuda:1 &
run_layer 21 cuda:2 &
wait

# Batch 2: remaining layers
echo "=== Batch 2: L22/L23 ==="
run_layer 22 cuda:0 &
run_layer 23 cuda:1 &
wait

echo "=== All expand_ratio.py jobs finished ==="

echo "=== Running multi_layer_scan.py on cuda:1 ==="
python multi_layer_scan.py \
  --model "${MODEL}" \
  --layers "19,20,21,22,23" \
  --group_counts "4,8,12,16" \
  --checkpoint_root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_ROOT}/multi_layer_scan" \
  --device cuda:1 \
  > "${OUTPUT_ROOT}/logs/multi_layer_scan.log" 2>&1

echo "=== Multi-layer scan complete ==="
echo "Results: ${OUTPUT_ROOT}/multi_layer_scan/multi_layer_scan.json"
