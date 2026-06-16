#!/usr/bin/env bash
# Sequentially run expand_ratio.py for layers 19-23, then multi_layer_scan.py.
#
# Usage (from v3/ directory on the GPU server):
#   bash run_expand_and_scan.sh          # uses cuda:1
#   bash run_expand_and_scan.sh cuda:0   # use another device

set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${1:-cuda:1}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_ROOT="outputs"
TARGET_COUNTS="4,8,12,16"
USED_GROUPS=""

mkdir -p "${OUTPUT_ROOT}/logs"

for layer in 19 20 21 22 23; do
  echo "=== expand_ratio.py L${layer} on ${DEVICE} ==="
  python expand_ratio.py \
    --model "${MODEL}" \
    --layer "${layer}" \
    --used_groups "${USED_GROUPS}" \
    --target_counts "${TARGET_COUNTS}" \
    --output_root "${OUTPUT_ROOT}" \
    --device "${DEVICE}" \
    | tee "${OUTPUT_ROOT}/logs/expand_ratio_l${layer}.log"
done

echo "=== multi_layer_scan.py on ${DEVICE} ==="
python multi_layer_scan.py \
  --model "${MODEL}" \
  --layers "19,20,21,22,23" \
  --group_counts "4,8,12,16" \
  --checkpoint_root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_ROOT}/multi_layer_scan" \
  --device "${DEVICE}" \
  | tee "${OUTPUT_ROOT}/logs/multi_layer_scan.log"

echo "=== Done ==="
echo "Results: ${OUTPUT_ROOT}/multi_layer_scan/multi_layer_scan.json"
