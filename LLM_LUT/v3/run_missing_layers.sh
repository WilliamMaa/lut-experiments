#!/bin/bash
# Run expand_ratio.py for missing layers (L15, L16, L26, L27 by default).
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_missing_layers.sh
cd "$(dirname "$0")"
LAYERS="${LAYERS:-15 16 26 27}"
for layer in $LAYERS; do
  echo "===================================="
  echo "Running expand_ratio for L$layer"
  echo "===================================="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python expand_ratio.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer $layer \
    --output_root outputs \
    --calib_size 512 --eval_size 128 \
    --batch_size 2 \
    --device cuda:0
  if [ $? -ne 0 ]; then
    echo "ERROR: L$layer failed"
    exit 1
  fi
done
echo "All missing layers done."


CUDA_VISIBLE_DEVICES=1 python quantize_lut.py \
    --summary_root ../v3/outputs \
    --checkpoint_root ../v3/outputs/checkpoints \
    --output_root ../v3/outputs_fp16 \
    --dtype fp16 \
