#!/usr/bin/env bash
# Phase 2 large-scale sequential deployment-aware build.
#
# down_proj: L15-L27 (v4 non-uniform group counts)
# o_proj:    L15/L16/L17 direct + L27 delta
#
# Run from LLM_LUT/v5.

set -e

export LD_LIBRARY_PATH=""
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-7B-Instruct"
CALIB_SIZE=512
EVAL_SIZE=128
MAX_SEQ_LEN=512
BATCH_SIZE=4

echo "=========================================="
echo "Phase 2 Large: Sequential Build + Joint Fine-Tune"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Sequential deployment-aware build
# ------------------------------------------------------------------
echo ""
echo "[1/2] Sequential build: L15-L27 down + L15/L16/L17 direct + L27 delta"
python build_lut_sequential.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_modes "15:direct,16:direct,17:direct,27:delta" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_sequential_large

# ------------------------------------------------------------------
# 2. Joint fine-tune
# ------------------------------------------------------------------
echo ""
echo "[2/2] Joint fine-tune on sequential checkpoints"
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_sequential_large \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_sequential_large \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/finetune_joint_sequential_large

echo ""
echo "=========================================="
echo "Phase 2 large complete."
echo "See: results/finetune_joint_sequential_large/summary.json"
echo "=========================================="
