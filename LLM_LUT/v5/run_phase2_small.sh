#!/usr/bin/env bash
# Phase 2 small-scale validation:
# Compare sequential deployment-aware build vs independent build
# on down L18-L23 + o_proj L15-L17.
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
echo "Phase 2: Sequential vs Independent Build"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Sequential deployment-aware build
# ------------------------------------------------------------------
echo ""
echo "[1/4] Sequential build: o L15-17 -> down L18-23"
python build_lut_sequential.py \
    --model "$MODEL" \
    --down_configs "18:8,19:8,20:8,21:8,22:8,23:8" \
    --o_configs "15:8,16:8,17:8" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --o_mode direct \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_sequential_small

# ------------------------------------------------------------------
# 2. Independent down_proj build
# ------------------------------------------------------------------
echo ""
echo "[2/4] Independent build: down L18-23"
python build_lut.py \
    --model "$MODEL" \
    --configs "18:8,19:8,20:8,21:8,22:8,23:8" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_independent_down_small

# ------------------------------------------------------------------
# 3. Independent o_proj build
# ------------------------------------------------------------------
echo ""
echo "[3/4] Independent build: o L15-17"
python build_lut_o_proj.py \
    --model "$MODEL" \
    --configs "15:8,16:8,17:8" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --mode direct \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_independent_o_small

# ------------------------------------------------------------------
# 4. Joint fine-tune: sequential build
# ------------------------------------------------------------------
echo ""
echo "[4/4a] Joint fine-tune on SEQUENTIAL checkpoints"
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "18:8,19:8,20:8,21:8,22:8,23:8" \
    --down_checkpoint_root ../v5/outputs_sequential_small \
    --o_configs "15:8,16:8,17:8" \
    --o_checkpoint_root ../v5/outputs_sequential_small \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/finetune_joint_sequential_small

# ------------------------------------------------------------------
# 5. Joint fine-tune: independent build
# ------------------------------------------------------------------
echo ""
echo "[4/4b] Joint fine-tune on INDEPENDENT checkpoints"
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "18:8,19:8,20:8,21:8,22:8,23:8" \
    --down_checkpoint_root ../v5/outputs_independent_down_small \
    --o_configs "15:8,16:8,17:8" \
    --o_checkpoint_root ../v5/outputs_independent_o_small \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/finetune_joint_independent_small

echo ""
echo "=========================================="
echo "Phase 2 complete."
echo "Compare:"
echo "  results/finetune_joint_sequential_small/summary.json"
echo "  results/finetune_joint_independent_small/summary.json"
echo "=========================================="
