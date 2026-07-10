#!/usr/bin/env bash
# v5 expansion v1: scale both down_proj and o_proj together.
#
# down_proj: L15-L27 tree (v4-style non-uniform group counts)
# o_proj:    L15/L16/L17 direct + L27 delta
#
# Run from LLM_LUT/v5.

set -e

export LD_LIBRARY_PATH=""
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

MODEL="Qwen/Qwen2.5-7B-Instruct"
CALIB_SIZE=512
EVAL_SIZE=128
MAX_SEQ_LEN=512

echo "=========================================="
echo "[1/3] Build down_proj L15-L27 tree LUTs"
echo "=========================================="
python build_lut.py \
    --model "$MODEL" \
    --configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_tree_l15_l27

echo "=========================================="
echo "[2/3] Build o_proj L15-L17 direct LUTs"
echo "=========================================="
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
    --output_root ../v5/outputs_o_proj_exp

echo "=========================================="
echo "[3/3] Build o_proj L27 delta LUT"
echo "=========================================="
python build_lut_o_proj.py \
    --model "$MODEL" \
    --configs "27:8" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --mode delta \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root ../v5/outputs_o_proj_exp

echo "=========================================="
echo "[4/4] Joint fine-tune"
echo "=========================================="
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --epochs 5 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size 4 \
    --output_dir results/finetune_joint_exp_v1

echo "Done."
