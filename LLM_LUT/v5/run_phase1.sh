#!/usr/bin/env bash
# Phase 1 control experiments for joint down_proj + o_proj failure analysis.
#
# Uses existing checkpoints:
#   ../v5/outputs_tree_l15_l27   (down_proj L15-L27 tree)
#   ../v5/outputs_o_proj_exp     (o_proj L15/L16/L17 direct + L27 delta)
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
BATCH_SIZE=4

echo "=========================================="
echo "Phase 1: Isolation Experiments"
echo "=========================================="

# ------------------------------------------------------------------
# Exp A: Down-only recovery
#   Install: down_proj L15-L27
#   Train:   down_proj L15-L27
#   Goal:    Is large-scale tree down_proj itself recoverable?
# ------------------------------------------------------------------
echo ""
echo "[Exp A] Down-only L15-L27 fine-tune"
python finetune.py \
    --model "$MODEL" \
    --configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --checkpoint_root ../v5/outputs_tree_l15_l27 \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/phase1_down_only_l15_l27

# ------------------------------------------------------------------
# Exp B: O-only recovery
#   Install: o_proj L15/L16/L17/L27
#   Train:   o_proj L15/L16/L17/L27
#   Goal:    Is this o_proj configuration itself recoverable?
# ------------------------------------------------------------------
echo ""
echo "[Exp B] O-only L15/L16/L17/L27 fine-tune"
python finetune_o_proj.py \
    --model "$MODEL" \
    --configs "15:8,16:8,17:8,27:8" \
    --checkpoint_root ../v5/outputs_o_proj_exp \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/phase1_o_only_l15_17_27

# ------------------------------------------------------------------
# Exp C: Down + frozen-o
#   Install: down_proj L15-L27 + o_proj L15/L16/L17/L27
#   Train:   down_proj only
#   Goal:    Can down_proj compensate for fixed o_proj perturbation?
# ------------------------------------------------------------------
echo ""
echo "[Exp C] Down train, o_proj frozen"
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --freeze_o \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/phase1_down_train_o_frozen

# ------------------------------------------------------------------
# Exp D: O + frozen-down
#   Install: down_proj L15-L27 + o_proj L15/L16/L17/L27
#   Train:   o_proj only
#   Goal:    Can o_proj compensate for fixed down_proj perturbation?
# ------------------------------------------------------------------
echo ""
echo "[Exp D] O train, down_proj frozen"
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --freeze_down \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir results/phase1_o_train_down_frozen

# ------------------------------------------------------------------
# Exp E: Layerwise drift diagnostic
#   Compare hidden states of original model vs joint-replaced model
#   Goal:    Identify where error starts to explode
# ------------------------------------------------------------------
echo ""
echo "[Exp E] Layerwise drift (original vs joint-replaced)"
python measure_layerwise_drift.py \
    --model "$MODEL" \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_json results/phase1_drift_joint_raw.json

echo ""
echo "=========================================="
echo "Phase 1 complete."
echo "=========================================="
