#!/usr/bin/env bash
# Phase 3: expand down_proj + o_proj using sensitivity scanner top 4% MAC selection.
# Based on results/sensitivity_scan_summary.md (down+o only, sorted by score_ppl_per_mac).
#
# down_proj: non-uniform groups across L15-L27
# o_proj   : non-uniform groups across L15-L27, all direct mode

set -e

export LD_LIBRARY_PATH=""
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

MODEL="Qwen/Qwen2.5-7B-Instruct"
CALIB_SIZE=512
EVAL_SIZE=128
MAX_SEQ_LEN=512
BATCH_SIZE=4

# Generated from sensitivity_scan.json for a 4% MAC target (down_proj + o_proj only)
DOWN_CONFIGS="15:17;0;1;8;9;16;28;30;32;38;43;44;47;48;49;52;53;55,16:11;2;7;17;19;22;25;31;40;43;47;50,17:13;0;1;2;3;4;13;14;18;20;27;33;49;54,18:15;3;6;7;8;9;18;22;24;26;29;41;47;49;51;52,19:15;4;14;15;19;23;28;29;35;37;39;46;48;49;51;55,20:24;1;4;6;7;8;9;12;20;21;22;24;28;34;35;37;38;41;44;45;48;49;50;52;54,21:17;0;1;4;5;8;9;11;12;13;15;18;29;31;33;34;37;41,22:9;4;7;17;19;25;26;33;49;54,23:11;2;5;7;10;11;21;33;34;46;49;53,24:8;2;5;7;21;29;33;42;51,25:10;6;10;14;18;21;31;32;33;45;51,26:3;15;24;49,27:12;1;6;10;21;22;26;29;36;42;44;48;53"

O_CONFIGS="15:17;3;12;13;15;16;17;18;21;24;27;33;34;39;40;47;50;55,16:21;1;2;3;5;6;10;12;15;18;20;21;24;25;27;29;31;37;38;39;49;53,17:20;0;7;13;15;17;18;21;23;24;25;26;27;30;34;39;42;43;44;46;49,18:32;1;2;4;6;8;10;12;13;16;17;19;22;24;25;28;29;32;33;36;37;38;39;40;41;43;44;46;49;50;52;53;55,19:20;1;3;4;6;10;12;13;15;22;24;26;28;29;31;37;39;40;47;50;52,20:25;4;9;11;15;16;19;20;21;22;23;25;27;30;35;37;38;39;40;43;44;45;46;49;50;54,21:18;3;4;5;8;9;10;12;18;19;20;28;32;33;34;41;45;50;52,22:18;0;6;11;12;13;21;22;24;27;32;34;35;43;45;47;49;52;54,23:24;1;2;4;6;8;9;10;11;15;17;22;28;33;34;35;36;38;40;44;45;46;53;54;55,24:13;0;6;7;9;16;21;22;24;37;42;44;45;54,25:26;0;1;2;4;6;12;13;17;19;20;22;23;26;29;35;36;37;40;41;42;44;48;50;53;54;55,26:14;4;7;8;9;10;12;13;25;31;40;42;44;45;47,27:19;1;3;5;6;9;12;16;17;19;21;27;28;30;31;33;45;47;53;54"

OUTPUT_ROOT="../v5/outputs_phase3_down_o_4pct"
FINETUNE_OUT="results/finetune_joint_phase3_down_o_4pct"

echo "=========================================="
echo "Phase 3: down_proj + o_proj expansion (4% MAC)"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Sequential deployment-aware build
# ------------------------------------------------------------------
echo ""
echo "[1/2] Sequential build..."
python build_lut_sequential.py \
    --model "$MODEL" \
    --down_configs "$DOWN_CONFIGS" \
    --o_configs "$O_CONFIGS" \
    --address_mode tree \
    --num_bits 10 \
    --channels_per_bit 4 \
    --tree_candidates 32 \
    --tree_min_samples 32 \
    --tree_max_samples 16384 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --output_root "$OUTPUT_ROOT"

# ------------------------------------------------------------------
# 2. Joint fine-tune
# ------------------------------------------------------------------
echo ""
echo "[2/2] Joint fine-tune..."
python finetune_joint.py \
    --model "$MODEL" \
    --down_configs "$DOWN_CONFIGS" \
    --down_checkpoint_root "$OUTPUT_ROOT" \
    --o_configs "$O_CONFIGS" \
    --o_checkpoint_root "$OUTPUT_ROOT" \
    --epochs 10 \
    --lr 5e-5 \
    --calib_size "$CALIB_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$FINETUNE_OUT"

echo ""
echo "=========================================="
echo "Phase 3 complete."
echo "See: $FINETUNE_OUT/summary.json"
echo "=========================================="
