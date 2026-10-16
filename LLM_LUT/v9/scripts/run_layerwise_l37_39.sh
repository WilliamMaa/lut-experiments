#!/usr/bin/env bash
# v9 第 2 步：三层逐层推进全流程（L37->L38->L39 采样+打标+训练+推进）
# 用法：bash scripts/run_layerwise_l37_39.sh   （在 v9/ 目录下，后台跑看 logs/v9_layerwise_l37_39.log）
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=${MODEL_PATH:-/data/mamingyu/models/Qwen3.6-35B-A3B}
OUT_ROOT=${OUT_ROOT:-/data/mamingyu/outputs/v9_layerwise_l37_39}
TEACHER_ROOT=${TEACHER_ROOT:-/data/mamingyu}

python scripts/propagation/collect_layerwise_propagation.py \
  --model_path "$MODEL_PATH" \
  --prompt_file ../v6/candidate_prompts.jsonl \
  --layers 37,38,39 \
  --teacher_weights \
    37=$TEACHER_ROOT/qwen_35b_shared_expert_l37.pt \
    38=$TEACHER_ROOT/qwen_35b_shared_expert_l38.pt \
    39=$TEACHER_ROOT/qwen_35b_shared_expert_l39.pt \
  --output_root "$OUT_ROOT"
