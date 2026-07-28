#!/usr/bin/env bash
# 从已有 100-prompt on-policy 结果增量扩展到 1000 prompt
#
# 前提：
#   - ./candidate_prompts.jsonl 里已经有 1000 条候选 prompt
#   - ./onpolicy_layer39_v4_tail_test/ 里有 100 条跑出来的：
#       candidate_features.jsonl, global_pca.pt, selected_stage1.json 等
#
# 行为：
#   - 复用前 100 条的 features 和 PCA
#   - 只对新 900 条跑 256-token short rollout
#   - 合并 1000 条 features，重新选 prompt，继续 Stage 2/3

nohup python -u collect_onpolicy_data.py \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --teacher_module_path "shared_expert" \
  --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints \
  --prompt_file ./candidate_prompts.jsonl \
  --output_root ./onpolicy_layer39_v4_tail_1000 \
  --layer_idx 39 \
  --hook_path "model.model.layers[39].mlp.shared_expert" \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --device cuda:0 \
  --max_candidate_prompts 1000 \
  --short_max_new_tokens 256 \
  --enable_medium_stage \
  --medium_max_new_tokens 1024 \
  --long_max_new_tokens 2048 \
  --n_select_stage1 160 \
  --n_select_final 64 \
  --n_held_out 64 \
  --resume_stage1_from ./onpolicy_layer39_v4_tail_test \
  > collect_onpolicy_v3.log 2>&1 &
