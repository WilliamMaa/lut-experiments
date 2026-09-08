# v8 运行指令（从本文档复制，不要在对话里手打）

日期：2026-09-08
远程目录：`/data/mamingyu/v8`（所有命令在该目录下执行）

评测集已升级到 `data/multi_turn_prompts_v3.jsonl`（5 个加长中文文档 + 2 英文 + 1 日文，53 轮）。
`--max_eval_samples` 同时截断 PPL 文本数和多轮文档数，**必须设 8 或以上**，否则只跑前 5 个文档，英文/日文文档不会执行。

## 1. 新评测集标定（v3 评测集，budget 256，500x）

```bash
nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 256 \
  --sink_tokens 4 \
  --recent_tokens 128 \
  --obs_window 64 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl \
  --prompt_file candidate_prompts.jsonl \
  --multi_turn \
  --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 \
  --max_new_tokens 128 \
  --max_length 4096 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/heavy_hitter_attn_l256_s4_r128_w64_multiturn_v3set.json \
  > heavy_hitter_attn_v3set.log 2>&1 &
```

## 2. budget 128 探测（1000x 压缩）

注意：budget 128 时 recent_tokens 必须小于 128 - sink，否则 heavy-hitter 预算为 0、
淘汰被静默跳过（heavy_hitter_cache.py:149,161），结果会假装等于 baseline
（2026-09-07 的 l128 v2set 运行就是这种情况，已作废）。这里 recent 用 64，hh 预算 60。

```bash
nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 64 \
  --obs_window 64 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl \
  --prompt_file candidate_prompts.jsonl \
  --multi_turn \
  --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 \
  --max_new_tokens 128 \
  --max_length 4096 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r64_w64_multiturn_v3set.json \
  > heavy_hitter_attn_l128_v3set.log 2>&1 &
```

注意：`--torch_dtype bfloat16` 和 `--logit_metrics` 之间必须有空格。输出文件名不要覆盖旧结果。
