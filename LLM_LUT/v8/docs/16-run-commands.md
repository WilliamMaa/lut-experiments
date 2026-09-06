# v8 运行指令（从本文档复制，不要在对话里手打）

日期：2026-09-07
远程目录：`/data/mamingyu/v8`（所有命令在该目录下执行）

## 1. 新评测集标定（multi_turn_prompts_v2，48 轮）

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
  --multi_turn_file data/multi_turn_prompts_v2.jsonl \
  --max_eval_samples 5 \
  --max_new_tokens 128 \
  --max_length 4096 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/heavy_hitter_attn_l256_s4_r128_w64_multiturn_v2set.json \
  > heavy_hitter_attn_v2set.log 2>&1 &
```

## 2. budget 128 探测（1000x 压缩）

```bash
nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 128 \
  --obs_window 64 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl \
  --prompt_file candidate_prompts.jsonl \
  --multi_turn \
  --multi_turn_file data/multi_turn_prompts_v2.jsonl \
  --max_eval_samples 5 \
  --max_new_tokens 128 \
  --max_length 4096 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r128_w64_multiturn_v2set.json \
  > heavy_hitter_attn_l128_v2set.log 2>&1 &
```

注意：`--torch_dtype bfloat16` 和 `--logit_metrics` 之间必须有空格。输出文件名不要覆盖旧结果。
