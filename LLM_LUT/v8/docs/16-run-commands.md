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

## 3. budget 128 内重分预算（1000x：recent 32 / hh 92）

2026-09-08 的 l128/r64 run 结论：1000x 越界（退化循环 + mid-document 事实错误，
EOS 0.717）。decode KL 与 500x 几乎相同但生成崩了，说明轨迹级指标不敏感，
应以退化轮次 + 事实正确率为主指标。先试把 middle 预算从 60 提到 92（recent 64→32），
专打事实丢失。

```bash
nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_multiturn_v3set.json \
  > heavy_hitter_attn_l128_r32_v3set.log 2>&1 &
```

## 4. 哨兵题修复探测（1000x：obs_window 64 → 256）

2026-09-09 结论：l128/r32 全面好于 r64（EOS 0.830、零循环、新加坡/积碳/温哥华全对），
middle 预算比 recency 值钱。但哨兵题 doc0 T4（全年指引 178-182 亿）在 500x/1000x、
hh 60/124 下全部答错——不是容量问题，是选择机制问题：obs_window=64 里约 40 行是
chat template，真正携带问题信息的 query 行太少。以下 run 只放大打分窗口。

```bash
nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 256 \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w256_multiturn_v3set.json \
  > heavy_hitter_attn_l128_r32_w256_v3set.log 2>&1 &
```

## 5. M1 补偿式淘汰（1000x，同 l128/s4/r32/w64 配置 + --merge_evicted）

与 2026-09-09 的 l128/s4/r32/w64 run 唯一差异是 folding。对比重点：
哨兵题 doc0 T4（178-182 亿）是否修复、退化轮次是否增加。patch 名带 `_m` 后缀。

注意：2026-09-09 首次 M1 run 在首次淘汰全部完成后崩于 device-side assert（index out of
bounds，异步上报，栈不可靠）。已对 fold 索引全部加钳制，并加 CUDA_LAUNCH_BLOCKING=1
重跑：若仍崩，栈会指向真实 kernel，把新的报错发回来。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 64 \
  --merge_evicted \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_multiturn_v3set.json \
  > heavy_hitter_attn_l128_r32_w64_m_v3set.log 2>&1 &
```

## 6. M1 v2 凸组合折叠（1000x，同 l128/s4/r32/w64 配置）

2026-09-10：M1 v1（加法折叠，权重上限 1）是负收益——保留 value 幅度膨胀 2-9 倍，
repetition +5.7pp，doc1 T1 从答对变成循环，哨兵题仍错。v2 改为质量加权凸组合
V_t' = (p_t·V_t + Σ p_i·V_i)/(p_t + Σ p_i)，幅度有界。若 v2 仍退化，folding 判死，
转 M2（INT8 value 量化）。日志关键字变为 "convex-folded"。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 64 \
  --merge_evicted \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m2_multiturn_v3set.json \
  > heavy_hitter_attn_l128_r32_w64_m2_v3set.log 2>&1 &
```

## 7. M2 INT8 KV 量化叠加（l128/s4/r32/w64 + merge + k8v8，标称 2000x）

M1 v2 凸组合折叠已在 1000x 验证（EOS 0.849 > baseline，零退化轮）。M2 把存储的
K/V 量化（K per-channel INT8、V per-token INT8，KIVI 式），同样字节数下有效容量
翻倍：l128 的 hh 92 相当于 184 个 bf16 槽。量化只作用于 decode 阶段的存储态，
prefill 仍 bf16，PPL 门槛逻辑不变。与 folding 正交，两者叠加。
若 2000x 仍可用，说明"选择 × 补偿 × 量化"三件套成立。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 64 \
  --merge_evicted \
  --k_bits 8 \
  --v_bits 8 \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_k8v8_multiturn_v3set.json \
  > heavy_hitter_attn_l128_m_k8v8_v3set.log 2>&1 &
```
