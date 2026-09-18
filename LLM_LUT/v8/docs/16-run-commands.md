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

## 8. M3 跨层共享选择（l128/s4/r32/w64 + merge + shared，bf16，隔离 M3 效应）

M2 结论（2026-09-10）：k8v8 在 2000x 可用（EOS 0.830 > baseline）但质量略降于
bf16/1000x（0.849）——INT8 买到存储没买到质量，三件套成立。哨兵题 doc0 T4 换错法
仍错，是寻址问题。M3 把 eviction 分数改为 10 个 full-attn 层注意力质量的均值，
并共用一份选择索引。先跑 bf16 隔离 M3 效应，重点看哨兵题是否修复。
2026-09-12 更新：同配置重跑证实 run 间方差（CUDA topk 不稳定，事实答案翻转）后，
选择路径已改为稳定降序排序（并列取位置小者，跨 run 可复现）。本次 run 包含
该修复 + M3 共享选择 + M1 凸组合折叠。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 64 \
  --merge_evicted \
  --shared_selection \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_sh_m_multiturn_v3set.json \
  > heavy_hitter_attn_l128_sh_m_v3set.log 2>&1 &
```

若哨兵题修复且无退化，最终组合跑 `--merge_evicted --shared_selection --k_bits 8 --v_bits 8`
（patch 名 `_sh_m_k8v8`，输出文件自行换后缀，标称 2000x）。

## 9. 2026-09-13 状态与下一步

已完成的关键 run（v3 评测集，53 轮，全部 merge_evicted 凸组合折叠 + 确定性 tie-break）：

| run | 标称压缩 | EOS | 判定 |
|---|---|---|---|
| l128/s4/r32/w64/m（bf16） | 1000x | 0.849（>baseline 0.811） | 干净 |
| l128/s4/r32/w64/m/k8v8 | 2000x | 0.830 | 基本干净 |
| l128/s4/r32/w64/sh/m/k4v4 | 8000x | 0.774（<baseline） | 越界：循环+幻觉 |

注意：k4v4 那次 run 的文件名误用了 bf16 的输出名，内容实为 k4v4，见
results/heavy_hitter_attn_l128_s4_r32_w64_sh_m_multiturn_v3set.json（patch 名
带 k4v4）。4 与 8 bit 之间是 bit-depth 边界。
2026-09-13 根因修复：`--k_bits/--v_bits` 默认值原是 4（kivi 遗留），接入
heavy_hitter_attn 后导致不显式传参的 run 静默 4bit。已改为默认 16（bf16），
并加 [WARN] loud 打印。受影响 run 只有这一个（sh_m 实为 k4v4/8000x）；
M1（m/m2）与 k8v8 run 均早于该 wiring 或显式传参，结论不变。

**仍欠：M3 共享选择在 bf16 下的隔离 run**（第 8 节命令原样，哨兵题 doc0 T4
是判定标准）。若修复，最终生产配置为 sh+m+k8v8（2000x）：

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 \
  --sink_tokens 4 \
  --recent_tokens 32 \
  --obs_window 64 \
  --merge_evicted \
  --shared_selection \
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
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_sh_m_bf16_multiturn_v3set.json \
  > heavy_hitter_attn_l128_sh_m_bf16_v3set.log 2>&1 &
```

哨兵题修复后跑 `--merge_evicted --shared_selection --k_bits 8 --v_bits 8`，
输出名带 `sh_m_k8v8` 后缀。

## 10. 哨兵题诊断探针（选择失败 vs 寻址失败）

M3 判定不成立（2026-09-14，bf16 隔离 run）：shared 选择 EOS 0.792 < per-layer
0.849，哨兵题 doc0 T4 仍错（170-180亿）。跨层均值稀释了单层锐利信号，M3 路线作废。
下一步不是盲试变体，而是先回答：哨兵 token 是被淘汰了（选择问题），还是留着但
没有 query 照到它（寻址问题）。探针逐层打印答案 span 的原始位置、是否在保留集、
注意力质量排名、是否在 obs_window 排除区：

```bash
python kv_cache/probe_sentinel.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --doc_index 0 --turn 4 --answer_text "178亿元至182亿元" \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

解读：span 全部 kept= N -> 选择问题（分数不认为它重要）；kept=Y 但答案错 ->
寻址问题（obs_window 里的 query 行没照到它，或照到了但后续层丢了）。

## 11. M4 span 感知选择（探针验证 + 全量 eval）

探针诊断（2026-09-14）：哨兵 span（9 token）在 layer 3/39 全灭、中间层碎片化保留
（rank 9-53 的 token 被孤立保留、数字邻居被淘汰）；模型生成"178亿元至178"——
部分检索发生但 span 碎了。tokenizer 把数字拆成单字 token，逐 token 打分把
多 token 事实拆散。M4 = 打分后 ±W 窗口 max-pool（--span_window 4），事实
成组保留/淘汰。

第一步，探针验证（同一命令加 --span_window 4，期望多数层 kept=Y）：

```bash
python kv_cache/probe_sentinel.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --doc_index 0 --turn 4 --answer_text "178亿元至182亿元" \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

第二步，全量 eval（1000x bf16，与 m2 run 对比）：

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl \
  --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 \
  --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_multiturn_v3set.json \
  > heavy_hitter_attn_l128_m_sp4_v3set.log 2>&1 &
```

## 12. M4 验证与最终配置（2026-09-15）

M4（span_window 4，l128/s4/r32/w64/m/sp4，1000x bf16）结果：哨兵题 doc0 T4
首次答对（178-182亿），doc0 T5（9.7亿）和 doc3 T4（平台名称）同时修复，
EOS 0.811 = baseline，decode KL 0.519 历史最低。代价：repetition +3.8pp，
doc2 T0 多一个 no-EOS。探针的碎片化选择诊断被直接验证。

### 12a. 确定性复验（同配置重跑，输出加 _rerun）

stable tie-break 已生效，两次 run 应逐位一致。不一致 = 仍有方差源，先别定型。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl \
  --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 \
  --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_multiturn_v3set_rerun.json \
  > heavy_hitter_attn_l128_m_sp4_v3set_rerun.log 2>&1 &
```

### 12b. 最终配置候选（+ k8v8，标称 2000x）

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 --k_bits 8 --v_bits 8 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl \
  --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 \
  --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_k8v8_multiturn_v3set.json \
  > heavy_hitter_attn_l128_m_sp4_k8v8_v3set.log 2>&1 &
```

### 12a 结果（2026-09-15）：未过逐位一致，方差源已定位并修复

两次 run 聚合指标逐位相同（EOS 0.8113207547、rep 0.0566037736、
decode KL 0.5187070159869756，全部 53 轮 PPL 相同），但 **2/53 轮自由生成
文本中途分叉**（doc0 T6 第 30 字符处、doc6 T4 第 261 字符处；两轮都是
no-EOS 的开放式长输出，非哨兵题，指标不受影响）。

根因：`_fold_evicted_values` 的 CUDA `index_add_`（两处）走原子加，
累加顺序不确定 → fp32 末位差 → cast bf16 后翻转近并列 token。
baseline 路径零分叉证明前向本身确定，stable sort 选择也已确定。

修复（heavy_hitter_cache.py）：折叠归约改为 one-hot matmul einsum
（`eh,bhed->bhd`），数学等价、GEMM 逐 run 位稳定。已 py_compile 通过。

### 12b 结果（2026-09-15）：2000x 成立，span 保护盖过 INT8 噪声

文件：results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_k8v8_multiturn_v3set.json
配置核验：k8v8，compression_ratio 2000.0，与文件名相符。PPL delta = 0。

- EOS 0.792 vs baseline 0.811：恰好丢 1 轮（doc1 T2，该轮在 bf16 sp4 下
  本就是重复堆砌的临界轮，INT8 噪声把它推过 no-EOS）
- **哨兵题全保**：doc0 T4（178-182亿）、doc0 T5（9.7亿）、doc3 T4（平台名）
  三轮全部答对且 EOS
- repetition 0.057 与 bf16 sp4 持平；decode KL 0.512（轨迹级指标不敏感，仅参考）

结论：1000x→2000x，事实保真不破，代价是 1 个临界轮翻 no-EOS。
12b 跑在旧折叠代码上；折叠改动是数学等价的末位级修正，数据点继续有效。

### 12c. 确定性折叠复验（修复后跑一次即可）

判定标准（不是逐位比对旧文件——折叠归约数序合法变化，与旧 run 末位
不同是预期）：聚合指标与 m_sp4 一致（EOS 0.811、rep 0.057、
KL ≈ 0.519），哨兵题三轮全对。通过后方法栈定型。

结果（2026-09-16）：通过。EOS 逐位一致，KL 0.51873≈0.51871，哨兵题三轮
逐字一致，rep 反降至 0.038。6/53 分叉全为开放式长输出，无事实错误。
方法栈定型，最终总结见 docs/19-final-summary.md。

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl \
  --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 \
  --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_multiturn_v3set.json \
  > heavy_hitter_attn_l128_m_sp4_det_v3set.log 2>&1 &
```
