# v8 KV 压缩：方法论路线图（2026-09-09）

## 现有证据（v3 评测集，53 轮，全部 multiturn）

| 配置 | 压缩率 | EOS | 退化情况 |
|---|---|---|---|
| l256 r128 w64 | 500x | 0.811（=baseline） | 1 轮事实错（哨兵题 178-182 亿）+ 轻微口吃 |
| l128 r64 w64 | 1000x | 0.717 | 2 轮退化循环 + 2 轮事实错 |
| l128 r32 w64 | 1000x | 0.830 | 0 循环，哨兵题仍错 |

三条定论：
1. **选择机制是瓶颈，不是容量**。哨兵题（全年收入指引 178-182 亿，文档靠后段单次出现）
   在 hh 预算 60/92/124 下全部答错。调 budget 分配（r64→r32）和 obs_window 只是绕着这个问题走。
2. **轨迹级指标（decode KL / top-1）不敏感**。1000x 和 500x 的 decode KL 几乎一样（0.54 vs 0.37），
   但生成质量一个崩一个可用。主指标必须是退化轮次 + 事实正确率（哨兵题是固定探针）。
3. middle 预算 > recency 预算（文档事实问答场景）。

## 方法推进（按优先级）

### M1 补偿式淘汰（compensation eviction）— v1 负收益，v2 凸组合待跑
硬淘汰的信息损失是失败根源：top-k 选不中 = 信息永久丢失。M1 把被淘汰 token 的
value 按 per-head attention mass 加权折叠进最近的保留后继。
改动：attention_scores.py（bank 存 per-head 分数 [H_kv, K]）、
heavy_hitter_cache.py（`_fold_evicted_values`）、CLI `--merge_evicted`。
PPL 门槛不受影响（folding 只发生在首次 decode 压缩时，prefill 无改动）。

- **v1（2026-09-10，加法，权重 min(1, p_i/p_t)）：负收益**。每槽平均折入 ~8 个 value，
  保留 value 幅度膨胀 2-9 倍，repetition +5.7pp，decode KL 0.56→0.61，doc1 T1
  从答对变重复循环，哨兵题仍错（错法变了）。幅度膨胀是主因。
- **v2（2026-09-10，凸组合）：验证通过，成为默认配置**。EOS 0.849 反超 baseline
  0.811，repetition 仅 +1.9pp，零退化轮，doc1 T1 / doc0 T5 / doc4 T0 均正确。
  哨兵题 doc0 T4 仍错——确认是寻址问题（信息折入但无 query 照到邻居），归 M3。
  已折叠 run 档案：results/heavy_hitter_attn_l128_s4_r32_w64_m2_multiturn_v3set.json

### M3 跨层共享选择（union index）— 判死（2026-09-14）
bf16 隔离 run（l128/s4/r32/w64/sh/m）：EOS 0.792 < per-layer 0.849，哨兵题仍错。
跨层均值稀释单层锐利信号，整体更差。唯一保留的遗产：索引共享思路（未验证收益）。
哨兵题进入诊断阶段：kv_cache/probe_sentinel.py 判定"被淘汰（选择问题）"还是
"保留但无 query 照到（寻址问题）"，两条路修复方向相反。

### M2 结果（2026-09-10）：成立，Pareto 移动
k8v8@l128+merge（标称 2000x）：EOS 0.830 > baseline 0.811，无退化轮，decode KL 0.564
≈ bf16 版 0.555。代价：repetition +3.8pp（bf16 版 +1.9pp），doc4 T0 重新出现轻度
功能幻觉。结论：INT8 买到存储没买到质量；"选择×补偿×量化"三件套在 2000x 成立。
档案：results/heavy_hitter_attn_l128_s4_r32_w64_m_k8v8_multiturn_v3set.json

### M3 跨层共享选择（union index）
10 个 full-attn 层各自独立选 hh，同一 token 在不同层的重要性高度相关。
用跨层并集选择 + 共享索引，省索引存储（CIM 相关），并避免"某层把别层的关键 token 删了"。

### M4 分层预算（PyramidKV 式）
M1-M3 做完后的分配细化，最后做。

## 方法学修正（2026-09-12）：run 间方差与确定性选择

500x 同配置重跑（l256/s4/r128/w64，旧 25 轮集）：EOS 稳定 0.96，但哨兵题级事实
答案翻转（T4 从"178-182亿"变成"170-175亿"）。根因：CUDA topk 在大量近并列质量下
跨 run 不稳定。处置：选择路径改用稳定降序排序（并列取位置小者），选集跨 run 可复现。
教训：质量边缘的单 run 比较不可信，关键结论需双 run 确认；结果文件不要覆盖。

### M4 结果（2026-09-15）：成立
l128/s4/r32/w64/m/sp4（1000x bf16）：EOS 0.811 = baseline，decode KL 0.519
历史最低，哨兵题 doc0 T4 首次答对，doc0 T5 / doc3 T4 同时修复；代价
repetition +3.8pp。探针的碎片化选择诊断被直接验证。
档案：results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_multiturn_v3set.json

叠加 k8v8（标称 2000x）：哨兵题全保，EOS 0.792 仅丢 1 个临界轮
（doc1 T2，bf16 下本就是重复堆砌边缘轮）。span 保护盖过 INT8 噪声。
档案：results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_k8v8_multiturn_v3set.json

遗留：12a 复验发现折叠 index_add_ 原子加是末位方差源（2/53 自由生成轮
分叉，指标无损），已改确定性 one-hot matmul，待 12c 复验后定型。

## 已放弃的方向（不要再回头的）

- 自定义 eager attention kernel（数值 drift 必死，sdpa wrapper 是唯一干净路径）
- key-norm 重要性（全面劣于 attn-score）
- 调参式扫描（obs_window 等）：w256 那个 run 跑完当参考，不再系统性扫
- 跨层共享选择 M3（稀释单层锐利信号，EOS 反降）
