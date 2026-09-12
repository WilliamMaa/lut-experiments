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

### M3 跨层共享选择（union index）— 已实现，待跑
10 个 full-attn 层各自独立选 hh，同一 token 在不同层的重要性高度相关。
M3 把 eviction 分数改为 10 层注意力质量（列和，天然跨层可比）的均值：某 token
在任一层重要就不被淘汰；10 层共用一份选择索引，索引存储省 ~10x（CIM 相关）。
folding 权重仍用 per-layer per-head 质量（折叠是逐层的）。CLI `--shared_selection`，
patch 名 `_sh` 后缀。攻哨兵题（doc0 T4 寻址问题）。
2026-09-10 状态：代码完成，语法通过，待远程数值验证。

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

## 已放弃的方向（不要再回头的）

- 自定义 eager attention kernel（数值 drift 必死，sdpa wrapper 是唯一干净路径）
- key-norm 重要性（全面劣于 attn-score）
- 调参式扫描（obs_window 等）：w256 那个 run 跑完当参考，不再系统性扫
