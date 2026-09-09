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

### M1 补偿式淘汰（compensation eviction）— 已实现，待跑
硬淘汰的信息损失是失败根源：top-k 选不中 = 信息永久丢失。M1 把被淘汰 token 的
value 按 per-head attention mass 加权折叠进最近的保留后继（权重上限 1，单 token
无法支配目标）。选择不再需要完美——被淘汰事实活在邻居的 value 里。
改动：attention_scores.py（bank 存 per-head 分数 [H_kv, K]）、
heavy_hitter_cache.py（`_fold_evicted_values`）、CLI `--merge_evicted`。
PPL 门槛不受影响（folding 只发生在首次 decode 压缩时，prefill 无改动）。

### M2 KV value INT8 量化 — 与 M1 正交，2x 有效容量
value 占存储大头（head_dim 256 × 2B）。value 降 INT8（per-head per-channel scale），
同样字节数下 budget 翻倍：l128 的 60 个 hh 相当于 120 个。工程为主、收益确定，
在 M1 结论出来后叠加。

### M3 跨层共享选择（union index）
10 个 full-attn 层各自独立选 hh，同一 token 在不同层的重要性高度相关。
用跨层并集选择 + 共享索引，省索引存储（CIM 相关），并避免"某层把别层的关键 token 删了"。

### M4 分层预算（PyramidKV 式）
M1-M3 做完后的分配细化，最后做。

## 已放弃的方向（不要再回头的）

- 自定义 eager attention kernel（数值 drift 必死，sdpa wrapper 是唯一干净路径）
- key-norm 重要性（全面劣于 attn-score）
- 调参式扫描（obs_window 等）：w256 那个 run 跑完当参考，不再系统性扫
