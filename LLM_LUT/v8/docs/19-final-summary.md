# v8 KV 压缩：最终总结（2026-09-16）

## 结论

Qwen3.6-35B-A3B 上，**"attn-score 选择 + 凸组合折叠 + span 感知"的 KV cache
压缩方法栈成立**：

| 配置 | 标称压缩 | EOS | repetition | 哨兵题 | 档案 |
|---|---|---|---|---|---|
| baseline（无压缩） | 1x | 0.811 | 0.019 | 对 | — |
| m_sp4（定型配置） | 1000x | 0.811（=bl） | 0.057 | 全对 | results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_multiturn_v3set.json |
| m_sp4 + k8v8 | 2000x | 0.792（丢 1 临界轮） | 0.057 | 全对 | results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_k8v8_multiturn_v3set.json |

评测：v3 多轮集（8 文档 × 53 轮），bf16，balanced_low_0。
decode KL 轨迹级指标不敏感（0.52–0.56 区间），仅作参考。

## 方法栈（定型）

1. **sdpa stash wrapper**（attention_scores.py）：前向逐位不变地截取 per-token
   attention 质量。硬门槛：PPL delta 恒为 0（所有 run 通过）。
2. **HeavyHitterCache**：budget 128 = sink 4 + recent 32 + hh 92，按 prefill
   attn score 选 hh，全层（10 个 full-attn 层）统一配置。
3. **merge_evicted**（凸组合折叠）：被淘汰 value 按质量加权折入最近保留后继，
   幅度有界。v1 加法折叠判死（幅度膨胀 2-9 倍 → 重复循环）。
4. **span_window 4**（M4）：±4 窗口 max-pool，多 token 事实成组保留——
   tokenizer 把数字拆成单字 token，逐 token 打分必然拆散事实，这是哨兵题
   之前所有配置全错的根因（probe_sentinel.py 探针钉死）。

配套：稳定降序排序替代 CUDA topk（跨 run 选择可复现）；折叠归约 one-hot
matmul 替代 index_add_ 原子加（跨 run 位稳定，2026-09-15 修复并验证）。

## Pareto 演化（同一 53 轮集）

| 阶段 | 压缩 | EOS | 问题 |
|---|---|---|---|
| 三段淘汰 l256 | 500x | =bl | 哨兵题错（选择瓶颈） |
| 三段淘汰 l128 | 1000x | 0.83 | 哨兵题仍错 |
| + 凸组合折叠 | 1000x | 0.85 | 哨兵题仍错（信息折入但寻址不到 → M4 动机） |
| + span_window | 1000x | =bl | **哨兵题修复**，定型 |
| + INT8 KV | 2000x | -1 临界轮 | 事实保真，存储再砍半 |

判死勿回头的方向：跨层共享选择 M3（稀释单层信号）、自定义 eager kernel
（数值 drift）、key-norm 重要性、4-bit KV（8000x 出退化循环）、调参式扫描。

## 代价与边界（诚实记录）

- repetition 0.019 → 0.057（+3.8pp）：压缩下偶发局部重复，哨兵修复的代价；
  det run 中降到 0.038。
- 开放式长输出（T6 观点题类）对折叠末位差最敏感，轨迹天然混沌，但 12c 证明
  不影响事实正确性与聚合指标。
- 2000x 的 EOS 损失集中在本身重复堆砌的临界轮，非事实性退化。

## 复现工具

- 分析：`python tools/analyze_result.py <run.json> [--compare <B.json>]`
  （聚合对比 / EOS 翻转 / 分叉点 / 哨兵题 / 文件名-配置核验）
- 哨兵探针：`kv_cache/probe_sentinel.py`（定位选择问题 vs 寻址问题）
- 全部 run 命令与判定标准：docs/16-run-commands.md；方法演进：docs/17-method-roadmap.md

## 下一步（可选，非阻塞）

- 定型配置的 k8v8 run 跑在折叠修复前（数学等价、末位级差异），如需档案严格
  一致可在修复后重跑一次 2000x。
- vqk 线（weight 表示实验）是独立工作线，本总结不涉及。
