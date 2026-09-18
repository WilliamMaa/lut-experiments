# ICN KV Allocation Simulator — 第一版实现计划

> 配套文档：`00-ideas.md`（方向与问题定义）。本文档把第 7/11 节的 simulator 想法落到具体实现。
> 状态：**已取代（superseded）**。方向已改为全真机 prototype，见 `02-real-prototype-plan.md`。本文档保留作为设计参考（其中的策略定义、指标清单、representation 数字锚点仍然适用）。

## 1. 目标

在 `LLM_LUT/v8/` 下从零搭建一个**纯事件驱动的 KV memory allocation simulator**（无模型推理、无 GPU 依赖，Python + 标准库即可运行），回答 `00-ideas.md` §11 的窄问题：

> 在多 GPU serving 中，如果 KV block 拥有 location-independent name，并让 allocation 同时考虑 existing KV locality、GPU load 和 KV transfer cost，是否能比传统 least-load / cache-affinity allocation 获得更好的并发与 memory utilization？

### v8 特有的主攻点（相对 00-ideas.md 的关键调整）

`00-ideas.md` 假设的经济学是"经典 KV cache 场景"——KV 很大、显存是真约束。但 v8 的实测数字改变了这个前提：

| representation | 常驻大小 | 质量（EOS rate，baseline 0.811） |
|---|---|---|
| BF16 全量 | 20 KiB/token（10 个 full-attn 层合计） | 1.000x（基准） |
| m_sp4（1000x） | 2.5 MiB/请求固定（128 槽 × 10 层，与上下文无关） | ≈ 持平（0.811） |
| m_sp4 + k8v8（2000x） | ~1.3 MiB/请求 | 0.792（Δ≈0.019） |

推论：

1. **"搬 KV"几乎免费**（KV follows compute 总是赢），"没缓存 → recompute"的经典痛点大幅缓解；
2. **真正的张力转移到 representation 选择**：近期前缀保 BF16 保质量，远期前缀降级到 m_sp4/k8v8 省空间。`00-ideas.md` §6 的 "where + which representation" 双决策从锦上添花变成**主要看点**；
3. **replication 近乎免费**，热点副本不再是问题。

因此 simulator 的核心科学问题调整为：

> **在压缩使 KV 存储不再稀缺的前提下，representation-aware 的 KV placement（哪些前缀保 BF16、哪些降级、降级发生在哪个 tier）能否在同等内存预算下保住生成质量，同时让 locality-aware routing 提升吞吐？**

## 2. 设计口径（已确认/默认值）

| 项 | 口径 |
|---|---|
| 硬件假设 | 8×H100（80GB HBM），NVLink 全互联 P2P 400 GB/s，host RAM、NVMe tier；全部可配置 |
| 质量 penalty | v1 用标量：m_sp4 Δ≈0，k8v8 Δ≈0.019（引用 `docs/19-final-summary.md` 实测值）；细粒度建模留 hook |
| 命名粒度 | 一个 KV object = `(doc_id, turn_idx, token_span)`，聚合 10 层；representation ∈ {bf16, m_sp4, k8v8} |
| workload | `data/multi_turn_prompts_v3.jsonl`：8 文档 × 每文档顺序 questions，全部轮次共享文档前缀 |
| 结果纪律 | 写 `results/`，文件名带 policy+config+日期后缀，不覆盖 |
| 红线 | 纯模拟，不加载模型，不涉及任何 `device_map` |

## 3. 关键数字锚点（成本模型标定）

| 项 | 值 | 来源 |
|---|---|---|
| bf16 KV | 20 KiB/token（H_kv=2 × head_dim=256 × 2B × 10 层） | `kv_cache/kv_cache_patch.py:63-66` |
| m_sp4 常驻 | 128 槽/层（sink4 + recent32 + hh92）→ 2.5 MiB/请求 | `kv_cache/heavy_hitter_cache.py:48`、`docs/19-final-summary.md` |
| k8v8 | m_sp4 的约一半（INT8 存储 + metadata） | `kv_cache/kivi_cache.py` |
| prefill 成本 | MoE A3B（3B 激活），按 ~20K tokens/s/GPU 起步估 | CLI 可配，待标定 |

## 4. 代码布局（拟新建 `LLM_LUT/v8/icn_sim/`）

```
icn_sim/
├── __init__.py
├── kvname.py         # KVName 数据类 + 命名/哈希 + representation 定义（bytes_per_object()）
├── workload.py       # 从 multi_turn_prompts_v3.jsonl 生成请求流：
│                     #   session(doc) → 顺序 turn；到达间隔 = think-time 分布（可配）；
│                     #   请求携带 prefix chain = [共享指令前缀, 文档前缀, 前序轮次...]
│                     #   token 长度：字符 heuristic 估算（tokenizer 插件留口，v1 用估算）
├── cluster.py        # 拓扑 + 带宽/容量表：GPU HBM / NVLink P2P / PCIe / RAM / NVMe；
│                     #   CostModel 常量集中处
├── index.py          # KVNameIndex：name → LocationSet[(gpu, tier, repr)]，
│                     #   元数据 reuse_count / last_access / bytes
├── policies.py       # P0 least-loaded / P1 cache-affinity / P2 ICN allocator
│                     #   P2 成本：C_j = queue + transfer + recompute + mem_pressure + quality_penalty
│                     #   P2 独有：representation 选择（近期窗口 bf16，窗外 m_sp4/k8v8）
├── sim.py            # 离散事件模拟（heapq）：GPU = 排队服务器，transfer job 占用带宽，
│                     #   事件：request_arrival / gpu_free / transfer_done
├── metrics.py        # 00-ideas.md §7 指标 + v8 新增：降级前缀占比、质量 penalty、
│                     #   各 representation 内存分布
└── run_experiment.py # CLI: --policy --gpus --arrival-rate --think-time --topo --seed
                      #   输出 results/icn_sim_<policy>_<config>_<date>.json
```

## 5. 实施步骤

### Step 1 — 骨架与 workload（kvname.py、workload.py、cluster.py）

- KVName 命名 + 三种 representation 的 bytes 表（bf16 = 20KiB × span_tokens；m_sp4 = 2.5MiB 固定；k8v8 = 1.3MiB 固定）。
- workload 解析 trace：8 个 session，每 session 内顺序 turn；输出 `Request(session_id, turn_idx, prefix_chain, new_tokens, arrival_time)`。
- **验收**：跑通脚本打印 8 session × 各 turn 的 prefix 长度分布，确认两个复用来源都在（文档前缀跨轮共享、轮次链式共享）。

### Step 2 — 事件模拟器 + P0/P1 baseline（sim.py、index.py、metrics.py、policies.py 的 P0/P1）

- GPU 简化为：队列 + 占用至完成；prefill 时长 = f(prefix_tokens, new_tokens)，decode 时长 = f(new_tokens)。
- P0：路由到队列最短的 GPU。P1：路由到 prefix 命中最多的 GPU（平手按负载）。
- 命中判定：请求 prefix chain 中已在目标 GPU 的 KV object 集合；未命中部分计入 recompute tokens。
- **验收**：单次 run 产出全部指标 JSON；P1 的 hit rate > P0，P0 的队列延迟方差 < P1（策略间大小关系符合直觉，证明模拟器没写反）。

### Step 3 — P2 ICN allocator + representation 选择（policies.py 的 P2）

- 每请求对每候选 GPU 计算 C_j（排队 + 传输 + 重算 + 显存压力 + 质量惩罚），取 argmin。
- 传输成本按目标 representation 的实际 bytes × 路径带宽（NVLink/PCIe/RAM 分档）。
- representation 决策：请求落卡后，recent_window（可配，默认 8K token）内 prefix 保 bf16，之外按内存压力选 m_sp4 / k8v8；index 记录每个 object 的 representation。
- 质量惩罚 = 请求使用降级 representation 的 prefix token 比例 × 该 representation 的 EOS delta。
- **验收**：transfer vs recompute 出现合理的策略切换（短前缀倾向 recompute、长前缀且 NVLink 可达倾向 copy），日志可观察。

### Step 4 — 实验矩阵与对比报告

- 3 policies × 3 seeds × 2 到达率（低/高负载）= 18 runs。
- 对比表：hit rate / recompute tokens / throughput / p99 queueing / 质量惩罚 / 内存分布。重点回答：
  - **a.** P2 是否 Pareto 优于 P1（同等质量下吞吐更高，或同等吞吐下质量更高）？
  - **b.** representation-aware 降级带来的内存节省是否换来可接受的 quality penalty？
  - **c.** 在什么到达率 / think-time 区间 ICN 收益最大？
- 结果写入 `results/`，文件名带 policy + config + 日期后缀。

### Step 5 —（stretch，本次不做）

hot-content replication、PIT 式 fetch aggregation、trace 外加共享 system prompt 的全局前缀对象——等 Step 4 数据确认基线成立后再加。

## 6. 验证方式

- 无 GPU 依赖，`python -m icn_sim.run_experiment ...` 直接跑。
- Step 2/3 各有一个行为 sanity check（见上），依赖策略间的大小关系与切换行为，不依赖精确数字。
- 最终验证 = 18-run 对比表能干净回答 §5 Step 4 的 a/b/c 三个问题。

## 7. 风险与注意

- **成本模型标定是最大不确定项**（prefill 吞吐、think-time 分布）。缓解：全部做成 CLI 可配参数，Step 4 对关键参数做敏感性扫描而非单点结论。
- trace 只有 8 个 session，到达过程靠合成分布。结论限定在"该 workload 形态下"，不外推绝对数字。
- 纯模拟不碰 v8 现有评测管线（`common/evaluator.py`），零回归风险；质量数字直接引用 `docs/19-final-summary.md` 实测值，不在 simulator 里重测。
