# 14 — ICN 机制 × LLM inference-state 映射矩阵（项目总纲）

> **结项总结论（原理/实现/数据/结论全文）：`16-conclusions.md`。**
> 本文是机制清单与判文状态；过程事故见 12 号，逐行判文见 13/15 号。

研究问题（唯一）：

> **ICN 的 information-centric resource-management 思想，有多少可以在
> LLM inference state 上成立？** 每个被搬过来的机制必须各自解决一个
> 真实存在的问题；不成立的机制也要给出 regime 边界。

不是"做一个更好的 KV scheduler"，不是 ours vs b3 的比分。

## 1. 机制矩阵

| # | ICN 机制 | LLM 对应 | 解决的真实问题 | 状态 | 证据 / 缺口 |
|---|---|---|---|---|---|
| 1 | Name / identity-location 解耦 | 前缀/KV/checkpoint 有 content identity，state 不属于 session/GPU | 状态焊死在算出它的进程上，跨实例只能重算 | ✅ 成立 | RQ1：35B hybrid 真模型上 token 级一致 |
| 2 | Name resolution（name → locator set） | 块名 → 多个 worker/tier 持有者 | "去哪找可复用 state"从应用问题变系统原语 | ✅ 成立 | b1→b2 阶梯：new_tok 5,852→4,558 |
| 3 | 跨 worker 按需取块（Interest/Data） | 缺块→点名取→送达注入 | 复用不局限于本机 | ✅ 成立且很强 | b3 在多种 workload 下稳定；06 证明其强度 |
| 4 | Content Store / 分层驻留 | HBM→DRAM→… 驱逐≠消失 | HBM 稀缺，驱逐=销毁导致重算回潮 | ✅ 成立（E2，2026-09-27 判文） | sp-1：外部性全吸收（b3=ours，P1 三次复现）；sp96：tier 满→LRU 踢→真消失，回潮 2.9–4.4×（P3）；trace 踢块计数为直接计量 |
| 5 | Replication（度数×位置） | 热 state 多副本，副本可落在不同 tier | 热点 state 单副本成为单点瓶颈/等待源 | 🔶 方向性成立 | HBM-only 已证伪（E1：虚空落点才是罪魁）；tier 有限时复制有保护价值的迹象（E2 P2：tier LRU 踢块高频 + evict 分担），缺"复制 vs 不复制"同预算严格对照 |
| 6 | Nearest / best-copy retrieval | topology/load-aware  locator 选择 | 所有副本一视同仁，无视距离与拥塞 | ❌ 未做 | 单机 4-worker 扁平拓扑测不了；需多机或模拟拓扑代价 |
| 7 | Interest aggregation（PIT） | 并发请求同一未就绪 state，合并只算/搬一次 | 高并发同前缀风暴下的重复劳动 | ✅ 成立（2026-09-28 判文，15 号 §8） | 闭链风暴 regime：served 41-47/64 turns、saved_tok 16.6-19 万（上界口径）、m_fetch=2、failed=0，b3/ours 两策略三 rep 复现；open-loop 机会量≈0 判否（poisson+think 错开）。残余：ours 稀有超时竞态（12 号事故六，backstop 已入码，不推翻判文） |
| 8 | Freshness / lifetime | model 版本 / adapter / cache salt 的 state validity | 错误 state 被复用 | ✅ 基础已有 | 块名含内容哈希，语义即 validity；不需要 ICN 原机制 |
| 9 | NDN 包平面（FIB/PIT 路由转发） | —— | —— | ❌ 不需要 | 明确排除：只借抽象，不碰网络栈 |

## 2. 数据面 vs 控制面（边界）

外部已有：vLLM 的内容哈希块命名（数据面）、LMCache 的跨实例
lookup/transfer（数据面）。

本项目的位置：**这些信息之上的 information-centric 控制面**——
locator 解析、驻留层级决策、复制放置、需求合并。不重复造数据面。

## 3. 现有实验在矩阵里的位置

| 实验 | 矩阵行 | 它回答的 |
|---|---|---|
| RQ1 / E0 | 1, 2, 3 | 命名+跨 worker 恢复在真模型上成立吗 |
| 06 / E1 | 5（HBM-only 版） | 复制进稀缺 HBM 为什么亏（驱逐外部性链条，过程计数器直接观察到） |
| E2（主体已判，2026-09-27） | 4（+5 的"tier 副本"前提） | 驱逐落点=便宜层打断 E1 链条（✅ P1/P3）；tier 稀缺边界在 tier 容量（✅）；落点=虚空的对照（s0）待 §5a 补齐 |
| E3（设计） | 6 的雏形 | fetch 代价 ratio 多少时 placement 开始划算（纯 regime 边界） |

## 4. 下一步优先级

1. **补齐 E2 的 s0 对照**（b3@s0 × {s1.0, s1.6}，runbook 11 §5a）：
   sp-1/sp96 两档已判（13 §4），缺落点=虚空的第三档，补齐后行 4
   的"落点决定链条"三角完整。**逐环过程证据，不看比分。**
   另有两件收尾不挡判文：`stale_resume_retries` 报告核验（11 §5b）、
   b3 sp96×s1.6 一格 wall=310s 查因（11 §7c）。
2. ~~**Interest aggregation（行 7）**~~ **✅ 已判文（2026-09-28，
   15 号 §8）**：闭链风暴 regime 成立，跨策略复现，open-loop
   判否。判文与计数器证据见 15 号；事故五/六（12 号）是过程
   中的两个工程bug，均已处置，不影响判文。
3. **Nearest-copy（行 6）**：需要多机或拓扑代价模拟，排在 testbed
   升级之后；E3 的 fetch-cost ratio 是它的单维前哨。

## 5. 负结果政策

任何一行判否都是有效结论，前提是：**否在哪个环节、regime 边界在
哪**，由过程计数器说清，不由终局比分说清。已有一例：行 5 的
HBM-only 复制（06/E1）——链条逐环有据，负得成立。

## 6. 总结论（2026-09-28，矩阵 9 行全判完结）

**"把推理状态从焊死在 GPU 进程上，变成按内容命名的可寻址共享
对象"——这个抽象在 35B 真模型上成立；每个机制省多少、什么时候
省，边界全部量出。**

1. **复用本身成立且很强**（行 1-3）：内容命名的 KV 块可跨
   worker 定位、按需取回、token 级一致注入，不是近似。
2. **驱逐落点决定一切**（行 4，E2 判文）：驱逐 ≠ 消失。落点是
   便宜层（DRAM tier）→ 重算回潮被完全吸收（P1）；落点是虚空
   → 回潮 2.9-4.4×（P3，tier 满 → LRU 踢 → 真消失）。
3. **HBM 稀缺不靠多复制解决**（行 5）：HBM-only 复制被驱逐
   外部性链条证伪（E1 的 20.8× 崩塌是虚空落点的锅）；tier 时代
   复制有方向性的保护价值，缺同预算严格对照，不写"成立"。
4. **并发同前缀风暴的重复劳动可合并消除**（行 7，15 号 §8）：
   闭链风暴 regime 下 2/3 的重复 prefill 服务被 PIT 聚合掉，
   b3/ours 一致复现、failed=0；open-loop 错开则一文不值。
   **价值由"共享度 × 并发度"的 regime 决定，不是机制有无。**
5. **判否同样是结论**（行 5 的 HBM-only 版、行 6 单机不可测需
   多机、行 8 内容哈希天然解决、行 9 明确不需要）。

一句话给外行：**ICN 式内容寻址是 serving 侧消除重复计算/重复
搬运的正道，且"什么时候值得上"有定量答案——闭链风暴、高共享、
落点便宜时收益最大。**
