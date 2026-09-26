# 14 — ICN 机制 × LLM inference-state 映射矩阵（项目总纲）

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
| 4 | Content Store / 分层驻留 | HBM→DRAM→… 驱逐≠消失 | HBM 稀缺，驱逐=销毁导致重算回潮 | 🔶 部分（E2 进行中） | sp96 格：tier 满→丢块→回潮 2.75×，边界已现；sp-1 档证据待补 |
| 5 | Replication（度数×位置） | 热 state 多副本，副本可落在不同 tier | 热点 state 单副本成为单点瓶颈/等待源 | 🔶 部分，HBM-only 版本已证伪 | 06/E1：复制进 HBM→挤占→驱逐→churn；**"副本落在便宜层"的组合未试** |
| 6 | Nearest / best-copy retrieval | topology/load-aware  locator 选择 | 所有副本一视同仁，无视距离与拥塞 | ❌ 未做 | 单机 4-worker 扁平拓扑测不了；需多机或模拟拓扑代价 |
| 7 | Interest aggregation（PIT） | 并发请求同一未就绪 state，合并只算/搬一次 | 高并发同前缀风暴下的重复劳动 | ❌ 未做 | **当前 testbed 可测**：open-loop Poisson + zipf 共享文档天然产生并发同前缀需求 |
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
| E2（进行中） | 4（+5 的"tier 副本"前提） | 驱逐落点=便宜层是否打断上述链条；tier 稀缺边界 |
| E3（设计） | 6 的雏形 | fetch 代价 ratio 多少时 placement 开始划算（纯 regime 边界） |

## 4. 下一步优先级

1. **先收尾 E2 的基底分解**（b3 × {s0, sp-1, sp96}，控制器关闭）：
   这是行 4 的干净验证——同一驱逐压力，落点=虚空时重算、落点=DRAM
   时 recall，**逐环过程证据**，不看比分。命令在 runbook 11 §5；
   跑时加 `--trace-dir`，用 trace_replay 逐块重放链条（11 §6）。
2. **Interest aggregation（行 7）**：当前 testbed 内价值最高的未做
   机制。scheduler 加 PIT 式 pending-object 表：一个 turn 正在
   fetch/prefill 某前缀时，后续同前缀需求挂为等待者，就绪后一并
   唤醒。观测量：重复劳动抑制次数。它解决的"并发同前缀风暴"是
   高并发 serving 的真问题，且比 proactive replication 更 ICN 正统。
3. **Nearest-copy（行 6）**：需要多机或拓扑代价模拟，排在 testbed
   升级之后；E3 的 fetch-cost ratio 是它的单维前哨。

## 5. 负结果政策

任何一行判否都是有效结论，前提是：**否在哪个环节、regime 边界在
哪**，由过程计数器说清，不由终局比分说清。已有一例：行 5 的
HBM-only 复制（06/E1）——链条逐环有据，负得成立。
