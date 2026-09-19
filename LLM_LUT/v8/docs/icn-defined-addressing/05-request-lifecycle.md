# Request 生命周期与两时间尺度控制（v3，实现前最终规格）

> 承接 `04-reflection.md`。v2 → v3 的四处结构性修订：
> ① block 是 **storage unit**，**contiguous prefix segment** 才是 placement-value unit（placement 不能按单块独立决策）；
> ② Match 输出 local prefix + fetch plan + fresh suffix，不是一个"可达长度"；
> ③ zero-replica 后 metadata 也 TTL 删除——**identity 既不依赖物理副本、也不依赖 directory entry，只依赖内容**；
> ④ baseline 按 **capability** 定义，不把 LMCache/Preble/DualMap 生硬绑定为某个开关。
> **本文档之后停止 architecture 讨论，直接实现 §6 第 1 步。**

## 0. 核心：fast/slow 反馈闭环（NRS 只是状态表示，闭环才是系统）

```text
           demand observations
Fast ──────────────────────────────→ Slow
 (per-request dispatch)               (content placement)
Fast ←───────────────────────────── Slow
            KV residency

Demand → Fast scheduling → observed access/load → Slow placement
→ residency changes → future fast scheduling → ...
```

- **Fast path**：Given current residency, where should request r execute NOW?
- **Slow path**：Given future demand, how should residency change?
- 两时间尺度优化**同一个目标**：固定硬件、固定 SLO 下的服务容量
  （sustainable QPS / throughput under TTFT SLO）。
- NRS/Directory 只是这个闭环的状态表示，不是贡献本身。

## 1. 对象模型：block chain，但两个 unit 必须分开

```text
B0 = H(model_version, tokens[0:b])
B1 = H(B0, tokens[b:2b])
B2 = H(B1, tokens[2b:3b])
...
```

| unit | 是什么 | 用途 |
|---|---|---|
| **block** | 不可变 prefix 块（b tokens 的 attn K,V） | **storage / transport / publish 粒度** |
| **contiguous prefix segment** | 一条连续块链 Bi:Bj | **placement-value 粒度**——reuse 价值是 chain-dependent 的，单独把 B17 复制到没有 B0:B16 的 worker 上几乎无价值 |

vLLM 查 cache 就是从左到右找最长连续命中、断链即停（[vLLM APC]）——placement 同理。

**Slow path 的决策单位是 segment**：
```text
replicate prefix B0:B17 → W3
extend W3's residency B0:B9 → B0:B17
G_rep(p, j) = λ̂_p · ΔC_future(p,j) − C_copy(p,j) − C_memory(p,j) > 0
```
存储仍按块；segment 是决策时的视图。

### Directory 结构

```text
KVBlock(name) → {
    parent_name,
    locators: {worker → tier},
    size_bytes,
    prefix_end_position,
    demand: { λ̂, per-locator pressure, last_access },   # 见 §3 spatial demand
}
```

**metadata 生命周期**（修正 v2 的永久保留规则）：
```text
resident object metadata
    ↓ evict 到零副本
short-lived historical demand record（供 slow path λ̂ 估计）
    ↓ TTL
彻底删除
```
因为 name 是 prefix 的确定性纯函数，即使 entry 完全不存在，下一个 request
也能重新 derive → lookup miss → compute → publish。**identity 由内容保证，
不花存储成本维持。**

## 2. Fast path：request 生命周期

```text
1. Arrive    Request token sequence 到达
2. Derive    派生有序 block name 链 B0 → … → Bk（纯函数，不查状态）
3. Resolve   Directory 返回各 block 的 locator set
4. Match     对每个 worker j 计算三张量：
               L_local(j)      连续本地 prefix 长度
               FetchPlan(j)    把本地链继续延伸所需搬运：
                               {source locator 集合, bytes, 分段数, 预计耗时}
               L_fresh(j)      最终必须 prefill 的 suffix 长度
5. Schedule  C_j = C_queue,j            # compute plane
                 + C(FetchPlan_j)      # 不是单一 acquire 成本：
                                       # 多 source/多次 transfer 都要计价
                 + L_fresh(j)/R_prefill
                 + C_mem,j             # KV 显存压力
             argmin → (j*, plan)
6. Dispatch  分配（排队期间 locator set 可变，执行时刻按当时状态重新 Resolve）
7. Acquire   按 FetchPlan 执行搬运；本地块直接复用
8. Execute   prefill 剩余 suffix + decode
9. Publish   按块发布新生成的 blocks
```

Match 的关键语义：**information plane 给 scheduler 的不是 hit/miss，
而是一张 distributed content map**。"B16@W1, B17@W2, B18@W3 理论可及"
与"W0 本地就有 B0:B18"是两种完全不同的 plan，必须分别计价。

无复用部分 = 正常 prefill（无特殊 recompute action）。

## 3. Slow path：Placement Controller（spatial demand，不是全局 popularity）

v2 的 λ̂ 只是开始。真正驱动 placement 的是**需求的空间分布**：

```text
A = 100 req/s，但需求已自然落在持有 A 且不拥堵的 W0/W1 → 不值得复制
A = 30 req/s，但 W0 队列爆炸、W3/W4 大量收到 A 的需求 → 非常值得复制
```

概念上 slow path 读的是 demand matrix `λ̂[segment][demand-source]`；
第一版实现可简化为 per-locator request pressure，但概念边界先立住：

> **Placement follows spatial demand, not global popularity.**
> （这是从 ICN caching/replication 借来的真正有用的东西。）

决策仍用 §1 的 segment-level `G_rep`，加上 evict 的对称判据
（`C_memory > λ̂·ΔC_future` 时降副本，含 recompute 兜底成本，零副本合法）。

## 4. Baseline 阶梯（按 capability 定义）

| Baseline | 能力 | 备注 |
|---|---|---|
| **B0** | load only（least-loaded） | 无 content awareness |
| **B1** | reactive locality + load routing | 实现/对比时参考 Preble、DualMap 的能力集 |
| **B2** | planned prefix-affinity routing（按历史 rate 规划 warm destination set，set 内按当前 load dispatch） | 实现/对比 CacheRoute |
| **B3** | routing + global remote KV lookup/fetch | 在同级 routing 上加 LMCache 式 lookup+fetch substrate；**LMCache 是 data plane 不是完整 scheduling policy** |
| **Ours** | routing + fetch + **显式 residency/replication 控制** | fast/slow 联合 |

公平表述：**Ours vs 同一 routing policy + LMCache-style global lookup/fetch
但无主动 residency 控制**。不说"打赢 LMCache"。

**论文 claim**：唯一新增的一层是 *request placement 与 KV physical residency
placement 的联合控制*（Preble/DualMap 做 affinity+load；CacheRoute 做 planned
affinity；LMCache 给 remote fetch 机制；都不控制 physical residency）。

**指标**（与 Preble/CacheRoute 的评测方式对齐，不用"总算力"这种抽象量）：
```text
primary : fixed TTFT SLO 下的 sustainable QPS / throughput
latency : P50/P95/P99 TTFT（及 TPOT）
quality : SLO attainment
cost    : redundant prefill tokens（FLOPs 记账）、KV transfer bytes、
          HBM residency、queue imbalance（max/mean）
```

## 5. 与 prototype 的差距

| 组件 | prototype 现状 | 差距 |
|---|---|---|
| 对象模型 | whole-turn snapshot | **第 1 步改造**：block chain + 按块 publish + segment 视图 |
| Directory | locators + reuse_count | parent_name、size、λ̂/per-locator pressure、metadata TTL |
| Match | "prev object 在不在" | L_local / FetchPlan / L_fresh 三张量 |
| Compute state | busy 布尔 | queue_length / kv_memory_free |
| Fast path cost | 三选一 | §2 四项式 + FetchPlan 计价 |
| Placement ctrl | ❌ | 整个 §3（segment-level economic trigger） |
| Data plane | zmq 玩具 | 评估够用；真机换 LMCache/NIXL |

## 6. 实现顺序

1. **Block-chain 对象模型**（本文档之后立即开始）：
   block naming（parent-hash 链）、worker 按块 extract/publish
   （attn K,V 按块切片；GDN recurrent state 作为 turn 边界的 checkpoint
   挂在最后一个块上——它是 S_t 不是块函数）、scheduler 的
   Resolve/Match 改链式；
2. Directory 元数据 + worker 上报 queue/memory/demand；
3. B0–B3 策略开关（同一 fast-path 状态机的 capability 子集）；
4. Placement Controller（segment-level replicate/evict）；
5. 实验矩阵：内容热度偏斜 × 并发度 × baseline 层级，指标按 §4。
