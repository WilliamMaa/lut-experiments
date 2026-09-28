# 16 — 总结论：ICN 机制 × LLM 推理状态共享（原理、实现、数据、结论）

> 本文是这条研究线的完整结项文档：只有原理、实现、ICN 映射、
> 实验流程、数据与结论。过程性的事故记录见 12 号，逐行判文
> 细节见 13/15 号，机制总纲见 14 号。本文自足，不依赖它们。

## 1. 研究问题与定位

**已有系统不是不会做 prefix cache**：vLLM 在同 engine 内已有
基于哈希的 automatic prefix caching，LMCache 已支持跨实例的
lookup/transfer。所以本研究的问题不是"能不能复用前缀"。

**真正的问题是**：现有系统把可复用推理状态（KV / 层间
checkpoint）当作 serving engine / cache 的**局部资源**来管理——
属于哪个实例、哪块卡、哪个进程。本研究把它提升为
**location-independent 的命名状态**（named state），然后系统性地
回答：**ICN 的哪些 sharing/control 原语在这个抽象上成立、哪些
不成立、价值由什么 regime 决定。**

**起点**：RQ1——在 35B 真模型（Qwen3.6-35B-A3B，hybrid 架构，
full-attention KV + recurrent state 混合）上验证内容命名的状态
块跨实例取回注入是 token 级一致的，不是近似。

**终点**：14 号矩阵 9 行机制全部终态判文（2026-09-28）。

**核心结论**：**ICN 机制不是整体搬得过来的。一部分直接成立，
一部分需要重新诠释，一部分强依赖 workload/hardware regime，
一部分应该直接丢弃。** named state 抽象消掉三类重复——算过之后
再重算、驱逐后因状态消失而重算、状态尚在生成/搬运时的并发重复——
三类收益分别受**空间共享度、时间重叠度、驻留层成本**控制。

## 2. ICN 机制 × LLM serving 的映射（四类分类）

| 类型 | 机制 | LLM serving 对应 | 结论 |
|---|---|---|---|
| **直接成立** | Name / identity-location 解耦 | 前缀块按内容哈希命名，状态不属于 session/GPU | ✅ token 级一致（含 hybrid recurrent checkpoint） |
| | Name resolution | 块名 → 多 holder 集合（worker/tier） | ✅ 全局可寻址 |
| | 跨节点按需取块（Interest/Data） | 缺块 → 点名取 → 送达注入 | ✅ 成立且很强 |
| **成立但强 regime-dependent** | Content Store / 分层驻留 | HBM→DRAM 驱逐 ≠ 消失 | ✅ 落点决定一切 |
| | Interest aggregation（PIT） | 并发同需求合并只算/搬一次 | ✅ 风暴 regime 成立，错峰判否 |
| **条件性 / 待完成** | Replication（度数×位置） | 热 state 多副本 | 🔶 HBM-only 证伪；tier 场景方向性，未做隔离对照 |
| | Nearest / best-copy retrieval | 拓扑/负载感知的 holder 选择 | ❌ 单机扁平拓扑测不了，留待多机 |
| **不应照搬** | Freshness 包语义 | state 有效性 | ✅ 被 namespace/version identity 吸收（重新诠释，非照搬） |
| | NDN 包平面（FIB/PIT 路由） | —— | ❌ 明确排除（只借抽象） |

## 3. 原理与实现

### 3.1 内容命名的状态块（直接成立的三行机制的基石）

推理前缀按 16-token 切块，块名 = 内容哈希（`.../i/<layer>/
span/<start>-<end>/repr/bf16`）。名字的同一性就是有效性的全部
保证：同名块在任何 worker 上算出/取回都逐 token 一致。

**对 hybrid 模型这一点比普通 APC 更强**：Qwen3.6 的 recurrent
state 只在精确的 prefix checkpoint 边界上有定义，不能像普通 KV
那样任意断点复用。本原型的块命名与 resume 协议把 recurrent
checkpoint 编进块身份与链式 tip（worker 侧 `linear_checkpoint`
语义），因此**内容寻址的推理状态对 attention/recurrent 混合模型
依然良定义**——前提是可复用状态携带正确的 checkpoint 边界。
这是 named state 行最实质的证据，不是"我们也做了哈希"。

### 3.2 系统结构

- **scheduler（控制面）**：块目录（name → holder 集合 + 字节数）、
  每 worker 的 resident/tips/spilled 视图、放置决策、活性管理
  （心跳标灰、stall 上限、死锁 backstop——任何等待有界）。
- **worker（数据面）**：35B 模型两卡均分明示放置（禁自动多卡
  切片），本地 HBM 驻留 + 主机 DRAM 溢出层，收到 assign 后按
  resume 集注入块、prefill 增量、decode、发布新块并上报。
- **resume 语义**：turn 带着"从块 E 处续算"的指派到某 worker，
  缺的块本地召回（spilled）/ 跨 worker 取（fetch）/ 重算
  （fresh）。复用率 hit = 免算 token 占比。
- **聚合是纯控制面行为，worker 协议零改动**（PIT 行）。

### 3.3 能力阶梯（增量归因的尺子）

b0（无内容感知，全量重算）→ b1（本地匹配）→ b2（全局可见 +
计划性亲和）→ b3（+跨 worker 按需取块）→ ours（b3 + 放置
控制：驱逐到 tier、热点复制）。每一档能力的增量可单独计量。

**归因纪律**：b1→b2 的 new_tok 5852→4558 量化的是"全局内容
可见性 + 计划性亲和"的合并收益，**不是 name resolution 单独的
收益**。name resolution 一行的成立证据是机制正确性：同名 →
解析出远端 holder → 取回 → resume → token 级一致。机制正确性
与性能归因不混用。

### 3.4 分层驻留（regime-dependent 第一行）

驱逐不再销毁块，而是落到主机 DRAM tier（容量可配：无限 sp-1 /
96MB sp96）。worker 召回走 PCIe（~20GB/s，代价模型远低于跨
worker 取块）；tier 满则 LRU 踢除、块真消失。

**一阶问题不是"驱逐什么"，而是"驱逐到哪"（where eviction
lands）**：同一个驱逐动作，落点便宜 → 状态可召回，重算回潮被
吸收；落点虚空 → 状态销毁，重算回潮数倍反弹。数据见 §5.1。

### 3.5 Interest aggregation / PIT（regime-dependent 第二行）

两个聚合点，共用一个 PIT 表：

- **A 计算聚合**：同前缀指纹的 turn 到达时，若同 fp 的 serving
  turn 在飞 → 挂为 waiter；serving 完成发布块后唤醒走正常放置。
  同一条链只算一次。
- **B 取块聚合**：两个 worker 同时要同批块 → 后者挂为 waiter
  骑在前者的 in-flight fetch 上；payload 到手后组播 deliver，
  同一批字节只搬一次线。

这正是 NDN PIT 的核心思想：同一个 named state 正在生成/传输时，
后续相同需求挂到同一个 in-flight 操作上。现实 serving 系统里
这个问题并未解决——并发请求命中同一前缀时，各自在 GPU 上
materialize 一份 KV 仍是常态，PIT 式聚合对准的正是这个缝隙。

记账口径：`saved_tok` = 每次 park × 当时 cum_tokens 的累加，
是上界口径（同一 turn 反复 park 会重复计），不是去重 token 数。

## 4. 实验流程与判文政策

- **workload**：闭链（16 session 同时起步，share=2 制造 8 对
  完全相同前缀链——风暴 regime）+ open-loop（poisson 到达 +
  zipf 文档热度 + think time——错峰 regime）。
- **单元**：matrix 一格 = 独立集群（4 worker × 2 A800-80GB）
  跑一个 (workload × 策略 × rep) 组合，出 summary JSON +
  逐 turn 记录 + 块生命周期 trace。
- **判文政策（事先写死）**：过程计数器优先，终局比分（rps/SLO）
  只在机制成立后参考；负结果必须说清"否在哪个环节、regime
  边界在哪"才算有效结论。

## 5. 数据

### 5.1 分层驻留（E2，16 格矩阵，spill ∈ {无限, 96MB} × zipf
s ∈ {1.0, 1.6} × {b3, ours} × 2 reps，SLO 全 1.000）

```text
wl            policy  hit    new_tok  xfer    repl  evict  recall  favoid
sp-1 × s1.6    b3     0.979   21776   139.5    0.0   340.5  27587    75.0
sp-1 × s1.6   ours    0.979   21656   117.5   23.5   305.0  24180    79.5
sp96 × s1.6    b3     0.933   64807   159.5    0.0   246.5  11990    25.5
sp96 × s1.6   ours    0.928   63024   133.0   43.0   187.5   7825    26.0
sp-1 × s1.0    b3     0.976   22473   148.5    0.0   357.0  27722    47.0
sp-1 × s1.0   ours    0.976   22188   137.5    8.0   312.0  25832    51.5
sp96 × s1.0    b3     0.884   97456   157.5    0.0   263.5  11717    23.5
sp96 × s1.0   ours    0.916   77198   169.5   15.5   215.5   9411    14.0
```

判读：

- **P1（tier 无限，复制有增量吗）→ 不成立**：sp-1 档 b3 与 ours
  的 hit 逐位相等（0.976/0.979），new_tok 差 <1.3%，repl 仅
  8-23 次/格无影响。落点可召回后"驱逐=消失"不复存在，按需取块
  拿走全部收益。**E1 时代 20.8× 的崩塌不是复制的锅，是虚空
  落点的锅。**
- **P3（tier 有限的回潮，regime 边界）→ 成立**：sp96 vs sp-1
  hit 从 0.976-0.979 跌到 0.884-0.933，new_tok 涨到 2.9-4.4×。
  容量瓶颈从 HBM 转移到 tier；tier 满 → LRU 踢 → 块真消失。
  trace 重放显示单块一生被踢数百次，是 tier 真实丢块的直接计量。
- **P2（tier 给复制创造新角色）→ 方向性成立，未隔离**：sp96 档
  ours 的 evict 更少（187-215 vs 246-263，复制分担驱逐压力）；
  缺"复制 vs 不复制"同预算严格对照，**不写"成立"**。

**Replication 的准确表述**：条件性成立（conditional）——其价值
取决于副本放置成本与内存层级，尚未被隔离为独立的正向贡献。
在受测 request-level workload 里，向稀缺 GPU 驻留复制一般不
值得。

### 5.2 Interest aggregation（行 7，闭链 s16 t3 share2 q40，
{b3, ours} × {pit off, on} × rep=3）

| 腿 | 兑现（served/64 turns） | saved_tok（上界） | fetch merge | failed |
|---|---|---|---|---|
| b3 pit=on ×3 | 41 / 47 / 45 | 175k / 190k / 188k | 2/格 | 0 |
| ours pit=on ×2 | 41 / 42 | 167k / 177k | 2/格 | 0 |

- 基线机会（pit=off，逐 rep 稳定）：A_pairs=4，A_re_tok=6082，
  B_pairs=1/格。pit=off 的时间窗重叠测量在高并发下系统性低估
  （合并本身拉长 serving 窗、制造新重叠），判文以 pit=on 计数器为准。
- 质量面：b3 的 hit/new_tok（0.969/4810）开不开 pit 逐位一致；
  ours 的 new_tok 被 pit 从 9296 拉回 4810——放置控制引起的
  迁移重算被聚合 resume 吸收，pit 对 ours 是净收益。
- **边界（本行最有价值的部分）**：open-loop（poisson + think
  错开）机会量 ≈0，判否。收益 ∝ 同一需求的**时间重叠度**——
  风暴成立，错峰一文不值。这不是"机制有时灵有时不灵"，而是
  regime 的定量边界。

## 6. 结论

**把可复用推理状态变成按内容命名的共享对象后，可以消掉三类
重复，收益各自由一个 regime 维度控制：**

1. **跨位置复用**（直接成立）：状态算过之后，别处按需取回，
   token 级一致——消掉"算过再重算"。收益受**空间共享度**控制。
2. **跨层级保留**（成立，落点决定一切）：驱逐落点是便宜层时
   状态不消失——消掉"驱逐后重算"。收益受**驻留层成本**控制；
   一阶问题是"驱逐到哪"，不是"驱逐什么"。
3. **并发需求聚合**（成立，强 regime-dependent）：同一状态尚在
   生成/搬运时，并发需求挂到同一 in-flight 操作——消掉"进行时
   的重复劳动"。收益受**时间重叠度**控制。

**机制层面的判定**：

- 直接成立：naming（含 hybrid recurrent checkpoint 边界）、
  resolution、跨 worker 按需取块。
- regime-dependent：分层驻留、PIT 聚合。
- 条件性：replication（HBM-only 已证伪；tier 场景未隔离）。
- 需要重新诠释而非照搬：freshness 被 namespace/version
  identity 吸收。
- 明确丢弃：NDN 包平面；单机不可测留待多机：nearest-copy。

**一句话（contribution statement）**：named inference state
使三种共享成为可能——after-production reuse、across-tier
preservation、in-flight demand aggregation；其价值分别由空间
共享度、时间重叠度、驻留层成本控制。

## 7. 边界与遗留（均不推翻以上结论）

按优先级：

1. **PIT 正式化**：把行 7 的实验做成最干净的一组正式结果
   （补 ours 第三 rep、固定记账口径的去重 token 数），这是
   目前最适合往论文推的一条。
2. **Replication 隔离对照**：同预算"除复制外一切相同"的严格
   对照，决定行 5 从"方向性"升级还是维持证伪。
3. **Nearest-copy（行 6）**：多机 / 拓扑代价模拟。datacenter
   层级拓扑（同卡 → 同机 NVLink → host DRAM → rack 内 →
   跨机 RDMA）下 `name → locator set` 才真正产生
   `argmin C(ℓ)` 的最副本选择，可能是把本工作放大到 GPU farm
   的关键一行。
4. 工程收尾：PIT 在 ours 下的稀有活性竞态（死锁 backstop 已
   保障秒级显形），修复后补一格 rep 即可。
