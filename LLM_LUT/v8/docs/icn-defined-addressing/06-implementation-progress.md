# 06 方法论、结果与进度（ICN-defined addressing，截至 2026-09-21）

> 本文档回答四件事：**我们用什么方法验证**（§1–2）、**到底实现了什么目标**
> （§3）、**目前的结果是什么**（§4）、**Step 5 矩阵要回答什么问题、怎么算赢**
> （§5）。工程细节（组件、协议、踩坑）压缩在 §6，环境事实在 §7。
> 代码：`LLM_LUT/v8/icn_proto/`（本地 `LLM_LUT/v8`，远程 `~/lut-experiments/LLM_LUT/v8`）。

## 1. 研究问题

ICN（ Information-Centric Networking ）给 KV cache 的核心启发是：
**内容的名字可以脱离内容的位置**——同一前缀的计算结果天然可共享、可寻址、
可按内容路由。本项目要回答三个递进的科学问题：

- **RQ1（命名与正确性）**：content-addressed 的 KV 块链命名，在真实 35B
  hybrid 模型（含 GDN linear attention）上是否端到端成立？跨 worker 共享
  同一内容是否产生逐 token 一致的结果？
- **RQ2（能力归因）**：serving 收益能否**按能力层归因**？即"locality 路由、
  planned affinity、global fetch"各贡献多少，而不是糊成一个"我们的系统 vs
  他们的系统"的总数。
- **RQ3（联合控制的增量价值）**：在需求偏斜 + 并发排队下，**显式 residency
  控制**（复制/驱逐的经济学决策）比同 routing 能力下的纯 reactive fetch
  （B3）多赚多少 QPS/SLO，代价是多少传输字节与 HBM？

RQ1/RQ2 已闭环（§3），RQ3 是 Step 5 矩阵的主题（§5）。

## 2. 方法论

### 2.1 阶梯式归因设计（capability ladder）

对照系统不按"实现"划分，而按**能力**划分（05 §4），每一层恰好比上一层
多一个能力，增益即可归因：

```text
B0  load-only .................... 无内容感知（调度器看不见 KV 内容）
B1  + reactive locality .......... 看得见本地有什么，命中就复用
B2  + planned affinity ........... 看得见"谁快空了且暖"，愿意等
B3  + global fetch ............... 看得见全网有什么，缺的去取
Ours + residency control ......... 不止取，还按需求信号预先摆放（复制/驱逐）
```

相邻两层的指标差 = 该能力的边际贡献。Ours vs B3 是**唯一**包含新科学内容
的比较（05 §4："唯一新增的一层是 request placement 与 KV physical
residency placement 的联合控制"），其余四层是与 Preble/DualMap/CacheRoute
能力集对齐的公平参照。

### 2.2 多指标互锁（防作弊设计）

单一指标总能被"作弊式优化"，所以每个 claim 至少两个指标互锁：

| 想证明的 | 主指标 | 锁它的副指标 |
|---|---|---|
| 算力节省 | `new_tokens_processed`（GPU prefill 记账） | transfer_bytes（不能靠疯狂传输换） |
| 延迟/QPS | P50/P95、SLO attainment | 算力（不能靠重复 prefill 换） |
| 复制器价值 | spill 场景的 P95 / QPS@SLO | replicated_bytes（复制必须有界） |
| 命名正确性 | decoded_ids 一致（CONSISTENT） | failed=0 |

### 2.3 正确性不变量

ICN 式共享的前提是"**同内容 ⇒ 同结果**"。每次 run 用
`check_consistency` 核验：相同前缀指纹（fp）的 turn 无论落在哪个 worker、
无论 resume 还是 fetch，decoded_ids 必须逐 token 一致，否则全 run 作废。
这把"分布式 KV 复用"从工程技巧升格为**可证伪的断言**。

### 2.4 噪声控制

- **环境噪声**：`balanced_low_0` 会被邻居租户重塑布局（Step 0 的 flake 根
  因），改为确定性 `explicit_even` 分片 + 加载自检（§6）。同一 workload
  的指标差异才能归因于 policy，而不是某次加载运气。
- **调度噪声**：每格 3 遍取均值（矩阵 runner 强制）；closed-loop workload
  消除到达流随机性（局限性见 §5.5）。
- **单元/集成双层验证**：链逻辑、策略门控、驱逐保护写成纯 Python 单测
  （stub zmq，本地可跑）；真机只跑标准入口，两层都绿才算数。

## 3. 实现了什么目标（对照 RQ）

| 目标 | 状态 | 证据 |
|---|---|---|
| RQ1：内容寻址块链在真实 35B hybrid 模型端到端成立 | **✅ 达成** | 块链 + per-block GDN checkpoint；E1 roundtrip + 全部 run `=> CONSISTENT`、failed=0 |
| RQ2：收益可按能力层归因 | **✅ 达成** | 五档梯度完美单调（§4 表），73.9% 算力节省可拆到层 |
| RQ3 的机制：经济触发的复制 + 对称判据的驱逐 | **✅ 机制就绪** | G_rep 复制/驱逐真机触发、`failed=0`、protection 语义生效（§6 Step 4） |
| RQ3 的增量：Ours > B3 的 QPS/SLO | **⏳ 待矩阵** | flat workload 下控制器正确地静默（无冷目标）；分化需要偏斜+并发（§5） |

一句话总结当前状态：**"ICN 式 KV 命名 + 能力阶梯"已被证明成立且可归因；
"联合 placement 控制"的机制已就绪并单独验证，它与 B3 的差异价值是下一个
实验的主题。**

## 4. 目前的结果

### 4.1 能力阶梯（同一 workload：4 sessions × 2 turns，doc-share=2，2 workers）

| policy | hit_rate | resumed | new_tok | xfer | 解读 |
|---|---|---|---|---|---|
| b0 | 0 | 0 | 17,450 | 0 | 无内容感知，全量 prefill（算力分母） |
| b1 | 0.667 | 8 | 5,852 | 0 | reactive locality：同 worker 命中即复用，省 66.5% |
| b2 | 0.75 | 9 | 4,558 | 0 | planned affinity：等暖 worker，再省掉一次冷启动 |
| b3 | 0.75 | 9 | 4,558 | 1×29MB | global fetch：跨 worker 共享改"重算"为"搬运" |
| ours | 0.75 | 9 | 4,558 | 1×29MB | 此刻 ≡ b3（复制器在 flat 负载下无冷目标，正确静默） |

**头条数字：ours 比 b0 少算 12,892 个 prefill token（73.9%）。**
这就是"ICN 系统级算力节省"的记账口径（B0 vs Ours 的 `new_tokens_processed`
差）。同时注意诚实边界：这个 workload 太小（12 turns），延迟五档没拉开
（27–29s wall 内），**算力指标可信，QPS/延迟指标要到矩阵里取**。

### 4.2 Placement controller 的单独验证

- **驱逐**：40MB 人为预算（工作集 74MB）下真机触发 5 次，`failed=0`；
  每次只清 1–2 个独占冷 tip，共享祖先（对齐 tip = 别人的内部链块）全部
  存活——protection 语义与单测一致。
- **复制**：planner 经 22 项单测（G 门控/全链段/冷却/broken-chain 跳过/
  ack 归属/驱逐保护）；真机上修复了"复制刚送达 segment"的陈旧视图问题后，
  flat 负载下不再出现冗余复制。
- **per-locator λ̂**：directory 每个块带 `loc{worker→λ̂}`，G_rep 用目标
  worker 本地观测到的需求率评估——05 §3 "placement follows spatial
  demand, not global popularity" 的 v1 落地。

## 5. Step 5 矩阵是什么

### 5.1 要回答的问题

RQ3：**偏斜 + 并发下，Ours 比 B3 多赚多少，代价多少？**

为什么现有 workload 回答不了：2 workers 时任何偏斜都会被"前两次到达各自
全量 prefill"自愈——热点 doc 很快处处都有，控制器没有工作可做（这是
**正确的**静默，不是 bug）。必须有"热点只住在部分 worker 上、且请求多到
要 spill 到冷 worker"的场景，复制器才有差异化空间。

### 5.2 设计

| 维度 | 取值 | 作用 |
|---|---|---|
| workers | **4**（8 卡 × 2 卡/worker） | 制造"部分持有"的常态 |
| 偏斜 | doc-share **2**（8/16 sessions 共享同一热点 doc）vs **8**（轻度复用） | 热点强度两个等级 |
| 并发 | 16 sessions × 3 turns，closed-loop | 4 worker 吃 16 个 session → 必然排队、必然 spill |
| policies | b0 b1 b2 b3 ours | 阶梯全量 |
| reps | 3 | 消调度噪声 |
| 预算 | 全关 | 隔离变量；eviction 交互另行研究 |

机制上的预期分化：doc-share=2 时每个热点 doc 自然落在 ~2/4 workers 上；
负载 spill 到冷 worker 时，**B3 每次在关键路径付 ~1.4s fetch，Ours 在
λ̂>G_rep 盈亏平衡后付一次离线复制，之后 local resume**（关键路径只剩
~0.05s inject）。复制有界：max-in-flight=2、按 (tip,target) 冷却、
demand 已在送达即跳过。

### 5.3 预注册预测（跑之前写下，防事后讲故事）

1. **share=2**：`new_tok` 各策略与 flat 实验同序（b0 ≫ b1 > b2≈b3≈ours）；
   差异出现在 **P95 与 SLO attainment**——ours > b3 > b2 > b1 > b0；
   ours 的 `replicated_bytes` 有界（≈ 热点 doc 数 × 冷 worker 数 × 段大小）。
2. **share=8**：偏斜减弱，ours 与 b3 的差距缩小但不消失（仍是 2/4 持有率）。
3. **算力**：b3 与 ours 的 `new_tok` 应基本持平——复制的价值是延迟与吞吐，
   不是再省算力（flat 实验已把可省的算力省完了）。

### 5.4 证伪条件（什么结果会推翻我们的 claim）

- ours 的 QPS@SLO **≤** b3 且 `replicated_bytes` 显著 >0 → 经济触发器
  在误触发，G_rep 判据或 λ̂ 估计有问题；
- ours 出现 `failed>0` 或 CONSISTENT 破 → 机制性 bug，先于结论修复；
- b3 与 ours 在所有指标上不可区分 → residency 控制在 closed-loop 场景
  无增量价值，claim 需降级为"机制等价，价值在开放到达流"（→ §8 下一步 4）。

### 5.5 分析计划与可信边界

按 (share, policy) 聚合 3 reps 均值：主看 QPS@SLO(2s) 与 P95，副看
`new_tok`（算力）、`transfer_bytes`/`replicated_bytes`（代价）、
`hit_rate`。当前结论的可信边界：closed-loop 到达（非泊松）、≤4 workers、
zmq 玩具 data plane、block=16 token、decode=4 步——矩阵结论在这个包络内
成立，外推到真实到达流需要 §8 的下一步 4。

## 6. 工程实现摘要

### 6.1 组件

| 组件 | 文件 | 职责 |
|---|---|---|
| 命名 | `blkchain.py` | content-addressed 块链，纯 Python 可单测 |
| 编解码 | `kvcodec_blk.py` | 块切/inject/extract；GDN checkpoint 挂链尾 tip 块 |
| 协议 | `msg.py` | zmq ROUTER/DEALER，JSON hdr + 二进制 payload |
| Worker | `worker.py` | 单模型副本；hello/status/fetch/deliver/**evict**/assign |
| Scheduler | `scheduler.py` | fast path（match/schedule/dispatch）+ slow path（placement controller，ours 专属） |
| 启动 | `run_cluster.py` | 显式发牌 GPU；`--no-share` 映射 b0 |
| 评测 | `matrix_step5.py` / `compare_policies.py` / `check_consistency.py` | 矩阵 runner / 对比表 / 一致性核验 |
| 单测 | `test_policy.py` / `test_controller.py` | 纯 Python（stub zmq），全绿 |

### 6.2 关键协议决策（踩坑后定型）

- `busy` 只归 scheduler 所有（status 天然过期，信它双重预订）。
- fetch ack 按 `(holder, names)` 配对；deliver 按 `(target, names)`；
  复制 ack 按回显 `repl` rid **优先**归属（同名并发不互吞）。
- 复制不占 busy、不跑 turn；demand-delivered 立即更新 scheduler 侧
  residency（乐观记账快于 trailing status）。
- eviction protection 按块统计**所有 resident tip 的完整 resume set**。

### 6.3 Step 0 的 flake 根因（值得记住的一类问题）

`balanced_low_0` 按邻居租户**当下剩余显存**动态分片 → 每次加载布局不同、
紧张时静默丢层到 CPU/meta → 间歇性 cat 跨设备崩溃。修复：`explicit_even`
确定性半半分（层数从 `model.safetensors.index.json` 数，config 没有
`num_hidden_layers`）+ 加载自检。判别清单：device_map 两 worker 同形
（43 条目）、`layout check OK`、resume turn 的 place_cache 全 accelerator。

### 6.4 Step 4 修过的 4 个 bug（均有回归测试）

复制 ack 被同名 demand fetch 吞掉；demand-delivered 不记账导致复制
刚送达的 segment；`_chain_names` 祖先序反；planner 缺 policy 门控。

## 7. 环境事实

远程 `mamingyu@u` / env `lut_py310` / 8×A800-80GB /
模型 `/home/u/downloads/models/Qwen3.6-35B-A3B`（40 层 hybrid）。
邻居常驻 ~20GB/卡。结果 `results/icn_proto/blkcluster_*.json`；
一致性 `python -m icn_proto.check_consistency <json>`。

## 8. 下一步

1. 跑完 30 格矩阵 → 按 §5.3 的预注册预测核对 → 出 RQ3 结论。
2. 预算模式：resident_bytes 主动扣减记账；session-affine 驱逐保护
   （pin 本 worker 下一 turn 要用的 tip，缓解人为预算下的 thrash）。
3. Data plane：zmq → LMCache/NIXL（05 §5 已留口）。
4. closed-loop → 泊松到达 + zipf popularity，把 λ̂/EWMA 的语义从轮内
   扩展到跨请求流——这是从"原型"到"论文实验"的最后一跳。
