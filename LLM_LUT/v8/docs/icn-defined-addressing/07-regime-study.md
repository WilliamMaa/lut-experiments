# 07 设计文档：Where Does Residency Control Become Economically Useful?

> 前置：06 的判决——RQ1/RQ2 成立，RQ3（proactive residency control 相对
> reactive global fetch 的增量）在请求级 serving 场景双场景证伪。本文档
> 回答下一个问题：**在什么 workload × 硬件 regime 下，主动摆放可复用
> inference state 才比"缺了再取"赚？** 三个实验全部预注册（预测+证伪条件
> 跑之前写下），是三个**正交的 regime 切片**，全部完成，最后合成 regime map。
>
> 日期：2026-09-23（v2，吸收评审修订：E1 迁移机会定义 / E2 host-DRAM
> backing tier 定位 / E3 成本交叉判据 / 决策树改为三切片全跑）。

## 1. 从阴性结论推出 regime 问题

06 的负结果不是"ICN 思路失败"，而是一个 regime 结论：

> 在"请求来了再查全局 KV、需要时 remote fetch"的 workload 里，B3 已经
> 强得惊人；proactive 复制没有足够的 amortization window，且在 HBM 紧张
> 时与已有 useful state 竞争，诱发 churn（复制→挤占预算→驱逐→零副本→
> re-derivation）。

两条让阴性变扎实的观测：

- **b3 随偏斜单调变好**（zipf 1.0→1.6：new_tok 11,337→9,618，hit
  0.859→0.880；ours 的 churn 成本不随偏斜缓解）——"热点不够热"被排除。
- **b3 的 evict 恒 0**——reactive fetch 只在需要时搬，天然把每卡常驻
  压在预算内；churn 是 proactive 复制独有的病。

关键背景：B3 不是 strawman。LMCache 已支持 P2P KV lookup + NIXL
transfer；vLLM APC 用 parent-hash block identity 做单 engine prefix
reuse。我们证伪的是**强 reactive 基线下的 proactive 增量**——阴性结论
的分量由此而来。

## 2. Regime 的形式化：amortization 不等式

现有 G_rep（05 §3）：复制触发于 `G_rep = λ̂·ΔC_future − C_copy − C_memory > 0`。
它在 06 的全部实验里都没有错——λ̂ 估计工作正常（复制确实打在真热点上）。
它缺失的是分母里的另外几项，这些项决定了什么 regime 下 G 能赢：

```
真实的主动摆放盈亏：

    λ̂ · H · P_reloc · ΔC_future   （复用视野 × 异地复用概率 × 每次关键路径节省）
  > C_copy                          （搬运成本）
  + R_scarce                        （稀缺租金：占用昂贵 tier 的机会成本）
  − R_tier                          （便宜 tier 把稀缺租金打下来）
  + L_churn                         （驱逐外部性：被挤掉 state 的 re-derivation 代价）
```

**v2 关键修订——`P_reloc`（异地复用概率）**：long-lived session ≠
residency opportunity。若 session 的所有 turn 都回落原 worker
（`A→W0→W0→W0…`），B3 天然全 local hit，ours 无事可做，H 再大也没用。
真正要扩大的量是 `H × P_reloc`——长寿命 state **且**因为负载/调度而
需要它出现在别的 worker。这不只是我们的推理：Continuum（multi-turn
agent 在 tool-call 暂停后的 KV TTL 决策）与 CacheScout（利用 agent
执行转移做 proactive prefetch/eviction）都以"暂停—重调度—异地复用"
为价值前提。

三个 regime 轴 + 一个隐式前提，各对应 06 的一个观测：

| 轴 | 符号 | 06 里的取值 | 翻转手段 | 实验 |
|---|---|---|---|---|
| **复用视野** | H × P_reloc | ~3 turns，migration≈0 | 长生命周期 agent + inter-turn 暂停重调度 + 负载驱动 spill | E1 |
| **摆放层成本结构** | R_scarce − R_tier | 单一昂贵 tier，驱逐=消失 | 真实 host-DRAM backing tier | E2 |
| **reactive 成本** | ΔC_future | 跨 worker fetch ~1.4s，但 b3 已够用 | 合成 remote/local 成本比 | E3 |
| （隐式）**稀缺度** | — | 无预算时复制无害 | 所有实验恒开 48MB 硬预算 | 全部 |

**核心 design implication（可证伪形式）**：

> Proactive replication into scarce per-GPU HBM is structurally unattractive
> when reactive remote reuse is cheap. **A cheaper backing tier may be
> necessary for proactive information-centric placement to pay off.**
> 失败的可能不是"主动"，而是"唯一可用的 residency tier 太稀缺"。

## 3. 证据等级声明

| 等级 | 含义 | 对应实验 |
|---|---|---|
| **A（真机实证）** | 真实 GPU、真实字节流、真实模型计算 | E1 长生命周期 agent |
| **B（真实 host-DRAM backing tier）** | DRAM residency 是真的（块真的住进 host 内存、真的召回、真字节账）；CXL/rack-shared memory 是其 architectural extension，由外部文献支撑（INFOCOM 2026 的 CXL hierarchical KV placement、TraCT 的 rack-scale CXL shared prefix cache），**不由本实验宣称** | E2 |
| **C（合成成本沙盒）** | 只调成本参数（速率档位），无真实字节语义；输出是成本假设下的分界曲线 | E3 拓扑成本 |

表述纪律：可以说"cheap tier matters" + "CXL can provide such a tier
（引文献）"；**不能说"we validated CXL"**。

## 4. 实验阶梯（三个正交切片，全部完成）

### E1 长生命周期 agent workload（A 级）

**科学问题（时间维度）**：更长的复用视野 × 真实的异地复用机会，是否足以
amortize proactive residency？

**动机**：reuse horizon 从 ~3 turns 拉到 30–50；inter-turn tool-call
暂停允许重调度；负载让 session 在 worker 间 spill。re-derivation 代价
随累积前缀长度显著增长（**实测 `C_recompute(L)`，不预设增长阶**——
hybrid 架构 + 真实 kernel 的 scaling 用数据说话，不承诺平方）。

**设计**：

| 项 | 取值 |
|---|---|
| session 形态 | 1 doc turn + 30–50 question turns，前缀渐进增长（每 turn +数十 token） |
| **迁移机制（v2 新增，实验成立的前提）** | inter-turn 思考暂停期间 session **不绑定 worker**：下一 turn 由调度器按当时负载重新指派（现有 choose() 天然支持）；负载 λ 越高，spill 越频繁 |
| 到达 | 泊松，λ 扫 {0.5, 1.0, 2.0} sessions/s |
| 热度 | zipf-16，s ∈ {1.0, 1.6} |
| 预算 | 48MB 硬预算（维持稀缺性） |
| 对照 | b3 vs ours，reps 3 |

**必须记录的新口径（v2 新增，没有它们 E1 白跑）**：

```
session_turn_migration_rate          # session 的 turn 换了 worker 的比例
remote_resume_opportunities          # 到达时目标前缀的最新 state 在别的 worker 的次数
remote_resume_served_local_due_to_replication
                                     # 其中因 ours 的复制而本地命中的次数（E1 的专属判决量）
rederivation_tokens                  # 降级重算的总 token 账（绝对值，随前缀变贵）
C_recompute(L)                       # 实测重算代价曲线（校准不等式）
state_lifetime_hist                  # publish → last-access 间隔分布
```

**预注册预测**：
1. λ=2.0（重载）下 migration_rate 显著 >0，remote_resume_opportunities
   与 λ 单调——E1 的实验条件本身成立（若 opportunities≈0，E1 无判决力，
   先修负载再谈结论）。
2. ours 的 `remote_resume_served_local_due_to_replication` > 0 且随 λ 上升；
   判决量：**new_tok 差（ours−b3）是否随 λ/horizon 收窄乃至翻正**。
   churn 的绝对成本随前缀变贵线性增长，能否被 amortization 盖过是 E1
   的判决量。
3. 若 E1 翻正，翻正必须主要由 replication-served-local 解释（而非 b3
   退化）——互锁防作弊。

**证伪条件**：三 λ × 两 s 全谱 ours 的 new_tok ≥ b3 × 1.5，**且**
remote_resume_served_local_due_to_replication 不占 remote opportunities
的多数 → "horizon ≤ 50 turns × P_reloc ≤ 实测值 内无正窗口"成立，进入 E2。

**实现面**：workload 生成器参数化（`--turns-per-session 40`、问题增量
`--q-tokens`）；**session-worker 亲和解除**——closed-loop 下 choose()
已对每个 turn 独立选 worker，无需改调度；但要确认 turn 完成→下一 turn
重新进 ready 时不携带亲和（现有 advance 不携带，验证即可）。口径补
上述统计。块链/协议/控制器零改动。

### E2 host-DRAM backing tier（B 级，与负结果机制贴得最近）

**科学问题（层级维度）**：便宜的 backing tier 能否消灭让 proactive
residency 有害的驱逐外部性？

**动机**：现在驱逐 = 块消失（zero-replica → re-derivation），这是 churn
反噬的**放大器**。改成驱逐到 per-worker host DRAM：不占 HBM 预算、召回
远便宜于 recompute。proactive controller 可能依然复制错，但**复制错误
不再意味着信息彻底消失**——整个经济学改变。定位修订：这是**真实
host-DRAM backing tier 实验**（LMCache 官方 long-doc benchmark 同样
在 GPU working set 溢出后用 CPU RAM 保存 KV，mean TTFT 757→185ms——
我们做的是同一层级的受控对照）；CXL/rack-shared memory 是它的
architectural extension（引 INFOCOM 2026 / TraCT），不由本实验宣称。

**设计**：

```text
驱逐路径改造：
  现在:  HBM evict → 块消失 → 下次要用 → 全网没有 → 整段重算
  E2:    HBM evict → host DRAM spill（真字节 residency）→ 下次要用 → spill 召回
         （spill 也有容量上限，v2.2 档同样按 λ̂ 判据清，测试 tier 稀缺边界）
```

| 项 | 取值 |
|---|---|
| spill 容量 | ∞（v2.1）与 2× 预算（v2.2，spill 也稀缺） |
| spill 召回 | 同 worker host 内存，计字节、速率 ≫ 跨 worker fetch |
| 对照 | b3 vs ours × {无 spill, spill∞, spill 受限}，预算恒 48MB |
| 判决量 | ours−b3 的 new_tok 差在 spill 开启后的变化；evict 后命中率 |

**预注册预测**：
1. spill 对 b3 也有益但小（b3 驱逐近零）——它是 **ours 的专属修复**
   才算数（ours 的改善 ≫ b3 的改善）。
2. design implication 成立的情形：spill 开启后 ours 的 rederivation
   趋零，new_tok 差收窄至 ≤1.2×，repl 上升（复制不再怕驱逐）→
   "便宜 tier 存在时 proactive 第一次呈现正经济学"。
3. spill 受限（v2.2）时中等偏斜出现第二级 churn——给出 tier 容量的
   regime 边界。

**证伪条件**：spill 开启后 ours 仍 ≥1.5× → design implication 在该
层级不成立，06 结论适用范围比预期更广（同样是有用信息）。

**实现面**：worker.py 加 spill 存储（name→bytes）+ spill-recall 协议
（复用 fetch 消息加 tier 标记）；scheduler 驱逐 action 写目标 tier；
召回记账进 records。预算账不变（spill 不进 HBM 账）。

### E3 remote/local 成本交叉（C 级，沙盒）

**科学问题（拓扑维度）**：reactive remote access 贵到什么程度，proactive
placement 才开始经济合理？

**v2 关键修订——判据**：E3 操作的是 `C_fetch`，**new_tok 看不见它**
（一次 fetch 无论假设 0.1s 还是 10s，prefill 账都是零）。主判据改为
**建模关键路径/总 serving 成本的交叉曲线**：

```
x 轴: remote/local access cost ratio r（跨 worker fetch 相对本地 resume
      的代价倍数，合成档位）
y 轴: modeled total cost  C_total = C_queue + C_fetch + C_prefill + C_place
      （队列 + 传输 + 重算 + 摆放，全部由真实记录的 events 回放计算，
       非现采现编）
曲线: b3(r) 与 ours(r) 的交叉点 r* 即 "fetch 贵到多少倍 proactive 开始赚"
```

new_tok 退居副指标（互锁：不许用 recompute 暴涨换传输下降）。

**设计**：fetch 两档成本（同 rack 快 / 跨 rack 慢，参数化），扫
r ∈ {1, 2, 4, 8, 16} × {b3, ours} × zipf {1.0, 1.6}。每格跑出
C_total(r) 曲线。无真实字节语义，**输出是成本假设下的分界曲线**。

**预注册预测**：存在临界 r*，ours−b3 的 C_total 差随 r 单调收窄并翻正；
r* 位置给出可引用结论"proactive 有值 ⇔ remote access 比 local 贵 ≥
r* 倍"，且 r* 随偏斜下降（热点越热临界越低）。

**证伪条件**：全 r 档位 C_total 不翻正 → proactive 的失败与 reactive
成本无关，06 结论升级为"与 fetch 成本无关的结构性阴性"。

**实现面**：run 记录的 events（assign/fetch/deliver/result 时间戳与字节）
导出，成本模型离线回放（`icn_proto.cost_replay` 模块——按给定 r 重算
C_total，同一 trace 扫全 r 档位，零额外真机时间）。

## 5. 里程碑与产出：regime map（v2 决策树修订）

v1 的顺序决策树（E1 否→E2→E3）改为：**三个正交切片全部完成**，因为
它们扫的是三根不同的轴，任何一根提前停都会让 regime map 缺面。最终
产出是三张二维 slice 合成的判定图：

```
E1: horizon × load        → 时间轴上的正/负域
E2: backing capacity × memory pressure
                          → 层级轴上的正/负域（含 tier 稀缺边界）
E3: remote/local cost ratio × skew
                          → 拓扑轴上的分界曲线 r*(skew)

合成: (H, R_tier, C_remote) → {reactive wins, proactive wins}
```

每条路径都有明确产出，不存在"全阴就白干"：

- E1 翻正 → "正价值域 ⊇ 长生命周期 × 高负载 spill"，regime 图第一面。
- E2 翻正 → "便宜 backing tier 是 proactive 的前提"，对接 CXL 文献。
- E3 出 r* → 可引用的经济分界，无论 proactive 是否翻正。
- 全部不翻正 → 06 结论升级为"与 horizon/tier/fetch 成本无关的结构性
  阴性"，负结果的边界条件完整，论文以负机制为主贡献。

每步沿用现有纪律：预注册预测、多指标互锁（A/B 级 new_tok 主、latency
仅参考——本机邻居噪声；C 级 C_total 主、new_tok 副）、CONSISTENT
不变量、manifest wl 签名、单测先行。

## 6. 不做的事（边界）

- **不做真实多机/机架拓扑**：拓扑轴只到 C 级合成；E3 的 r* 是成本模型
  结论，不外推为硬件实证。
- **不宣称 CXL 实证**：E2 是真实 host-DRAM 实验；CXL/rack-shared 由
  INFOCOM 2026、TraCT 等外部文献支撑定位。
- **不做数值量化/稀疏化**：与 LUT 主线无关，不引入。
- **不回头调 G_rep 参数**：λ̂ 估计已被 06 排除为病根；E1–E3 不改控制器
  判据本身（E2 只改驱逐落点，E3 只改成本回放）。
- **E1 不预设重算代价增长阶**：`C_recompute(L)` 实测校准，不承诺平方。

## 7. 相关工作锚点（预登记，写作时用）

- **LMCache**：P2P KV sharing（NIXL）、long-doc benchmark 的 CPU-RAM
  backing（757→185ms mean TTFT）、bench CLI 单列 multi-round-chat——
  B3 的能力锚 + E2 的层级先例 + E1 的 workload 现实性，三处都要引。
- **vLLM APC**：parent-hash block identity、单 engine prefix reuse——
  RQ1 增量的对照面（我们多跨 worker 可恢复 + hybrid recurrent state）。
- **Continuum / CacheScout**（2026 agent-serving）：tool-call 暂停、
  KV TTL、执行转移驱动的 proactive prefetch——E1 的 `P_reloc` 价值
  前提已由他人独立提出，我们必须做差异（content-addressed 块粒度的
  经济摆放 vs 整 session 粒度）。
- **CXL KV 工作**（INFOCOM 2026 hierarchical KV placement、TraCT
  rack-scale shared prefix cache）：E2 的 architectural extension 锚点。
