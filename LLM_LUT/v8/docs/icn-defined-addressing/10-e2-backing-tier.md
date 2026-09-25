# 10 设计文档：E2 — host-DRAM backing tier（驱逐外部性的消除实验）

> 前置：`07-regime-study.md` §4 E2（预注册框架）；`09-e1-results.md`
> §8（动机数据）。本文档是 E2 的**设计**，预注册预测与证伪条件在
> 开发前写下。运行手册落 `11-e2-runbook.md`。
>
> 日期：2026-09-24（v1，待评审）

## 1. 科学问题（层级维度）

> **便宜的 backing tier 能否消灭让 proactive residency 有害的驱逐外部性？**

E1 已经证明：在"唯一可用的 residency tier 是稀缺 HBM"的系统里，
proactive 摆放结构性失败（抢跑 / 需求摊薄 / churn，09 §4）。E2 把
层级轴翻过来：**驱逐不再等于消失**。

## 2. 动机数据（全部来自 E1 实测）

- **饥饿 regime（48MB）**：912MB 唯一 state vs 4×48MB 常驻，驱逐
  摧毁 holder 与需求信号 → 96% 重算（09 §3.2）。驱逐 = 信息彻底
  消失。
- **重复搬运放大（fetch-fest regime）**：5.4GB / 1.85GB 传输 vs
  912MB 唯一 state ≈ 2–6× 重复 fetch（驱逐→再从 peer 抓）。跨
  worker fetch（~10–100MB/s）远贵于 host DRAM 召回（~20GB/s PCIe）。
- **churn（健康 regime）**：ours repl 76–85 → evict 218–362 →
  new_tok 劣化 4.6–7.2×。复制的错误代价被"驱逐即消失"放大。
- b3 在全部 E1 格中 `evict=0`——**spill 天然是 ours 的专属对照修复**，
  b3 是完美的控制组（它不受 spill 影响，预测见 §4）。

## 3. 设计

### 3.1 驱逐路径改造

```text
现在:   HBM evict → 块消失 → 下次要用 → 全网没有 → 整段重算
E2:     HBM evict → host DRAM spill（真字节 residency）→ 下次要用
        → 本地召回（spill-recall，host→device）
```

- **spill 存储在 worker 进程内**（`dict: block_name → 序列化字节`），
  用的是 host RAM——这是**真实的 DRAM residency**：块真的住进 DRAM、
  真的被召回、字节账全部真实记录。
- 预算账不变：spill 不进 HBM 预算（48MB 硬预算照旧，稀缺性保留）。
- **定位纪律（B 级证据）**：本实验是真实 host-DRAM backing tier
  实验（LMCache long-doc benchmark 的 CPU-RAM backing 同层级先例，
  mean TTFT 757→185ms）。CXL / rack-shared memory 是它的
  architectural extension（INFOCOM 2026 hierarchical KV placement、
  TraCT rack-scale shared prefix cache），**只引文献，不由本实验
  宣称**。

### 3.2 协议与调度集成

```text
worker.py:
  self.spill: dict name→bytes;  self.spill_bytes
  evict 处理:  容量允许 → 序列化进 spill（kvcodec 同一套字节流），
               否则真正丢弃（v2.2 受限档：spill 满后按 LRU 丢弃最冷）
  assign 处理: resume_names 中在 spill 的 → 先召回进 GPU cache
               （recall，计 recall_bytes），再走正常 prefill
  fetch 处理:  被请求块在 spill → 先召回再发送（holder 可以是
               "spill 持有者"）
  status:      携带 spilled 名单 + spill_bytes（名单 ~数百条 × 60B，
               每 turn 一次，量级可忽略）

scheduler.py:
  WorkerState: +spilled(set), +spill_bytes
  match_local/choose(): resume 可行性 = resident ∪ spilled；
               成本模型加召回项 spilled_bytes / spill_rate
               （spill_rate ≫ xfer_rate ⇒ 本地召回归因天然优先于
               跨 worker fetch）
  _plan_repl 的 holder 判定: resident ∪ spilled（spill 持有者也是
               合法复制源，worker 侧先召回再发送）
  _plan_evict: 不变（策略无关的共享基底）；经济变化自然涌现——
               被驱逐的 state 仍在系统里，no_holder 饿死与 churn
               放大器同时被拆除
```

关键性质：**协议零新消息**。evict/assign/fetch/status 全部复用，
只是语义扩展（worker 内部多一层 spill 存储）。调度器只在成本模型
和 holder 判定上感知 tier。

### 3.3 实验矩阵（预登记）

| 档 | spill 容量 | 含义 |
|---|---|---|
| s0 | 0（关） | **E1 基线直接复用**（16 格已在 matrix_e1.json），不重跑 |
| s1 | ∞ | 驱逐永不消失——backing tier 的纯效果 |
| s2 | 96MB（2× HBM 预算） | tier 也稀缺——第二级 churn 边界 |

- 固定：λ=2.0（E1 里最易 churn 的负载档），s ∈ {1.0, 1.6}，
  policies {b3, ours}，reps 2 → **2 档 × 2 偏斜 × 2 策略 × 2 reps
  = 16 新格**（~3.5 小时）。
- 判决量：`new_tok`（主）、`rederivation_tokens`、`evict`、
  `repl`、`repl_served_local`、spill 新口径（`spill_evicted_bytes` /
  `recall_count` / `recall_bytes` / `fetch_avoided_by_recall`）。
- manifest：`results/icn_proto/matrix_e2.json`（wl 签名加 spill 档）。

## 4. 预注册预测与证伪条件

**P1（对照纯净性）**：spill 对 b3 几乎无影响（E1 实测 b3 evict 恒 0）。
若 b3 在 spill 档显著变化，说明实现有泄漏，先修再判。

**P2（design implication 成立的情形）**：spill∞ 下 ours 出现
① `rederivation_tokens` 趋零；② new_tok 差收窄至 ≤1.2×；
③ `repl` 上升（复制不再怕驱逐）且 `repl_served_local` > 0
（E1 恒 0 的量首次变正，即 proactive 第一次有正经济学）。
三条同时成立 ⇒ "便宜 backing tier 是 proactive residency 的前提"
在真实 DRAM 层级上成立，对接 CXL 文献。

**P3（tier 稀缺边界）**：spill 96MB 档，s=1.6 出现第二级 churn
（spill 也满 → 丢弃 → 重算回潮），给出 tier 容量的 regime 边界。

**证伪条件**：spill∞ 下 ours 仍 ≥ b3 × 1.5 ⇒ design implication
在该层级不成立，06+E1 结论适用范围比预期更广（同样是有用信息，
写作时作为"backing tier 也不救 proactive"的强阴性）。

**互锁**：不许用 fetch 流量暴涨换 new_tok（`transfers` 与
`transfer_bytes` 同报）；latency 类照旧只作参考（邻居噪声）。

## 5. 实现计划（文件级）

| 文件 | 改动 |
|---|---|
| `worker.py` | spill 存储 + evict 落点改造 + assign/fetch 召回 + status 携带 spilled（全部 worker 内部，协议字段不变） |
| `scheduler.py` | `--spill-mb`（0 关 / -1 无限）、`--spill-rate`（成本模型召回速率，默认 20e9 B/s）；WorkerState spilled/spill_bytes；match_local/choose 的 resident∪spilled 可行性与召回成本项；_plan_repl holder 判定含 spilled；spill 口径统计与 summary |
| `test_e2.py`（新） | spill 感知的 match/choose（stub state）、召回成本排序（本地召回优先于跨 worker fetch）、_plan_repl 接受 spill 持有者、预算账不进 spill、LRU 丢弃（受限档）、FakeSock 协议测试 |
| `matrix_step5.py` | `--spill-mb` 透传 + wl 签名加 spill 档 |
| `11-e2-runbook.md`（新） | 同步/验证/冒烟/矩阵命令（显式逐条） |

worker 侧 torch 逻辑本地不可测，全部走远程冒烟验证；scheduler 侧
纯逻辑单测先行（沿用 stub zmq 模式）。

**冒烟通过标准**（铺矩阵前）：`failed=0`；spill 档 evict 后
`spill_bytes` > 0 且 `recall_count` > 0；b3 档行为与 E1 基线一致
（P1 的冒烟面）。

## 6. 不做的事（边界）

- **不宣称 CXL 实证**：本实验只到真实 host DRAM；CXL 定位由文献承担。
- **不做 spill 的跨 worker 共享**：spill 是 worker 私有的（CXL 共享
  tier 是延伸，不在此列）。
- **不改 G_rep 判据**：E2 只改驱逐落点与成本模型，控制器经济学不变
  ——这正是"层级变化是否足够"的干净测试。
- **不重跑 E1 基线**：s0 档直接引用 matrix_e1.json。
- **不引入真实 CXL 硬件/多机拓扑**：层级轴到 host DRAM 为止。

## 7. 里程碑

1. 本文档评审通过 → 实现（§5）→ 本地单测全绿。
2. 远程冒烟（ours + b3 各一格 spill∞）→ P1 检查 → 16 格矩阵。
3. 判决对照 §4 → 写 `12-e2-results.md`，regime map 补上层级轴
   （E1 时间轴 + E2 层级轴；E3 拓扑轴成本回放可并行或随后）。
