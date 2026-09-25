# 09 E1 结果：长生命周期 agent workload 下的主动摆放判决

> 设计文档：`07-regime-study.md` §4 E1（预注册）。运行手册：
> `08-e1-runbook.md`。本文档是 E1 的**结果与判决**记录。
>
> 日期：2026-09-24（E1 矩阵 16 格全绿后定稿）

## 1. 判决（TL;DR）

**E1 预注册证伪条件成立：长复用视野 × 真实异地复用机会，没有让
proactive residency 在请求级 serving 里翻盘。四格矩阵 ours 全面
显著劣于 b3（4.6×–20.8×），负载轴与偏斜轴均无翻案。**

机制不是"控制器没调好"，而是三层结构性原因（§4）：
① reactive fetch 抢跑了 proactive 的可行集；② 需求信号 per-target
摊薄，低于复制盈亏；③ 稀缺 HBM 下 proactive 一开火就诱发 churn，
比不放更糟。

## 2. 实验设置（一句话版）

Qwen3.6-35B-A3B hybrid，4 worker × 2×A800；8 sessions × 40 turns
（doc + 40×40tok 问题，前缀渐增至 ~3200 token）；泊松到达
λ ∈ {1.0, 2.0}，zipf-16 热度 s ∈ {1.0, 1.6}；48MB/卡硬预算；
b3（reactive fetch）vs ours（+G_rep 主动复制控制器），2 reps。
对照 smoke：2 worker 同 workload ×5 + 256MB 预算探测 ×1。

数据：`results/icn_proto/matrix_e1.json`（16 格 manifest）+
 `blkcluster_s8t40_2026092{3,4}_*.json`（冒烟 6 次）。

## 3. 主结果

### 3.1 确认矩阵（48MB，new_tok = 重算 token 账，越低越好）

| λ | s | b3 | ours | ours/b3 | ours repl | ours evict | b3 hit | ours hit |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 20,970 | 107,504 | 5.1× | 84.0 | 288.0 | 0.979 | 0.841 |
| 1.0 | 1.6 | 20,054 | 145,019 | 7.2× | 76.0 | 362.5 | 0.982 | 0.752 |
| 2.0 | 1.0 | 22,188 | 103,141 | 4.6× | 84.5 | 218.5 | 0.976 | 0.861 |
| 2.0 | 1.6 | 20,054 | 416,313 | 20.8× | 7.5 | 482.5 | 0.982 | 0.439 |

b3 的 new_tok ≈ 20–22K（≈ 全部真实新增 token，几乎零重算、零驱逐）；
ours 在复制开火（repl 76–85）时把 new_tok 推到 b3 的 4.6–7.2 倍，
在 s=1.6/λ=2 落进饥饿 regime 时崩到 20.8 倍。

### 3.2 冒烟的 regime 谱系（2 worker，ours）

| regime | 预算 | transfers | hot | no_missing | g_nonpos | repl | 现象 |
|---|---|---|---|---|---|---|---|
| fetch-fest | 48 | 158（5.4GB） | 52,506 | 58,019 | 46,993 | 0 | b3 抢跑一切 |
| 饥饿 | 48 | 7 | 2,765 | 5,530 | 0 | 0 | 96% 重算，栈饿死 |
| 饥饿(矩阵4w) | 48 | 30 | — | — | — | 7.5 | evict 482，hit 0.44 |
| 预算放宽 | 256 | 136（1.85GB） | 32,856 | 50,382 | 14,573（g_best −0.0112） | 1 | 复制开火 1 次，served 0 |

跨全部 6 次冒烟 + 16 格矩阵：`remote_resume_served_local_due_to
_replication` **恒为 0**（累计机会 400+）。

## 4. 机制链（为什么翻不了盘）

1. **抢跑（pre-emption）**：turn 落冷 worker → b3 立即 demand fetch
   （worker 转 busy）→ 控制器下一拍只看到 `no_missing`（5–5.8 万次，
   直方图最大门）。reactive 的控制环延迟毫秒级，proactive 的经济门
   永远慢半拍——**demand fetch 的每次临场搬运本身就是摆放**。
2. **需求摊薄**：λ̂ 纯观测不预测（红线），session 私有 tip 的
   per-target 需求只在第一次命中后才存在，而第一次命中已被 reactive
   服务；近似均匀路由下 loc 需求 ~0.1/s，低于校准后的复制盈亏
   （`g_nonpos` 主导剩余缺口；256MB 格 g_best=−0.0112 说明存在
   差之毫厘的 near-miss，但主导项仍是 no_missing）。
3. **饥饿（48MB 特有）**：912MB 唯一 state vs 4×48MB 常驻，驱逐
   摧毁 holder 与需求信号（tip 要 resume 到才计数，resume 不到
   永远冷门）——b3 fetch 也只有 7 次，全栈 96% 重算。
4. **churn（开火即反噬）**：预算内 replication 挤占有用 state →
   驱逐 → 零副本 → 重算。矩阵里 repl 76–85 次对应 evict 218–362
   次、new_tok 劣化 4.6–7.2×——06 的机制在长 horizon、4 worker
   下原样复现，且偏斜越大越糟（热点 tip 更多 → 复制更多 →
   预算更爆）。

## 5. 预注册对照（07 v2 §4 E1）

| 预注册预测 | 结果 |
|---|---|
| 1. λ=2.0 下 migration_rate>0、opportunities 随 λ 单调 | **成立**（0.33–0.52，机会 104–166）——实验条件有效 |
| 2. ours 的 repl_served>0 且随 λ 上升 | **证伪**（恒 0） |
| 2 判决量：new_tok 差随 λ/horizon 收窄/翻正 | **证伪**（4.6–20.8× 劣化，无收窄趋势） |
| 3. 翻正必须由 repl-served 解释 | 不适用（未翻正） |
| 证伪条件：全谱 ours ≥ b3×1.5 且 repl_served 不占多数 | **成立**（最劣化格远超 1.5×，served 恒 0） |

## 6. 实验过程中的两个合法修复（均有测试与数据支撑）

1. **复制粒度**：全链 → 缺失后缀（与 demand fetch 对齐）。证据：
   修复前 40-turn 全链 ~100MB 超预算，`no_holder` 4,376；修复后
   `no_holder=0`。3-turn 行为不变，06 结论不受影响。
2. **ΔC_future 校准**（07 §4 预注册项）：占位近似（= 搬运时间，
   break-even 1 hit/s）→ 实测重算代价 `missing_tokens/prefill_rate`。
   证据：48MB 饥饿格 `g_nonpos=0`（经济门全过）；256MB 格复制首次
   开火。`c_recompute` 实测曲线同时归档（小 prefill 固定开销极大：
   ≤64tok 桶仅 80–90 t/s）。

两个修复都没有让 proactive 翻盘——它们排除的是"控制器坏了"的
备择解释，让阴性归因于 regime 结构。

## 7. Caveats

- latency 类指标受邻居负载污染，全系列只作参考；判决只看
  计数类（new_tok / xfer / repl / evict / hit / served）。
- 256MB 探测格 prefill_rate EWMA 被拖到 197.8 t/s（注入变慢+
  邻居），不影响计数指标。
- b3 的 new_tok 四格近似常数（≈ 真实新增 token），因 evict 恒 0
  且 fetch 总能命中——这是 b3 的上限表现，对比基准偏强，阴性
  结论在此强基准下取得。
- zipf 模式下 `--doc-share` 无效，`shares=2` 仅作占位。

## 8. 对 E2 / E3 的导出

- **E2（host-DRAM backing tier）动机增强**：① 饥饿 regime 里
  驱逐 = 信息消失（no_holder/重算 96%）；② fetch-fest regime 里
  5.4GB/1.85GB 传输对 912MB 唯一 state 有 ~2–6× 重复搬运放大
  （驱逐→再从 peer 抓）。便宜 tier 同时打这两个点：复制错了不消失、
  驱逐了召回避重算/重抓。
- **E3（成本交叉）判据不变**：E1 阴性是在"跨 worker fetch 相对
  便宜且总在关键路径上可接受"的假设下取得的；E3 的 r* 扫描将给出
  fetch 贵多少倍时结论翻转。
- E1 阴性结论的边界：请求级 serving、逐 turn 随机重派发、无
  placement-demand 预测。带粘性的 session 路由 / 可预测放置需求
  的场景不在本次证伪范围内（写作时作为 future work 表述）。
