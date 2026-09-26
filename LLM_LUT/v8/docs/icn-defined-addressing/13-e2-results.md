# 13 — E2 结果记录

> 项目总纲（我们在复现 ICN 的哪些机制）：`14-icn-mechanism-matrix.md`。
> E2 = 矩阵第 4 行"分层驻留"的实验。

## 0. 一页状态（只看这里就够，更新于 2026-09-27）

### E2 要回答什么

E1 的结论：只给每块 GPU 自己的 HBM 当驻留层时，主动复制（ours）
干不过按需取块（b3），亏在"复制→挤占 HBM→驱逐=消失→重算"这条链。
E2 验证：给被驱逐的块加一层便宜驻留层（主机 DRAM），这条链断不断。

### 矩阵进度（2026-09-27 修复后重跑中）

| 格 | 状态 |
|---|---|
| sp-1 × s=1.0 / s=1.6 | ✅（retries=0，干净，保留） |
| sp96 × s=1.0 | ✅ 已补跑（b3 新数 new_tok 5.99 万 / hit 0.939；ours 保留旧干净格） |
| sp96 × s=1.6 | 🔄 补跑中：b3 新格 SLO **1.000**（崩塌消失）；ours 1 旧 1 新，还缺一次干净跑；b3 rep1 又一次 failed=1，错误待查 |
| b3@s0 × s=1.0 / s=1.6 | 🔄 s=1.0 rep0 ✅；rep1 两次坏格已清；s=1.6 未跑 |

### 已有的数说明了什么（机制语言）

- **tier 无限（sp-1）时 ours 和 b3 完全重合**（s=1.0 差 0.2%、s=1.6 差
  0.2%，hit 双双相等，E1 时代 ours 差 20.8× 的崩塌彻底消失）：
  机制含义——**驱逐落点不再是虚空后，驱逐外部性对两个策略都被
  吸收，proactive 复制失去了要修的问题，退化为 b3**（repl 仅
  10–20 次且不影响结果）。
- **tier 有限（sp96）时回潮真实出现**：s=1.0 档 new_tok ~6.0–7.1 万
  （sp-1 档的 3 倍）——驱逐落点从虚空变 DRAM 后，容量瓶颈转移到
  tier 层，这是真实的 regime 边界。
- **"延迟崩塌"初步判定为事故二的症状而非独立机制**：sp96×s1.6
  b3 干净重跑后 SLO 0.613 → **1.000**、p50 1.6s → 0.97s。clobber
  造成的重试→重排队堆积足以解释崩塌的大部分。ours 同档还差一次
  干净跑才能定稿。
- ours 在 sp96 档略差于 b3：复制的搬运没有换回重算节省。

### 接下来该干什么

1. 跑 runbook §7a 看新 BAD 格（`..._225708`，sp96×s1.6 b3）的错误，
   贴回来；然后 `--drop-bad` 重跑 runbook §5 第 4 条补完这格；
2. §5a 两条 b3@s0（若还没跑）；
3. 全部补完后重读数：重点确认 sp96×s1.6 ours 干净格 SLO 是否也回
   1.0（回到 1.0 = 崩塌完全是 bug 症状；回不到 = 有真实残余，
   再 trace 重放定位）；
4. 判 P1/P2/P3（口径在 10 §4），结论我来写。

---

## 1. 聚合数据（2026-09-27 矩阵输出原文，8 格）

```text
wl                     policy  runs     rps    hit  new_tok   xfer  repl  evict   recall  favoid     p50     p95  SLO@2.0s   wall
sp-1 × s=1.6             b3      2   1.9861  0.979    21690  126.5   0.0  347.5  25636.0    80.0   0.972   1.138     1.000   170.0
sp-1 × s=1.6            ours     2   2.0480  0.979    21650  123.5  20.0  302.0  25897.0    78.0   0.964   1.117     1.000   164.8
sp96 × s=1.6             b3      2   1.4124  0.959    46993  174.5   0.0  213.5  10134.5    35.5   1.626   3.210     0.613   237.5
sp96 × s=1.6            ours     2   1.1584  0.942    52832  184.5  26.0  180.5   8317.5    44.5   2.387   3.510     0.233   288.4
sp-1 × s=1.0             b3      2   1.9816  0.976    22233  150.5   0.0  356.0  28525.5    50.0   0.957   1.095     1.000   170.2
sp-1 × s=1.0            ours     2   2.0371  0.976    22188  133.5  10.0  307.5  25323.0    49.5   0.979   1.116     1.000   165.7
sp96 × s=1.0             b3      2   2.0029  0.921    67662  162.0   0.0  231.0  10094.5    13.0   0.975   1.289     1.000   168.4
sp96 × s=1.0            ours     2   1.8103  0.927    71280  161.0  29.0  224.5   9286.5    14.0   0.978   2.175     0.959   186.6
```

（列含义：hit=复用率；new_tok=每格重算+新算的 token 总量；xfer=跨
worker 取块次数；repl=主动复制次数；evict=驱逐次数；recall=从
DRAM 层召回块数；favoid=召回替代掉的跨 worker 取块次数。）

## 2. 逐格文件索引

trace 目录：`results/icn_proto/traces/cell_<wl>_s2_<b3|ours>_r<rep>/`
（每格 sched.jsonl + w*.jsonl）。聚合 JSON 与 manifest 行一一对应
（`results/icn_proto/matrix_e2.json`）。

## 3. 缺口（事实清单）

- b3@s0 两条未跑（runbook §5a，跑完落点三档对照齐全）。
  **更新（2026-09-27）**：s=1.0 腿 rep0 ✅（hit 0.415 / new_tok 41 万，
  与 E1 时代 b3 基线一致，虚空落点对照成立）；rep1 两次跑均
  `STALE_RESUME` 硬失败（1 次 → 2 次）。注意：报错带**完整缺失块
  名单**（修复后的新路径），且进了 failed 记录 = **每 turn 一次的重试
  已经跑过、第二次仍旧缺块**。只剩 s=1.6 一条未跑。
- **s0 腿 STALE_RESUME 根因：已证实并已修复**（trace 重放 + 修复
  详情见 `12-e2-diag.md` 事故二）。一句话：worker 的 status 是
  发送时现取的全量快照，**在 worker 执行 scheduler 的驱逐之前生成
  的晚到快照**会把乐观驱逐掉的块复活回视图（盲替换不认快照时间），
  下一轮派工据此选了 local 模式 → worker 已真删 → stale。修复 =
  快照带时间戳 + scheduler 减去比快照新的本端驱逐。同一 bug 全
  档位存在（sp96 格 stale_purge 10），sp96 被 recall 兜住、
  sp-1 不驱逐，只有 s0 暴露。
- **已确认的事实（2026-09-27）**：
  1. 失败格 `..._211638` 的 `stale_resume_retries = 5`、`failed = 2`
     —— 重试路径确实工作；5 次竞态里 2 个 turn 第二次又撞上 →
     硬失败。竞态率 ~5 次/格，不是零星。
  2. sp96 × s1.6 ours r0 格的 trace `--summary` 显示 `stale_purge 10`
     且该格 `failed = 0` —— **同一竞态在 sp96 格同样发生**，只是
     重试+召回全部吸收（evict 数 ~2 万/格、每 worker spill_drop
     1.4–3.8 千，也印证了 tier 层自身在二级 churn）。
  3. 定位过程留档：
- **定位指令**（s0 两格都带 trace）：
  1. 先确认重试计数：`python -m icn_proto.e1_report results/icn_proto/blkcluster_s8t40_20260926_211638.json`，
     看 `spill` 里的 `stale_resume_retries`（应 >0，证明重试路径
     工作）；
  2. 重放失败块一生（完整命令，直接复制）：
     `python -m icn_proto.trace_replay results/icn_proto/traces/cell_pp2_z16s1_t2_tps40_q40_sd0_s2_b3_r1 "parent/4add2dcf28508bf7"`
     —— 收窄到失败所在的文档根链，避免 `span/0-16` 匹配到其它
     文档的同名 span。注意第二参数里的斜杠不可省。要看的是：
     该块 demand 之后、resume 之前有没有同 worker 的 evict 事件。
- sp96 × s=1.6 两格 SLO 崩塌（0.613 / 0.233），链条未重放定位。
- sp96 × s=1.6 ours rep0 格 `rc=2`（run_cluster 看门狗：某 worker
  进程异常死亡退出码）；JSON 正常、数值与 rep1 一致。死因待查，
  日志：`results/icn_proto/cell_logs/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_ours_r0_b48_p0.log`。
- `repl_served_local` 是否仍为 0：从 ours 格 JSON 的
  `remote_resume_served_local_due_to_replication` 读。

## 4. 判决口径核对（10 §4，只对已出的格）

- **P1 公平性**：sp-1 两档 b3/ours 重合（new_tok 差 0.2%、hit 相等）
  → 同预算同 tier 同构 ✅
- **P2 核心**（tier 无限时主动复制变得有角色）：
  ours ≈ b3 + 微量 repl → **tier 吸收了外部性，但没有给 proactive
  创造出增量角色** ❌（判读完成，等 b3@s0 补齐后定稿）
- **P3 边界**：sp96 回潮（new_tok 3×）+ s=1.6 延迟崩塌 ✅ 边界有数
