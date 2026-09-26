# 13 — E2 结果记录

数据快照：2026-09-26。判决口径：`10-e2-backing-tier.md` §4 预注册。

**记号对照**：

| 记号 | 含义 |
|---|---|
| b3 | 对照策略：worker 之间按需互相取块（reactive） |
| ours | 实验策略：在 b3 基础上加主动复制（proactive residency） |
| s0 / sp-1 / sp96 | spill 档：s0=关闭，sp-1=无限容量，sp96=容量 96MB（超出按最久未用丢弃） |
| s=1.0 / s=1.6 | zipf 偏斜参数：1.6 时热点文档更集中 |
| λ=2.0 | 泊松到达：平均每秒 2 个 turn |

## 1. 进度

| 格 | 状态 |
|---|---|
| sp-1 × s=1.0 | ✅ 完成（b3 只有 1 rep，缺 r1） |
| sp-1 × s=1.6 | ⬜ 未跑 |
| sp96 × s=1.0 | ⬜ 未跑 |
| sp96 × s=1.6 | ✅ 完成（2026-09-26） |
| b3@s0 两条 | ⬜ 未跑 |

## 2. 聚合数据（2026-09-26 矩阵输出原文）

```text
wl                  budget   price  share policy  runs     rps    hit  new_tok  xfer  repl  evict recall favoid     p50     p95 SLO@2.0s    wall
pp2_z16s1.6_t2_tps40_q40_sp96_sd0      48       0      2 b3         2  1.9771  0.938    61799 146.5   0.0  284.015029.5   46.0   0.965   1.328    1.000   170.6
pp2_z16s1.6_t2_tps40_q40_sp96_sd0      48       0      2 ours       2  1.9817  0.928    66871 149.0  44.5  223.5 8104.0   26.0   0.973   1.300    1.000   170.2
pp2_z16s1_t2_tps40_q40_sp-1_sd0      48       0      2 b3         1  1.9776  0.976    22444 161.0   0.0  577.027969.0   51.0   0.958   1.095    1.000   170.6
pp2_z16s1_t2_tps40_q40_sp-1_sd0      48       0      2 ours       2  2.0232  0.976    22939 150.0  46.5  533.527110.0   49.5   0.971   1.099    1.000   166.8
```

## 3. 逐格文件索引

| 格 | policy | rep | JSON |
|---|---|---|---|
| sp96 × s=1.6 | b3 | 0,1 | blkcluster_s8t40_20260926_{151235,151528}.json |
| sp96 × s=1.6 | ours | 0,1 | blkcluster_s8t40_20260926_{151820,152118}.json |
| sp-1 × s=1.0 | b3 | 0 | （1 rep，文件见 manifest） |
| sp-1 × s=1.0 | ours | 0,1 | （见 manifest） |

修复有效性（ours sp96 r1 格）：`failed: 0`、
`stale_resume_retries: 0`。

## 4. 关键指标摘录（判决口径用量）

sp96 × s=1.6（r1 格，blkcluster_s8t40_20260926_152118.json）：

| 指标 | b3 | ours |
|---|---|---|
| hit | 0.938 | 0.928 |
| new_tok | 61,799 | 66,871 |
| repl | 0 | 44.5 |
| recall（块） | 15,029.5 | 8,104 |
| dropped_bytes/格 | 有（1.2–1.7GB/worker） | 有 |
| remote_resume_served_local_due_to_replication | — | 0 |

## 5. 缺口（事实清单）

- sp-1 × s=1.0 的 b3 只 1 rep：补跑 runbook §5 命令 1 补齐 r1。
- sp96 ours 两格 `remote_resume_served_local_due_to_replication` = 0。
- sp-1 × s=1.6 未跑（P2 ③ 的判读格）。

## 6. 判决口径核对（只对已出的格）

- **P1**：sp-1 × s=1.0 格 b3/ours 同区间（new_tok 1.022×、hit 同
  0.976）。b3 补齐 r1 前不下结论。
- **P2**：sp-1 档 new_tok 差 1.022× ≤1.2× ✅；`repl_served_local`
  = 0 ❌（判读格 sp-1 × s=1.6 未跑）。
- **P3**：sp96 × s=1.6 出现 dropped > 0 + recall 8,104 + b3 new_tok
  较 sp-1 档回潮 2.75× ✅。
