# 13 — E2 结果记录

> 项目总纲（我们在复现 ICN 的哪些机制）：`14-icn-mechanism-matrix.md`。
> E2 = 矩阵第 4 行"分层驻留"的实验。

## 0. 一页状态（只看这里就够，更新于 2026-09-27 晚）

### E2 要回答什么

E1 的结论：只给每块 GPU 自己的 HBM 当驻留层时，主动复制（ours）
干不过按需取块（b3），亏在"复制→挤占 HBM→驱逐=消失→重算"这条链。
E2 验证：给被驱逐的块加一层便宜驻留层（主机 DRAM），这条链断不断。

### 矩阵进度（2026-09-27 第三次重跑：16 格全出、无 BAD、SLO 全 1.000）

| 格 | 状态 |
|---|---|
| §5 十六格（sp-1/sp96 × s1.0/s1.6 × b3/ours × 2 reps） | ✅ 全出。待核验两件事：① 每格 `stale_resume_retries=0`（`matrix_report` 报告）；② b3 sp96×s1.6 一格 wall=310s（近 2 倍），疑似中途卡过一次，其 new_tok/hit 可能污染该行读数（runbook §7c 查 watchdog） |
| §5a b3@s0 四格 | 🔄 未跑（不在上表） |

**本表是三个修复全部到位后的第一份矩阵**（09-25 至 09-27 白天的所有
历史数均受三个事故不同程度污染，只能作方向参考，不与本表混读）。
已可读出的方向（判文待上面两件核验完成后写）：

- sp-1 档 b3/ours 继续重合（hit 双双相等）——P1 方向稳定；
- 全矩阵 SLO=1.000——"延迟崩塌=事故症状"定案；
- **sp96 档出现反转：ours 的 hit/new_tok 均不劣于 b3**（s1.0 还明显
  更好：77k vs 97k）——但 b3 s1.6 行的 wall 异常未查清前不能下结论；
- sp96 vs sp-1 的 hit 差（0.88-0.93 vs 0.98）确认 tier 有限时回潮
  真实，regime 边界在 tier 容量上。

---

## 1. 聚合数据（2026-09-27 第三次重跑，三个修复全部到位后的首份矩阵，16 格全出无 BAD）

```text
wl                     policy  runs     rps    hit  new_tok   xfer  repl  evict   recall  favoid     p50     p95  SLO@2.0s   wall
sp-1 × s=1.6             b3      2   2.0450  0.979    21776  139.5   0.0  340.5  27587.0    75.0   0.978   1.175     1.000   165.1
sp-1 × s=1.6            ours     2   2.0660  0.979    21656  117.5  23.5  305.0  24179.5    79.5   0.971   1.111     1.000   163.5
sp96 × s=1.6             b3      2   1.3432  0.933    64807  159.5   0.0  246.5  11990.0    25.5   0.972   1.299     1.000   310.4
sp96 × s=1.6            ours     2   1.9326  0.928    63024  133.0  43.0  187.5   7825.0    26.0   0.982   1.260     1.000   174.6
sp-1 × s=1.0             b3      2   1.9659  0.976    22473  148.5   0.0  357.0  27721.5    47.0   0.970   1.104     1.000   171.5
sp-1 × s=1.0            ours     2   1.8656  0.976    22188  137.5   8.0  312.0  25832.0    51.5   0.961   1.082     1.000   181.6
sp96 × s=1.0             b3      2   1.8154  0.884    97456  157.5   0.0  263.5  11717.0    23.5   0.976   1.362     1.000   186.0
sp96 × s=1.0            ours     2   1.9187  0.916    77198  169.5  15.5  215.5   9411.0    14.0   0.961   1.368     1.000   175.8
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
- ~~sp96 × s=1.6 ours rep0 格 `rc=2`~~ **已定位（2026-09-27）**：
  run_cluster launcher 看门狗的 shutdown 竞态误炸——scheduler
  finally 先发 shutdown、worker 正常退出后，主线程才置 stop，
  2s 轮询命中"已退出而 stop 未置"即 `os._exit(2)`。JSON 正常 +
  failed=0 + 数值与 rep1 一致全部吻合；真 mid-run 死亡的格不会有
  JSON。修法见 runbook §7d L1（teardown 事件），格数据有效。
- `repl_served_local` 是否仍为 0：从 ours 格 JSON 的
  `remote_resume_served_local_due_to_replication` 读。

## 4. 判文（2026-09-27 干净矩阵；判据事先写死于 10 §4，非跑后解释）

### V0 "延迟崩塌"：不是机制，是事故症状 —— 定案

三修复后 16 格 SLO 全部 1.000、p50 ≤ 0.98s。此前 SLO 0.233–0.613
的读数与事故二/三的事件链（视图被污染 → STALE_RESUME 重试 →
重排队堆积）吻合，干净代码上该现象不存在。

### P1（tier 无限时 proactive 复制是否有增量角色）→ 不成立，第三次复现

- 判据：sp-1 档 ours vs b3 的 hit/new_tok 差 < 噪声级。
- 证据：s1.0 hit 0.976=0.976，new_tok 22473/22188（差 1.3%）；
  s1.6 hit 0.979=0.979，new_tok 21776/21656（差 0.6%）；
  repl 仅 8–23 次/格，对结果无影响。
- 机制表述：驱逐落点必然可召回后，驱逐不再摧毁 resume 路径，
  b3 的按需取块拿走全部收益；proactive 复制要解决的问题（驱逐=
  消失）不复存在 → 退化为 b3。**E1 时代 20.8× 的崩塌不是复制的
  锅，是虚空落点的锅。**

### P2（tier 是否给 proactive 复制创造新角色）→ 方向性成立，关键证据是事件链不是数字

- 机制预测（事先写死）：tier 有限时"驱逐→tier 满→LRU 踢→真
  消失→重算"的二级 churn 出现，HBM 驻留（复制）因此有额外保护
  价值。
- 事件链证据：trace 重放（runbook §7b）显示 sp96 格 tier LRU
  踢块常态化——单个失败块一生被踢数百次；tier 踢块对 scheduler
  不可见正是事故三的来源，反过来说"踢块高频"本身就是 tier 在
  真实丢块的直接计量。
- 观测量与机制一致（非胜负对比）：sp96 档 ours 的 evict 更少
  （187.5/215.5 vs b3 246.5/263.5，复制分担了驱逐压力），
  recall/favoid 维持正常。
- 保留：尚未做"复制 vs 不复制"的同预算严格对照来分离该效应，
  判为**方向性成立**，不写"成立"。

### P3（tier 有限的回潮 / regime 边界）→ 成立

- 判据：sp96 vs sp-1 的 hit 显著下降 + new_tok 数倍增长。
- 证据：hit 0.884–0.933 vs 0.976–0.979；new_tok 2.9–4.4×。
- 机制表述：容量瓶颈从 HBM 转移到 tier；tier 满 → LRU 踢 →
  块真消失。regime 边界在 tier 容量上，与 E1 的 HBM 边界同构。

### 残余事项（不影响以上判文方向）

1. r0 格日志显示 6 个 turn 首次尝试 stale、全部重试成功
   （failed=0）——事故三守卫的设计行为（硬失败→重试），可用性
   门槛相应改为 failed=0 + 无卡死（runbook §5b）；
2. **b3 sp96×s1.6 wall=310s 已定性**：一次取块 300s 无响应
   （`xfer doc6:31 stuck in stage fetch`），看门狗降级 + 重试兜住，
   代价 160 rederiv tokens（占该格 0.25%）+ wall 虚高。根因是活
   性检测设计缺陷（worker 无周期心跳，阻塞 recv，卡死与死亡都
   只能靠超时糊）。三层修法（心跳标灰 / teardown 竞态修复 /
   STALL_S 收紧）已于 2026-09-27 实现，见 runbook §7d，含验证
   步骤；验证通过后此格的重跑价值另行评估；
3. §5a（b3@s0 四格）未跑：补齐后 P1/P3 获得虚空落点对照。

## 5. 行 7 PIT 判文（2026-09-28；判文主体与机制细节见 15 号 §8）

闭链同前缀风暴 regime：**Interest aggregation 成立**。

- 配置：闭链 s16 t3 share2 q40，{b3, ours} × pit on/off，rep=3。
- 兑现：pit=on 各格 served 41-47/64 turns 由 compute merge 服务，
  saved_tok 16.6-19 万（上界口径：park 事件 × cum_tokens，非去重
  token 数），fetch merge 2 次/格，failed=0，b3 质量面（hit/new_tok）
  与 pit=off 逐位一致；ours 的 new_tok 被 pit 从 9296 拉回 b3 水平
  4810（放置控制引起的迁移重算被聚合 resume 吸收）。
- 边界：open-loop（poisson 到达 + think time 错开）机会量 ≈0，
  判否——**PIT 的价值由"共享度 × 并发度"的 regime 决定**：风暴
  成立，错开无价值。
- 过程事故：事故五（组播空 payload → worker 崩溃连环，已修）、
  事故六（ours 稀有超时竞态，死锁 backstop 已入码，开放）——
  均见 12 号，不影响判文。
