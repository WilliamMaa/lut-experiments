# 12 — E2 事故：根因与修复记录

## 事故一（2026-09-25，sp96 首跑）

状态：**已修复、已重跑通过**（2026-09-26 四格全绿；结果在
`13-e2-results.md`）

## 1. 现象

§5 命令 4（sp96 × s=1.6）四格全部 rc=1、failed=3/8/5/8（b3 与 ours
同炸），错误一律：

```
RuntimeError: resume block not resident: <块名>
```

失败 JSON：
`blkcluster_s8t40_20260925_{175925(b3 r0), 180218(b3 r1), 180523(ours r0), 180816(ours r1)}.json`

**关键判别**：出事的是 **sp96 档，不是 sp-1**——各 worker
`dropped_bytes` 0.3–1.9GB 只可能来自 `_spill_put` 的 LRU 级联丢弃，
而 sp-1 档（cap=inf）永不触发 LRU。

## 2. 根因（两个叠加）

1. **spill LRU 丢弃对 scheduler 不可见 + 决策窗口滞后**。worker 处理
   evict 时 `_spill_put` 级联丢块，随后 `report_status` 带全量
   `spilled` 列表排队发往 scheduler；但主循环每轮 **先 dispatch 后
   只 recv 一条消息**——dispatch 内 `choose()` 用上一周期的
   `w.spilled`（含已丢块）做 placement 决策，assign 的 resume 集
   超出 worker 实际持有。worker 端 `run_turn` recall 缺失 → raise。
   竞争窗口 = 一次 dispatch，open-loop Poisson 突发释放多个 turn 时
   可连击（与 3–8/320 的失败率吻合）。
2. **无重试路径**。旧 raise 只报第一个缺失块，scheduler 把它当硬
   失败记 failed——一次视图竞争直接毁掉一格。

排除的假设：fetch-deliver 用 `xfer["names"][-1]` 反推 tip end
（原主嫌疑）——fetch 路径 holder 端 recall-then-send + fetched
ok=False 降级已覆盖，不是本次根因。

## 3. 修复（回归测试 `test_stale_resume_retry`，五测全绿）

- `scheduler.py` 主循环：dispatch **前** `sock.poll(0)` drain 全部
  排队消息，消除决策窗口的视图滞后；
- `worker.py` `run_turn`：收集**全部**缺失块，
  raise `STALE_RESUME: <逗号分隔名单>`；
- `scheduler.py` result 处理：识别 STALE_RESUME → 从该 worker 的
  `resident/spilled/tips/via_repl` 清除缺失块（result 后的 status
  尚未处理，必须主动清）→ 删除失败 record → turn 重新入队、
  `choose()` 重决策（降级为更低 E 或全量重算，语义正确）；
  每 turn 最多重试一次，第二次硬失败（真不变量破坏）；
- summary spill 块新增 `stale_resume_retries`（健康格 ≈0；>0 表示
  重试在工作，turn 不再失败）。

## 4. 处置结果

2026-09-26 `--drop-bad` 清 4 格后重跑命令 4：四格全绿、
`failed=0`、`stale_resume_retries=0`。结果与读数在
`13-e2-results.md`。

## 5. 备注

- 命令 1（sp-1 × s=1.0）绿格跑在修复前代码，但竞争只在 spill 丢块
  时出现，sp-1 格**保留有效**。
- 本事故的 system 含义：tier 有限容量引入第二级 churn（P3 要量的
  东西）时，控制器视图必须与 tier 内容强一致——96MB 档的
  dropped_bytes 本身就是 P3 的 regime 信号，不是纯噪声。

---

## 事故二（2026-09-27，s0 腿 STALE_RESUME 硬失败）

状态：**根因已定位（trace 证实）+ 已修复（2026-09-27，六测全绿），
待远程重跑验证**

### 1. 现象

b3@s0 × s=1.0 rep1 连续两次跑 `failed=1/2`，错误一律
`STALE_RESUME: <完整缺失名单>`。失败格
`blkcluster_s8t40_20260926_211638.json`：`stale_resume_retries=5`，
即重试路径工作正常、2 个 turn 重试后第二次又撞上才硬失败。

### 2. 根因（trace 证实，2026-09-27 重放）

trace：`results/icn_proto/traces/cell_pp2_z16s1_t2_tps40_q40_sd0_s2_b3_r1`，
匹配 `parent/4add2dcf28508bf7`（共享文档根链）。三个 stale_resume
实例同一结构，以 `doc0:23` 为例：

```text
t+132.895  sched evict_plan {w0}      ← scheduler 自己下令驱逐该块
t+132.895  w0     evict (dropped)
t+132.911  sched demand doc0:23 → w0 ← 16ms 后把需要该块的 turn 派给 w0
t+132.916  w0     stale_resume
```

**不是视图滞后，也不是驱逐/放置决策互斥**——trace 里有个
278ms 间隔的实例（493.677 驱逐 → 493.955 才派工），这么长的间隔
里 choose 的 `match_local` 全集检查不可能通过。逐条排除后的真正
根因：

**乐观驱逐 vs 全量快照覆盖的因果冲突**。
`_apply_evict` 乐观地把块从视图丢弃（worker 还要几毫秒才真正执行
删除）；而 worker 的 status 是发送时现取的全量快照
（`worker._status_hdr`），**在 worker 执行驱逐之前生成的快照**可能
在这之后才到 scheduler，status handler 的 `w.resident = set(...)`
整个替换视图，把刚驱逐的块**复活**。trace 实例完全吻合：

```text
t+490.5   w0 收到 deliver → 发出 status S1（快照含块 X）
t+493.677 scheduler 驱逐 X（视图乐观丢弃）
t+493.7   下一轮 drain 处理 S1 → 视图被覆盖回"X 在驻留"
t+493.955 派工 doc1:17 local（视图说 X 在）→ worker 已真删 X → stale
```

09-25 的"dispatch 前 drain 全部消息"修复反而把这个 clobber 引进了
决策路径：drain 本身没错，错的是盲替换不认快照时间。

**为什么只有 s0 暴露**：同一 bug 全档位存在——sp96 × s1.6 ours r0
格 `stale_purge 10 / failed 0`，recall 把缺块从 DRAM 层捞回兜住了；
sp-1 永不驱逐。只有 s0（虚空落点）无处兜底。

### 3. 修复（2026-09-27 实现，回归测试 `test_status_snapshot_clobber`）

- worker `_status_hdr` 增加快照时间戳 `"t"`；
- scheduler `WorkerState` 增加 `evict_t`（本端发起的驱逐
  name→time 日志），`_apply_evict` 记录；
- status handler：替换视图时减去"比快照时间新的本端驱逐"
  （resident 和 tips 都减）；快照确认了丢失（名字不在快照里）
  或超过 600s 的日志条目退役；
- 复活路径（result 的 published、deliver 的 fetch/repl）把名字
  从 `evict_t` 弹出，允许真复活；
- 六测全绿（含新回归测试：旧快照不能复活已驱逐块、新快照正常
  生效、日志正确退役）。

**原拟的 pending-interest pin 方案未采用**：trace 显示驱逐全部发生
在派工之前，pin 护不住已驱逐的块；clobber 修复才是对症的。
pin（有 pending interest 的 content 不可选为驱逐 victim）作为
PIT 语义的独立机制，留待 interest aggregation 实验时一并评估。

### 4. 备注

- s=1.0 腿 rep0 绿格（hit 0.415 / new_tok 41 万）数值与 E1 基线
  一致，是否保留由 `matrix_report` 的 retries 扫描判决（runbook
  §5b）；rep1 两格坏 JSON 随 §5b 流程清掉重跑。
- b3@s0 × s=1.6 一条未跑，同样等修复后一起跑。
