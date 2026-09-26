# 12 — E2 sp96 首跑事故：根因与修复记录

日期：2026-09-25
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
