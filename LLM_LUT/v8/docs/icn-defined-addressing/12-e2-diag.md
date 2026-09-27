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

- worker `_status_hdr` 携带因果序号 `"seq"`（每处理一条命令递增）；
- scheduler `WorkerState` 增加 `evict_t`（驱逐 name→**驱逐命令的
  seq**）与 `cmd_sent`（`_send` 统一发命令并编号）；
- status handler：快照序号**小于**某条驱逐命令序号的快照，证明
  拍于 worker 处理该驱逐之前，替换视图时减去这些驱逐名
  （resident 和 tips 都减）；快照序号覆盖到的日志条目退役；
- 复活路径（result 的 published、deliver 的 fetch/repl）把名字
  从 `evict_t` 弹出，允许真复活。

**演进记录**：第一版用 wall-clock 时间戳（快照 `t` vs 驱逐时间），
当晚补跑 sp96×s1.6 b3 仍出现 1 次 STALE_RESUME——时间戳防不住
"驱逐命令还在飞行、worker 已按驱逐前状态拍了快照"的交叉窗口
（快照时间比驱逐时间新，内容却是旧的）。第二版改为因果序：
序号序比较的是"worker 到底处理到哪条命令"，与飞行延迟无关，
从原理上闭住。

**原拟的 pending-interest pin 方案未采用**：trace 显示驱逐全部发生
在派工之前，pin 护不住已驱逐的块；pin（有 pending interest 的
content 不可选为驱逐 victim）作为 PIT 语义的独立机制，留待
interest aggregation 实验时一并评估。

六测全绿（含回归测试：旧快照不能复活已驱逐块、新快照正常生效、
日志正确退役）。

### 4. 备注

- s=1.0 腿 rep0 绿格（hit 0.415 / new_tok 41 万）数值与 E1 基线
  一致，是否保留由 `matrix_report` 的 retries 扫描判决（runbook
  §5b）；rep1 两格坏 JSON 随 §5b 流程清掉重跑。
- b3@s0 × s=1.6 一条未跑，同样等修复后一起跑。

## 事故三（2026-09-27，sp96×s1.6 b3 两格修复后仍 STALE_RESUME）

状态：**根因已定位 + trace 已确认 + 已修复（2026-09-27，六测全绿），
待远程重跑验证**

trace 确认（runbook §7b 重放 r1 格失败块 span/848-864）：该块一生
中 `spill_drop (reason=lru)` 出现数百次——tier 不断把它踢掉，每次
踢都落在 scheduler 视野的盲窗里。机制链与代码因果分析一致。

### 1. 现象

事故二修复（因果序）当天重跑 sp96×s1.6：b3 rep0/rep1 仍各
`failed=1`，错误仍是 `STALE_RESUME`，`mode=fetch`。两格失败形态：

- `..._181143.json` doc1:19→w3：缺 3 块（span 2224-2272，一段连续）。
- `..._181451.json` doc2:19→w3：缺 0-109 与 115-117，**中间的
  110-114 却在**——块被零散交付，scheduler 对"w3 缺多少前缀"的
  视图和实际不一致。

### 2. 根因（代码因果分析）

事故二的因果序修复只保护了 **resident/tips 轴**（驱逐目标有
`evict_t` 记录，可做 ghost 减法）。**spilled 轴完全没有对等保护**：

1. worker 的 tier 在 `_spill_put` 里为了给驱逐块腾地方会 LRU 踢
   collateral 旧块（还有 oversize 拒绝）。这些被踢的名字**不是驱逐
   目标**，不在任何因果日志里，scheduler 只能通过全量 status 的
   spilled 列表间接知道。
2. 全量替换本身在 zmq FIFO 下是对的，但**踢块发生 → post-evict
   status 被 scheduler 处理**之间有一个在飞窗口。窗口内
   `choose()`/`match_local()` 把已被踢的名字当作"可召回"，从 fetch
   的 `need` 里排除 → 不 fetch → 派工 resume → worker `_recall`
   拿不到 → STALE_RESUME。
3. 重试再失败的路径：purge 把缺块从视图清掉后，块可经 deliver
   重新落到 w3 并被 status 确认回视图；下一次未确认驱逐窗口又把
   它踢掉——同一 turn 连撞两次窗口即硬失败。

加重因素：sp96 档 tier 常满 + 驱逐频繁，窗口被击中概率最高；tier
满时 LRU 踢的是整条冷链，解释了"大块连续缺失名单"。

**连带发现（同一现场暴露）**：worker 的 fetch 失败 ack 没带
`names` 字段，而 scheduler 按 `(holder, names)` 配对——失败 ack
永远配不上等待中的 xfer，计划的即时 degrade 路径是死代码，turn
只能等 120s 看门狗兜底。

### 3. 修复（2026-09-27，回归测试 3 条新增）

- worker `_spill_put(name, obj, dropped=...)` 收集本次 collateral
  踢块名单（LRU + oversize），evict handler 随 post-evict status
  上报 `"spill_dropped"`；scheduler status handler 从 `w.spilled`
  减去（`test_status_spill_dropped`）。
- scheduler 新增 `_spill_confirmed(w)`：存在**未确认驱逐**且按
  投影字节 tier 装不下时，spilled 名字一律不可信——
  `match_local` 不把 spilled-only 当可 resume，`choose` 把这类块
  计入 `need`（宁可多 fetch 不可硬失败）；tier 空闲、装得下、
  或驱逐已确认时保持 E2 原语义（`test_spill_window_conservative`）。
- fetch 失败 ack 补 `"names"` 字段使配对生效；degrade 同时按
  缺块名单清 holder 的 resident/spilled/tips 视图
  （`test_fetch_fail_purges_holder_view`）。

### 4. 备注

- 行为变化范围：只触及 spill tier 路径（sp96 格）和 fetch 失败
  路径。按项目规则（行为修复→旧格全作废），2026-09-27 当天跑出的
  全部格子再次作废，见 runbook §5b。
- 事故二/三的共同教训：scheduler 的每个"可用性判断"都必须问
  "这条信息最后由谁、经哪条消息确认"——resident 轴和 spilled 轴
  各栽一次。


## 事故四（2026-09-27，活性检测设计缺陷：心跳缺失 + teardown 竞态）

### 1. 现象

两案并查：

- **案 A**（b3 sp96×s1.6 r0）：`xfer doc6:31 stuck in stage fetch`
  持续 300s，`STALL_S=300s` 看门狗降级兜住（failed=0，代价 160
  rederiv tokens + wall 虚高 310s）。
- **案 B**（ours sp96×s1.6 rep0）：格 `rc=2` 但 JSON 正常、
  failed=0、数值与 rep1 一致。

### 2. 根因

不是协议行为，是活性检测的设计缺陷：系统把"死亡"和"卡死"两种
本质不同的事件统一用超时糊住（STALL_S=300s），且细查后两种都
没有及时检测路径。

1. **死亡**：launcher（`run_cluster.py` watchdog 线程）每 2s
   `poll()` 子进程——事件检测本来就有。案 A 不是死亡（死亡的
   话 launcher 会 abort，格不会有 JSON；该案有 JSON、正常出
   summary）。案 B 的 rc=2 才是 launcher 看门狗触发的，但它是
   **误炸**：`scheduler.run()` 的 finally 先向 worker 发
   shutdown、worker 收到即退出；而主线程要等 run() 返回才进
   finally 调 `stop.set()`——run() 的 finally 在发完 shutdown 后
   还 sleep 0.5s、关 socket，且 `summary()`（写 JSON + 打印
   "summary ->"）是在这一切**之后**才执行的。于是存在 0.5s 以上
   的稳定窗口：worker 已全部正常退出、stop 未置、launcher 的 2s
   轮询必然命中"worker 已退出而 stop 未置"即 `os._exit(2)`。
   有 JSON + failed=0 + 数值与 rep1 一致，全部符合"正常跑完、
   teardown 时被误炸"；真 mid-run 死亡的格不会有 JSON。
2. **卡死**：worker 主循环是**阻塞 recv**，**没有周期心跳**——
   协议注释（worker.py 头部）里写的 "periodic + on change" 的
   periodic 从未实现。status 只在命令处理后发送：空闲 worker
   完全无信号，忙 worker 卡死时也无信号。scheduler 对失联的
   检测因此只剩命令超时一条路，这就是 300s 沉默的成因（holder
   在那之后整体沉默，进程活着但 wedged）。
3. 同族小坑：旧 xfer degrade 路径把 turn 重新 `send_assign` 给
   **同一个失联 worker**（`x["target"]`），worker 真卡死时会
   二次挂起。

### 3. 修复（2026-09-27 实现；验证 = runbook §2 五条单测 + §7d 三条）

三层修法，死亡走事件、卡死走秒级心跳，超时只剩一个 60s 兜底：

- **L1 死亡 = 事件**：launcher poll 保留，响应维持"立即终止实
  验"——mid-run worker 死亡是基础设施故障，不是协议变量，格作
  废重跑（BAD）是正确语义，不该降级混进数字。修的是竞态：
  Scheduler 加 teardown 钩子，`run()` 的 finally 第一行先置
  teardown 事件；watchdog 仅在"未 teardown 且未 stop"时 abort
  ——正常跑完不再误炸 rc=2。
- **L2 卡死 = 心跳失联**：worker 主循环改 `poll(timeout=1s)`，
  命令空闲 ≥2s（`HB_INTERVAL`）主动发 status——协议注释里的
  periodic 从此是真的。scheduler 侧任何 worker 消息（hello /
  status / result / fetched / delivered）都刷新 `t_status`；
  housekeeping tick 对**非 busy** 且心跳静默 >8s（`HB_GREY_S`）
  的 worker 调 `_grey()`：在途 turn 按旧看门狗语义判失败、在途
  xfer 降级到 fetch 前本地边界、repl 中止；重派一律 re-queue 走
  `choose()` 重选，**绝不回灰 worker**（根因 3 的坑顺带修掉）。
  choose() / controller 的 idle、tips、holders、replication 目标
  全部过滤 grey；全灰且有 ready turn 时 raise 快速坏格，不挂起
  等 cell timeout。超时在此不可避免（卡死无事件），但语义从
  "判死"改为"降级转移"，且从 300s 收紧到秒级。
- **L3 兜底**：`STALL_S` 300s→60s，只覆盖"busy worker 卡死在
  长 assign 内"——run_turn 内联执行无法心跳，只能靠这个平界；
  busy 打印阈值 120/180s 不变（仅日志）。

实现落点：`worker.py`（serve 心跳 + `_t_status`）、
`scheduler.py`（WorkerState.grey/t_status、`_grey`、
`_watchdog_fail` 重构、choose/controller 过滤、teardown 钩子）、
`run_cluster.py`（teardown 事件传入看门狗）。

### 4. 备注

- 案 A 那类 holder wedged 场景，修复后检测时间从 300s → ~8s；
  若未来再现，日志签名是 `GREY (heartbeat silent ...s)` +
  `degraded to E_loc=... (grey ...)`，取证命令见 runbook §7d。
- 与事故二/三同一类教训：scheduler 每个活性/可用性判断都要问
  "这条信息的最后确认时刻是什么"——resident 轴、spilled 轴、
  活性轴各栽一次。




## 7. 常见异常处置

| 现象 | 处置 |
|---|---|
| 某格 `BAD` / 超时 | `python -m icn_proto.matrix_step5 --manifest results/icn_proto/matrix_e2.json --drop-bad`（清坏格），然后重跑那一条命令 |
| 同一格清完重跑还 `BAD` | 先别重跑。跑 §7a 的命令看失败错误，贴回来再定处置 |
| worker 起不来 / hello 超时 | 有孤儿进程：`pkill -9 -f icn_proto`，sleep 3，重跑 |
| 想看重跑某格的完整日志 | `results/icn_proto/cell_logs/cell_<wl>_s2_<pol>_r<rep>_b48_p0.log`（wl 含 `_sp-1`/`_sp96`） |
| 聚合表 spill 列全 0 | 确认命令带 `--spill-mb`、§2 的 5 条测试命令全过；再查该格日志里 `recalled` / `evicted ... spill +` 行 |
| 96MB 档 `dropped_bytes` 恒 0 | LRU 没触发；s=1.6 格应出现第二级 churn，若无记录到结果文档即可 |
| 格内 failed>0，错误含 `resume block not resident` / `STALE_RESUME` | 根因见 `12-e2-diag.md` 事故二+三：确认 `git pull` 拿到最新修复 → §5b 清格重跑；修复后仍出现则跑 §7a 贴完整错误回来 |
| b3 的 spill 格质量指标大幅偏离 ours 同档 | 泄漏或新 bug，停止矩阵，贴日志回来修 |
| 格显示 `OK rc=2`（看门狗退出码，JSON 可能正常） | JSON 数值照常读，但要看门狗为什么杀 worker：贴该格 cell log 最后 50 行回来（路径见上一行），文件名里 `_b48_p0.log` 的 b48 对应 budget=48 |

### 7a. 看格内失败错误（最近 4 个有 failed 的 JSON）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.failed_report
```

只打印有 failed 的 JSON。每条失败记录给 request_id、worker、
decision mode 和完整错误文本（STALE_RESUME 会列出所有缺失块，
末尾就是要拿去 trace 重放查的块名，别截断）。

不带参数扫最新 4 个 JSON；`python -m icn_proto.failed_report 8` 扫
最新 8 个；`python -m icn_proto.failed_report 路径.json` 只看一个。

模块不存在（老代码）时用这个等价 heredoc，但要把 `[:200]` 改成
完整打印，否则看不到错误末尾的块名：

```bash
python - <<'EOF'
import json, glob, os
for p in sorted(glob.glob('results/icn_proto/blkcluster_*.json'),
                key=os.path.getmtime)[-4:]:
    d = json.load(open(p))
    if not d.get('failed'):
        continue
    print('==', p, 'failed', d['failed'])
    for r in d['records']:
        if not r.get('ok'):
            print(' ', r.get('request_id'), '->', r.get('worker'))
            print('   ', r.get('error'))
EOF
```

### 7b. 事故三机制确认（trace 重放已知失败块）

修复是按代码因果洞实施的，定稿前要事件链证据。§7a 查到
STALE_RESUME 后，用错误里**任意一个缺失块**的 span 重放它的一生。
预期事件序列（事故三成立则必见）：

```text
publish → evict (outcome=spilled) → spill_drop (reason=lru)
→ demand（scheduler 派工）→ stale_resume
```

2026-09-27 两格 b3 失败（sp96×s1.6）的现成重放命令：

```bash
cd ~/lut-experiments/LLM_LUT/v8
# r0 格（doc1:19 → w3，缺 span 2224-2272）
python -m icn_proto.trace_replay \
  results/icn_proto/traces/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r0 \
  "span/2224-2240"
# r1 格（doc2:19 → w3，缺 0-109 等，取其中一块即可）
python -m icn_proto.trace_replay \
  results/icn_proto/traces/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r1 \
  "span/848-864"
```

输出贴回来。看不到 `spill_drop` 就说明根因判错，回炉。

### 7c. 一格 wall 明显长于其它格（近 2 倍）

先查该格日志里有没有看门狗/降级记录（以 b3 sp96×s1.6 为例，
文件名里 b48 对应 budget=48）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
grep -iE "watchdog|degrade|FAILED" \
  results/icn_proto/cell_logs/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r0_b48_p0.log \
  results/icn_proto/cell_logs/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r1_b48_p0.log
```

有 `watchdog` 行就把两行日志的这部分贴回来；没有就贴
`grep -c evicted` 的计数即可。

### 7d. 活性异常：日志签名与取证

活性语义（死亡=事件立即炸格；卡死=8s 心跳失联标灰降级）的根因
与修法记录见 **12 号事故四**；本节只放操作性的签名与取证。

日志里看到以下签名时的含义：

| 签名 | 含义 | 处置 |
|---|---|---|
| `[launcher] worker pid=... exited ... aborting` + 格无 JSON | mid-run worker 死亡（事件路径） | 格本就 BAD，按 §7 清格重跑 |
| `GREY (heartbeat silent ...s)` / `degraded to E_loc=... (grey ...)` | worker 卡死被标灰，在途工作已降级转移 | 格可正常完成；grep 该 worker 有无 `Traceback` |
| `GREY (stall ...s > STALL_S)` | busy worker 卡死在长 turn 内（60s 兜底） | 同上 |
| rc=2 但格有 JSON、failed=0 | ~~teardown 竞态误炸~~ 已修复；再出现才算新问题 | 贴日志回来 |

修复后验证标准：§2 五条单测全 `ALL PASS`；人为 kill worker →
2s 内 abort 无 JSON；注入卡死 → ~8s 标灰、格 failed=0 完成；
连跑 3 格无 rc=2。

取证命令（以 b3 sp96×s1.6 r0 为例）：

1. **死亡**：launcher（`run_cluster.py` watchdog 线程）每 2s
   `poll()` 子进程——事件检测本来就有。案 A 不是死亡（死亡的
   话 launcher 会 abort，格不会有 JSON；该案有 JSON、正常出
   summary）。案 B 的 rc=2 才是 launcher 看门狗触发的，但它是
   **误炸**：`scheduler.run()` 的 finally 先向 worker 发
   shutdown、worker 收到即退出；而主线程要等 run() 返回才进
   finally 调 `stop.set()`——run() 的 finally 在发完 shutdown 后
   还 sleep 0.5s、关 socket，且 `summary()`（写 JSON + 打印
   "summary ->"）是在这一切**之后**才执行的。于是存在 0.5s 以上
   的稳定窗口：worker 已全部正常退出、stop 未置、launcher 的 2s
   轮询必然命中"worker 已退出而 stop 未置"，每格竞态概率不低。有
   JSON + failed=0 + 数值与 rep1 一致，全部符合"正常跑完、
   teardown 时被误炸"；真 mid-run 死亡的格不会有 JSON。
2. **卡死**：worker 主循环是**阻塞 recv**（worker.py），
   **没有周期心跳**——协议注释（05 号 §2）里写的 "periodic +
   on change" 的 periodic 从未实现。status 只在命令处理后发送：
   空闲 worker 完全无信号，忙 worker 卡死时也无信号。scheduler
   对失联的检测因此只剩命令超时一条路，这就是 300s 沉默的成因
   （holder 在那之后整体沉默，进程活着但 wedged）。
3. 同族小坑：xfer degrade 路径把 turn 重新 `send_assign` 给
   **同一个失联 worker**（`x["target"]`），worker 真卡死时会
   二次挂起。

**三层修法**（随行 7 interest aggregation spec 一起实现，改
scheduler/worker 消息路径时顺带）：

- **L1 死亡 = 事件**：launcher poll 保留，响应维持"立即终止实
  验"——mid-run worker 死亡是基础设施故障，不是协议变量，格作
  废重跑（BAD）是正确语义，不该降级混进数字。修的是竞态：
  Scheduler 加 teardown 钩子，`run()` 的 finally 第一行先置
  teardown 事件；watchdog 仅在"未 teardown 且未 stop"时 abort
  ——正常跑完不再误炸 rc=2。
- **L2 卡死 = 心跳失联**：worker 主循环改 `poll(timeout=1s)`，
  每 ~2s 主动发一次 status（约 10 行改动，顺带兑现协议注释里
  早已写下的 periodic 承诺）。scheduler 在现有 1s housekeeping
  tick 里记每 worker 最后 status 时间：>8s 未更新 = 失联，标灰
  ——不再派工、移出 holder 集合、在途 turn/xfer/repl 走 degrade
  路径，**重派时必须排除灰 worker**。超时在此不可避免（卡死无
  事件），但语义从"判死"改为"降级转移"，且从 300s 收紧到秒级。
- **L3 兜底阈值**：xfer/repl 挂起打印阈值 120s 不变（仅日志）；
  强制 degrade 的 STALL_S 300s → 60s，作为 L2 的兜底（心跳
  消息自身丢失的极端情况）。

**验证标准（实现后跑）**：

先跑 §2 五条单测（`test_controller / test_e1 / test_e2 /
test_policy / test_openloop`，全 `ALL PASS`），再过下面三条：

1. 人为 `kill` 一个 worker 进程：launcher 2s 内 abort、格 BAD
   无 JSON（事件路径）；
2. 人为给 worker 注入 120s 卡死：scheduler ~8s 标灰、在途 turn
   degrade、格正常完成 failed=0、日志有标灰记录（超时路径）；
3. 连续 3 格正常完成：无 rc=2（竞态修复）。

**实现落点（2026-09-27，随本修法入库）**：

- `worker.py`：主循环改 `sock.poll(1000)` tick，命令空闲
  ≥2s（`HB_INTERVAL`）主动发 status——协议注释里 "periodic +
  on change" 的 periodic 从此是真的；
- `scheduler.py`：`WorkerState` 加 `grey` / `t_status`；
  任何 worker 消息（含 hello/ack）都刷新 `t_status`；
  housekeeping tick 对非 busy 且心跳静默 >8s（`HB_GREY_S`）的
  worker 调 `_grey()`：在途 turn 按旧看门狗语义判失败、在途
  xfer/repl 降级或中止，**重派一律 re-queue 走 choose()，绝不
  回灰 worker**（旧 watchdog 会把 turn 重新 send_assign 给
  卡死的原 worker，顺带修掉）；choose()/controller 的 idle、
  tips、holders、replication 目标全部过滤 grey；全灰且有 ready
  turn 时 raise 快速坏格，不挂起等 cell timeout；
  `STALL_S` 300s→60s，只兜"busy worker 卡死在长 assign 内"
  （run_turn 内联执行无法心跳，只能靠这个平界）；busy 打印阈值
  120/180s 不变（仅日志）；
- `run_cluster.py`：`Scheduler(..., on_teardown=...)`，
  `run()` 的 finally 第一行先置 teardown 事件，launcher 看门狗
  在 stop 或 teardown 已置时不再 abort——正常跑完不再误炸
  rc=2。

**再现任一格时定位 holder 的取证命令**：

```bash
grep -E "xfer .doc6:31 fetch|fetched|Traceback|Killed" \
  results/icn_proto/cell_logs/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r0_b48_p0.log
```
