# 11 E2 实验运行手册（runbook）

只记录要跑的指令。结果看 `13-e2-results.md`，事故根因看
`12-e2-diag.md`，设计与判决口径看 `10-e2-backing-tier.md`。

## 1. 同步代码

```bash
cd ~/lut-experiments/LLM_LUT/v8
git pull
```

## 2. git pull 之后：确认拉下来的代码没坏（5 条命令，<10 秒）

这 5 条是项目的单元测试，不碰 GPU、不加载模型，几秒钟跑完。作用：
**抓代码错误，不针对操作人**。两类原因都会 FAIL——

1. 我（改代码的一方）把别的东西改坏了——这是主要防范对象，
   一次 FAIL 等于替所有下游实验挡了一个会作废全部结果的 bug；
2. 你这边代码状态乱了（没拉全 / 拉错版本 / 本地改动冲突）——
   真发生了重置一下就好，不是操作失误的问题。

任何一条 FAIL = 不要开始跑实验。

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.test_controller
python -m icn_proto.test_e1
python -m icn_proto.test_e2
python -m icn_proto.test_policy
python -m icn_proto.test_openloop
```

每条最后一行应打印 `ALL PASS`，5 条全过再继续往下走。

## 3. 跑前检查

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

有自己的僵尸进程（同事的不动）：

```bash
pkill -9 -f icn_proto
sleep 3
```

## 4. 冒烟（两格，各 ~13 分钟，每次矩阵前必跑）

4a：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.run_cluster --policy ours --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 48 --spill-mb -1 --seed 0
```

4b：

```bash
python -m icn_proto.run_cluster --policy b3 --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 48 --spill-mb -1 --seed 0
```

每格跑完读数：

```bash
python -m icn_proto.e1_report
```

通过标准（全满足才铺矩阵）：
1. 两格 `failed: 0`；
2. ours 格 `spill_bytes` > 0 且某 worker `recall_count` > 0；
3. b3 格与 ours 格 hit 都 ~0.97、new_tok 都 2.2–2.4 万；两格 worker
   `resident_bytes` 都 ~48MB。不满足 = 停止，贴输出回来。

## 5. 确认矩阵（16 格 + b3 重基线 4 格，每格 ~13 分钟）

逐条执行，一条跑完再跑下一条（manifest 断点续跑，中断后重跑同一条）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --spill-mb -1 --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb -1 --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --spill-mb 96 --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb 96 --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

b3 重基线（2 条）：

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3 --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3 --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --trace-dir results/icn_proto/traces \
  --manifest results/icn_proto/matrix_e2.json
```

## 5b. 修复后重跑（清掉修复前的全部格子）

背景见 `12-e2-diag.md`：事故二（2026-09-27 白天）和事故三
（2026-09-27 晚，spilled 轴因果洞 + fetch 失败 ack 缺 names）
各作废一次全部格子。修复前的格子全部作废重跑（逐格保留的旧格
会把新旧数据混在一起，2026-09-27 已踩过两次）。步骤：

1. 跑 §2 的 5 条测试命令，全过再继续；
2. 清掉全部格子（两次修复当天跑出的也在内）：

```bash
python -m icn_proto.matrix_report --drop-before 20260928
```

3. 重跑 §5 的 4 条和 §5a 的 2 条（manifest 只补被清的格）。
   可用性门槛：**failed=0**、无 `xfer ... stuck` 卡死记录即可用。
   少量 stale_resume 重试（重试后成功）是事故三守卫的设计行为
   （宁可重试不可硬失败），不单独作废格；重试多得反常
   （>20）或同一格反复出现才贴回来。

## 6. 块生命周期追踪（机制观测）

§5 的矩阵命令已带 `--trace-dir results/icn_proto/traces`，每格追踪写在
子目录里，命名规则：

```
results/icn_proto/traces/cell_<wl>_s2_<policy>_r<rep>
```

当前已跑完的 16 格（E2 manifest 全部）：

| wl | b3 目录 | ours 目录 |
|---|---|---|
| `pp2_z16s1_t2_tps40_q40_sp-1_sd0` | `cell_pp2_z16s1_t2_tps40_q40_sp-1_sd0_s2_b3_r0` / `_r1` | `..._s2_ours_r0` / `_r1` |
| `pp2_z16s1.6_t2_tps40_q40_sp-1_sd0` | 同上换 wl | 同上换 wl |
| `pp2_z16s1_t2_tps40_q40_sp96_sd0` | 同上换 wl | 同上换 wl |
| `pp2_z16s1.6_t2_tps40_q40_sp96_sd0` | 同上换 wl | 同上换 wl |

`cell_pp2_z16s1_t2_tps40_q40_sd0_s2_b3_r0` / `_r1` 和
`cell_pp2_z16s1.6_t2_tps40_q40_sd0_s2_b3_r0` / `_r1`（无 `_sp` 段）。

（格 → summary JSON 的对应关系在 manifest 里，下面命令直接查。）

### 6a. 选一个热点块（以 sp96 × s1.6 延迟崩塌格为例）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -c "
import json
m = json.load(open('results/icn_proto/matrix_e2.json'))
for r in m:
    if 'z16s1.6' in r.get('wl','') and '_sp96_' in r.get('wl','') \
       and r['policy']=='ours' and r['rep']==0:
        print('JSON:', r['json'])
        d = json.load(open(r['json']))
        for e in d['directory']['top_lambda'][:10]:
            print('  lambda=%-9s count=%-4s %s' % (e['lambda'], e['count'], e['name']))
"
```

输出每行末尾是块名（截断显示，但 `span/起点-终点` 部分完整）。记下
要查的块的 span，如 `1968-1984`。把上面条件换成 `policy=='b3'`、
`rep==1`、或 `z16s1_`/`_sp-1_` 可查其它格。

### 6b. 先看仪器是否完整（每格必做）

```bash
python -m icn_proto.trace_replay \
  results/icn_proto/traces/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_ours_r0 \
  --summary
```

应列出每个进程的事件计数（worker 的 publish/evict/recall，
scheduler 的 demand/fetch_send/delivered 等），总数 > 0。全是 0
说明该格没带追踪，重跑对应 §5 命令。

### 6c. 重放单块完整一生

```bash
python -m icn_proto.trace_replay \
  results/icn_proto/traces/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_ours_r0 \
  "span/1968-1984"
```

（第二参数是名字子串，用 6a 里记下的 span，加引号。）输出按时间
归并了该块的所有事件：publish → demand → evict（落点 spilled 还是
dropped）→ recall / fetch_send → delivered。同一 span 若匹配到多块
会全部打出，正常。

### 6d. 要贴回来的东西

1. 6a 的 top-lambda 列表；
2. 6b 的 `--summary` 输出；
3. 6c 对 **sp96 × s1.6 的 ours 和 b3 各一格** 各重放 2–3 个高
   lambda 块的完整输出。

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

### 7d. 活性设计缺陷与三层修法（2026-09-27 定稿）

两案并查：

- **案 A**（b3 sp96×s1.6 r0）：`xfer doc6:31 stuck in stage fetch`
  持续 300s，`STALL_S=300s` 看门狗降级兜住（failed=0，代价 160
  rederiv tokens + wall 虚高）。
- **案 B**（ours sp96×s1.6 rep0）：格 `rc=2` 但 JSON 正常、
  failed=0、数值与 rep1 一致。

**定性：不是协议行为，是活性检测的设计缺陷**。当前系统把"死亡"
和"卡死"两种本质不同的事件统一用超时糊住（STALL_S=300s），且
细查后两种都没有及时检测路径：

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

1. 人为 `kill` 一个 worker 进程：launcher 2s 内 abort、格 BAD
   无 JSON（事件路径）；
2. 人为给 worker 注入 120s 卡死：scheduler ~8s 标灰、在途 turn
   degrade、格正常完成 failed=0、日志有标灰记录（超时路径）；
3. 连续 3 格正常完成：无 rc=2（竞态修复）。

**再现任一格时定位 holder 的取证命令**：

```bash
grep -E "xfer .doc6:31 fetch|fetched|Traceback|Killed" \
  results/icn_proto/cell_logs/cell_pp2_z16s1.6_t2_tps40_q40_sp96_sd0_s2_b3_r0_b48_p0.log
```
