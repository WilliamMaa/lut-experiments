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
