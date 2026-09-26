# 11 E2 实验运行手册（runbook）

只记录要跑的指令。结果看 `13-e2-results.md`，事故根因看
`12-e2-diag.md`，设计与判决口径看 `10-e2-backing-tier.md`。

## 1. 同步代码

```bash
cd ~/lut-experiments/LLM_LUT/v8
git pull
```

## 2. 远程验证（每次同步后必跑，<10 秒）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.test_controller
python -m icn_proto.test_e1
python -m icn_proto.test_e2
python -m icn_proto.test_policy
python -m icn_proto.test_openloop
```

五行 `ALL PASS` 才继续。任何 FAIL 停止，把输出贴回来。

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
| 聚合表 spill 列全 0 | 确认命令带 `--spill-mb`、§2 五测试全过；再查该格日志里 `recalled` / `evicted ... spill +` 行 |
| 96MB 档 `dropped_bytes` 恒 0 | LRU 没触发；s=1.6 格应出现第二级 churn，若无记录到结果文档即可 |
| 格内 failed>0，错误含 `resume block not resident` / `STALE_RESUME` | 根因见 `12-e2-diag.md`：确认 `git pull` 拿到修复 → `--drop-bad` 清格重跑；修复后仍出现则贴日志 |
| b3 的 spill 格质量指标大幅偏离 ours 同档 | 泄漏或新 bug，停止矩阵，贴日志回来修 |

### 7a. 看格内失败错误（最近 4 个有 failed 的 JSON）

```bash
cd ~/lut-experiments/LLM_LUT/v8
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
            print(' ', r.get('request_id'), '->', r.get('worker'),
                  repr(r.get('error'))[:200])
EOF
```
