# 11 E2 实验运行手册（runbook）

> 前置：`10-e2-backing-tier.md`（设计 + 预注册判决 P1/P2/P3）。
> 本文档只写**要跑的命令**，每条完整、可直接整段复制。
> E2 = host-DRAM backing tier：驱逐不再等于消失（10 §1）。
>
> 日期：2026-09-25

## 0. 实验配置（所有命令共用）

| 项 | 取值 |
|---|---|
| 模型 | `/home/u/downloads/models/Qwen3.6-35B-A3B` |
| 机器 | mamingyu@u，8×A800，conda env `lut_py310` |
| workload | 8 sessions × 40 turns，`--q-tokens 40`，泊松 λ=2.0 + zipf-16 |
| 预算 | `--worker-mem-budget-mb 48`（HBM 稀缺性恒开，spill 不进预算） |
| 新轴 | `--spill-mb`：0=关（E1 基线），-1=无限，96=受限档 |
| 对照 | b3 vs ours（s0/E1 基线 16 格已在 `matrix_e1.json`，不重跑） |
| 结果目录 | `results/icn_proto/`，manifest `results/icn_proto/matrix_e2.json` |

单格资源：2 worker × 2 卡（`--gpu-pool 0,1,2,3`），留 4 卡给同事。
矩阵：8 卡（`--gpu-pool 0,1,2,3,4,5,6,7`，4 worker）。

固定 λ=2.0 的理由：E1 矩阵中 s=1.6/λ=2 是最深 churn 格
（ours new_tok 劣化 20.8×，evict 482），backing tier 效果在这个
regime 里最有判别力。

## 1. 同步代码（git，本地提交推送 → 远程 pull）

远程只做：

```bash
cd ~/lut-experiments/LLM_LUT/v8
git pull
```

本次同步带的改动（2026-09-25，E2 实现，详见 10 §5）：
- `worker.py`：spill 存储（`--spill-mb`，0 关 / -1 无限 / 受限档 LRU
  丢最老）+ evict 落点改造 + assign/fetch 前召回 + status 携带
  spilled 名单与口径。**协议零新消息**。
- `scheduler.py`：`--spill-rate`（成本模型召回速率，默认 20e9 B/s）；
  resume 可行性 = resident ∪ spilled；choose 加召回成本项；
  `_plan_repl` holder 判定含 spill 持有者；`_plan_evict` 只驱逐
  resident（spilled tip 保护链但不再被重复计字节）；spill 口径统计
  与 summary `spill` 块。
- `test_e2.py`（新，21 断言）；`matrix_step5.py`：`--spill-mb` 透传 +
  wl 签名加 spill 档（`_sp-1` / `_sp96`）；`e1_report.py` 加 spill 读数。
- **实现期修正**（单测暴露）：`_plan_evict` 的链推导 pool 原来只看
  resident，spill 后 tip 的链不可走会把整个 worker 跳过（驱逐停摆）；
  现 pool = resident ∪ spilled，最终过滤仍只驱逐 resident。

## 2. 远程验证（每次同步后必跑，<10 秒）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.test_controller
python -m icn_proto.test_e1
python -m icn_proto.test_e2
python -m icn_proto.test_policy
python -m icn_proto.test_openloop
```

期望五行 `ALL PASS`。任何 FAIL 停止实验，把输出贴回来。

## 3. 跑前检查（每次）

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

若有自己的僵尸进程（同事的不动）：

```bash
pkill -9 -f icn_proto
sleep 3
```

## 4. 冒烟（两格，各 ~13 分钟，每次矩阵前必跑）

### 4a. ours @ spill∞（P2 的主力面）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.run_cluster --policy ours --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 48 --spill-mb -1 --seed 0
```

### 4b. b3 @ spill∞（P1 对照纯净性）

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

**冒烟通过标准**（全满足才铺矩阵）：
1. 两格均 `failed: 0`；
2. ours 格的 `spill` 块：`spill_bytes` > 0 且某 worker 的
   `recall_count` > 0（tier 真的在收块、真的在召回）；
3. **P1**：b3 格的计数类指标（hit / new_tok / xfer / transfers）
   与 E1 基线同配置（matrix_e1.json 里
   `pp2_z16s1_t2_tps40_q40_sd0` 的 b3 行：hit 0.976，new_tok
   22,188，xfer 155.5）**无明显变化**——b3 在 E1 里 evict≈0，spill
   不该改变它的行为；若显著变化说明实现有泄漏，先修再判。
4. ours 格的 `rederivation_tokens` 应明显低于 E1 同配置 ours 基线
   （103,141 new_tok 那格的重算部分）——tier 接住被驱逐的 state
   的直接证据。

latency 类指标照旧只作参考（邻居噪声）。

## 5. 确认矩阵（2 spill 档 × 2 偏斜 × {b3, ours} × 2 reps = 16 格，~3.5 小时）

**逐条执行**，一条跑完再跑下一条（manifest 断点续跑，中断后重跑
同一条即可）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --spill-mb -1 --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb -1 --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --spill-mb 96 --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb 96 --manifest results/icn_proto/matrix_e2.json
```

每格约 13 分钟。wl 签名带 spill 档（`_sp-1` / `_sp96`），与
matrix_e1.json 的 s0 格（`_sp` 后缀缺省）互不串格。最后一条跑完
打印聚合表，新增 `recall`（召回块数）与 `favoid`（召回替代的跨
worker fetch 次数）两列。

### §5b 结果记录（跑完贴聚合表到这里）

（待填）

## 6. 常见异常处置

| 现象 | 处置 |
|---|---|
| 某格 `BAD` / 超时 | `python -m icn_proto.matrix_step5 --manifest results/icn_proto/matrix_e2.json --drop-bad`（清坏格），然后重跑那一条命令 |
| worker 起不来 / hello 超时 | 有孤儿进程：`pkill -9 -f icn_proto`，sleep 3，重跑 |
| 想看重跑某格的完整日志 | `results/icn_proto/cell_logs/cell_<wl>_s2_<pol>_r<rep>_b48_p0.log`（wl 含 `_sp-1`/`_sp96`） |
| 聚合表 spill 列全 0 | tier 没生效：确认命令带 `--spill-mb`、§2 五测试全过；再查该格日志里 `recalled` / `evicted ... spill +` 行 |
| 96MB 档 `dropped_bytes` 恒 0 | LRU 没触发：tier 没满过，s=1.6 格应出现第二级 churn，若无则 P3 需要更大压力（记录即可，不回退） |
| b3 的 spill 格指标显著偏离 E1 基线 | P1 违反：实现泄漏，停止矩阵，贴日志回来修 |

## 7. 判决口径（10 §4 预注册，跑完对着读）

**P1（对照纯净性）**：b3 在 spill 档与 E1 基线无显著差异。违反 =
先修实现再判。

**P2（design implication 成立）**：spill∞ 档 ours 同时满足
① `rederivation_tokens` 趋零；
② ours−b3 的 `new_tok` 差收窄至 ≤1.2×（E1 同配置基线：s=1.0 差
4.6×、s=1.6 差 20.8×）；
③ `repl` 上升且 `remote_resume_served_local_due_to_replication` > 0
（E1 恒 0 的量首次变正）。
三条同时成立 ⇒ "便宜 backing tier 是 proactive residency 的前提"
在真实 DRAM 层级上成立。

**P3（tier 稀缺边界）**：96MB 档 s=1.6 出现第二级 churn（spill 满
→ `dropped_bytes` > 0 → 重算回潮），给出 tier 容量的 regime 边界。

**证伪条件**：spill∞ 下 ours 仍劣于 b3 ≥1.5× ⇒ backing tier 也救
不了 proactive residency，06+E1 阴性适用范围比预期更广——强阴性，
照样成文（12-e2-results.md 按"层级轴也不救"写）。

**互锁**：不许用 fetch 流量暴涨换 new_tok（`transfers` /
`transfer_bytes` 与 new_tok 同报，两边都要看）；latency 类只作参考。
