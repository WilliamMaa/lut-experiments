# 08 E1 实验运行手册（runbook）

> 前置：`07-regime-study.md`（设计 + 预注册判决）。本文档只写**要跑的命令**，
> 每条完整、可直接整段复制。E1 = 长生命周期 agent workload（07 §4 E1）。
>
> 日期：2026-09-24

## 0. 实验配置（所有命令共用）

| 项 | 取值 |
|---|---|
| 模型 | `/home/u/downloads/models/Qwen3.6-35B-A3B` |
| 机器 | mamingyu@u，8×A800，conda env `lut_py310` |
| workload | 8 sessions × 40 turns，`--q-tokens 40`，泊松到达 + zipf-16 |
| 预算 | `--worker-mem-budget-mb 48`（稀缺性恒开） |
| 对照 | b3 vs ours |
| 结果目录 | `results/icn_proto/`，manifest `results/icn_proto/matrix_e1.json` |

单格资源：2 worker × 2 卡（`--gpu-pool 0,1,2,3`），留 4 卡给同事。
矩阵：8 卡（`--gpu-pool 0,1,2,3,4,5,6,7`，4 worker）。

## 1. 同步代码（git，本地提交推送 → 远程 pull）

项目本来就是 git 管理的
远程只做：

```bash
cd ~/lut-experiments/LLM_LUT/v8
git pull
```

本次同步带的改动（2026-09-24）：
- `scheduler.py`：缺失后缀复制粒度；ΔC_future 用实测 C_recompute 校准；
  `repl_reject` 八道门直方图 + `g_best`；E1 全部口径统计。
- `e1_report.py`（新）：一条命令读判决块。

## 2. 远程验证（每次同步后必跑，<10 秒）

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.test_controller
python -m icn_proto.test_e1
python -m icn_proto.test_policy
python -m icn_proto.test_openloop
```

期望四行 `ALL PASS`。任何 FAIL 停止实验，把输出贴回来。

## 3. 跑前检查（每次）

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

若有自己的僵尸进程（同事的不动）：

```bash
pkill -9 -f icn_proto
sleep 3
```

## 4. 单格冒烟（~13 分钟，每次矩阵前必跑）

ours @ λ=2.0, s=1.0（机会最多、最可能开火的配置）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.run_cluster --policy ours --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 48 --seed 0
```

跑完读结果（自动抓最新 JSON）：

```bash
python -m icn_proto.e1_report
```

**冒烟通过标准**（三条全满足才铺矩阵）：
1. `failed: 0`
2. `session_turn_migration_rate` 显著 > 0（前三次实测 0.33–0.52）
3. `remote_resume_opportunities` > 50（前三次实测 106–166）

`repl_planned` / `replications` 预期是 0——**这就是 E1 阴性结果本身**，
不是故障。关键看 `repl_reject`：
- `g_best`（离 0 最近的被拒 G）：负得深 ⇒ 阴性决断；接近 0 ⇒ 差之毫厘；
- `no_missing` 高 ⇒ b3 的 demand fetch 抢跑（机制第 1 层）；
- `no_holder` 回潮 ⇒ 同步丢了代码。

## 4b. 预算探测格（~13 分钟，48MB 饥饿 regime 的对照）

48MB 是 06 时代按 3-turn session 定的；E1 的 40-turn session 工作集
~912MB，4×48MB 常驻把系统压进饥饿 regime（transfers≈0、96% 重算、
reactive/proactive 一起失效）。用 256MB/卡（仍只有工作集 ~1/4，稀缺
性保住）验证阴性是否预算的人造产物：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.run_cluster --policy ours --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 256 --seed 0
python -m icn_proto.e1_report
```

读数重点：`transfers`（holder 可用性是否恢复）、`hot` 与 `g_nonpos`
（session tip 的需求信号是否活过来）、`repl_planned`（经济门过线后
控制器是否终于有活干）。三档结局：仍 0 ⇒ 饥饿不是主因，阴性更硬；
开火 ⇒ 矩阵要加预算轴；中间态（fetch 恢复但 repl 仍 0）⇒ 抢跑机制
仍是主因，按阴性走。

## 5. 确认矩阵（4 配置 × {b3, ours} × 2 reps = 16 格，~3.5 小时）

**逐条执行**，一条跑完再跑下一条（manifest 断点续跑，中断后重跑同一条即可）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e1.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e1.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 1.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e1.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 1.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e1.json
```

每格约 13 分钟。最后一条跑完会打印聚合表（wl / rps / hit / new_tok /
xfer / repl / evict / p50 / p95 / SLO）。

## 6. 常见异常处置

| 现象 | 处置 |
|---|---|
| 某格 `BAD` / 超时 | `python -m icn_proto.matrix_step5 ... --drop-bad --manifest results/icn_proto/matrix_e1.json`（清坏格），然后重跑那一条命令 |
| worker 起不来 / hello 超时 | 有孤儿进程：`pkill -9 -f icn_proto`，sleep 3，重跑 |
| 想看重跑某格的完整日志 | `results/icn_proto/cell_logs/cell_<wl>_s2_<pol>_r<rep>_b48_p0.log` |
| 聚合表 ours 行缺数据 | 该格 JSON 没产出，按 BAD 处置 |
| 想换 reps / 补跑 | 同一条命令改 `--reps`，manifest 键含 rep 序号，只补新格 |

## 7. 判决口径（07 v2 预注册，跑完对着读）

**主判据**：ours−b3 的 `new_tok` 差随 λ 收窄/翻正，且翻正主要由
`remote_resume_served_local_due_to_replication` 解释。

**预期（结构性阴性）**：ours ≡ b3（`repl_planned=0`），因为
① b3 demand fetch 抢跑（`no_missing` 主导）② 剩余缺口 per-target λ̂ 低于
break-even（`g_nonpos`）。若三配置×两 reps 全部如此，E1 阴性成立，
机制链由 `repl_reject` 直方图支撑，进入 E2（host-DRAM backing tier）。

**翻正的处置**：若任一配置 `repl_served_local > 0` 且 new_tok 差收窄，
先查 `no_holder` 与 g_best 边际，再决定是否扩大矩阵——不回头调控制器。
