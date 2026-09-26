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
- **2026-09-25 二次修正（公平性，重要）**：`controller()` 入口的
  policy 门原来把**驱逐**也闸掉了——驱逐只在 ours 下运行，b3 实际
  享有无限 residency（b3 冒烟实测 resident_bytes 800MB vs 预算
  48MB），违反 AGENTS 红线 2 的同等预算原则，且该红利渗入了 E1 的
  全部 b3 基线。修复：驱逐对全策略运行（`_plan_evict` 注释与
  10 §3.2"策略无关共享基底"的本意），policy 门只留 `_plan_repl`。
  **影响**：E1 的 b3 格保留为历史运行（其 hit 0.976 含"永不驱逐"
  红利，写 12 号结果文档时必须标注）；E2 的所有 b3 格都在修复后的
  共享预算下重跑（§5 新增 b3@s0 两条重基线命令）。ours 的 E1/E2
  格不受影响（ours 本来就走驱逐路径）。

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

### 4b. b3 @ spill∞（共享基底下的对照——修 gate 后重跑）

**注意**：`controller()` 驱逐门修复后，b3 也遵守 48MB 预算、被驱逐
的块进 tier、recall 生效——**这正是 P1 该有的样子**：两策略同预算、
同 tier，只差 proactive 复制。旧语义下的 b3 冒烟
（20260925_153335）是最后一格"无限 residency"运行，仅作历史存档。

```bash
python -m icn_proto.run_cluster --policy b3 --sessions 8 \
  --turns-per-session 40 --q-tokens 40 --doc-chars 4000 \
  --doc-repeat 2 --doc-repeat-alt 2 --port 5671 --gpu-pool 0,1,2,3 \
  --gpus-per-worker 2 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 \
  --think-s 2.0 --worker-mem-budget-mb 48 --spill-mb -1 --seed 0
```

**P1 判读（修复后语义）**：b3 与 ours 在**计数质量指标上应同处一个
区间**——hit 都 ~0.97、new_tok 都 ~2.2–2.4 万、rederivation 都千
级（tier 吸收驱逐损失，两个策略都受益）；差别只在 `repl`（b3 恒 0）
和各自 spill 口径的量。b3 的 `evicted_bytes`/`recall_count` 应 > 0
（tier 是共享的，b3 也在用）、`resident_bytes` 应 ~48MB。若 b3 质量
指标大幅偏离 ours 或预算又失控，才是泄漏/新 bug，贴日志回来。

每格跑完读数：

```bash
python -m icn_proto.e1_report
```

**冒烟通过标准**（全满足才铺矩阵）：
1. 两格均 `failed: 0`；
2. ours 格的 `spill` 块：`spill_bytes` > 0 且某 worker 的
   `recall_count` > 0（tier 真的在收块、真的在召回）；
3. **P1（修复后语义）**：b3 格与 ours 格计数质量指标同区间——hit
   都 ~0.97、new_tok 都 2.2–2.4 万、rederivation 都千级；差别只
   在 `repl`（b3 恒 0）。b3 的 spill 口径 > 0（tier 共享，b3 也
   在收块召回）、`resident_bytes` ~48MB。若 b3 质量大幅偏离或
   预算失控 = 泄漏，先修再判。（旧语义 b3 基线 hit 0.976 /
   new_tok 22,188 含无限 residency 红利，只作存档，见 §1。）
4. ours 格的 `rederivation_tokens` 应明显低于 E1 同配置 ours 基线
   （103,141 new_tok 那格的重算部分）——tier 接住被驱逐的 state
   的直接证据。
5. **两格 e1_report 的 worker 行 `resident_bytes` 都应回到 ~48MB
   量级**（HBM 预算内）。若冲到数百 MB，说明驱逐保护被 spilled
   tip 撑住，驱逐基底停摆—— regime 无效，停止，贴输出回来
   （首冒烟 20260925 就栽在这，见 §4c）。

latency 类指标照旧只作参考（邻居噪声）。

### §4c 首冒烟记录（2026-09-25，blkcluster_s8t40_20260925_095834.json，ours@spill∞）

**作废重跑格**。P2 ①② 强烈向好：`rederivation_tokens` 仅 1,722
（vs E1 同配置 ours 灾难格）、hit 0.973 ≈ b3 基线 0.976、
new_tok 23,910 vs b3 22,188 = 1.08×（≤1.2× 以内）、
recall_count 629/821、`fetch_avoided_by_recall` 20。

**但 worker resident_bytes 669/681MB，预算 48MB——稀缺性 premise
失效，整格作废**。根因：worker 把 spilled tip 报进 `tips`（为
match 可见性，这部分对），而 `_plan_evict` 保护集取 `w.tips` 全
部 → 197 个 spilled tip 的链把 resident 块全"保护"住，驱逐只清出
1,837 块。修复（违反 10 §3.2"_plan_evict 不变"，已对齐）：
① 保护与驱逐 Universe 只算 **resident tip**（spilled tip 留在
w.tips 供 match/choose，但不保护链）；
② 新增 **orphan 清扫**：不被任何 resident tip 链覆盖的 resident
块（spilled 段的残留、状态滞后孤儿）按 λ 最冷先行驱逐——pre-E2
该不变量成立时为空操作，tier 打破它后必须有这个清扫。
回归测试 `test_evict_spilled_tip_chain_is_swept`。

重跑 §4a 时期望：resident_bytes ≈ 45–50MB，spill 口径不变，
rederivation 保持千级。

## 5. 确认矩阵（2 spill 档 × 2 偏斜 × {b3, ours} × 2 reps = 16 格
## + b3@s0 重基线 4 格 = 20 格，~4.5 小时）

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

### §5a b3@s0 重基线（驱逐门修复后必须重跑，4 格）

E1 的 b3@s0 格是在"驱逐被闸"的代码下跑的（无限 residency 红利），
不能再用。ours@s0 的 E1 格不受影响（ours 本来就走驱逐路径），s0
对比组 = E1 ours 格 + 下面重跑的 E3 b3 格：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3 --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.0 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e2.json
```

```bash
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3 --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --manifest results/icn_proto/matrix_e2.json
```

预期：b3@s0 进入饥饿 regime（与 E1 ours@s0 同构：transfers→0、
rederivation 巨大、hit 塌）——这是 s0→s∞ 的 tier 效果对照里 b3
一侧的必要基线。

### §5b 结果记录（跑完贴聚合表到这里）

（待填）

### §5c sp96 首跑事故（2026-09-25，已修复，重跑指令在此）

**现象**：§5 命令 4（sp96 × s=1.6）四格全部 rc=1、failed=3/8/5/8，
错误一律 `RuntimeError: resume block not resident`（b3 和 ours 都炸）。
**注意：出事的是 sp96 档，不是 sp-1**——`dropped_bytes` 0.3–1.9GB/格
只可能在有限容量档出现（spill∞ 的 cap 是 inf，LRU 永不触发）。

**根因**（两个叠加）：
1. `_spill_put` 的 LRU 级联丢弃对 scheduler 不可见——status 虽然带全
   量 `spilled` 列表，但主循环每轮 dispatch 前只 recv 一条消息：
   worker 处理 evict（级联丢块）→ 发 status 排队 → scheduler 下一轮
   先 dispatch（用旧视图做 placement 决策）→ 才 recv status。
   决策窗口内 `w.spilled` 含已被丢的块 → assign 的 resume 集超出
   worker 实际持有 → `run_turn` recall 缺失 → raise。
2. worker raise 时只报第一个缺失块、scheduler 也没有重试路径，
   一次视图竞争直接变成 failed turn。

**修复**（提交见 git log，回归测试 `test_stale_resume_retry`）：
- scheduler 主循环 dispatch **前** drain 全部排队消息（`sock.poll(0)`），
  消除决策窗口的视图滞后；
- worker `run_turn` 收集**全部**缺失块，raise `STALE_RESUME: <名单>`；
- scheduler result 处理识别该错误：从 `w.resident/spilled/tips/via_repl`
  清除缺失块（worker _result 后的 status 尚未处理，必须主动清）、
  删除失败 record、turn 重新入队重决策（最多一次；第二次视为真
  不变量破坏，硬失败）；
- summary spill 块新增 `stale_resume_retries`（健康格应 ≈ 0；
  >0 说明仍有视图竞争，drain 后应极少）。

**重跑指令**（按顺序）：

```bash
# 0. 远程同步修复
cd ~/lut-experiments/LLM_LUT/v8 && git pull

# 1. 冒烟回归（§4a + §4b，各 ~13 分钟；通过标准照旧）
#    ——修的是 dispatch 主循环，必须先确认 E2 主面没退化

# 2. 清掉 4 格坏数据（failed>0 的行会被 --drop-bad 删掉）
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb 96 --manifest results/icn_proto/matrix_e2.json --drop-bad

# 3. 重跑 §5 命令 4（sp96 × s=1.6；manifest 断点续跑，已完成格自动跳过）
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 8 --turns-per-session 40 --q-tokens 40 \
  --shares 2 --policies b3,ours --reps 2 --budget-mb 48 --cell-timeout 1800 \
  --arrival poisson --arrival-rate 2.0 --zipf-n 16 --zipf-s 1.6 --think-s 2.0 \
  --spill-mb 96 --manifest results/icn_proto/matrix_e2.json
```

**有效性确认**（命令 3 跑完逐格查）：

```bash
python -m icn_proto.e1_report    # failed 必须全 0
python -c "
import json
d = json.load(open(sorted(__import__('glob').glob('results/icn_proto/blkcluster_s8t40_*.json'), key=__import__('os').path.getmtime)[-1]))
print('stale_resume_retries:', d['spill'].get('stale_resume_retries'), 'failed:', d['failed'])"
```

判读：`stale_resume_retries` 可以 >0（重试在工作，turn 不再失败），
但 `failed` 必须为 0；若某格 failed>0 且报错含 `STALE_RESUME`，说明
同一 turn 连续两次视图竞争——停下来贴日志，别再重跑。

**已完成格的有效性**：命令 1（sp-1 × s=1.0，20260925 绿）跑在修复前
代码。该竞争只在 spill 丢块时出现（spill∞ 无丢块），且 drain/重试
不改变 sp-1 格的决策语义，**格数据保留有效**。若之后发现
`stale_resume_retries` 在 sp-1 格也频繁 >0，再回来重跑命令 1–2。

## 6. 常见异常处置

| 现象 | 处置 |
|---|---|
| 某格 `BAD` / 超时 | `python -m icn_proto.matrix_step5 --manifest results/icn_proto/matrix_e2.json --drop-bad`（清坏格），然后重跑那一条命令 |
| worker 起不来 / hello 超时 | 有孤儿进程：`pkill -9 -f icn_proto`，sleep 3，重跑 |
| 想看重跑某格的完整日志 | `results/icn_proto/cell_logs/cell_<wl>_s2_<pol>_r<rep>_b48_p0.log`（wl 含 `_sp-1`/`_sp96`） |
| 聚合表 spill 列全 0 | tier 没生效：确认命令带 `--spill-mb`、§2 五测试全过；再查该格日志里 `recalled` / `evicted ... spill +` 行 |
| 96MB 档 `dropped_bytes` 恒 0 | LRU 没触发：tier 没满过，s=1.6 格应出现第二级 churn，若无则 P3 需要更大压力（记录即可，不回退） |
| 格内 failed>0，错误含 `resume block not resident` / `STALE_RESUME` | 调度器 spilled 视图滞后（sp96 事故，§5c）：确认 `git pull` 拿到修复 → §5c 重跑指令；修复后仍出现则贴日志 |
| b3 的 spill 格质量指标大幅偏离 ours 同档 | 泄漏或新 bug，停止矩阵，贴日志回来修 |

## 7. 判决口径（10 §4 预注册，跑完对着读）

**P1（对照纯净性，修复后语义）**：b3 与 ours 同预算同 tier，计数
质量指标同区间（hit ~0.97、new_tok 2.2–2.4 万、rederivation 千
级），b3 的 spill 口径 > 0（tier 共享）、`repl` 恒 0。若 b3 质量
大幅偏离或预算失控 = 泄漏，先修再判。
（历史注记：驱逐门修复前的 b3 基线 hit 0.976 含无限 residency
红利，只作存档不作对照——§5a 重跑后的 b3@s0 才是合格基线。）

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
