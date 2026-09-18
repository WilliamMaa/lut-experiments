# ICN KV Allocation — 真机 Prototype 实现计划（v2）

> 配套文档：`00-ideas.md`（方向与问题定义）、`01-simulator-implementation-plan.md`（已取代的纯模拟方案，策略/指标/数字锚点仍适用）。
> 状态：**待评审**，未开始编码。
> 与 v1 模拟方案的核心差异：**不做离散事件模拟，直接在有 GPU 的远程机器上搭一个多 worker 的 serving prototype**，routing / allocation / KV 传输全部真实发生，系统指标全部实测。

## 1. 目标与不变的核心问题

问题定义不变（`00-ideas.md` §11）：

> KV block 拥有 location-independent name，allocation 同时考虑 KV locality、GPU load、transfer cost 时，是否优于 least-loaded / cache-affinity？

主攻点不变（v8 特有）：**representation-aware placement**——近期前缀保 BF16、远期前缀降级到 m_sp4/k8v8，allocation 同时决定 "where + which representation"。

数字锚点不变（引自 `01` §3，出处见该文档）：bf16 = 20 KiB/token；m_sp4 = 2.5 MiB/请求固定；k8v8 ≈ 1.3 MiB/请求；质量 delta：m_sp4 ≈ 0，k8v8 ≈ 0.019（`docs/19-final-summary.md` 实测）。

变化的是**测量方式**：延迟、吞吐、传输时间、命中率从"成本模型估算"变为"真实测量"。

## 2. 系统架构

```text
                 ┌──────────────────────────────────────┐
 trace replay →  │  Scheduler (CPU 进程)                 │
 (8 sessions,    │  - KVNameIndex: name → locations[]   │
  Poisson/think) │  - policy: P0 / P1 / P2               │
                 └───────┬───────────────┬───────────────┘
              控制消息(zmq)             控制消息(zmq)
        ┌───────┴───────┐       ┌───────┴───────┐
        │  Worker 0     │       │  Worker 1     │     ... ×W
        │  model shard  │◄────►│  model shard  │
        │  resident KV  │ KV传输 │  resident KV  │
        │  (HBM)        │(host或 │  (HBM)        │
        └───────────────┘ NVLink)└───────────────┘
```

- **Scheduler**：单 CPU 进程。接收请求流（trace 回放），查 KVNameIndex，按 policy 选 worker，下发 prefill / 传输 / decode 指令，回收指标。本身不做重活，所有决策延迟可忽略。
- **Worker**：每 worker 一份模型副本（多卡放置，固定确定性方案，见 §7-1），HBM 里驻留 named KV objects，同一时刻只处理一个请求（v1 不做 continuous batching——排队效应真实存在且实现简单）。
- **KV 传输**：worker 间搬 KV object 走**真实传输**。默认 host-mediated（sender `.cpu()` → zmq → receiver `.to(gpu)`），这同时也是设计里的 Tier-2，且绕开 NCCL 组管理的工程复杂度；同节点带宽实测后再决定是否加 GPU-direct 路径。
- **通信**：控制面 pyzmq（localhost IPC，足够）；不复用 v8 评测管线的 `common/evaluator.py`（那是串行单请求），但**复用** `kv_cache/` 的缓存实现来产出真实 representation。

## 3. KV Object 与命名（真实实现，不是模拟）

```text
name = /session/<doc_id>/turn/<t>/span/<start>-<end>/repr/<bf16|m_sp4|k8v8>
payload = 10 个 full-attn 层的 tensor 列表（per-layer [1, 2, T, 256]）
```

- **bf16 object**：prefill 后从 `DynamicCache` 直接取的真实全量 KV。
- **m_sp4 / k8v8 object**：worker 用 `kv_cache/heavy_hitter_cache.py` 的 `HeavyHitterCache`（budget 128，sink4+recent32+hh92，`merge_evicted` + `span_window 4`）真实跑 prefill 得到的紧凑驻留态；k8v8 叠加 `kv_cache/kivi_cache.py` 的 INT8 存储。这正是 v8 评测里 1000x/2000x 的那两个东西，搬运它们 = 搬运真实压缩结果。
- 一个 session 的 prefix chain = [指令前缀(全 session 共享), 文档前缀, turn1 增量, ..., turnN-1 增量]，链上每个 span 是一个 object，可各自有不同的 representation——representation 混布是天然支持的。

## 4. Policies（与 `01` §4 一致，落到真机语义）

| Policy | 路由规则 | KV 行为 |
|---|---|---|
| P0 least-loaded | 队列最短（空闲）的 worker | 无共享：prefix 全部本地 recompute，驻留 bf16 |
| P1 cache-affinity | prefix 命中 object 数最多的 worker（平手按负载） | 命中部分直接复用；未命中 recompute；驻留 bf16 |
| P2 ICN | 每候选 worker 计算 C_j = 排队 + 传输 + 重算 + 显存压力 + 质量惩罚，argmin | 额外做 representation 决策：recent_window（可配，默认 8K token）内保 bf16，窗外新 object 按目标 worker 显存压力生成 m_sp4 / k8v8；跨 worker 命中时比较"搬 object" vs "recompute" 的真实成本 |

- 质量惩罚（系统实验内）：请求命中降级 object 的 prefix token 比例 × 该 representation 的 EOS delta（标量，实测值）。
- 采样轮次另跑哨兵题探针（复用 `kv_cache/probe_sentinel.py` 思路）验证压缩表示下的真实质量，不放主循环里拖慢吞吐测量。

## 5. Workload 与实验矩阵

- **Trace**：`data/multi_turn_prompts_v3.jsonl`，8 sessions × 每 session 顺序 turns。turn 间到达间隔 = think-time 分布（对数正态，可配）。
- **每请求**：prefix = 文档 + 前序轮次（真实 tokenized 长度）；生成长度固定 cap（如 128 tokens）以控制变量；记录请求级延迟分解（queue / prefill / transfer / decode）。
- **矩阵**：3 policies × 3 seeds × 2 到达率 = 18 runs（W=2）。预估单 run = 424 turns ÷ 2 workers × 每 turn 数秒 ≈ 40–80 分钟，全程约两个窗口/一个周末。比模拟贵得多，所以参数敏感性靠**少量关键扫描**（到达率、recent_window），不做全网格。
- **指标**（全部实测）：KV hit rate、recompute tokens、transfer bytes/time、HBM 占用与 representation 分布、queueing delay、throughput、quality penalty 期望、哨兵题正确率。

## 6. 实施步骤

### Step 0 — 环境侦察（你跑命令，我来填表）

远程机器信息目前文档里完全没有记录。请先跑：

```bash
nvidia-smi
free -g; df -h /data
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import zmq; print('zmq ok')"   # 没有就 pip install pyzmq
ls /home/u/downloads/models/Qwen3.6-35B-A3B
```

产出：硬件表（2026-09-17 实测）：

| 项 | 值 |
|---|---|
| GPU | 8× NVIDIA A800-SXM4-80GB（CUDA 12.4，driver 550.90.07） |
| 机器状态 | **共享机，非独占**：GPU 0-3 被一套 vLLM（TP4，~71GB/卡）占用，GPU 4 被 VLLM::EngineCore（72GB）占用，GPU 6-7 被另一套 vLLM（TP2，~75GB/卡）占用；**仅 GPU 5 空闲** |
| RAM | 2015 GB（available 1787 GB）；/data 余量 9.4T |
| 软件 | torch 2.6.0+cu124，transformers 5.14.1，zmq 可用，conda env `lut_py310`（Python 3.10） |
| 模型 | `/home/u/downloads/models/Qwen3.6-35B-A3B`（26 shards，文件齐全；含 VL preprocessor，v8 管线已适配） |

**Worker 拓扑决策（已拍板：35B bf16，W=2）**：35B bf16 ≈ 70GB 权重，A800-80GB 单卡放不下权重+KV → **每 worker = 2 卡**（`balanced_low_0` 固定到这 2 卡）。矩阵阶段 **W=2 worker（共 4 卡）**，窗口需求从 8 卡降到 4 卡，协调容易很多。代价：负载均衡/排队维度只有 2-way，比 W=4 弱；缓解：Step 5 用 session 副本（trace 重放 2×）拉高到达负载来补偿并发压力。当前机器被占用时，Step 1-2 用小模型（GPU 5 单卡）做机制验证，窗口内再上 35B。

### Step 1 — KV object 层（`icn_proto/kvname.py` + `icn_proto/kvcodec.py`）

- KVName 命名/哈希；bf16/m_sp4/k8v8 三种 payload 的打包/解包（serialize → bytes → deserialize，含 crc 校验）。
- 从 `DynamicCache` / `HeavyHitterCache` 提取 payload、注入还原的适配函数。
- **验收**：单机单卡上，对一个 session 连续两轮 prefill，第二轮从第一轮注入的 m_sp4 object 恢复，logit 差异在 v8 已公布的容差内（对照 `kv_cache/inspect_kernel_ab.py` 的验证口径）。

### Step 2 — Scheduler + Worker 骨架（`icn_proto/scheduler.py`、`worker.py`、`msg.py`）

- zmq 控制协议（JSON 消息）：assign / transfer_push / transfer_pull / finish / metrics。
- Worker 状态机：IDLE → PREFILL → (TRANSFER) → DECODE → report。
- **验收**：1 scheduler + 2 worker（单卡加载即可）跑通 1 个 session 的 3 个 turn，日志里能看到 KVNameIndex 增长、turn2 命中 turn1 的 object、真实 transfer 发生。

### Step 3 — P0 / P1 上真机 + 指标管线

- 三 policy 里的前两个先跑通全 trace，metrics 汇总成 run JSON（文件名带 policy/config/日期，不覆盖）。
- **验收**：P1 hit rate > P0；P1 单 turn 延迟均值 < P0（复用命中收益）；指标分解合理（transfer 时间占比小）。

### Step 4 — P2 ICN allocator + representation 选择

- 实现 §4 P2 行；recent_window 与降级阈值可配。
- **验收**：日志可见 transfer vs recompute 的策略切换；P2 的 HBM 占用显著低于 P1（representation 混布生效）；质量惩罚期望 < k8v8 delta 上限。

### Step 5 — 18-run 矩阵 + 对比报告

- 按 §5 跑完，出对比表，回答：`a.` P2 是否 Pareto 优于 P1；`b.` 降级的内存收益是否值得质量代价；`c.` 收益最大的负载区间。
- 结论写回 `docs/icn-defined-addressing/`（新文档 03），更新 AGENTS.md 当前阶段。

## 7. 决策记录与核对

1. **模型放置（已拍板）**：单卡放不下就用多卡，红线的要点是**显式定义、不允许未定义的自动行为**，不是禁多卡。沿用 v8 现行做法 `device_map="balanced_low_0"`（固定、确定性的多卡放置，非 auto），每个 worker 绑定一组固定的卡。35B bf16 ≈ 70GB 权重，Step 0 确认 GPU 型号/数量后定 W（worker 数）；若卡是 80GB 则每 worker 占 2 卡，若 H200-141GB 可尝试单卡一 worker。任何情况下不写 `device_map="auto"`、不引入 accelerate 自动切片。
2. **think-time 分布**：无真实数据，先用对数正态（中位 20s），Step 5 做敏感性扫描。
3. **并发度**：v1 每 worker 串行。之后若要拉负载，优先加 session 副本（trace 重放 2×）而不是上 continuous batching。
4. **与主线的关系**：这是系统方向实验，不改 `kv_cache/` 现有评测结论；所有新代码隔离在 `icn_proto/`，质量数字仍引用 `docs/19-final-summary.md`，不重测。
5. **GPU 资源（Step 0 发现，已拍板）**：机器为共享机，2026-09-17 时点 8 卡中 7 卡被 vLLM 实例占用，仅 GPU 5 空闲。**已与卡主确认可协调出空闲窗口**（如夜间/周末）。模型与占用拍板：**不换模型、不量化，35B bf16，矩阵阶段 W=2（共 4 卡）**。执行模式：现在就把全部代码写完；Step 1-2 的正确性验证用小模型（Qwen3 小尺寸，GPU 5 单卡够）先做机制验证，35B 验证和 Step 3 起的矩阵放到协调好的 4 卡窗口里跑。

## 8. 风险

- **工程量显著大于模拟方案**（估计 2 周量级）：分布式进程、真实传输、缓存适配三层都要写。缓解：每步有独立验收，Step 2 之前不碰模型多卡。
- **真实实验一次只能测一个配置**：矩阵规模受限，结论以趋势为主，不报绝对数字。
- **传输实现是简化项**（host-mediated）：可能高估跨 worker 搬运成本；若 Step 3 发现 transfer 时间主导决策，再补 GPU-direct。
