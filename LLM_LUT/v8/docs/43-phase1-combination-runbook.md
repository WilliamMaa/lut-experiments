# 43 - Phase 1 runbook：v8 + LUT 组合假设（minimum viable replacement）

> 状态：已批准开工（2026-10-08）。依据：`v8/docs/42-reflection.md`（三阶段计划，
> runbook 15 已降级延后）。核心问题只有一个：
>
> **v8 释放的 memory headroom，能否投资到 LUT compute 替换上，让 v8+LUT 系统性地
> 优于 v8-only？**
>
> 不追 40 层，不碰 routed，不追最大 coverage。先证明组合假设有生命力。

## 0. 阶段划分（来自 docs/42）

| Phase | 内容 | 本文档 |
|---|---|---|
| **Phase 1** | 现有 3 层 v6 资产 + GPU-native LUT + v8 → 组合假设 | 本文件 |
| Phase 2 | 测 teacher rollout `t(k)`（0/1/2/3 LUT），定 on-policy 扩展成本 | 待 Phase 1 出信号后写 |
| Phase 3 | 有收益才扩 coverage（3→5→8…），每步算 Δthroughput/Δtraining cost | 待 Phase 2 |

## P1.1 — CUDA Graph microbench（小时级，当天出答案）

**假设**：`naive ≈ flat ≈ 55ms` 表明瓶颈是 kernel-launch dispatch 地板，不是计算。
若成立，把 flat 路径整图 capture 后应收薄到亚毫秒级。

**命令**（remote，`vllm_py310`）：

```bash
python v6/scripts/benchmarks/micro_lut_vs_gemm.py \
    --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
```

输出新增 `graph ms` 和 `graph/dense` 两列（脚本已改，含 P1.1 判据打印）。

**判据**（decode 尺寸 N≤64 的 worst graph/dense）：

| graph/dense | 结论 | 下一步 |
|---|---|---|
| ≤ 1.0 | LUT 执行本身不输 GEMM，组合假设的执行前提成立 | 直接 P1.3 |
| 1.0–2.0 | production 边际，但 teacher 成本已砍两个数量级 | P1.2 尝试 triton；P1.3 照跑拿真实系统数 |
| > 2.0 | 时间真在 kernel 本体（memory-bound gather） | 必须 P1.2 triton；triton 也 >2× 则 Phase 1 以"组合假设未证实"结案 |

**注意**：graph 列没算输入拷贝（[N,2048] bf16 拷贝 ~µs 级，忽略）；prefill N=8192
一行仅供参考，decode 尺寸才是判据。

## P1.2 — Triton fused kernel（条件触发，天级）

触发条件：P1.1 graph/dense > 1.0。目标：单 kernel 完成"树遍历 + 双级 gather"，
消除 per-level temporaries 和随机读的有效带宽损失。decode N≤64 优先，prefill
可走 eager/graph 混合。成功线：graph/dense ≤ 1.0；若 >2× 则全路线停，写结案文档。

设计要点（开工时再细化）：14 级 coarse 遍历 + 32×16 级 group 遍历全部 in-kernel；
表常驻 HBM（coarse 64MiB + residual 256MiB/层，vLLM 集成时计入显存预算）。

## P1.3 — vLLM 插件集成（天级）

把 v6 引擎的 LUT 查表逻辑移植进 `vllm_plugin` 的 Qwen3_5Moe patch 路径。

**硬要求**：
1. **替换 module forward，不是 hook 后置覆盖**。v6 HF 引擎的 hook 是先算完原 FFN
   再覆盖（MAC 节省恒 0，docs/40 记录在案）；在 vLLM 里必须直接改写
   `shared_expert.forward` 为查表，原 GEMM 不执行，compute 才算真被替掉。
2. 层：L37–39（现有 `_as_v4/checkpoints`，步骤 0 已确认远端数据在；checkpoint
   确认命令见下）。先 1 层（L39）再 3 层。
3. 显式单卡/显式 TP，**禁 device_map="auto"**（AGENTS.md 红线 5）。
4. CUDA graph 兼容：先用 `--enforce-eager` 验证正确性，再试 vLLM 全图模式；
   若 capture 失败，LUT 层退回 eager（P1.4 两种模式都报数）。
5. 升版三处同步：`vllm_plugin/config.py` PLUGIN_VERSION、`tools/repro_concurrency.sh`
   must-say 行、`docs/28-vllm-plugin-spec.md` §8.2。

**checkpoint 确认**（P1.3 前置）：

```bash
ls -d ~/lut-experiments/LLM_LUT/v6/outputs_ffn_lut_layer3*/checkpoints 2>/dev/null
```

## P1.4 — 系统对比（天级）

四个配置跑同一 bench（复用 docs/38 口径，数字直接可比）：

```text
full           （无 v8 无 LUT）
v8-only        （= docs/38 的 v8-1024，118.4 sess/h @N=16 基准）
v8 + LUT×1     （L39）
v8 + LUT×3     （L37–39）
```

**命令骨架**（每配置：起 server → capacity bench → fact_acc → 收尾检查）：

```bash
# server（LUT 配置加 --v8-lut-layers 39 或 "37 38 39"，参数名以 P1.3 实现为准）
python -m vllm_plugin.serve /home/u/downloads/models/Qwen3.6-35B-A3B \
  --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
  --tensor-parallel-size 2 --max-num-seqs 16 --gpu-memory-utilization 0.88 \
  --v8-slots 1024 --port 18002

# bench（与 docs/38 同数据同并发档）
python tools/bench_concurrency.py --base-url http://localhost:18002 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data data/longctx_multi_turn_65536_64docs.jsonl \
  --currency 16 --max-docs 64 --out results/phase1_<cfg>_c16.json

# fact_acc（v8 协议红线 ≥0.734）
python tools/eval_longctx_server.py --base-url http://localhost:18002 \
  --data data/longctx_multi_turn_65536_64docs.jsonl --max-docs 16
```

**指标**：sess/h、P50/P95 时延、peak HBM（`nvidia-smi` 采样）、fact_acc、
OOM/preemption/rewind 计数（server 存活 + 0 contract raises 是硬门，同 06j）。

**判据**（来自 docs/42）：

- **v8+LUT×3 ≥ v8-only + 数个百分点** → 组合假设成立，启动 Phase 2（t(k) 测量）。
  不要求超过 full；方向正确 + 质量不崩即可。
- **v8+LUT ≈ v8-only（±噪声内）** → 看 HBM：若 LUT 表吃掉的显存没挤占 KV pool
  （capacity 数字不变），组合中性偏正，仍进 Phase 2。
- **v8+LUT 明显更差** → 诊断：慢在 LUT 执行（回 P1.1/P1.2 数字）还是慢在显存挤占
  （KV pool 缩小 → 看启动日志 GPU KV cache size）；写诊断报告再定。

**质量门**：v8+LUT×3 的 fact_acc 掉 >5pt 或触发 06j 不可用区间 → 停，回 H1
（on-policy 复验议题，不是直接判死）。

## 时间预算

| 步 | 预估 |
|---|---|
| P1.1 microbench | 小时级 |
| P1.2 triton（条件） | 1–2 天 |
| P1.3 插件集成 | 1–2 天 |
| P1.4 四配置 bench | ~1 天 |
| **合计（不含 P1.2）** | **~2–3 天** |

## 范围红线

- 不训新层（只用 L37–39 现有资产）。
- 不动 v8 的 KV allocator / attention 路径（06j 冻结，升版只做加法）。
- 不重开 HF 24 层扫描（runbook 15 冻结延后）。
- 不碰 routed experts。
