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

**P1.1 结果（2026-10-08，已跑）**：graph capture 把 55ms 砍到 **11.2ms**
（dense 的 **208×**）。55ms 的构成被拆开：**~44ms 是纯 dispatch**，剩 ~11ms 是
~2600 个小 kernel 的图内执行地板（每 kernel ~4µs 时长地板）。→ 触发 P1.2。
推论：每 token 实际计算仅 ~2k FLOPs / ~8KB 访存，11ms 纯粹是 kernel 数量问题，
融合成个位数 kernel 后有 realistic 路径压到百 µs 以内、甚至接近 dense（53µs）。

**注意**：graph 列没算输入拷贝（[N,2048] bf16 拷贝 ~µs 级，忽略）；prefill N=8192
一行仅供参考，decode 尺寸才是判据。

## P1.2 — Triton fused kernel（已开工，待真机数字）

**设计（已实现，在 microbench 内）**：3 个 kernel——
`_k_coarse`（grid (N,)，14 级树遍历 + 2048 宽行 gather）、`_k_resid`
（grid (N,32)，16 级遍历 + 64 宽切片 gather，写 [N,2048] 的组切片）、
`_k_add`（grid (N,)，fp32 加和转 bf16）。eager 两列 `triton ms` /
`t_graph ms`（graph capture 后）+ 对 flat 的数值校验（atol 0.05）。

```bash
python v6/scripts/benchmarks/micro_lut_vs_gemm.py \
    --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
```

**P1.2 结果（2026-10-08，已跑）**：数值校验全 OK（max|Δ|=0.016 < 0.05）。
**t_graph/dense decode 尺寸 0.11–0.39×（比 dense 快 2.5–9 倍）**，eager triton
已与 dense 打平（~0.05ms）。N=8192 prefill 4.7× dense（gather 随机读在
大 N 下效率下降），serving 时 prefill 可保持 GEMM 或后续优化，不挡 decode 结论。
→ **P1.2 成功，按判据进 P1.3 插件集成。**

真机首跑候选报错点（本机无法验证）：triton 标量 loop-carried 变量
（`node` 从 python int 0 起被 tensor 重赋值）在 triton 3.x 合法，但旧版
编译器可能报 type unification 错——报错就把 `node = 0` 改成
`node = tl.zeros((), dtype=tl.int32)`（或升级 triton）。

## P1.3 — vLLM 插件集成（已实现 2026-10-08k，待真机验证）

实现要点（与硬要求的对应）：
1. **替换 module forward**：包装 `Qwen3NextSparseMoeBlock.__init__`，实例属性遮蔽
   `shared_expert.forward` 为 triton 查表，原 GEMM 不执行（满足硬要求 1）。
2. **TP 正确性**：`SharedFusedMoE` 对 shared 输出做 all-reduce，故只有 tp_rank 0
   返回 LUT 结果、其余 rank 返回 zeros（all-reduce 后恰为单次结果）。
3. **显存**：表 tensor 在构造期 `register_buffer(persistent=False)` 挂到
   shared_expert 实例，随 .to(device) 移动、profile_run 前计入 non-KV memory。
   已知冗余：replacer 惰性 _to(device) 产生第二份副本（~640MiB/层而非 320MiB），
   P1.4 若 HBM 敏感再去重。
4. **配置**：环境变量 `V8_LUT_LAYERS`（"39" / "37,38,39"，空=关闭）、
   `V8_LUT_BUNDLE_DIR`。**无 CLI 参数**（对齐插件现有风格；runbook 早先的
   `--v8-lut-layers` 字样作废）。
5. **版本 2026-10-08k**：config.py / repro_concurrency.sh 注释 / docs/28 §8.2
   三处已同步；本机 py_compile + test_ast_names + test_forbidden + bash -n 全过。

新增/改动文件：`vllm_plugin/lut_ffn.py`（3 kernel：`_k_leaf` 树遍历（不动点
自环处理早停树，coarse T=1 / residual T=32 复用）、`_k_gather_add` coarse+residual
fp32 加和）、`vllm_plugin/tests/test_lut_ffn.py`（数值对照）、
`v6/scripts/conversion/prep_lut_bundle.py`（checkpoint 目录 → 单文件 bundle，
 pickled 树 → padded 自环张量）、`config.py`、`__init__.py`。

真机验证序列（按序，每步过了再走下一步）：

```bash
# 0. 确认 checkpoint 在（P1.3 前置，一直没人确认过）
ls -d ~/lut-experiments/LLM_LUT/v6/outputs_ffn_lut_layer3*/checkpoints 2>/dev/null

# 1. 打包 L39（lut_py310）
python v6/scripts/conversion/prep_lut_bundle.py \
  --checkpoint_dir ~/lut-experiments/LLM_LUT/v6/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --output /data/mamingyu/lut_bundles/layer39.pt

# 2. 数值测试（kernel vs torch reference，vllm_py310）
python vllm_plugin/tests/test_lut_ffn.py

# 3. 带 LUT 起 server（vllm_py310），grep 确认 "LUT shared_expert layer 39 installed"
V8_LUT_LAYERS=39 V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles \
python -m vllm_plugin.serve /home/u/downloads/models/Qwen3.6-35B-A3B \
  --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
  --tensor-parallel-size 2 --max-num-seqs 4 --gpu-memory-utilization 0.88 \
  --port 18002 2>&1 | tee logs/lut_l39.log | grep -E "LUT shared_expert|KV cache size"

# 4. 冒烟：同 repro_concurrency 的 2-doc 小 bench + fact_acc，与 v8-only 对比
```

注意：checkpoint 目录名以步骤 0 的实际输出为准（`_onpolicy_as_v4/checkpoints`）。

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
