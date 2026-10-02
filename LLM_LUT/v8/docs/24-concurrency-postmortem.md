# v8 并发测试 Postmortem（结论与排查记录）

日期：2026-09-30
本文档只放**结论与机制记录**。怎么跑见 `docs/22-concurrency-runbook.md`，
完整结果表、统计检验与 Pareto 分析见 `docs/21-concurrent-serving.md`。

---

## 1. 显存墙机制（64k/128k 高并发 OOM 根因，已定案）

两条机制，memlog 实证（`SERVE_MEMLOG=1`，N=32/64k，chunk=1032）：

1. **resident 累积（泄漏型）**：分块 prefill 期间 `heavy_hitter_cache.py`
   的 update() 只追加不淘汰（deferred eviction 语义），满血 KV 无约束
   增长（N=32×60k token × 20KB/token ≈ 39.6GB，单卡攒 ~30GB——memlog
   上 gpu1 +0.10GiB/chunk、gpu5 +0.55GiB/chunk 两条斜率）。**128-slot
   的存储红利只在 decode 第一步后兑现，对 prefill 峰值毫无帮助**——
   这是 32k 矩阵"压缩配置峰值 HBM 反超 full"倒挂的根源。
2. **peak 棘轮（transient 型）**：HF `sdpa_attention_forward` 的
   `repeat_kv` 把 GQA 的 K/V 从 2 头 expand+reshape 拷贝成 16 头
   （2×B×16×K×256×2B，与 chunk 无关）：每 chunk 棘轮 ~0.52GiB/层，
   N=32/64k ≈ 28-31.7GB、N=64/64k ≈ 64GB。**full 的 N=64 OOM 也是
   同一堵墙**，不是稳态 KV。OOM 发生在 prefill 第 52/59 chunk
   （k≈54.7k，88%），N=16 能通过只是因为同样的棘轮减半。

**墙定律**（docs/21 发现 4）：两种路径的墙都近似随 B×K 缩放，但
compressed 常数约为 full 的一半（B×K@墙 ~1.0M vs ~2.0-2.4M）。
repeat_kv + 满血 KV 是**公共项**，解释不了这一倍差距；64k N=16 峰值
full 205.4GB vs m4_k8v8 279.4GB 的 **74GB 差额（prefill 期、两配置
同条件）仍未解释**——compression-specific 的 b·B·K 项（stash fp32 /
折叠簿记 / 量化中间张量）是最大嫌疑。
**docs/25 后降级为可选**：若 vLLM 路线走通（docs/26），这 74GB 属于
旧 harness 的成本结构，不再值得解释；只有走自写 engine 路线才需要
继续 profiling（runbook §12f）。

## 2. 已排除的假说（探针实证，torch 2.6.0+cu124）

- ❌ **"4D additive mask 把 sdpa 打到 math fallback 物化巨大 scores"**：
  `tools/probe_sdpa.py` 强制逐 backend 测 64k 下 N=8/16/32/64 精确形状，
  efficient 全部接受（B=64/64k peak 63.7GiB 也能跑）；math 才会 OOM 但
  不会被选中。
- ❌ **"chunk 太大"**：efficient 的 peak 构成是 repeat_kv 拷贝，与 chunk
  无关。曾据此加过 12e9 的 chunk cap，方向错误，已回退。
- ❌ **`enable_gqa=True`**：torch 2.6 efficient kernel 要求 dense 输入
  Q/K/V 同头数，带 mask 直接拒（flash 同样拒 mask）。
- ❌ **5-D stride-0 视图**（kernel 报错建议的 expand 路线）：B=8 形状
  全部 OOM（29.75GiB 单次分配 = fp32 scores 物化，静默落到 math 系
  backend）；小形状能跑但 `bitwise_equal=False`（max diff 2.4e-4），
  违反 stash wrapper"逐位委托 HF 原函数"的正确性铁律。

**结论**：在 torch 2.6 + 逐位委托铁律下，repeat_kv 拷贝是**不可绕过的
已知开销**；要破墙只能走 kernel 级路线（升级 torch / flash varlen GQA /
vLLM 类栈），软件层面无绕过。

## 3. docs/23 反馈处理状态（10 条，2026-09-30 全部闭环）

| # | 反馈 | 状态 | 落纸位置 |
|---|---|---|---|
| 1 | "并发 serving 安全"过度声称 | ✅ 收回，改为"无超出压缩本身的额外退化" | docs/21 发现 2 |
| 2 | sustainable 定义不含质量 | ✅ 改三项分报（memory/EOS/feasible + fact 曲线），m4_k8v8 在 quality-preserving 口径 N=1 也不达标 | docs/21 判定标准 |
| 3 | 64k EOS 0.953 不叫噪声 | ✅ 3 次 reps **逐位一致** = 确定性退化；修复路径是 budget 不是降并发 | docs/21 64k 节 |
| 4 | "墙与压缩无关"自相矛盾 | ✅ 改为"都 ∝B×K，compressed 常数小一半" | docs/21 发现 4 |
| 5 | 74GB 差额未解释 | ✅ 归因重写，b·B·K 列为最优先 profiling（未定位） | docs/21 发现 4 + 本文 §1 |
| 6 | 压缩省 HBM 无直接证据 | ✅ v2 稳态列闭环：KV 常驻实测 ~500×，进程 HBM 差 ~24GB（权重封死） | docs/21 v2 重跑 |
| 7 | 512GB 不是 KV budget | ✅ 改 `--hbm-budget-gb`（旧名保留别名），固定 KV budget 由 budget 扫描回答 | runbook §12g |
| 8 | TPOT 3.4-3.9× 定性 | ✅ 明确为 decode-step latency reduction，非 throughput | docs/21 发现 1 |
| 9 | 应扫 budget Pareto | ✅ 五点扫完：质量 2048 档追平 full，TPOT 红利 1024 内全拿，工作点推荐 512-1024（实测 32-64×） | docs/21 Pareto 节 |
| 10 | ladder 叙事不成立 | ✅ McNemar 闭环：INT8 中性、M4 显著（对无 merge）、merge 弱 | docs/21 配对分析 |

## 4. 实验批次状态总表

| 批次 | 结果目录 | 状态 | 结论位置 |
|---|---|---|---|
| 32k 主矩阵 v1 | `results/concurrency/` | ✅ 完成（早批部分 cell 在邻居污染期，以 09-30 凌晨重跑批为准） | docs/21 结果节 |
| 64k / 128k 两端 | `results/concurrency_65536/`、`_131072/` | ✅ 完成 | docs/21 64k/128k 节 |
| 32k v2（新字段） | `results/concurrency_v2/` | ✅ 完成 | docs/21 v2 重跑节 |
| budget 扫描 256-2048 | `results/budget_{256,512,1024,2048}_32k/` | ✅ 完成（128 档取 v2） | docs/21 Pareto 节 |
| 64k N=16 EOS 复现 ×3 | `results/repro_64k_n16_run{1,2,3}/` | ✅ 完成（逐位一致） | docs/21 64k 节 |
| 64k/128k v2 cell | — | 未跑（可选） | — |
| budget 扫描迁 64k/128k | — | 未跑（可选，验证边界随 K 外推） | — |
| HBM 五分解 memlog 对照 | — | 未跑（待办，回答 74GB） | runbook §12f |

**已完成批次不要重跑**。重跑前先看 runbook §12a 的 resume 语义
（ok/oom cell 会被 skip，需换 output-dir）。

---

## 5. 第二轮反馈（docs/25）：定位变更与下一阶段

docs/25（2026-09-30）判断成立并已被采纳：**本 harness 测的是测试框架的
prefill memory wall，不是 KV 压缩的 serving capacity wall**；lockstep、
deferred eviction、HF repeat_kv 三者在真实 serving 栈里都不存在。据此：

- 本文 §1–4 与 docs/21 的全部结果**降级为 diagnostic experiment**
  （证明 decode cost 下降 + 暴露旧 harness 局限），不是 serving capacity
  结论；
- 下一阶段交付物只有一个：**在 continuous-batching serving 栈上，质量
  达标条件下 v8 能把真实 concurrency / QPS 提高多少**（max concurrent
  seqs / QPS 对 offered load 的表）；
- 路线与 spike 步骤见 `docs/26-next-phase-serving.md`（Route A：vLLM
  集成四步 spike；Route B：最小 continuous-batching engine fallback）；
- 工作点改用 Pareto 甜点 512/1024 slots，不再用 128-slot 极端配置。
