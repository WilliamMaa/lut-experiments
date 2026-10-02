# v8 下一阶段：真实 serving 栈上的 concurrency 测量（docs/25 定案）

日期：2026-09-30
依据：`docs/25-feedback.md`（第二轮反馈）。结论记录见 `docs/24`，旧 harness
结果与 Pareto 分析见 `docs/21`。

---

## 1. 定位变更（docs/25 核心判断，接受）

旧 lockstep harness 的全部 concurrency 结果降级为 **diagnostic experiment**：

> 它证明了 decode cost 随压缩下降（TPOT 3.4–3.9×，最可靠正结果），并
> 暴露了旧 harness 不适合测 capacity——peak concurrency 先被 full-prefill
> residency、compression-specific workspace 和 repeat_kv transient 撞死，
> 测到的是**测试框架的 prefill memory wall**，不是 KV 压缩后 serving
> capacity 的 wall。

三个结构性问题（不重跑旧 harness 能修，只能换架构）：

1. **lockstep batch ≠ 真实并发**：N 个 session 同步超长 prefill + 同步
   decode，人为制造 B×K 峰值；生产是 continuous batching，prefill/decode
   混合、请求异步进出，由 KV capacity 决定 active sequences。
2. **deferred eviction 与产品目标冲突**：声称把 KV 压到 512/1024 slots，
   就要在 serving 生命周期内真正控制 residency；"首次 decode 才压"等于
   prefill 阶段完全没省。→ **cache 语义必须改成 prefill 内滚动淘汰**
   （这改变与 docs/16 档案的一致性，spike 阶段单独标定）。
3. **HF SDPA/GQA transient 是 harness 成本**：vLLM/FlashInfer 有 paged KV
   和专用 GQA kernel，"torch 2.6 必须 repeat_kv" 不是产品结论。

## 2. 交付物（唯一）

一张表 + 一句话数字：

| 配置 | KV/seq | max concurrent seqs | sustainable QPS | P95 TTFT | TPOT | quality |
|---|---:|---:|---:|---:|---:|---|
| Full BF16 | X GB | ? | ? | ? | ? | baseline |
| v8-1024 | Y MB | ? | ? | ? | ? | pass |
| v8-512 | Z MB | ? | ? | ? | ? | pass/fail |

> 例（目标形态）：64k context、相同 7×A800、相同 quality threshold 下，
> 1024-slot v8 将 max active concurrency 从 18 提高到 46（~2.6×），
> sustainable QPS 提高 1.9×。若只有 1.2×，也是真实答案——再去看瓶颈
> 是不是 compute 而不是 KV。

**不再扫学术指标。**

## 3. 实验协议（定死，防口径漂移）

- **栈**：vLLM 类 continuous-batching serving（Route A）；失败则自写最小
  engine（Route B，§5）。
- **固定**：相同 GPU（7×A800）、模型（Qwen3.6-35B-A3B）、context
  distribution（32k/64k/128k 三档负载，复用现有 jsonl）、输出长度、SLO。
- **工作点**：512 / 1024 slots（Pareto 甜点，32k 实测 32–64× 压缩、fact
  0.94+）。**不再用 128-slot 极端配置。**
- **硬约束**：压缩后的 KV 必须是 **allocator 看到的 physical KV
  footprint**——不是另存一份压缩 state、full KV 还留着。prefill 期间
  滚动淘汰，residency 从 append 时刻起受 budget 控制。
- **爬升**：逐步提高 offered load / max-num-seqs，直到 OOM、TTFT/TPOT
  SLO 超标、或 quality gate 超标三者居一。
- **quality gate**：fact ≥ full − 5pp（或 EOS < full − 2pp），判定负载
  复用现有 longctx_multi_turn 题库。

## 4. Route A：vLLM 集成 feasibility spike（第一步只做这个）

**不要先实现完整集成。** 四步 spike，每步有明确的 go/no-go：

1. **摸底**：找到 vLLM 当前 Qwen3.6 hybrid 模型的 KV-cache allocation 与
   attention backend 路径。已知有利事实：该模型族（Qwen3-Next/Qwen3.6
   hybrid Gated DeltaNet）vLLM 已支持，hybrid KV cache manager 按层类型
   分 KV group（GDN 层固定大小 recurrent state，full-attn 层标准 GQA
   paged KV）。要确认的是：10 个 full-attn 层的 KV group 能否替换/包裹
   成自定义 compressed-cache backend。
2. **替换可行性**：v8 只压 full-attn 层（GDN 层状态不动）。需要在 backend
   层挂接：per-key attention score 获取（heavy-hitter 选择的输入——
   **最大技术风险**：vLLM 的 FlashAttention/FlashInfer kernel 默认不输出
   per-key scores，需要在线近似（如 sink/window 启发式）或 kernel
   改造，spike 优先验证这条）、prefill 内滚动淘汰、paged 化存储。
3. **单请求对拍**：1 个 request，输出与现有 harness 512/1024 档逐字一致
   （允许已有的 turn≥1 浮点分叉记录，turn 0 必须一致）。
4. **接 continuous batching**：通过后进入正式实验（§3 协议）。

## 5. Route B（fallback）：最小 continuous-batching engine

仅当 Route A 集成成本不可接受时启用，且**不是回到 lockstep**。最低要求：

```text
requests asynchronous        # 异步到达，不锁步
prefill chunked              # 分块
decode interleaved           # prefill/decode 混合调度
KV pages dynamically allocated/freed
compressed residency applied during prefill   # 淘汰发生在 prefill 期
```

lockstep harness 一个字节都不要往这个方向改——它的价值已经兑现完了
（diagnostic）。

## 6. 开放问题（spike 期间必须回答）

1. per-key scores 在 vLLM backend 里怎么拿（§4.2）——拿不到就要改
   heavy-hitter 选择算法或改 kernel，成本重估。
2. prefill 内滚动淘汰 vs docs/16 两阶段档案的一致性标定（方法语义变化，
   需重新跑 B=1 回归门）。
3. 压缩 KV 的 paged 布局：v8 的 sink|hh|recent 三段布局映射到 vLLM
   block table 的方式（固定 512/1024 slots 其实可以一请求一页，工作量
   可能比想象小）。
4. quality gate 在 serving 栈上的自动化（离线判题管线复用现有
   fact 判定）。

## 7. 与既有文档的关系

- 旧 harness 结论**全部保留**，定位改 diagnostic：docs/24 §1–2（机制）、
  docs/21（结果、Pareto、配对分析）。
- 12f（HBM 五分解、74GB）**降级为可选**——如果 Route A 走通，那 74GB
  是旧 harness 的成本结构，不再值得解释；如果走 Route B，才需要。
- 下一轮反馈前，本文档 §4 的 spike 是唯一主线。
