# 37 · 06j 设计：fixed-budget KV allocation（让压缩 residency 成为一等记账对象）

日期：2026-10-08 · 版本：2026-10-08j（06j）· 前置：docs/36（定向）、vLLM 0.19.1 源码调研（见下"源码证据"）

## 目标

> Make v8 compressed residency a first-class vLLM KV allocation policy.

之前（06i 及以前）我们只是 **compress data inside a full-KV allocation model**：vLLM scheduler 仍按 `cdiv(logical_tokens, B_g)` 给每请求记账，压缩对 allocator 不可见，容量收益恒为零（docs/36 第三点）。06j 让 scheduler 按**固定小预算**记账。

## 源码证据（v0.19.1 tag，调研 2026-10-08）

- 每请求块数公式：`SingleTypeKVCacheManager.get_num_blocks_to_allocate` = `cdiv(num_tokens, block_size)`（`vllm/v1/core/single_type_kv_cache_manager.py:88`），**与 num_kv_heads/head_size 无关**——所以"把 spec 改小"路径无效。
- 增量语义：该函数算的是 required **总量**，manager 内部减已持有块返回**增量**。因此 fixed-budget 的正确实现是 `required = min(cdiv(tokens, B_g), B_target)`，**不是**每次调用返回固定值。
- 机制地图：spec 由插件层自己的 `get_kv_cache_spec` 产生（`gpu_model_runner.py:6901-6931`），**vLLM 无"spec 必须等于 HF config"的检查**；spec 经 `KVCacheManager.allocate_slots`（`kv_cache_manager.py:218`）→ coordinator 汇总各 group → BlockPool。Scheduler、runner、pool 全部不需要改。
- hybrid 记账：`kv_cache_size_tokens` / `kv_cache_max_concurrency` 口径不同（`kv_cache_utils.py:1289-1331`），"GPU KV cache size(tokens)"不能拿来除 64k（docs/36 第一点）。
- 先例：`MambaManager` 每请求固定块数（`single_type:447-470`）——"某类 state 物理需求不随 token 长度线性增长"在 vLLM 架构里是被原生接受的。
- 未采用的路径：SlidingWindowSpec（residency 上界被 chunk 稀释到 ~(window−1+8192)/16 ≈ 577 块，且语义是"连续窗口"不是 `[sink|HH|recent]`，只能做 sanity 不能当主线）；缩小 spec heads（无效，见上）。

## 内存契约（实现前钉死）

```
每请求 vLLM 记账 = B_target manager blocks，与 logical length 无关
  B_target = cdiv(V8_COMPRESS_SLOTS + V8_STAGING_TOKENS, B_g) + BLOCK_MARGIN
  默认：SLOTS=1024, STAGING=16384(=2×chunk 8192), B_g=1056 → ~17 块
  对照：full 64k = 63 块，128k = 122 块 → 常驻并发 ~3.7× / 7×
```

- **物理池即记账池**：vLLM 块的物理容量恰好 = SLOTS+STAGING，compact 布局继续写在 certified rows 里，不新增私有张量、无隐藏内存。slots `[0, 1024)` = steady compact；`[1024, 17408)` = prefill staging。
- **驱逐规则**：`allowance = min(deferred_allowance(C, budget, MAX_SEQ), certified_capacity − C)`，plan（blockplan）与 impl（eviction 数学）消费同一函数（docs/32 铁律）。staging 不紧张时与 06i **逐字节一致**（deferred、question-aware）；prompt 超过 SLOTS+STAGING 触发**压力驱逐**（snap-so-far 打分）；每次 decode（C==1）驱逐到 1024，steady state 恒定。
- **fail-closed 不变**：certify_kernel 在任何张量索引前校验 plan span ≤ certified capacity；plan/impl 任一分歧 → UnitError。
- **preemption**：vLLM free 全部块 → num_computed=0 → 重放 prefill。compact 内容是 token 历史的确定性函数，staging 随 chunk 重放重写，幂等；rewind 判据 `comp < max(snap_len, compact_len)`（06g/06h 定案）继续兜底。plugin 不跨 preemption 缓存 block id。

## 质量风险的唯一来源

压力驱逐时，先被赶走的内容看不到后续 question 的 attention mass（deferred 设计的核心价值）。缓解旋钮：`V8_STAGING_TOKENS` 调大（容量收益换质量）。这正是 Gate B 存在的意义。

## 改动面（预估）

| 文件 | 改动 |
|---|---|
| `config.py` | 新增 `V8_STAGING_TOKENS`（env 覆盖）；版本 06j |
| `spec.py` | `blocks_per_request` 新公式 + docstring |
| `blockplan.py` | `deferred_allowance` 加 capacity 上限（可选参，缺省退化现行为） |
| `impl.py` | allowance 走带 capacity 的单一入口；fail-closed 信息增强 |
| `__init__.py` | 核对 allocator patch 语义为"总量 clamp + 增量返回" |
| 版本三处 | config.py / `tools/repro_concurrency.sh` must-say / docs/28 §8.2 |
| tests | blockplan 新用例；integration 新增"长序列压力驱逐" case；fake certify 补容量语义 |

Scheduler / KVCacheManager / BlockPool / gpu_model_runner / block_table 管道：**零改动**。

## 验收 Gate（不过不往下走）

**2026-10-08 实测：A ✅ D ✅ 冒烟 ✅（B/C 见 phase 2）**
- Gate A：远程 `test_integration.py` 全 PASS，含 case 10 "long-seq pressure eviction (fixed budget)"；property 随机交错 2194 步 × 3 层全过。
- Gate D：`Maximum concurrency for 131,072 tokens per request` 从 27.08x → **174.55x**（6.4×）——custom accounting 确实进入了 admission path。
- 冒烟：`tools/repro_concurrency.sh` 版本 06j，0 崩溃 / 0 contract raise / fact_acc 1.0（32k 数据）。
- **Phase 2（2026-10-08，Gate B 首点 + 质量）**：16 并发 × 64k（压力驱逐全程激活）：`peak_kv_usage=0.168`、preemption 0、**fact_acc 0.9922**（127/128，压力驱逐未兑现质量风险）、吞吐 **118.5 sess/h**（06i 同格只有 65.0——06i 的"2× slowdown"主因是 prefill 对全 prompt 注意力，06j 容量上限把 prefill 注意力压到 ≤17408 后消失，v8 已接近 full 的 ~128）。单点 usage 因 hybrid 口径不可换算块数，形态证明见 phase 3。

- **Gate A — 记账单测**（远程 integration）：logical 1k/8k/64k/128k 后每请求持有块数 = B_target，chunk1/chunk2/decode 后总量不变。纯 fake 环境可测大部分。
- **Gate B — 物理 residency**：N=1/2/4/8，32k/64k/128k 同 slots 请求 steady decode 常驻 footprint 近似相同、∝ N 不随 logical length 线性增长。
- **Gate C — preemption/rebuild**：A running → preempt → B 吃容量 → A resume → 结果正确、记账正确。
- **Gate D — capacity 数字**：启动日志 `Maximum concurrency for 131,072 tokens` 相对 full 的 27.29x 大幅上升；不升 = 没进 admission path，直接停。
- 之后冒烟：`bash tools/repro_concurrency.sh`（64k 2 并发 0 错误），再跑一遍 capacity sweep 的 N=16 一格对比 full。

## 最终实验（06j 全过后）

同 docs/35 条件（2×A800、64k、64 会话 backlog、full vs v8-1024），但这次是 "same hardware, different physical KV residency model"。两个可能结局都成立且都算数：compute-bound 下 memory headroom 不涨吞吐（可信的产品结论）；或 KV-bound 部署下 v8 提升 QPS。只有这时才能回答："KV 压缩到底能把实际可驻留并发提高多少。"
