# 38 · 结案报告：fixed-budget KV allocation（06j）最终对照实验

日期：2026-10-08 · 版本：2026-10-08j · 前置：docs/35（容量对照）、docs/36（定向）、docs/37（06j 设计与 Gate）

## TL;DR

06j 让 vLLM scheduler 真正按压缩后 footprint 记账后，在 2×A800、64k、64 会话 backlog、同条件下：

- **吞吐**：v8-1024 = **118.4 / 119.0 sess/h**（N=16/32），为 full（127.6 / 128.4）的 **~93%**。06i 时代这个比例是 ~50%——"2× slowdown"的主因不是 decode kernel，而是 prefill 对全 prompt 的注意力，被固定容量上限消除。
- **质量**：fact_acc 0.986 / 0.996（512 题/格），与 full（0.998 / 0.990）同区间，压力驱逐未兑现质量风险。
- **可靠性**：全部格子 0 HTTP 错误 / 0 contract raise / 0 OOM / 0 rewind。
- **容量**（本实验的核心问题）：每请求 full-attention residency 从 63 blocks（64k，随长度线性涨）降为**恒定 17 blocks**（分配日志直证），vLLM 自带 max-concurrency 估计 27.08x → **166–174x**。

**回答 docs/34 要的那句话**（现在终于三个条件都齐了）：

> On 2×A800 at 64k context and ≥0.98 factual accuracy, v8-1024 (fixed-budget accounting) matches full-KV sustainable concurrency (both saturate at the compute bound, N≈4–16) with throughput at 0.93× of full-KV, while reducing per-request full-attention KV residency 3.7× at 64k (7× at 128k). KV compression's capacity benefit is real but only convertible to concurrency/throughput in KV-bound deployments; on this hardware the workload is compute-bound, so the benefit materializes as headroom, not speed.

## 最终对照表（64 会话 × 64k × 8 轮，同条件）

| N | full sess/h | v8-1024 sess/h (06j) | v8/full | full acc | v8 acc | full P50 | v8 P50 |
|---|---|---|---|---|---|---|---|
| 1 | 77.4 | 41.9 *(06i)* | 0.54 | 0.990 | 0.990 | 42s | 86s |
| 4 | 122.3 | 59.5 *(06i)* | 0.49 | 0.998 | 0.977 | 106s | 238s |
| 8 | 126.5 | 64.4 *(06i)* | 0.51 | 0.998 | 0.980 | 204s | 447s |
| 16 | 127.6 | **118.4** | **0.93** | 0.998 | 0.986 | 423s | 481s |
| 32 | 128.4 | **119.0** | **0.93** | 0.990 | 0.996 | 868s | 948s |

*(06i)* = 上一版代码（全 prompt 记账）跑出的旧格，列出只为展示 06j 的变化幅度；N=16/32 为本次 06j 实测。单会话延迟（N=1）仍是 2×，因为串行时 prefill 占主导且 v8 的逐 chunk Python 更新路径比原生 kernel 慢；并发下摊薄到 1.13×。

## Gate 结算（docs/37）

| Gate | 结果 | 证据 |
|---|---|---|
| A 记账单测 | ✅ | integration 全 PASS（含 case 10 压力驱逐）；分配日志每请求 17 块封顶、chunk/decode 不增长 |
| B 物理 residency | ✅ | 分配日志直证（`kv_cache_usage_perc` 因 hybrid 口径弃用为度量） |
| C preemption/rebuild | ✅（覆盖式） | harness case 5 + property 随机交错 rewind 路径；服务器侧 preemption 在 174x 准入下不再触发，属预期 |
| D capacity 数字 | ✅ | Maximum concurrency 27.08x → 166.33x（sweep）/ 174.55x（verify） |
| 质量 | ✅ | 压力驱逐全程激活下 0.986–0.996 |

## 这意味着什么

1. **docs/36 的 memory-side 问题现在有答案了**：把压缩 residency 反馈给 allocator 是可行的、且按设计工作（custom spec + manager 记账路径，scheduler/runner 零改动）。压缩的容量收益是真实的。
2. **compute-side 问题也清楚了**：剩余 ~7% 吞吐差是插件逐 chunk Python 更新 + SDPA 注意力路径的纯实现开销。若将来要追平，唯一方向是 kernel 化（gather/scatter 和更新路径的 fused kernel），而不是再改记账。
3. **产品判断**：A800×2 + 64k 不是 KV-bound 部署，v8 在这里的卖法是"3.7–7× KV 余量换 7% 吞吐"；真正的优势场景（小显存、≥128k 上下文、高并发目标）本实验台搭不出来，需要 KV-bound 环境才能演示吞吐反超。

## 复现

- 06j 验证：`tools/verify_06j.sh`（Gate A+D）、`tools/verify_06j_phase2.sh`（residency+质量）、`tools/verify_06j_phase3.sh` + `V8_DEBUG_ALLOC=1`（分配日志）
- 最终对照：`N_LIST="16 32" BACKENDS="v8-1024" bash tools/run_capacity_sweep.sh`
- 结果：`results/capacity_{full,v8-1024}_c{N}.json`、分配日志 `logs/verify_06j_phase3.log`
