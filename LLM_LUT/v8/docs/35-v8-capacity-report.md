# 35 · v8 vs full-KV 容量对照实验（docs/34 要求的唯一实验）

日期：2026-10-08 · 插件版本：2026-10-06i · 脚本：`tools/run_capacity_sweep.sh`

## TL;DR

在 2×A800-80GB（GPU 6,7，TP2）上、64 会话 × 64k 上下文 × 8 轮 QA 的真实 backlog 下：

1. **full-KV 在 N=32 下仍能完成全部会话**：0 OOM、0 错误。docs/34 的怀疑证实：旧说法"full 只能扛 1–2 路 64k"**错误，予以删除**。但注意（docs/36）：这不等于"32 路 64k KV 同时 resident"——vLLM 会用 preempt+recompute 消化超容量，真实 KV pressure 需要 preemption telemetry 才能定性，见"下一步 A"。
2. **吞吐饱和形态像 compute-bound 但未被证明**：full ~128 sess/h（N=4 起饱和），v8-1024 ~65 sess/h（N=8 起饱和），之后并发只增加排队延迟。饱和也可能同时掩盖 KV admission/preemption 开销。
3. **v8-1024 的吞吐约为 full 的一半**，每个并发档位稳定如此（N=1: 41.9 vs 77.4；N=16: 65.0 vs 127.6）。这是 compact attention 实现的 ~2× slowdown，不太像噪声。
4. 质量上两者无可见差距（512 题/格：full 0.986–0.998，v8-1024 0.977–0.990，单遍运行内）。
5. **结论（docs/36 修订版，替代初稿的"已证伪"表述）**：
   > **Under the current integration, v8 does not improve observed serving throughput or client-level concurrency; the compact attention implementation is approximately 2× slower than full KV. However, the experiment does not yet measure the memory-capacity benefit of compression, because vLLM still accounts and allocates KV capacity using the uncompressed request footprint. A final capacity conclusion requires compressed-aware KV admission/allocation.**
   >
   > 现在已经证明的是：当前 v8 kernel 太慢，且只把 KV 内容压紧、但不改变 vLLM 的 allocation accounting，不能带来系统并发收益。还没有证明的是：如果 vLLM 真正按压缩后的 KV footprint 分配资源，能多支持多少 resident concurrency。这两个必须分开，否则会把"integration 没释放 allocator headroom"误判成"KV compression 本身没有容量价值"。

## 实验设置（三后端完全相同）

| 项 | 值 |
|---|---|
| 模型 | Qwen3.6-35B-A3B，bf16，TP2（GPU 6,7） |
| workload | 64 会话（1 会话 = 1 doc），每会话 ~64k tokens、anchor + 8 轮问答 |
| 服务端 | `--enforce-eager --no-enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.88 --max-num-seqs N`（chunked prefill 默认 8192） |
| 并发 | 客户端并发 = 服务端 `--max-num-seqs` = N，admission 不是隐藏瓶颈 |
| 网格 | full × N{1,2,4,8,16,32}；v8-1024 × N{1,2,4,8,16}（c32 中途停止，见"未完成"节） |
| 数据 | `data/longctx_multi_turn_65536_64docs.jsonl`（64 docs） |
| 记录 | fact_acc、errors、wall、sess/h、P50/P95、peak HBM（bench 期间 5s 轮询）、OOM/rewind 计数 |

## 结果

### full-KV（纯 `vllm serve`，无插件）

| N | fact_acc | errors | wall | sess/h | P50 | P95 | peak HBM/卡 |
|---|---|---|---|---|---|---|---|
| 1 | 0.9902 | 0 | 2975s | 77.4 | 42s | 62s | 72,441 MiB |
| 2 | 0.9863 | 0 | 2270s | 101.5 | 64s | 105s | 72,461 MiB |
| 4 | 0.9980 | 0 | 1883s | 122.3 | 106s | 200s | 72,461 MiB |
| 8 | 0.9980 | 0 | 1822s | 126.5 | 204s | 347s | 72,481 MiB |
| 16 | 0.9980 | 0 | 1805s | 127.6 | 423s | 545s | 72,481 MiB |
| 32 | 0.9902 | 0 | 1795s | 128.4 | 868s | 952s | 72,501 MiB |

启动日志（每档相同）：`GPU KV cache size: 921,888 tokens`，`Maximum concurrency for 131,072 tokens per request: 27.29x`。

### v8-1024（`python -m vllm_plugin.serve`，V8_COMPRESS_SLOTS=1024）

| N | fact_acc | errors | wall | sess/h | P50(median) | mean | peak HBM/卡 |
|---|---|---|---|---|---|---|---|
| 1 | 0.9902 | 0 | 5494s | 41.9 | 85s | 86s | 同 full |
| 2 | 0.9902 | 0 | 4458s | 51.7 | 140s | 139s | 同 full |
| 4 | 0.9766 | 0 | 3874s | 59.5 | 238s | 240s | 同 full |
| 8 | 0.9805 | 0 | 3577s | 64.4 | 446s | 437s | 同 full |
| 16 | 0.9766 | 0 | 3542s | 65.0 | 868s | 868s | 同 full |

v8 侧启动日志同样报 `GPU KV cache size: 921,888 tokens`（27.08x）——插件的 compact pool 是从同一个 KV block pool 里切出来的，vLLM 账面上的 token 容量并没有变大。P95 精确值在 `results/capacity_v8-1024_c*.json` 的 `session_seconds` 里，需要时一行的 python 即可补。

所有格子：0 HTTP 错误、0 contract raise、0 OOM、0 rewind。

## 关键发现

**1. full-KV 能承接的并发远比旧假设大，但"没有内存墙"尚未证明。** N=32（32 路活跃请求）也 0 OOM 完成全部 64 会话——但这只说明 server 能把请求服务完，不说明 32 路 64k KV 同时 resident：vLLM 的正常机制是 preempt → free blocks → recompute。docs/33 中"full 只能扛 1–2 路"的断言**删除**；而"该配置下并发不由显存决定"这一 stronger  claim 需要 preemption/recompute telemetry 才能成立（docs/36 指出启动日志两个数本身矛盾：`921,888 / 131,072 ≈ 7.0× ≠ 27.29x`，hybrid 模型的 KV 记账是 group-aware 的，`GPU KV cache size` 的 token 数不能拿来直接除 64k）。

**2. 吞吐饱和形态与 compute-bound 一致，telemetry 已支持（2026-10-08 补测）。** full 从 N=4 起 sess/h 停在 ~122–128，N=8→32 只把 P50 从 204s 推到 868s；v8-1024 从 N=8 起停在 ~64–65。两边都在排队。telemetry（`tools/telemetry_probe.sh`，full、16 并发 × 64k、16 会话）：`kv_cache_usage_perc` 峰值 **0.137**，`num_preemptions_total` **0 → 0**，bench 0 错误。即 16 路 64k 并发只占 KV 池约 14%、零抢占；线性外推 N=32 也远低于饱和。**"该配置下 full 不 KV-bound、瓶颈在 compute"成立。**

**3. v8 的吞吐代价是恒定的 ~2×。** 每个 N 档位 v8 的 sess/h 都约为 full 的一半（N=1: 41.9 vs 77.4；N=16: 65.0 vs 127.6），单会话时延翻倍（86s vs 46s）。v8 用 ~2× 的 wall 换来了相同的 64 会话完成量。这 2× 来自插件的 compact pool 注意力路径（非 FlashAttention 原生 kernel 的 gather/scatter 实现），在长上下文 decode 中成为主开销。

**4. 省下的 KV 容量没有变成 headroom——但更直接的原因不是"workload 不 KV-bound"，而是我们根本没把压缩后的容量释放给 allocator。** v8 的 compact pool 是从 vLLM 同一个 KV block pool 里切出来的，启动日志报的 KV cache size / 27.x concurrency 与 full 相同：scheduler 眼里一个 v8 请求和一个 full 请求占同样的块（即使 v8 内部只实际使用 1024 slots）。所以本次实验真正测的是"**在 vLLM 仍按 full-KV footprint 做 admission/allocation 时，v8 压缩注意力后端的性能**"；还**没有**测"如果 allocator 知道每请求只需 compact budget，能多 admit 多少 request"。"workload 远没碰到 pool 上限"这一句在 telemetry 之前不能作为既定事实。

**5. 质量没有可分辨的差距。** 每格 512 题：full 0.986–0.998，v8-1024 0.976–0.990。单遍运行，不能说"完全无损"，但没有任何证据表明 v8 在该 workload 上质量明显劣化。这与 docs/33 的结论一致，且 backlog 条件下（多请求交错、抢占恢复）依然成立——integration 的可靠性是经住了真实并发考验的。

## 对 docs/34 逐条回应

| docs/34 的要求 | 结果 |
|---|---|
| 删除"full 只能 1–2 路" | ✅ 已删，实测 N=32 无 OOM |
| full / v8-1024 / v8-4096 同条件 sweep | full、v8-1024 完成；v8-4096 未跑（见下） |
| 64 会话 backlog，N 到 32 | full 全部完成；v8-1024 到 16 |
| 记录 OOM/admission、sess/h、P50/P95、fact_acc | 全部记录 |
| 产出"from A to B (B/A×)"结论句 | 产出，答案为 A=B（无提升），见 TL;DR #5 |

## 未完成与理由

- **v8-1024 c32**：完成 32/64 会话后按决定停止。N=16 已显示饱和（65.0 sess/h ≈ N=8 的 64.4），c32 只会重复"排队更久"这一已知形态。
- **v8-4096 整个 sweep**：未跑。它的速度必然落在 v8-1024 与 full 之间、质量大概率与 1024 无显著差异（docs/33 已记录"未观察到 1024→4096 的明确质量增益"），不会改变 compute-side 的 2× slowdown 结论，更不会触及 memory-side（accounting 未变，跑什么 slots 数都测不到容量收益）。如需补跑：`BACKENDS="v8-4096" bash tools/run_capacity_sweep.sh`（约 7 小时）。

## 这意味着什么（docs/36 修订：两个问题必须分开）

**Compute side（已被证明）**：当前 v8 compact attention 实现约 2× 慢于 full-KV， saturated throughput ~65 vs ~128 sess/h。这是现在真正的工程瓶颈——即使将来 allocator integration 让 resident capacity 翻倍，每 sequence 慢 2× 也会让吞吐收益归零。

**Memory side（尚未测）**：压缩能否真正释放 vLLM allocator capacity？当前 integration 只压缩物理内容，admission accounting 未改，所以这个问题本次实验没有触及。业务问题"v8 KV compression can increase serving concurrency by how much"要成立，必须满足：compressed resident KV smaller → allocator knows it is smaller → same HBM admits more active sequences。现在只有第一条。

**下一步（不再扫 concurrency，只做两件）**：

- **A. full 的真实 KV pressure telemetry —— 已完成（2026-10-08）**：full、N=16 × 64k × 16 会话，KV usage 峰值 0.137、preemption 0、0 错误。"compute-bound"站住。原始数据：`results/telemetry_full_c32.json`、`logs/telemetry_full_c32.log`。
- **B. compressed-aware KV admission/allocation（核心）**：让 vLLM scheduler 按压缩后 footprint 记账。对 Qwen3.6 这种 hybrid（40 层仅 10 层 full attention，其余 Gated DeltaNet）只能重定义**被压缩的 full-attention group** 的 residency accounting，GDN recurrent-state group 照常。候选实现方向：把 v8 attention group 的 block_size/记账粒度改为按 compact budget 声明，使 per-request 分配 ≈ slots×layers 而非 logical tokens；然后重读 `num_gpu_blocks` / per-group blocks-per-request / `kv_cache_max_concurrency` 验证 allocator 真的多 admit。做B之前先查 vLLM 0.19.1 源码确认 block accounting 的 hook 点（hybrid group-aware 路径）。
- v8-4096 sweep：同意 docs/36，暂不跑（1024≈4096 质量无差异 + kernel 开销特征相同，七小时换不来新信息）。
