对，你这个纠正非常重要。我们现在的目标不是写一篇“看起来合理”的论文，而是做出一个**能回答工作问题的 serving 方案**：

> **在真实 LLM serving 场景里，我们的 KV cache 压缩到底能把可支持并发提高多少？**

如果现有 harness 回答不了这个问题，那不是接受一个“Pareto frontier”就结束，而是应该**换实验架构直到能回答**。

你现在这批结果告诉我们的，其实不是“压缩不能提升并发”，而是：

> **当前 HF + lockstep + deferred eviction harness 根本不适合测这个问题。**

因为它把真正的 KV 存储收益挡住了。你自己的结果已经证明：压缩后的 steady-state KV 很小，但 peak concurrency 先被 full-prefill residency、compression-specific workspace 和 `repeat_kv` transient 撞死。也就是说，测到的是 **这个测试框架的 prefill memory wall**，不是“KV 压缩后 serving capacity 的 wall”。这正是为什么当前结果里 full 反而比 compressed 支持更高 N。:chatgpt-content-reference{index="0"}

而现实 serving engine 并不是这么工作的。vLLM 明确按可用 GPU memory 配置 **KV-cache capacity**，甚至直接给出 `kv_cache_max_concurrency` / `kv_cache_size_tokens` 这样的概念；它使用 paged KV 管理，而不是让所有 session 的全长 KV 在一个 lockstep tensor 里一路增长。:chatgpt-content-reference{index="1"} FlashInfer 也直接提供 paged KV cache 的 GQA/XQA kernel。:chatgpt-content-reference{index="2"}

所以答案是：**是的，我们现在的场景仍然不够真实，而且不真实的地方刚好会毁掉我们最想测的指标。**

最主要有三个问题。

第一，**lockstep batch 不是真实 concurrency。** 现在 N 个 session 同步做一个超长 prefill、同步 decode，这会人为制造一个巨大的 \(B\times K\) 峰值。生产 serving 是 continuous batching，请求不同步进入、prefill 和 decode 混合执行，KV capacity 才真正决定 active sequences 能留下多少。

第二，**deferred eviction 和我们的产品目标冲突。** 如果方法号称把 KV 从 32k/64k/128k 压到 128/512/1024 slots，那它必须在 serving 生命周期中真正控制 residency。现在等到“首次 decode 才压”，等于在最需要省内存的 prefill 阶段完全没省。这当然测不出 concurrency benefit。

第三，**HF SDPA/GQA transient 是 harness 成本，不是我们最终系统必须承担的成本。** vLLM/FlashInfer 这种 serving stack 已经有 paged KV 和专门 attention backend，所以“当前 PyTorch 2.6 必须 repeat_kv”不能作为最终产品结论。:chatgpt-content-reference{index="3"}

所以我会把下一步彻底改成一个非常工程化的目标：

> **把 v8 插到一个真正按 KV capacity 限制 concurrency 的 serving path，然后直接测 max concurrent sequences / QPS。**

不是再扫一个学术指标。

具体应该这样做：

- 用 **vLLM 类 continuous-batching serving 栈**作为 baseline；
- 固定相同 GPU、模型、context distribution、输出长度、SLO；
- baseline 用 full BF16 KV；
- ours 用例如 512 / 1024 slots 的压缩版本——先选已经能满足质量要求的实际工作点，不要再用 128-slot 极端配置；
- 必须让压缩后的 KV **真实成为 allocator 看到的 physical KV footprint**，而不是另存一份压缩 state、full KV 还留着；
- 逐渐提高 offered load / max-num-seqs，直到达到：
  - OOM，或者
  - TTFT/TPOT SLO 超标，或者
  - quality gate 超标。

最后只要回答一张表：

| 配置 | KV/seq | max concurrent seqs | sustainable QPS | P95 TTFT | TPOT | quality |
|---|---:|---:|---:|---:|---:|---:|
| Full | X GB | 16 | ... | ... | ... | baseline |
| v8-1024 | Y MB | ? | ... | ... | ... | pass |
| v8-512 | Z MB | ? | ... | ... | ... | pass/fail |

**这才是我们要的结果。**

比如最终能够说：

> 在 64k context、相同 7×A800、相同 quality threshold 下，1024-slot v8 将最大 active concurrency 从 18 提高到 46，约 2.6×；sustainable QPS 提高 1.9×。

这才是 job 需要的答案。

如果最后只有 1.2×，那也是真实答案；然后去看瓶颈是不是 compute 而不是 KV。

而且现实 vLLM 的设计本来就是按这一逻辑工作的：它会根据可供 KV cache 使用的 memory 计算 cache capacity / max concurrency，并支持显式的 `kv_cache_memory_bytes`；现在还支持 CPU/多级 KV offloading。:chatgpt-content-reference{index="4"} 所以迁过去以后，我们测出来的 concurrency 才有产品意义。

还有一个关键判断：**不要先实现整个 vLLM integration。** 先做一个小的 feasibility spike：

1. 找到 vLLM 当前 hybrid Qwen3.6 的 KV-cache allocation / attention backend；
2. 确认我们能不能在 full-attention 10 层上替换 cache read/write 或提供自定义 compressed-cache backend；
3. 跑 1 个 request，确认 output 和目前 512/1024 档一致；
4. 再接 continuous batching。

如果这条 integration 成本太高，第二选择也不是继续用现在的 lockstep harness，而是**自己写一个最小 continuous-batching simulator/engine**，至少做到：

```text
requests asynchronous
prefill chunked
decode interleaved
KV pages dynamically allocated/freed
compressed residency applied during prefill
```

然后再测 concurrency。

所以我现在会把旧的 concurrency 结果定位成：

> **diagnostic experiment：证明 decode cost 会下降，并暴露旧 harness 不适合测 capacity。**

它不是最终 serving capacity 结果。

而下一阶段的交付目标就一个：

\[
\boxed{\text{在质量达标条件下，v8 能把真实 serving concurrency / QPS 提高多少？}}
\]

这才值得继续做。