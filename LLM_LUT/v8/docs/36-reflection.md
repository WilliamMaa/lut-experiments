这版我觉得**还不能下“v8 对并发没有任何收益，已经证伪”这个最终结论**。不是替 v8 找借口，而是当前实验里有两个非常关键的 capacity-accounting 问题，先把它们钉死，才能说到底是不是 compute-bound。

第一，**`GPU KV cache size: 921,888 tokens` 和 `Maximum concurrency ... 27.29x` 这两个数本身就是矛盾的。**

如果真是：

\[
921{,}888 / 131{,}072
\]

只有大约 **7.0×**，不可能是 27.29×。

而 vLLM 对 hybrid 模型的 KV capacity 记账本来就不是普通 `num_blocks × block_size`。当前源码专门有 `kv_cache_size_tokens` 和 `kv_cache_max_concurrency` 两套 group-aware 指标，因为 hybrid 模型不同 KV group 的 page/block 占用不同，简单 token 数会误导。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py?utm_source=chatgpt.com)

更关键的是，vLLM 0.19.x 的 Qwen3.5/hybrid cache 日志就已经有人报告过类似问题：`GPU KV cache size` 和 `Maximum concurrency` 对不上，正是 hybrid group accounting 导致的。[GitHub](https://github.com/vllm-project/vllm/issues/40691?utm_source=chatgpt.com)

所以这句：

> 921,888 tokens ≈ 14 路裸 64k KV

**现在应该删掉。**

你不能拿这个数直接除 64k。

真正应该看的，是 vLLM 内部 group-aware：

```text
kv_cache_max_concurrency
num_gpu_blocks
per-group blocks/request
actual KV usage
```

而不是那个表面 token 数。

---

第二，**N=32 无 OOM，也不能直接推出 full-KV“没有内存墙”。**

vLLM 的 scheduler 在 KV cache 不够时，本来就会：

```text
preempt request
→ free KV blocks
→ later recompute
```

这是 vLLM 的正常机制，不一定 OOM。官方 tuning 文档明确说 KV cache 不够时会 preempt，并在后续重新计算 KV。[GitHub](https://github.com/AI-App/VLLM-Project.VLLM/blob/main/docs/configuration/optimization.md?utm_source=chatgpt.com)

所以：

```text
N=32
0 OOM
```

只能说明：

> server 能把 32 个客户端请求最终服务完。

它不能说明：

> 32 个 64k KV 同时 resident。

这两个概念差很多。

而且你现在的数据其实很像：

```text
N=4:
122 sess/h

N=8:
126

N=16:
128

N=32:
128
```

完全饱和。

这当然可能是 compute-bound。

但也可能同时存在：

```text
KV admission / preemption / recompute
```

只是被吞吐饱和掩盖。

所以这次实验必须补一个非常便宜的 telemetry：

```text
total preemptions
recomputed tokens due to preemption
GPU KV cache utilization over time
running / waiting requests
```

如果 full N=32：

```text
preemptions = 0
KV usage nowhere near pressure
```

那我会同意：

> **full 在这个 workload 下确实不是 KV-bound。**

如果有大量 preemption/recompute，那么结论会变成：

> full 能承接 N=32，但靠 scheduler recycling KV，不等于没有 capacity pressure。

这是系统意义上很不一样的结论。

---

### 第三个更大的问题：v8 实际上根本没有把“压缩后的容量”反馈给 vLLM allocator

你自己已经写出来了：

> v8 启动日志仍然报同一个 KV cache size / 27.x concurrency，compact pool 是从同一个 block pool 里切出来的。

这实际上意味着：

> **当前 v8 integration 只压缩了物理内容，却没有改变 vLLM scheduler 对每个 request 的 KV allocation accounting。**

那它当然不可能让 vLLM admit 更多 requests。

因为从 scheduler 看：

```text
full request:
requires X blocks

v8 request:
still requires X blocks
```

即使 v8 内部只实际使用 1024 slots。

这就非常关键了。

你现在真正测出来的是：

> **在 vLLM 仍按 full-KV footprint 做 admission/allocation 时，v8 的 compressed attention backend 性能如何。**

你还没有真正测：

> **如果 allocator 知道 v8 每请求只需要 1024 retained slots，它能多 admit 多少 request。**

所以：

> “省下的 KV 容量没有变成 headroom”

这句话是对的。

但原因不能直接归结为：

> workload 根本不 KV-bound。

还有一个更直接的原因：

> **我们根本没有把压缩后的容量释放给 allocator。**

这是当前实验最重要的缺口。

---

## 所以真正业务问题其实还没完全回答

你想回答的是：

\[
\boxed{
\text{v8 KV compression can increase serving concurrency by how much?}
}
\]

要回答这个问题，必须满足：

```text
compressed resident KV smaller
        ↓
vLLM allocator knows it is smaller
        ↓
same HBM admits more active sequences
```

现在只有第一条。

第二条没有。

那么第三条自然不会出现。

---

# 但是这次实验依然非常有价值

它已经回答了另一件很关键的事情：

> **当前 v8 attention implementation 太慢。**

full 饱和：

```text
~128 sessions/hour
```

v8：

```text
~65 sessions/hour
```

这个约 2× 差距非常稳定，不像噪声。

所以就算以后 allocator integration 真让 capacity 提高：

```text
2× active sequences
```

如果每个 sequence 本身慢 2×：

```text
throughput 还是可能没有收益
```

这才是现在真正的工程瓶颈。

换句话说，项目现在出现了两个完全独立的问题：

### Memory side

> 压缩能不能真正释放 vLLM allocator capacity？

**尚未测，因为 admission accounting 还没改。**

### Compute side

> compact attention backend 值不值得？

**当前答案很差：大约 2× slowdown。**

这两个必须分开。

---

# 4096 我反而同意不用现在跑

这一点文档判断我基本同意。

如果：

```text
1024 quality ≈ 4096
```

而 v8 backend 的主要计算损失来自：

```text
gather / scatter
custom attention path
```

那么再花七小时跑 4096 capacity sweep 确实不是当前优先级。

除非你怀疑：

> 4096 能走不同 kernel / overhead 特征。

否则价值有限。

---

# 下一步我不会再扫 concurrency

我会只做两个实验/改动。

## A. 先确认 full 到底有没有真实 KV pressure

N=1,4,8,16,32 不用重跑全部 workload。

选 N=16 / N=32，记录：

```text
preemption count
recompute due to preemption
KV cache utilization timeline
running/waiting
```

如果：

```text
preemption = 0
```

那么“compute-bound”这个结论就真正站住。

如果不是 0，就重新解释。

---

## B. 做真正的 compressed-capacity accounting

这是核心。

我们必须让 vLLM scheduler 明白：

```text
full:
64k logical tokens → X KV blocks

v8-1024:
64k logical tokens → only compact budget worth of
full-attention KV residency
```

对于 Qwen3.6 这种 hybrid 模型尤其要小心，因为 40 层里只有 10 层是 full attention，其余是 Gated DeltaNet；官方结构就是 10 × [3 linear-attention + 1 full-attention]。[Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blame/main/config.json?utm_source=chatgpt.com)

所以不能简单：

```text
all KV block requirement / 64
```

必须只重新定义被 v8 压缩的 full-attention group 的 residency accounting，而 GDN recurrent-state group 仍正常占空间。

然后重新计算：

\[
\text{blocks per v8 request}
\]

和：

\[
\text{kv\_cache\_max\_concurrency}
\]

这样才能真正看到：

```text
full max resident seq = A
v8 max resident seq = B
```

---

## 最终我会把 docs/35 的结论暂时改成

不是：

> **KV compression 没有提升并发，这个假设已经证伪。**

而是：

> **Under the current integration, v8 does not improve observed serving throughput or client-level concurrency; the compact attention implementation is approximately 2× slower than full KV. However, the experiment does not yet measure the memory-capacity benefit of compression, because vLLM still accounts and allocates KV capacity using the uncompressed request footprint. A final capacity conclusion requires compressed-aware KV admission/allocation.**

中文就是：

> **现在已经证明的是：当前 v8 kernel 太慢，而且仅把 KV 内容压紧、但不改变 vLLM 的 allocation accounting，不能带来系统并发收益。**
>
> **还没有证明的是：如果 vLLM 真正按照压缩后的 KV footprint 分配资源，能多支持多少 resident concurrency。**

这两个必须分开。

否则我们很容易把：

> integration 没释放 allocator headroom

误判成：

> KV compression 本身没有容量价值。

这次我觉得已经离最终答案很近了：**现在不要再改算法，先把“压缩后的内存真的交还给 vLLM allocator”这件事做实。** 然后再跑一次 full vs v8，就真的可以结案。