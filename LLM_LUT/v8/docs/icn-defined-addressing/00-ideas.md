我觉得可以，而且**memory allocation 正好是最适合先切进去的点**。因为它既能利用你现有的 KV 压缩结果，又不需要一上来就重做整个 serving stack。

现在业界这边其实已经有三个相邻方向：

* vLLM/PagedAttention：把 KV 切成 block，用 block table 做逻辑块到物理块映射，解决碎片和动态分配问题。([vLLM][1])
* LMCache：把 KV 从单个 engine 的临时状态提升成可跨 GPU/CPU/节点存储、迁移、共享的对象。([GitHub][2])
* CacheRoute：开始显式做 prefix-affinity routing，也就是请求调度会考虑“KV 已经在哪里”。([arXiv][3])

所以 ICN 的切入点不能只是“KV 有个名字”，而应该放在 **logical identity、placement、replication、lookup 和 allocation policy** 上。ICN 本来的核心就是“按名字找内容，而不是按地址找主机”，同时允许缓存、复制和 Interest 聚合。([RFC 编辑器][4])

我先给你一个可以直接拿去继续扩展的切入文档。

---

# ICN-Inspired KV Memory Allocation for Distributed LLM Serving

## 1. 问题定义

当前 LLM serving 中，KV cache 通常被看成某个 request / sequence 在某张 GPU 上的私有运行时状态。

典型流程：

```text
request
  ↓
scheduler chooses worker
  ↓
worker allocates KV blocks
  ↓
block table maps logical positions
to physical GPU addresses
```

PagedAttention 已经把 KV 从“连续物理地址”解耦成 block/page，通过 block table 做非连续物理分配，这极大减少了碎片。([vLLM][1])

但它本质上还是：

> **先决定在哪个 worker 上算，再在这个 worker 上找/分配 KV。**

我们的切入点是把这个顺序反过来一部分：

> **KV 本身成为可命名、可发现、可复制的 cluster-level object；KV 的已有位置参与 request routing 和 memory allocation。**

---

## 2. 核心抽象：Named KV Object

不再把 KV block 的主要身份定义为：

```text
GPU 2 / physical block 18421
```

而是定义一个与位置无关的逻辑名字。

例如：

```text
/model/qwen35b
/session-or-prefix/8af31...
/layer/3
/token-span/256-511
/encoding/m_sp4-k8v8
/version/17
```

物理位置只是这个逻辑 object 当前的 locator：

```text
GPU0 HBM
GPU3 HBM
CPU RAM
NVMe
remote node
```

这个思想和 ICN 的“Name → Data，而不是 Name → Host”一致；如果使用 name-resolution 架构，则 name 可以进一步解析成 locator / off-path cache pointer。([IETF Datatracker][5])

---

## 3. 先只做 Memory Allocation，不碰复杂网络协议

第一版不需要 NDN packet，不需要 PIT/FIB 全套。

只借 ICN 的三个核心思想：

1. **content identity 与 location 解耦**
2. **同一个 KV object 可以有多个 replica**
3. **allocation / routing 根据 object locality 决策**

做一个逻辑上的：

```text
KVName → KVLocationSet
```

例如：

```text
KVName X
→ GPU1:block_77
→ GPU4:block_21
→ CPU:offset_0x...
```

这个东西可以理解成一个非常简化的 Name Resolution Service。

RFC 9236 对 ICN 的 NRS 定义本身就是把 content name 映射到 routable prefix、locator 或 off-path cache pointer。([IETF Datatracker][5])

---

## 4. Allocation 不再只是“找空 block”

传统 allocator 大概是：

```text
new request arrives
→ find free GPU blocks
→ allocate
```

新的 allocator 可以变成：

```text
new request arrives
→ derive required KV names
→ lookup existing replicas
→ estimate reuse / transfer / queue cost
→ choose allocation strategy
```

可以有三种选择：

### A. Compute follows KV

如果 GPU3 已经有这个 request 需要的大量 prefix KV：

```text
route request → GPU3
```

即使 GPU3 不是当前最空闲的 worker。

这就是 locality-aware scheduling。

CacheRoute 的结果已经说明这种 cache affinity 能显著提高 KV hit rate，并在其 60×H100 实验里带来明显吞吐收益。([arXiv][3])

### B. KV follows compute

如果 GPU3 很忙，而 GPU7 很空：

```text
copy compressed KV
GPU3 → GPU7
```

再在 GPU7 执行。

你们现在 v8 已经把 KV 大幅压缩，这点反而非常关键：

> KV 越小，迁移成本越低，越有可能选择“搬 KV”而不是“等有缓存的 GPU”。

### C. Recompute

如果：

```text
transfer cost > recompute cost
```

则直接重新 prefill。

于是 allocator 的决策可以写成：

$$
C_j =
C_{\text{queue},j}
+
C_{\text{transfer},j}
+
C_{\text{recompute},j}
+
C_{\text{memory pressure},j}
$$

选择：

$$
j^*=\arg\min_j C_j
$$

这里真正有意思的地方就是：

> **KV locality 不再是一个 binary cache-hit 条件，而是 memory allocation 的 cost term。**

---

## 5. Memory Tier 也可以 ICN 化

现在 LMCache 已经支持：

```text
GPU
CPU RAM
local SSD
remote storage
```

之间的 KV offload / sharing / transfer。([LMCache][6])

所以我们可以定义类似：

```text
Tier 0: local GPU HBM
Tier 1: peer GPU HBM
Tier 2: host RAM
Tier 3: NVMe
Tier 4: remote node
```

每个 named KV block 带 metadata：

```text
name
size
encoding
quality
locations[]
last_access
reuse_count
transfer_cost
```

allocator 不再只做 eviction，而是做：

```text
promote
demote
replicate
move
drop
```

这已经非常接近 ICN 的 managed caching / replication 思路。RFC 8793 也明确区分 opportunistic caching 与 managed caching。([RFC 编辑器][4])

---

## 6. 你们 v8 给这个思路增加了一个很特别的维度

普通 KV serving 系统通常只有：

```text
有 KV / 没 KV
```

你们现在实际上可能有：

```text
BF16 KV
INT8 KV
m_sp4 compressed KV
m_sp4+k8v8 KV
```

也就是说一个逻辑 KV object 可以有多个 representation：

```text
/KV/X
 ├── BF16 @ GPU2
 ├── INT8 @ GPU4
 ├── compressed @ CPU
 └── compressed replica @ remote node
```

这就很像 ICN 中“同一内容的多个缓存/副本”，但这里还多了一层：

> **representation quality / size trade-off**

于是 allocation 可以同时决定：

```text
where?
+
which representation?
```

目标函数可以变成：

$$
C =
\alpha \cdot latency
+
\beta \cdot transfer\_bytes
+
\gamma \cdot memory\_pressure
+
\delta \cdot quality\_penalty
$$

这我觉得是最有新意的地方之一。

---

## 7. 第一版真正要做的 Memory Allocation Prototype

第一版完全没必要碰网络。

就做一个单机多 GPU simulator / runtime：

```text
4 or 8 GPUs

每张卡:
- capacity
- queue length
- resident KV names
- free blocks
```

请求到达时：

```text
request
→ required_prefix_hash / KV names
→ query name index
→ choose:
   local hit
   peer copy
   recompute
→ allocate physical blocks
```

和普通策略比较：

```text
Baseline 1:
least-loaded GPU

Baseline 2:
cache-affinity only

ICN-like:
name-aware allocation
+ replica-aware placement
+ transfer/recompute decision
```

先看几个系统指标：

```text
KV hit rate
remote KV fetch bytes
recompute tokens
GPU memory utilization
queueing delay
throughput
request migration count
```

这已经可以回答这个切入有没有东西。

---

## 8. 再往后可以加入 Replication

如果某个 prefix / KV object 很热：

```text
reuse_count ↑
```

系统不必一直把请求往同一个 GPU 塞。

可以主动：

```text
replicate KV X
GPU2 → GPU5
```

然后：

```text
future requests
→ GPU2 or GPU5
```

这就是非常 ICN 的 hot-content replication。

而且最近的 CacheRoute 也已经暴露了 locality 和 load balance 的冲突：固定 affinity 虽然提高缓存命中，但可能造成热点 worker 过载。它的解决办法就是对热点 key 做 planned placement / multiple destinations。([arXiv][3])

所以这里我们可以直接定义：

$$
Benefit_{\text{replica}}
=
ExpectedReuse \times SavedCost
-
CopyCost
-
MemoryCost
$$

当收益大于阈值：

```text
create replica
```

这会比单纯“cache-aware routing”更接近 ICN placement。

---

## 9. PIT 思路放到第二阶段

PIT 我觉得有意思，但不要第一版就上。

以后如果很多并发请求同时需要：

```text
KVName X
```

而 X 当前不在本地：

普通系统可能：

```text
request1 → fetch X
request2 → fetch X
request3 → fetch X
```

可以做成：

```text
pending[X] = {req1, req2, req3}
```

只发一次 remote fetch：

```text
fetch X once
↓
wake all waiters
```

这就是非常直接的 Interest aggregation。

NDN 的 PIT 本来就是把多个针对同一个 Data name 的未满足 Interest 聚合起来，只向上游转发一次。([RFC 编辑器][4])

但这个等 memory allocation / name lookup 跑起来以后再加就好。

---

## 10. 这个方向真正应该避免的坑

最重要的是不要做成：

> “PagedAttention block 加一个 content hash，所以我们用了 ICN。”

这个没东西。

真正的区别应该体现在：

```text
普通：
address-oriented block allocation

我们：
name-oriented KV allocation
+ location-independent resolution
+ multi-replica placement
+ cache-aware request routing
```

如果再结合你们自己的压缩：

```text
+ multiple KV representations
+ representation-aware movement
```

那就开始有自己的东西了。

---

## 11. 我会把第一阶段问题定义得很窄

先只问：

> **在多 GPU serving 中，如果 KV block 拥有 location-independent name，并让 allocation 同时考虑 existing KV locality、GPU load 和 KV transfer cost，是否能比传统 least-load / cache-affinity allocation 获得更好的并发与 memory utilization？**

这个问题足够具体。

如果第一阶段能看到：

```text
更高 cache hit
更少 recompute
更平衡 GPU utilization
更高 concurrency
```

再加：

```text
dynamic replication
PIT-like fetch aggregation
compressed representation selection
```

就很自然。

我觉得这个方向最有趣的一点，是你们现在不是空想“ICN + LLM”。PagedAttention 已经完成了**虚拟地址化**，LMCache 已经把 KV 做成**跨 engine 可迁移的 cache object**，CacheRoute 已经证明**cache locality 可以指导 request routing**。([vLLM][1])

你们可以继续往前走一步：

> **把 KV 从可迁移的 block，提升为真正 location-independent、name-addressed 的 serving object，并让 memory allocation / replication / routing围绕它统一决策。**

这个作为切入点，我觉得是成立的。

[1]: https://vllm.ai/blog/2023-06-20-vllm?utm_source=chatgpt.com "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention | vLLM Blog"
[2]: https://github.com/LMCache/LMCache?utm_source=chatgpt.com "GitHub - LMCache/LMCache: LMCache: Supercharge Your LLM with the Fastest KV Cache Layer · GitHub"
[3]: https://arxiv.org/abs/2608.19677?utm_source=chatgpt.com "CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving"
[4]: https://www.rfc-editor.org/info/rfc8793/?utm_source=chatgpt.com "RFC 8793: Information-Centric Networking (ICN): Content-Centric Networking (CCNx) and Named Data Networking (NDN) Terminology | RFC Editor"
[5]: https://datatracker.ietf.org/doc/html/rfc9236?utm_source=chatgpt.com "RFC 9236 - Architectural Considerations of Information-Centric Networking (ICN) Using a Name Resolution Service"
[6]: https://docs.lmcache.ai/getting_started/quickstart/index.html?utm_source=chatgpt.com "More Examples | LMCache"
