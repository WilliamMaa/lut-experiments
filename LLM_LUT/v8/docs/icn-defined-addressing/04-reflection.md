对，现在最应该做的就是把 **architecture 本身重新立起来**。先不讨论 E1–E5，不讨论压缩，也先不讨论“move/recompute 三选一”这种局部策略。我们先回答一个更高层的问题：

> **一个真正 ICN 化的 LLM serving architecture，到底应该比现有 vLLM / LMCache 多哪一层？**

我查完现有系统后，我觉得架构其实可以比较清楚地画出来。

现有 vLLM 的基本逻辑仍然是：

```text
Request
   ↓
Scheduler
   ↓
Worker / Engine
   ↓
Worker-side KV cache
```

Automatic Prefix Caching 能根据 token prefix 的 hash 复用已经存在的 KV block，典型 workload 就是 long-document query 和 multi-round conversation。([vLLM][1])

LMCache 则进一步把 KV 从 GPU-local cache 扩展成：

```text
GPU
 ↕
CPU DRAM
 ↕
NVMe / remote store
```

而且现在的 MP architecture 已经能让多个 vLLM instance 共用一个 cache service；P2P 模式甚至明确把不同节点的 KV cache 变成一个“logical cache”，本地 miss 后可以直接从 peer 用 RDMA 读取。([LMCache][2])

所以这些能力已经存在：

```text
content hashing       ✓
prefix reuse          ✓
cross-instance KV     ✓
KV transfer           ✓
multi-tier storage    ✓
P2P lookup            ✓
```

这意味着我们的 architecture **绝对不能只是把这些东西重新拼一次。**

---

## 我现在认为最合理的 ICN architecture 是两平面

核心不是“加 ICN cache”。

而是把整个 serving system 分成：

```text
              REQUEST / COMPUTE PLANE
                       │
                       │
                       ▼
              ┌─────────────────┐
              │ Global Scheduler│
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      GPU 0          GPU 1          GPU 2
      Worker         Worker         Worker


              INFORMATION PLANE
                       │
                       ▼
             ┌────────────────────┐
             │ KV Object Directory│
             │                    │
             │ Name → Locator Set │
             └────────────────────┘
                      │
       ┌──────────────┼───────────────┐
       ▼              ▼               ▼
     GPU0 KV        GPU1 KV         GPU2 KV
       │              │               │
     CPU/NVMe       CPU/NVMe        CPU/NVMe
```

最重要的变化是：

### compute resource 和 information resource 分开建模。

传统 scheduler 主要管理：

```text
worker
queue
GPU load
memory availability
```

ICN architecture 再维护一套独立状态：

```text
KV object identity
location set
replica count
residency
popularity
lifetime
```

然后这两个 plane **联合决策**。

我觉得这才是 architecture-level contribution。

---

# 1. 第一核心组件：Global Content Namespace

不是 session cache。

例如：

```text
KVName =
H(model_version,
  prefix_tokens,
  inference-relevant metadata)
```

得到：

```text
KVA
KVB
KVC
...
```

关键是这些对象**不属于某个 worker**。

不是：

```text
GPU0.KV_A
```

而是：

```text
KV_A
  locators:
    GPU0/HBM
    GPU3/HBM
    Node2/DRAM
```

这就是最直接的 identity/location separation。

RFC 9236 的 NRS 本质上就是这种 name → locator / locator-set 思想，一个名字可以解析到实际持有内容的位置。([LMCache][3])

这才是我们应该真正借 ICN 的地方。

---

# 2. 第二核心组件：KV Object Directory / NRS

例如：

```text
KV_A
├── GPU0 / HBM
├── GPU3 / HBM
└── Node1 / DRAM

KV_B
├── GPU1 / HBM
└── GPU2 / HBM
```

但是 directory 不能只是：

```text
name → address
```

真正 serving architecture 还需要：

```text
name
→ {
    locators,
    size,
    prefix_length,
    last_access,
    access_frequency,
    replica_count
  }
```

同时 compute plane 有：

```text
worker
→ {
    queue_length,
    active_sequences,
    memory_free,
    load
  }
```

然后 scheduler 同时看两套信息。

---

# 3. 第三个核心：Joint Compute–Information Scheduler

我现在觉得这才应该是论文算法的中心。

请求 \(r\) 到达：

```text
request r
   ↓
derive reusable prefix objects
   ↓
KVName lookup
   ↓
ContentState + ComputeState
   ↓
Joint Scheduler
```

它不是单独问：

> 哪个 GPU 最空？

也不是只问：

> 哪个 GPU 有 cache？

而是：

$$
j^*
=
f(
\text{compute load},
\text{KV locality},
\text{memory pressure},
\text{transfer cost},
\text{reuse expectation}
)
$$

这就是我们一直缺失的 architecture abstraction。

---

# 4. 第四核心：Placement / Replication Manager

这个我觉得不能只是“第二版功能”。

如果我们真想做 ICN architecture，**placement 和 replication 本来就是 architecture 的一部分**。

因为只做 lookup：

```text
KV_A 在 GPU0
→ request 去 GPU0
```

那其实就是 affinity routing。

真正 ICN 化以后，系统还要主动决定：

```text
KV_A 非常热门
→ GPU0 queue overloaded
→ replicate KV_A to GPU3
```

或者：

```text
KV_B 很冷
→ GPU2 pressure high
→ evict GPU2 copy
```

于是形成：

```text
               KV_A
             /      \
          GPU0      GPU3

               KV_B
                 |
               GPU1
```

这才是真正意义上的 **distributed content placement**。

LMCache 已经证明 cross-node KV sharing 和 P2P fetch 是现实机制，不需要我们重新造 transport。它明确描述：不同节点各自保存自己服务请求的 KV，本地 miss 后可以通过 RDMA 从持有 prefix 的 peer 获取。([LMCache][4])

我们的价值应该是：

> **谁应该保存什么、保存几份、请求应该跟哪一份走。**

---

# 5. 第五核心：KV Data Plane

这里我反而觉得**完全不应该创新**。

直接：

```text
LMCache
NIXL
RDMA
NVLink / PCIe
```

谁现成就用谁。

Data plane 只负责：

```text
fetch
store
transfer
evict
```

不负责策略。

这非常重要，因为这样论文贡献就不会散。

Architecture 可以明确写：

```text
Control Plane
─────────────
Name resolution
Compute scheduling
Placement
Replication

Data Plane
──────────
Existing KV transfer/storage substrate
```

非常清楚。

---

# 我觉得第一版不要 P/D disaggregation

Mooncake 是典型的：

```text
Prefill Cluster
       ↓ KV
Decode Cluster
```

而且它已经有 KVCache-centric scheduler，并用 CPU/DRAM/SSD 构成 disaggregated KV cache。([arXiv][5])

如果我们第一版也从 P/D 开始，很容易和 Mooncake 的 problem definition 缠在一起。

第一版我更建议：

```text
                   Frontend
                      │
               ICN Scheduler
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
  vLLM W0         vLLM W1         vLLM W2
  GPU0            GPU1            GPU2
  prefill+decode   prefill+decode   prefill+decode
```

每个 worker 都是正常完整 serving instance。

然后另有：

```text
          Cluster KV Plane
       ┌──────────────────┐
       │ KV Object Index  │
       └──────────────────┘
          │       │      │
          W0      W1     W2
```

这样我们只研究一个问题：

> **How should reusable KV state and compute be jointly allocated across serving replicas?**

非常干净。

等这个做成以后再扩：

```text
P/D disaggregation
multi-node
CPU/NVMe tier
```

---

# 而且这时候，“并发”就自然成为 architecture 的工作环境

不是为了测试硬造一个 concurrency。

而是：

```text
100 concurrent requests
        ↓
   content demand
A A A A A B B C C D ...
        ↓
  KV object distribution
        +
 compute load distribution
```

scheduler 同时面对：

```text
GPU0:
  KV A
  queue 20

GPU1:
  KV B
  queue 2

GPU2:
  KV A, C
  queue 8
```

现在真正的问题才出现：

```text
新的 A request 到来
```

应该：

```text
→ GPU0?
→ GPU2?
→ GPU1 + fetch A?
→ replicate A?
```

这就是 architecture 在运行。

---

# Baseline 也因此自然出来了

以后不用再造一些很奇怪的 P0/P1/P2。

直接：

```text
Load-aware scheduler
        |
        | + KV affinity
        |
        | + global KV location
        |
        | + active placement/replication
        v
Full ICN architecture
```

换句话说：

### B0

普通 least-loaded。

### B1

prefix-affinity / sticky scheduling。

### B2

global location-aware scheduling。

### Ours

**joint scheduling + placement + replication。**

这比：

> route/move/recompute 三选一

更像 architecture research。

---

# 我现在最认可的 architecture 图就是这个

```text
                        ┌───────────────┐
Incoming Requests ─────►│ Frontend      │
                        └───────┬───────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │ Information-Centric      │
                 │ Cluster Scheduler        │
                 │                          │
                 │ Compute State            │
                 │ Content State            │
                 │ Placement Policy         │
                 └─────────────┬────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
             Worker 0       Worker 1       Worker 2
             GPU 0          GPU 1          GPU 2
               │              │              │
               └────── KV Data Plane ────────┘
                    LMCache / NIXL / RDMA

                         ▲
                         │
                 ┌───────┴────────┐
                 │ KV Name Service │
                 │                │
                 │ Name→Locators  │
                 │ Popularity     │
                 │ Replicas       │
                 └────────────────┘
```

然后整个 paper 的一句话可以变成：

> **We decouple reusable KV state from serving workers and jointly manage information placement and compute placement at cluster scale.**

这个比我们前几轮讨论的东西干净太多了。

而且有一点现在也很明确：**vLLM APC、LMCache、Mooncake 都不是敌人，它们是在帮我们搭 substrate。** APC 已经证明 prefix 是 reusable content；LMCache 已经提供跨 instance 的存储/搬运；Mooncake 已经证明 KV state 对 cluster scheduling 很重要。([vLLM][6])

我们真正需要新增的 architecture 层，是：

$$
\boxed{
\text{global content identity}
+
\text{content placement}
+
\text{compute placement}
+
\text{joint control}
}
$$

我觉得现在先把这个 architecture 定住，然后下一步再讨论 **“一个 request 从进入系统到退出，到底经过哪些状态转换”**。那一步做完以后，才应该开始写 scheduler 和实验。

[1]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
[2]: https://docs.lmcache.ai/mp/?utm_source=chatgpt.com "Overview | LMCache"
[3]: https://docs.lmcache.ai/zh_CN/mp/p2p.html?utm_source=chatgpt.com "P2P KV 缓存共享 | LMCache"
[4]: https://docs.lmcache.ai/kv_cache/p2p_sharing.html?utm_source=chatgpt.com "P2P KV Cache Sharing | LMCache"
[5]: https://arxiv.org/abs/2407.00079?utm_source=chatgpt.com "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
[6]: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
