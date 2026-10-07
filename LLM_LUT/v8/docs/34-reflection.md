这版结果里，**最需要立刻停下来核对的不是 v8，而是你对 full-KV baseline 的那句“2 卡只能扛 1–2 个 64k 请求”**。按 Qwen3.6-35B-A3B 的官方结构，这个说法现在看起来非常可疑，甚至大概率不成立。

Qwen3.6-35B-A3B 有 40 层，其中只有 10 层是 full attention；每层 2 个 KV heads，head dim=256。[Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json?utm_source=chatgpt.com) 粗算 bf16 full-attention KV：

\[
2\;(K,V)\times2\;(KV heads)\times256\times2\text{B}
=2048\text{B/token/layer}
\]

10 个 full-attention 层就是：

\[
20\,480\text{B/token}\approx20\text{KB/token}.
\]

64k context 一个 request 的 full-attention KV 大约：

\[
65536\times20\text{KB}\approx1.25\text{GB}
\]

这是**整个模型合计**；TP2 后每卡大约 0.625 GB 左右。8 个 64k request 也就是约 **5 GB KV / GPU** 的数量级。即便再加 hybrid recurrent state，它也不应该突然变成“只能放 1–2 路”。

而两张 A800-80GB 上 35B bf16 权重粗略约 70GB 总量，即每卡约 35GB，理论上还剩几十 GB 给 KV/runtime。vLLM 本身也会基于 KV-cache capacity 计算 `kv_cache_max_concurrency`，而 hybrid 模型还专门使用 group-aware token capacity，说明这里应该直接读取真实 engine capacity，而不是凭之前的单请求结果推断。[vLLM](https://docs.vllm.ai/en/latest/api/vllm/config/index.html?utm_source=chatgpt.com)

所以你目前这句：

> full-KV 同时只能扛 1–2 个 64k 请求

**在 full sweep 跑出来以前应该删除。**

这会连带影响：

> “v8 以 0.95–0.98 fact_acc 扛 8 并发”

这个事实本身成立，

但：

> “因此并发能力大幅超过 full”

**现在还完全没证明。**

---

第二个明显问题是：**目前 1.43× throughput improvement 也不是 v8 相对 full 的 improvement。**

你现在比较的是：

```text
v8 N=1  → 46 sess/h
v8 N=8  → 63~66 sess/h
```

这是：

> v8 自己从 sequential serving 到 batched/concurrent serving 的收益。

它说明 vLLM batch scheduling 有收益，但不能归因给 KV compression。

真正工作目标应该是直接跑：

```text
full KV:
N = 1,2,4,8,16,...

v8-1024:
N = 1,2,4,8,16,...

v8-4096:
N = ...
```

然后在**相同 offered workload / quality gate / 两张 GPU**下比较：

\[
N_{\max,\text{full}}
\quad vs\quad
N_{\max,\text{v8}}
\]

和：

\[
QPS_{\text{full}}
\quad vs\quad
QPS_{\text{v8}}.
\]

现在 vLLM 本身就有 `max_num_seqs`、`max_num_batched_tokens` 和 chunked prefill 等真正控制并发的 scheduler 参数。[vLLM](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/?utm_source=chatgpt.com) 所以**full sweep 是下一步第一优先级，不是“后续补曲线”。**

---

第三，**1024 vs 4096 的速度解释不合理。**

你写：

> 4096 在高并发略快，符合“更多保留块 = 更少信息丢失重建”的直觉。

但按照你当前方法描述，decode 并不存在“信息丢失以后重新构建被淘汰 KV”这一过程。

1024 slots 理论上意味着 attention 读取更少的 retained KV：

\[
1024 < 4096
\]

如果其他条件相同，**1024 应该在 decode attention 上更便宜，而不是更慢**。

所以：

```text
N=8:
1024 → 429s
4096 → 398s
```

这 7% 目前只能先叫：

> run-to-run / scheduling / workload variance，或者存在别的 implementation overhead。

不能用“少重建”解释。

而且每档看起来只有一遍 run；至少需要相同配置重复 2–3 次再判断这个 7% 是否真实。

---

第四，**N=16 这一行基本没有实验意义。**

你自己已经指出只有 8 个 session，所以 N=16 实际不会有 16 个 active requests。

这意味着：

```text
N=8 ≈ N=16
```

根本不能用来说明 saturation。

我会直接把 N=16 从主结果表里拿掉，标成：

> invalid as a concurrency data point because workload cardinality is 8.

否则读者很容易误读。

真正测 concurrency 需要至少：

```text
32 / 64 sessions
```

最好 requests 比最大 N 多很多，让 scheduler 始终有 backlog。

---

第五，质量这一块反而是目前比较好的。

同一数据下：

```text
1024: 0.953–0.984
4096: 0.969
```

每格 64 题，确实没有足够证据说 4096 比 1024 更好。

所以现在可以暂时说：

> **在当前 64k workload 上，没有观察到 1024→4096 带来明确 factual-quality 增益。**

而不要说：

> 1024 已经足够。

“足够”取决于业务质量门槛。

如果我们要求：

\[
Acc\ge0.95
\]

那当前两档都过。

如果要求：

\[
Acc\ge0.99
\]

那都没过。

这个门槛应该由应用目标定，而不是实验自己决定。

---

## 现在其实离你真正想要的答案已经很近了

之前最大的问题是 vLLM integration 不可靠。

现在 Gate 1–4 都通过，所有点：

- 0 crash；
- 0 contract error；
- 0 HTTP error；

说明 integration 至少已经可以开始做真正 capacity test 了。

所以我不会再继续折腾方法。

**下一步只干一个实验：full-KV baseline sweep。**

而且 workload 要改成足够大的 backlog，例如：

```text
64 sessions
64k context
8 turns
same 8-doc distribution

max concurrency:
1, 2, 4, 8, 16, 32
```

跑：

```text
full
v8-1024
v8-4096
```

所有配置同样：

- TP2；
- chunked prefill=8192；
- prefix caching off；
- enforce-eager；
- 相同 memory utilization；
- 相同 prompts；
- 相同 max tokens。

记录：

```text
OOM / admission failure
peak KV residency
peak process HBM
completed sessions/hour
P50/P95 end-to-end latency
fact accuracy
```

然后我们终于可以得到你最开始要的那句话：

> **On 2×A800 at 64k context and ≥X factual accuracy, v8 increases supported active concurrency from A to B (B/A×) and sustainable throughput from C to D (D/C×).**

如果最后 full 也能轻松 N=8，完全没关系。

继续往：

```text
16
32
64
```

压，直到某一边出现真实 memory/admission wall。

这才叫“v8 能增加多少并发”。

---

我现在最重要的判断就是：

**这批结果说明 v8 integration 已经开始可信，但还没有回答核心业务问题。最大的缺口不是再调 v8，而是缺 full-KV 同条件容量基线。**

而且根据模型结构粗算，我会特别警惕“full 只能 1–2 路”这个旧假设——它很可能低估了 full baseline。现在千万不要把这句话带进总结里，直接用实测把它钉死。