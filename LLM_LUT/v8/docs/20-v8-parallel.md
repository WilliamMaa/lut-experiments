对，我觉得 **v8 现在必须重新作为一条独立主线继续推进**，不能让它沦为 ICN 项目的“一个 representation 选项”。

而且看 2026 最近这批工作以后，我反而觉得这条线还有明显空间。现在 KV compression 已经不只是“省显存”这么简单：有工作开始强调 attention-aware distortion、long-reasoning information preservation、multi-turn agent workload、以及现实场景里的 instruction degradation。([arXiv][1])

你们现在 v8 的独特性其实很清楚：

> **attention-score selection + convex merge + span-aware preservation**

特别是 M4 解决了 multi-token fact 被 tokenizer 拆散后逐 token HH 选择失效的问题。这个跟现有很多 “只看 token importance / head importance / quantization precision” 的路线并不完全一样。

我会把 v8 后续拆成三步。

### 第一，先重新把 evaluation 做强

你现在的 53-turn 自建集已经证明“方法不崩”，但不够支撑后续投稿。

下一轮优先补：

* **parallel structured retrieval**：我们之前说的 single prefill + parallel fields/candidates；
* **multi-instruction prompting**：ACL 2026 已经专门指出 H2O、SnapKV 等压缩方法在这类场景会让某些 instruction 快速失效，甚至出现 system-prompt leakage；这正好是你们 span-aware preservation 应该攻击的场景。([ACL 论文集][2])
* **long reasoning / long factual context**：InfoKV 这类 2026 工作已经开始强调 attention score 并不能完全描述长程信息价值。([arXiv][3])

所以不要只继续扩大原来的 EOS / repetition 集。

应该问：

> **M4 到底保住了什么类型的信息？**

---

### 第二，把 v8 放进真正 memory-constrained serving 场景

这一点 UltraQuant 很值得参考。它不是只报 perplexity/accuracy，而是直接放到 **context-heavy multi-round agents** 里测 cache residency、TTFT、throughput；它报告 4-bit KV 在 cache-pressure late rounds 下 P50 TTFT 改善 3.47×，整体 output throughput 提高 1.63×。([arXiv][4])

你们也应该有一条：

```text
Full KV
HH
HH + merge
HH + merge + M4
M4 + k8v8
```

在固定：

```text
GPU memory budget
concurrency
long context
multi-turn
```

下面比较：

* sustainable concurrency；
* TTFT；
* throughput；
* HBM footprint；
* factual/structured preservation。

这样 1000× / 2000× 才不只是“标称 compression ratio”。

---

### 第三，再决定算法要不要继续演进

我现在不建议立刻加新机制。

先用新的 evaluation 去找真正的 failure mode。

因为最近工作已经给了几个很明确的竞争方向：

* **CompressKV**：semantic retrieval heads + layer-wise budget，LongBench 上只留 3% KV 还能保 >97% full-cache performance。([arXiv][5])
* **HqeKV**：quantization + eviction 联合优化，而不是只做一种压缩。([ACL 论文集][6])
* **AATC**：开始从 attention-aware rate-distortion / transform coding 的角度分配 bits。([arXiv][1])

所以 v8 下一步最重要的不是：

> 再发明 M5。

而是先回答：

> **为什么 M4+merge 在极端 compression 下还能工作，而这些信息到底在哪类任务中最容易丢？**

如果 parallel / multi-instruction 测试发现：

```text
HH         short fact OK / span fact 崩
merge      probability recovery
M4         structured fact recovery
```

那你们的方法解释会比现在强很多。

---

所以我会把现在两个项目完全分开：

```text
Track A — ICN
where should reusable inference state reside?

Track B — v8 compression
what information can be removed/merged while preserving behavior?
```

互相**不依赖**。

最后如果两个都做成，再考虑：

```text
compressed KV as one possible object representation
```

但那只能是后续 integration，不应该再倒回来定义任何一条主线。

对 v8 来说，我觉得现在下一步最合理的就是我们之前说的 **parallel test + stronger benchmark expansion**。然后再决定要不要动算法。当前这套方法还远没到“测试完了”的程度。

[1]: https://arxiv.org/abs/2608.14191?utm_source=chatgpt.com "KV Cache Compression Through the Lens of Transform Coding"
[2]: https://aclanthology.org/2026.acl-long.1926/?utm_source=chatgpt.com "The Pitfalls of KV Cache Compression - ACL Anthology"
[3]: https://arxiv.org/abs/2606.26875?utm_source=chatgpt.com "Information-Aware KV Cache Compression for Long Reasoning"
[4]: https://arxiv.org/abs/2606.20474?utm_source=chatgpt.com "UltraQuant: 4-bit KV Caching for Context-Heavy Agents"
[5]: https://arxiv.org/abs/2606.24467?utm_source=chatgpt.com "CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference"
[6]: https://aclanthology.org/2026.findings-acl.201/?utm_source=chatgpt.com "HqeKV: Towards Hybrid Quantization and Eviction for KV Cache in Long-Context LLM Inference - ACL Anthology"
