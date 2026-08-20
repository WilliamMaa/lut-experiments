# KV Cache Compression Experimental Development Plan

## 1. Goal

当前 LUT 路线已经验证了一个重要现象：局部 hidden-state cosine similarity 并不能可靠预测最终 PPL 和 generation quality。下一阶段除了继续测试多层 LUT 外，单独开启一条 **KV Cache Compression** 路线，目标是寻找比直接 LUT 化 attention 更稳、更容易部署的 attention-side 优化方法。

核心目标不是单纯压缩 KV cache，而是找到：

> **在显著降低 KV memory / bandwidth 的前提下，尽可能保留 attention 行为、PPL 和长上下文能力。**

第一轮不追求提出最终新方法，而是先建立完整 baseline 和 sensitivity map，再决定最有价值的研究方向。

---

## 2. Background

KV cache 会随着 sequence length 和 batch size 线性增长，并且 autoregressive decoding 每一步都需要访问历史 K/V，因此除了显存容量，它也是实际 inference memory-bandwidth 的重要来源。

现有工作的几个主要方向如下。

### 2.1 KV Quantization

KIVI 对 KV cache 的元素分布进行了专门分析，发现：

* Key cache 更适合 **per-channel quantization**；
* Value cache 更适合 **per-token quantization**；
* 基于这一观察，可以实现 tuning-free 2-bit KV cache quantization。

KIVI 报告在其测试模型上可显著降低 peak memory，并提升可支持 batch size 和 throughput。

这意味着 K 和 V 不应该简单采用完全相同的压缩策略。

---

### 2.2 Token Eviction / Importance Retention

Scissorhands 的核心观察是 **Persistence of Importance**：历史上已经表现出较高重要性的 token，未来仍更可能继续重要，因此可以在固定 KV budget 下优先保存 pivotal tokens。其论文报告 KV memory 可减少 2–5× 而维持较好的模型质量。

H2O 则观察到少数 Heavy-Hitter tokens 对 attention 输出贡献很大，并动态保留：

* recent tokens；
* historically important heavy hitters。

其工作把 KV eviction 建模成动态 submodular problem，并验证了保留重要 token 对生成质量的重要性。

---

### 2.3 Layer-Adaptive KV Budget

PyramidKV 观察到不同 Transformer layer 的 attention information distribution 不同：

* lower layers 的注意力更加分散；
* higher layers 更集中于少数关键 token。

因此它不是所有层使用同样 KV budget，而是采用 pyramidal allocation：低层保留更多、高层保留更少。

DynamicKV 进一步说明，不同 task 的 layer-wise KV requirement 也可能不同，因此固定的 layer allocation 并非总是最佳策略。

---

## 3. Research Questions

第一阶段主要回答五个问题：

1. Qwen3.6-35B-A3B 的 K 和 V 对 quantization 分别有多敏感？
2. 删除 token 和降低 bit-width，哪种方式对模型功能破坏更大？
3. 不同 layer 是否存在明显不同的 KV sensitivity？
4. learned / compositional codebook 是否能够比普通 scalar quantization 获得更好的 memory-quality trade-off？
5. 是否可以组合：

   * high-precision important KV；
   * compressed normal KV；
   * evicted unimportant KV；

   构成分层 KV cache？

---

# 4. Phase 0 — Evaluation Infrastructure

首先统一 KV compression evaluation framework。

## 4.1 Baseline

必须保留：

```text
Full BF16 KV Cache
```

所有压缩结果都相对于它比较。

---

## 4.2 Core Metrics

不要再只使用 cosine similarity。

每个配置至少输出：

### Model quality

```text
PPL
Logit KL
Top-1 agreement
Top-5 agreement
Generation quality
EOS success rate
Repetition rate
```

### Attention behavior

```text
Attention output cosine
Attention score correlation
Top-k attention recall
Retained attention mass
```

其中：

[
\text{Attention Mass Recall}
============================

\sum_{i\in S_{\text{kept}}}p_i
]

相比简单 token recall，这个指标更能表示被保留的 KV 是否真正重要。

### Memory

```text
KV bytes / token
Total KV cache size
Compression ratio
Peak GPU memory
```

### Runtime

```text
Prefill latency
Decode latency / token
Tokens / second
KV read bandwidth
```

第一阶段如果 runtime kernel 尚未优化，可以先把 latency 标为 simulation/prototype result，不要过早作为最终硬件结论。

---

# 5. Phase 1 — KV Quantization Baselines

这一阶段只回答：

> Qwen 本身允许把 KV 压到什么程度？

## 5.1 Experiments

至少测试：

```text
BF16 KV
INT8 KV
INT4 KV
INT3 KV
INT2 KV
```

然后增加 KIVI-style asymmetric variant：

```text
K: per-channel
V: per-token
```

分别测试：

```text
4-bit
3-bit
2-bit
```

KIVI 的 K/V 非对称量化设计就是基于两者不同的 outlier / distribution characteristics，因此应该作为重要 baseline，而不是简单统一 per-token quantization。

---

## 5.2 Required Output

输出：

```text
bit width
K quantization method
V quantization method
KV compression ratio
PPL
logit KL
long-context score
generation quality
```

形成第一条：

```text
Memory ↔ Quality
```

曲线。

---

# 6. Phase 2 — Token Eviction Baseline

第二阶段不改变 K/V 数值，只减少保存的 token。

## 6.1 Recent Window Baseline

最简单：

```text
keep last N tokens
```

测试：

```text
100%
75%
50%
25%
12.5%
```

这是最弱 baseline。

---

## 6.2 Recent + Sink

增加：

```text
initial sink tokens
+
recent window
```

观察模型对前部 token 和最近 token 的依赖。

---

## 6.3 Heavy-Hitter Retention

参考 H2O / Scissorhands 的思路，根据历史 attention importance 累积 token importance：

[
I_i(t)
======

\sum_{\tau \le t}
A_{\tau,i}
]

保留：

```text
recent tokens
+
top importance historical tokens
```

H2O 和 Scissorhands 都支持这样一个基本判断：KV cache 中不同 token 的价值高度不均匀，少数关键 token 应获得更高 retention priority。

---

## 6.4 Output

重点画：

```text
Retained KV %
vs
PPL
vs
Long-context accuracy
```

尤其看：

```text
50%
25%
12.5%
6.25%
```

几个压缩点。

---

# 7. Phase 3 — Layer Sensitivity

不要默认每层同样重要。

## 7.1 Single-Layer Ablation

针对每一层：

```text
只压缩 Layer i
其他 layer 保持 Full KV
```

分别测试：

```text
4-bit
2-bit
50% retention
25% retention
```

得到：

[
S_i=\Delta \text{PPL}_i
]

以及：

[
S_i^{long}
==========

\Delta \text{LongContextScore}_i
]

---

## 7.2 Layer Sensitivity Map

最终形成：

```text
Layer
↓
Quantization sensitivity
Eviction sensitivity
Attention entropy
Attention concentration
```

PyramidKV 的工作表明 lower / higher layers 可能具有明显不同的信息聚合模式，因此这一实验非常值得在目标模型上重新验证，而不能直接照搬已有模型结论。

---

# 8. Phase 4 — Layer-Adaptive KV Budget

根据 Phase 3 的结果设计三个 baseline。

## A. Uniform

```text
每层相同 KV budget
```

---

## B. Pyramidal

例如：

```text
Lower layers: 100%
Middle layers: 50%
Upper layers: 25%
```

具体比例由 sensitivity result 决定。

---

## C. Sensitivity-Aware

根据实验得到的 (S_i) 分配：

[
B_i
\propto
S_i
]

更敏感的层得到更多 KV budget。

最终比较：

```text
相同总 KV memory
不同 layer allocation
```

看 PPL 和 long-context performance。

---

# 9. Phase 5 — Learned Codebook KV

这一阶段开始进入我们自己的主要方向。

核心思想：

> 不再只使用 scalar quantization，而是让 K/V vector 通过小型离散 codebook 表示。

---

## 9.1 Basic Product-Codebook

将一个 K vector：

[
K\in\mathbb R^d
]

切成 (G) 个 group：

[
K=
[K_1,K_2,\dots,K_G]
]

每组找到一个 code：

[
c_g
===

\arg\min_j
|K_g-C_{g,j}|
]

KV cache 中不再保存完整 vector，而只保存：

```text
[c1, c2, ..., cG]
```

实际需要时：

[
\hat K
======

[
C_{1,c_1},
C_{2,c_2},
\dots,
C_{G,c_G}
]
]

V 同样处理。

---

## 9.2 Initial Configurations

建议第一轮不要扫太多。

例如：

```text
Group count:
8 / 16 / 32

Code width:
4 bit
6 bit
8 bit
```

形成大约：

```text
32–256 states / group
```

---

# 10. Phase 6 — Attention-Aware Codebook Training

不要只优化：

[
|K-\hat K|^2
]

因为 FFN 实验已经说明：

> Local reconstruction quality 不一定对应最终模型质量。

因此增加 downstream-aware objective。

## K Loss

除了 reconstruction：

[
L_K^{rec}
=========

|K-\hat K|^2
]

加入 attention score preservation：

[
L_{score}
=========

|
QK^\top-Q\hat K^\top
|^2
]

---

## V Loss

对 V：

[
L_V^{rec}
=========

|V-\hat V|^2
]

再加入 attention output：

[
L_{attn}
========

|
AV-A\hat V
|^2
]

---

## Final Objective

进一步可以加入：

[
L
=

\lambda_1L_{KV}
+
\lambda_2L_{attn}
+
\lambda_3L_{\text{logit}}
]

其中：

[
L_{\text{logit}}
================

KL(
p_{\text{teacher}}
|
p_{\text{compressed}}
)
]

这个目标应该比简单 KV cosine 更符合最终任务。

---

# 11. Phase 7 — On-Policy KV Data Collection

复用目前 FFN LUT 已经采用的 rollout 方法。

流程：

```text
Current compressed-KV model
        ↓
真实生成
        ↓
记录真实访问到的 K/V
        ↓
记录 attention importance
        ↓
记录 compression error
        ↓
记录 logit KL
        ↓
挑选高价值 KV states
```

优先保留：

```text
high attention mass
high reconstruction error
high logit KL
long-context positions
EOS附近
language switching
rare retrieval states
```

而不是随机收集所有 KV。

---

# 12. Phase 8 — Hot / Warm / Cold KV

如果单纯 learned codebook 有效果，再做真正有研究价值的混合方案。

将 token 分成三档。

## Hot KV

非常重要的 token：

```text
BF16
or
INT8
```

包括：

```text
attention sinks
recent tokens
heavy hitters
retrieved critical tokens
```

---

## Warm KV

普通历史 token：

```text
learned compositional code
```

例如：

```text
4–8 bit / group
```

---

## Cold KV

长期低 attention contribution：

```text
evict
```

最终：

[
KV_i=
\begin{cases}
KV_i^{high}, & i\in H\
C(code_i), & i\in W\
\varnothing, & i\in C
\end{cases}
]

这实际上把：

```text
Quantization
+
Learned compression
+
Eviction
```

组合成一个统一框架。

---

# 13. Possible Dynamic Policy

进一步可以定义 token importance：

[
I_i
===

\alpha A_i
+
\beta R_i
+
\gamma Recency_i
]

其中：

* (A_i)：historical attention importance；
* (R_i)：retrieval / reuse probability；
* `Recency`：时间局部性。

根据两个阈值：

[
\tau_{hot},\tau_{cold}
]

分配：

```text
I > τ_hot
→ Hot

τ_cold < I ≤ τ_hot
→ Warm

I ≤ τ_cold
→ Cold / evict
```

后续甚至可以让阈值随：

```text
layer
context length
task
memory pressure
```

动态变化。

DynamicKV 已经说明 task-dependent layer behavior 可能值得建模，因此如果静态 policy 有效果，可以把动态预算作为后续扩展。

---

# 14. First Evaluation Suite

不要第一轮就跑非常大的 benchmark。

先准备固定小测试集。

## Short / Normal Generation

```text
Wikipedia-like knowledge
Chinese reasoning
English reasoning
technical explanation
code generation
math
JSON structured output
```

---

## Long Context

至少包括：

```text
Needle-in-a-Haystack
long-document QA
long summarization
multi-turn history
```

因为 KV compression 的真正价值必须在长 context 下验证。

---

# 15. Experiment Matrix

第一轮建议只跑：

| ID | Method                      |
| -- | --------------------------- |
| B0 | BF16 Full KV                |
| B1 | INT8 KV                     |
| B2 | INT4 KV                     |
| B3 | KIVI-style 2-bit            |
| B4 | Recent-window 50%           |
| B5 | Heavy-hitter 50%            |
| B6 | Heavy-hitter 25%            |
| B7 | Layer-adaptive 25% average  |
| M1 | PQ / codebook KV            |
| M2 | Attention-aware codebook KV |

先不要做 Hot/Warm/Cold。

只有 M1/M2 明显有价值以后再进入混合方案。

---

# 16. Decision Criteria

## Quantization Line

如果：

```text
2–4 bit KV
```

已经几乎无损，则 learned codebook 必须在：

```text
同等 memory 下质量明显更好
```

或：

```text
同等质量下 memory 更低
```

才能继续。

---

## Eviction Line

如果：

```text
25% retained tokens
```

仍然保持非常高的 long-context performance，那么 token selection 可能比 vector compression 更值得做。

---

## Learned Codebook Line

继续推进的条件：

[
\text{Codebook Quality}

>

\text{Scalar Quant Quality}
]

在相同 KV bytes/token 下成立。

---

# 17. Main Metrics for Final Comparison

最终必须形成两个 Pareto 图。

## Memory vs Quality

[
x=
\text{KV Cache Size}
]

[
y=
\text{PPL / Long-context Accuracy}
]

---

## Bandwidth vs Quality

[
x=
\text{KV Bytes Read per Generated Token}
]

[
y=
\text{Task Quality}
]

第二张尤其重要。

KV optimization 真正的系统价值不只是：

> “cache 占多少显存”

而是：

> **每生成一个 token，需要从 memory hierarchy 搬多少 KV 数据。**

---

# 18. Current Recommended Development Order

当前不需要同时实现全部方案。

建议严格按以下顺序推进：

```text
Phase 0
统一 KV evaluation framework

↓

Phase 1
2/3/4/8-bit quantization sensitivity

↓

Phase 2
Token eviction sensitivity

↓

Phase 3
Layer sensitivity map

↓

Phase 4
Layer-adaptive budget

↓

Phase 5
Basic learned codebook KV

↓

Phase 6
Attention/logit-aware codebook

↓

Phase 7
On-policy hard-state calibration

↓

Phase 8
Hot / Warm / Cold hybrid KV
```

---

# 19. Expected Research Story

如果最终 learned KV 方案成功，论文主线不应该只是：

> “We compress KV cache using a codebook.”

而应该是：

> Existing KV compression methods primarily exploit either reduced numerical precision or token-level importance. We investigate whether KV states can instead be represented using compact compositional discrete codes, while allocating precision according to downstream importance. The proposed method combines learned state representation with attention-aware and on-policy calibration, aiming to reduce both KV storage and memory traffic while preserving end-to-end model behavior.

这个方向和当前 FFN LUT 工作有一个统一的核心思想：

> **不要求内部状态逐元素精确，而是寻找小型离散表示，只保留最终模型功能真正需要的信息。**

如果最后 FFN LUT 和 KV compression 两边都能成立，它们甚至可以进一步统一成一个更大的方向：

> **Compositional discrete representations for reducing LLM inference computation and memory movement.**
