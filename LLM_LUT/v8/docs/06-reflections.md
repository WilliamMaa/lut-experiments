# Attention Compact Memory / LUT Development Plan

## 1. 目标

FFN LUT 已经证明：高维连续计算不一定需要逐元素精确恢复，只要找到合适的离散/组合表示，就可以用较小状态近似整个子层功能。

Attention 现在要解决的是同一个层级的问题：

> **不是优化 QK 的一部分，也不是优化某个乘法，而是把完整历史 KV memory 压缩成少量 memory states，使 QK、softmax、V aggregation 和 KV storage 一起下降。**

原始 causal attention：

$$
y_t=
\sum_{i=1}^{N}
\operatorname{softmax}(q_t^\top k_i)_i v_i
$$

目标变成：

$$
\hat y_t=
\operatorname{Attention}
(q_t,\tilde K_{1:M},\tilde V_{1:M})
$$

其中：

$$
M \ll N
$$

如果能做到：

```text
4096 KV states → 128 memory states
```

则同时缩小：

```text
QK calculation      32×
softmax length       32×
V aggregation        32×
KV state count       32×
KV reads             32×
```

这才是 Attention 对应 FFN LUT 的“整块替代”问题。

已有工作已经证明历史 KV 并非同等重要：H2O 发现少数 heavy-hitter token 占据主要 attention contribution；StreamingLLM 发现初始 attention sinks 和最近窗口具有特殊作用；PyramidKV 则发现不同层的 KV 需求明显不同。

我们不复现这些方法。它们只说明一个事实：

> **历史 memory 存在结构性冗余。**

---

# 2. 核心研究问题

只回答三个问题。

### Q1. Attention memory 是否可以被“大比例合并”？

不是删除，而是：

```text
多个历史 KV
→
一个 compressed memory state
```

如果 16:1 / 32:1 merge 都完全炸掉，那么后续 learned memory / LUT memory 很难成立。

---

### Q2. 如果能压缩，决定质量的是“保留哪些 token”，还是“怎么合并信息”？

这是最重要的结构判断。

如果 oracle 选最重要 token 很好，而 merge 很差：

> 重点应该做 routing / selection。

如果 merge 明显比单纯保留 token 好：

> 重点应该做 compact memory representation。

WeightedKV 的观察与这个问题直接相关：Keys 和 Values 的冗余性质不同，简单同时 eviction 会丢失 value information，因此它保留 key anchors、同时把被删 token 的 value 信息合并回去。

---

### Q3. Compact memory 能否最终变成离散 / LUT representation？

只有 Q1/Q2 得到积极结果后才进入这一阶段。

目标结构：

```text
raw KV history
      ↓
memory merge
      ↓
M compact states
      ↓
discrete / compositional codes
      ↓
LUT reconstructed memory
      ↓
attention
```

---

# 3. 不做什么

这一阶段明确停止：

```text
VQK
weight quantization
activation quantization
product LUT
QK-only LUT
softmax LUT
bit slicing
```

这些都不解决完整 Attention 的结构问题。

也暂时不实现：

```text
H2O
PyramidKV
StreamingLLM
KIVI
SnapKV
```

这些可以以后作为论文 baseline。

**探索阶段不花工程时间复现它们。**

---

# 4. 只保留两个 Diagnostic Baseline

## B0 — Full Attention

原始 BF16 KV。

用途：

```text
teacher / reference
```

没有别的意义。

---

## B1 — Oracle Top-M

这个 baseline 很重要，而且实现非常便宜。

用 Full Attention 的真实 attention score：

$$
A_i
$$

直接取：

$$
S=\operatorname{TopM}(A)
$$

只保留这些 KV，再重新计算 attention。

例如：

```text
N = 4096

M = 2048
1024
512
256
128
64
```

这不是一个可部署方案。

它只回答：

> **如果我们拥有完美 selection，在只保留 M 个 memory units 时，模型理论上还能保留多少能力？**

这是一个真正有用的 upper bound。

### 决策

如果：

```text
Oracle 128/4096
```

都已经非常差：

> 不应该继续研究 token-selection sparse attention。

因为连完美选择都救不了。

如果 Oracle 128 很好：

> selection/routing 有巨大空间。

---

# 5. 第一个真正的方法 baseline：Segment Merge

这不是为了发表，也不需要优化。

它只回答：

> **memory merge 这个基本假设是否成立。**

把连续 B 个 KV 合并：

$$
\tilde k_j
=
\frac1B\sum_{i\in C_j}k_i
$$

$$
\tilde v_j
=
\frac1B\sum_{i\in C_j}v_i
$$

测试：

```text
B = 2
4
8
16
32
64
```

因此：

```text
4096 → 2048
4096 → 1024
4096 → 512
4096 → 256
4096 → 128
4096 → 64
```

只需要一天级别代码。

---

# 6. Stage 0 — 建立 Attention Probe

不要先完整生成。

先做一个离线 probe。

## 数据

使用真实模型 rollout：

```text
100 prompts
每条 2048–4096 tokens
```

记录少量层：

```text
early  : layer 8
middle : layer 24
late   : layer 39
```

不需要40层全部记录。

每层记录：

```text
Q
K
V
attention score
attention probability
attention output
residual input/output
```

目的只是避免 layer39 单点结果误导整个方向。

---

# 7. Stage 1 — Memory Compressibility Test

只跑：

```text
Full
Oracle Top-M
Segment Mean Merge
```

memory budget：

```text
100%
50%
25%
12.5%
6.25%
3.125%
1.56%
```

即：

```text
4096
2048
1024
512
256
128
64
```

---

## 指标

不要只看 cosine。

### Attention 层

```text
attention output relative error
attention output cosine
```

### Transformer 层

将压缩后的 attention output 注回模型：

```text
post-attention hidden cosine
post-layer hidden cosine
```

### 最终输出

最重要：

```text
logit KL
top-1 agreement
teacher-token probability
```

第一轮甚至不需要 PPL。

---

# 8. Stage 1 的结果如何决定方向

这是整个实验最关键的地方。

## 情况 A

```text
Oracle 很好
Segment Merge 很差
```

例如：

```text
128 Oracle → 几乎无损
128 Merge  → 很差
```

结论：

> memory cardinality 可以大幅下降，但不能简单 merge。

下一步：

# → Plan A：Selection / Routing

不研究 reconstruction。

---

## 情况 B

```text
Oracle 好
Segment Merge 也不错
```

结论：

> Attention 历史存在很强的可压缩 representation。

下一步：

# → Plan B：Learned Compact Memory

这是最理想结果。

---

## 情况 C

```text
Oracle 差
Merge 也差
```

例如降到25%就明显崩。

结论：

> 当前 pretrained softmax attention 强依赖大量独立历史 states。

停止 token-memory compression。

下一步直接研究：

# → Plan C：Recurrent / Linearized Attention State

不再浪费时间优化 clustering。

---

# 9. Plan A — Selection / Routing

只有 Stage 1 情况 A 才做。

目标不是做 H2O clone，而是确定：

> 能不能用非常廉价的 representation 找到 oracle-important memory。

第一版只尝试：

```text
recent window
+
key similarity
```

例如：

$$
score_i
=
\alpha q^\top k_i
+
\beta Recency(i)
$$

然后取 Top-M。

如果普通 key similarity 已经接近 Oracle：

> routing 很简单。

如果差很多：

> 说明历史 importance 不能仅从当前 Q/K 判断，需要 learned router。

这时候才考虑：

```text
small LUT router
```

形式：

```text
Q state
+
K memory code
→
importance
```

这里 LUT 才真正有价值：

> 它负责决定哪些完整 memory unit 值得进入 attention。

---

# 10. Plan B — Learned Compact Memory

这是目前最值得期待的方向。

不是删除 KV，而是：

```text
K1,V1
K2,V2
...
KB,VB
   ↓
one memory state
   ↓
K~, V~
```

第一版不训练 neural network。

---

## 10.1 Key-based clustering

在一个历史窗口内：

$$
\{k_i\}
$$

按 key similarity 聚类。

例如：

```text
4096 KV
→
128 clusters
```

保存：

$$
\tilde k_j
$$

作为 cluster key。

---

## 10.2 Value aggregation

不能简单平均 V。

保存：

$$
\tilde v_j
=
\sum_{i\in C_j}w_i v_i
$$

第一版：

```text
uniform
```

第二版：

```text
historical attention-weighted
```

WeightedKV 已经给出有用的经验：保留 key anchor，同时合并被压缩 token 的 value 信息，比简单 KV eviction 更有潜力。

---

## 10.3 Softmax mass correction

如果一个 memory state代表：

```text
m_j
```

个原始 token，那么它不应该和一个单token state有完全相同的softmax质量。

第一版加入：

$$
s_j
=
q^\top\tilde k_j+\log m_j
$$

因为如果cluster内各key非常接近：

$$
\sum_{i\in C_j}
e^{q^\top k_i}
\approx
m_j e^{q^\top\tilde k_j}
$$

所以：

$$
\log
\sum_i e^{q^\top k_i}
\approx
q^\top\tilde k_j+\log m_j
$$

这个实验非常重要。

比较：

```text
Cluster Merge
Cluster Merge + log(mass)
```

如果 `log(mass)` 明显改善结果：

> 我们已经找到了 compact attention state 的必要结构。

---

# 11. Plan B 第一轮只比较三个方法

相同 memory budget：

```text
Segment Mean
Key Cluster
Key Cluster + Mass Correction
```

不要再增加十个baseline。

例如固定：

```text
4096 → 256
4096 → 128
```

只测两个 compression ratios。

---

# 12. Stage 2 的决策

## Cluster ≈ Segment

说明：

> KV相似性不是主要结构。

不要继续调 clustering。

直接进入 hierarchical / recurrent state。

---

## Cluster >> Segment

说明：

> 相似 key 可以被合并。

继续研究：

```text
how to merge V
how to preserve softmax mass
```

---

## Mass Correction >> Cluster

这是最有价值的结果。

意味着我们已经得到一个明确模型：

> **Attention memory可以被表示成 anchor key + aggregated value + probability mass。**

这会成为整个研究的核心。

---

# 13. Plan B2 — Hierarchical Memory

如果 merge 成立，再扩大。

不要所有历史统一压缩。

使用三级：

```text
Recent
Medium
Old
```

例如：

```text
Recent:
last 256 tokens
exact KV

Medium:
previous 1792 tokens
8:1 merge
→ 224 states

Old:
remaining history
32:1 merge
```

于是 context 可以增长：

```text
4K
32K
128K
```

但实际 attention units 保持几百个。

StreamingLLM说明 recent tokens 与初始 sink tokens具有特殊稳定作用；因此保留 recent exact memory 是一个有文献依据、同时几乎零工程成本的设计选择。

这不是要复现 StreamingLLM。

只借用：

> **不要把所有历史一刀切。**

---

# 14. 真正接 LUT 的位置

只有 compact memory 工作以后再做。

假设每个 compressed memory state：

$$
(\tilde k,\tilde v,m)
$$

仍然很大。

然后使用我们已有的 compositional representation：

$$
\tilde k
\rightarrow
(c_1,\dots,c_G)
$$

$$
\tilde v
\rightarrow
(d_1,\dots,d_H)
$$

只保存 codes：

```text
K codes
V codes
mass
```

查询时：

```text
code
↓
small LUT
↓
reconstruct / approximate K,V
```

因此最终：

```text
Original Attention

4096 × full K/V
        ↓

Compact Memory

128 × full K/V
        ↓

LUT Compact Memory

128 × discrete codes
```

这是两级压缩。

---

# 15. 为什么这和 FFN LUT 是同一条研究路线

FFN：

```text
high-dimensional activation
→
compact discrete representation
→
approximate whole FFN
```

Attention：

```text
large historical state set
→
compact memory representation
→
approximate whole attention
```

共同思想不是：

> 查表代替乘法。

而是：

> **丢掉内部不重要的自由度，用低维/离散状态保留最终功能。**

---

# 16. Plan C — Recurrent Attention State

只有 Stage 1 情况 C 才进入。

如果历史memory无法通过merge压缩，那么彻底换表示。

目标：

$$
S_t = f(S_{t-1},k_t,v_t)
$$

然后：

$$
y_t=g(q_t,S_t)
$$

history不再以token为单位保存。

这类似 linear/recurrent attention 的基本思想。

但是第一阶段不实现。

原因很简单：

> 如果简单memory compression已经能工作，就没有必要立即修改softmax attention的基本数学结构。

---

# 17. 第一周具体开发任务

## Day 1 — Attention collector

新增：

```text
collect_attention_states.py
```

保存：

```text
layer
head
position
Q
K
V
attention_probs
attention_output
```

只采：

```text
layers = [8, 24, 39]
```

---

## Day 2 — Compression evaluator

新增：

```text
evaluate_compact_attention.py
```

支持：

```text
full
oracle_topk
segment_mean
```

参数：

```text
--memory_budget 2048
--memory_budget 1024
--memory_budget 512
--memory_budget 256
--memory_budget 128
```

---

## Day 3 — 得到第一张核心表

形式：

| Method  | Memory states | Compression | Attn error | Post-layer cos | Logit KL | Top-1 |
| ------- | ------------: | ----------: | ---------: | -------------: | -------: | ----: |
| Full    |          4096 |          1× |          0 |              1 |        0 |  100% |
| Oracle  |           256 |         16× |            |                |          |       |
| Segment |           256 |         16× |            |                |          |       |
| Oracle  |           128 |         32× |            |                |          |       |
| Segment |           128 |         32× |            |                |          |       |

**只有这张表。**

不要画20张图。

---

## Day 4–5

根据结果只进入一条路线。

### A：Oracle好 / Merge差

写：

```text
routing_attention.py
```

---

### B：Merge也好

写：

```text
cluster_merge_attention.py
```

支持：

```text
key clustering
value merge
mass correction
```

---

### C：全部差

停止memory merge。

开始设计：

```text
recurrent_attention_state.md
```

---

# 18. 第二阶段真正的方法实验

如果进入 Cluster Merge，只测：

```text
M = 256
M = 128
```

方法：

```text
Segment
Key Cluster
Key Cluster + Mass
```

不再扫：

```text
cluster algorithm
20 distance metrics
10 window sizes
50 seeds
```

除非一个结果明确显示这些参数是主要瓶颈。

---

# 19. 一个非常重要的 Stop Rule

以后任何实验开工前，都必须有：

```text
Result A → 下一步 A
Result B → 下一步 B
Result C → 停止
```

例如：

### Segment Merge

```text
好
→ learned merge

差，但 oracle 好
→ routing

oracle 也差
→ recurrent state
```

所以这个实验无论结果是什么，都提供信息。

---

# 20. 不允许再出现的实验

例如：

```text
K=2
K=4
K=8
K=16
```

如果我们事先说不清楚：

> K=2好说明什么？
> K=8好说明什么？
> 两者都差又说明什么？

那就不跑。

同理：

```text
32
64
128 block
```

如果只是“看看哪个最好”，也不跑。

---

# 21. 当前第一目标

现在不要想着论文最终方法。

这周只需要回答：

> **对于一个预训练 Qwen Attention，4096 个历史 KV state 能不能在不大幅破坏最终 logits 的情况下，压成 128–256 个有效 memory states？**

也就是：

$$
16\times\sim32\times
$$

memory-unit compression。

如果答案是 Yes：

> 我们终于有了 Attention 对应 FFN LUT 的基础结构。

如果答案是 No：

> 我们立即知道 token-wise compact memory 不是方向，转 recurrent/linear state。

这两个结论都足够重要，不会再出现“做三天发现这个实验本身就毫无意义”的情况。

---

# 22. 最终预期结构

如果路线成立，目标不是普通 KV compression，而是：

```text
                    Raw Attention
                         |
                  historical KV
                         |
                 memory formation
                         |
          -----------------------------
          |             |             |
        exact        merged         coarse
        recent       medium          old
          |             |             |
          -------- compact memory -----
                         |
                  discrete coding
                         |
                       LUT
                         |
                compressed attention
```

最终希望做到：

$$
O(N)
$$

历史 attention units 变成近似：

$$
O(M),\qquad M\ll N
$$

甚至在 hierarchical memory 下让 \(M\) 随 context length 增长得非常慢。

这才是下一阶段真正值得做的 Attention 工作。
