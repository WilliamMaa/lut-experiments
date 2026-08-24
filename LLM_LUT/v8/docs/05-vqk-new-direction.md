# Transformer VQK Next-Step Development Plan

## 1. Objective

接下来不再做“VQK feasibility test”。

当前目标改为：

> **围绕 Transformer 低 bit 数值计算，设计并比较多种 VQK-based implementation，找到最适合 decoder-only LLM 的 weight / activation representation 和 arithmetic organization。**

VQK 的核心价值不是单纯把 BF16 weight 压成 INT4，而是把主要矩阵乘法转成：

```text
low-bit integer representation
+
block-level scale / exponent
+
integer accumulation
```

原始 DSConv 就是利用低 bit VQK 与低 bit activation mantissa 做 block-wise dot product，再用 KDS / exponent 恢复尺度。

现有 LLM 工作也表明，真正困难的地方不是“weight 能不能4bit”，而是 decoder-only 模型中的 activation fluctuation、token variation 和 block distribution mismatch。I-LLM 专门通过 dynamic integer-only MatMul 处理 token/channel activation 波动；BCQ 则使用 block clustering 和不同 codebook 处理不同 block 的统计差异。

---

# 2. Current Target

第一阶段统一使用：

```text
Model:
Qwen3.6-35B-A3B

Layer:
39

Module:
model.model.layers[39].self_attn.o_proj
```

暂时不换 module。

原因不是因为 `o_proj` 一定最好，而是为了固定实验对象，让不同 VQK 设计能够公平比较。

只有确定一种较好的 numerical representation 后，再迁移到：

```text
v_proj
q_proj
k_proj
down_proj
gate_proj
up_proj
```

---

# 3. What We Are Actually Replacing

原始：

[
y = Wx
]

其中：

```text
W: BF16
x: BF16
accumulation: BF16 / FP32
```

目标逐渐变成：

[
W \approx s_w Q_w
]

[
x \approx s_x Q_x
]

因此：

[
Wx
\approx
s_ws_x(Q_wQ_x)
]

真正的大规模 dot product：

[
Q_wQ_x
]

使用低 bit integer arithmetic。

我们要优化的不是单独的 weight reconstruction，而是整个：

```text
representation
→ arithmetic
→ scaling
→ output
```

pipeline。

---

# 4. Plan A — Static Block VQK

这是最接近原始 DSConv 的版本。

## 4.1 Weight

对每个 block：

[
W_B \approx s_B Q_B
]

其中：

```text
Q_B: signed INT4
s_B: FP16 / BF16
```

第一轮：

```text
weight bits = 4
block size = 32 / 64 / 128
```

先不要同时扫 bit 和 block。

固定：

```text
W4
```

只看 block size。

---

## 4.2 Activation

第一版使用：

```text
A8
```

即：

```text
W4A8
```

activation 每个 token 动态计算一个 global scale：

[
x_t \approx s_tQ_t
]

这样主体执行：

```text
INT4 × INT8
→ integer accumulation
```

最后乘：

[
s_Bs_t
]

恢复尺度。

---

## 4.3 第一组实验

```text
A1: W4A8 block=32
A2: W4A8 block=64
A3: W4A8 block=128
```

记录：

```text
PPL
logit KL
top-1 agreement
top-5 agreement
generation
weight memory
scale overhead
effective bits / weight
```

这一组的目的不是决定“VQK行不行”，而是找到 static VQK 的合理 block granularity。

---

# 5. Plan B — Dynamic Activation VQK

如果 activation 使用单一 token scale 仍然产生较明显误差，不结束实验，而进入 dynamic block activation。

I-LLM 指出 LLM activation 在不同 token 和 channel 之间存在明显波动，这是低 bit integer-only inference 的核心困难之一。

因此改成：

[
x_{t,B}
\approx
s_{t,B}Q_{t,B}
]

即：

> 每个 token、每个 activation block 独立 scale。

---

## 5.1 Configurations

固定最好的 weight block size。

然后测试：

```text
B1:
W4
A8
activation scale = per-token

B2:
W4
A8
activation scale = per-token/per-block

B3:
W4
A6
activation scale = per-token/per-block

B4:
W4
A4
activation scale = per-token/per-block
```

不要从一开始直接 W4A4。

先：

```text
A8
→ A6
→ A4
```

找 activation bit 的下降边界。

---

# 6. Plan C — Clustered VQK

如果不同 block 的 optimal scale / quantization error 差异明显，则进入 clustered representation。

BCQ 的核心观察就是：不同 tensor block 的统计分布不同，用一个统一 quantizer 会损失明显；它通过根据 block statistics 聚类，再为不同 cluster 使用不同 codebook。

我们不需要完全复制 BCQ，而是把这个思想融合到 VQK。

---

## 6.1 Basic Form

首先为每个 weight block提取统计特征：

```text
mean
std
max_abs
kurtosis
outlier ratio
```

然后聚类：

```text
cluster_count = 2 / 4 / 8
```

不同 cluster 使用不同量化规则：

[
W_B
\approx
s_BQ_{c(B)}
]

例如：

```text
Cluster 0:
normal symmetric INT4

Cluster 1:
narrow-range INT4

Cluster 2:
outlier-aware INT4

Cluster 3:
alternative codebook
```

---

## 6.2 第一轮不要训练复杂 codebook

先做最简单版本：

```text
4 clusters
+
each cluster has its own clipping threshold / quantization range
```

如果有效，再升级成 learned codebook。

---

# 7. Plan D — Outlier-Aware VQK

这是 clustered VQK 的另一条更简单分支。

如果分析发现误差主要来自少数 extreme weight / activation values，则不需要改变全部block。

可以：

```text
normal values
→ VQK INT4

outliers
→ INT8 / FP16 side channel
```

即：

[
W = W_{\text{VQK}} + W_{\text{outlier}}
]

但 `W_outlier` 必须非常稀疏。

第一轮测试：

```text
top 0.1%
top 0.5%
top 1%
```

权重保留成高精度。

观察能否用非常小的 side-channel storage 换明显更低的 PPL。

---

# 8. Plan E — VQK + Bit-Sliced Arithmetic

如果 VQK representation 已经稳定，就继续减少真正的 multiplication cost。

例如：

```text
weight: INT4
activation: INT8
```

将 activation：

[
Q_x = 16H + L
]

拆成两个 4-bit component。

于是：

[
Q_wQ_x
======

16Q_wH
+
Q_wL
]

这里：

```text
Q_w ∈ INT4
H ∈ UINT4
L ∈ UINT4
```

所以 multiplication 可以用：

```text
4-bit × 4-bit tiny product LUT
```

---

## 8.1 Product LUT Size

输入组合：

[
16\times16=256
]

因此乘法表只有：

```text
256 entries
```

而不是之前 Hex-Dual Split 那种：

```text
in_dim × 16 × out_dim
```

巨大 response table。

这条线的重点是：

> **LUT 只替换 arithmetic primitive，而不是存整个 Linear response。**

---

## 8.2 Compare

统一比较：

```text
Integer MAC VQK

vs

VQK + 4×4 product LUT

vs

VQK + shift/add implementation
```

这里才真正开始测：

```text
latency
memory traffic
power / op estimate
```

---

# 9. Development Sequence

接下来严格按以下顺序执行。

## Stage 1 — Static VQK Design

实现：

```text
W4A8

block:
32
64
128
```

不要跑别的 module。

输出：

```text
v8_stage1_results.json
```

必须包含：

```json
{
  "block_size": 64,
  "weight_bits": 4,
  "activation_bits": 8,
  "ppl": 0,
  "ppl_delta": 0,
  "logit_kl": 0,
  "top1_agreement": 0,
  "top5_agreement": 0,
  "weight_bytes": 0,
  "scale_bytes": 0,
  "effective_bits_per_weight": 0
}
```

---

# 10. Stage 2 — Dynamic Activation Scaling

选 Stage 1 最好的 block。

跑：

```text
W4A8 global-token scale
W4A8 token-block scale
W4A6 token-block scale
W4A4 token-block scale
```

这里最重要的曲线是：

[
\text{Activation bits}
\rightarrow
\text{PPL}
]

如果 W4A4明显恶化，但 W4A6稳定，那么直接保留 W4A6。

不存在“必须做到4bit”。

现有 Transformer 研究也发现，decoder-only 模型对直接 W4A4 更敏感，因此把 activation bit 当成设计变量而不是硬目标更合理。

---

# 11. Stage 3 — Distribution Analysis

对 Stage 2 最好的配置采集真实 rollout activation。

至少：

```text
100 prompts
1024 tokens / prompt
```

记录每个 block：

```text
min
max
mean
std
max_abs
99%
99.9%
outlier ratio
quantization error
output error
```

然后回答：

> error 是普遍的，还是由少数 block / token / outlier 主导？

---

# 12. Stage 4 — Choose Plan C or D

根据 Stage 3 自动分叉。

### 如果 error 与 block distribution 强相关

做：

```text
Clustered VQK
```

先：

```text
2 clusters
4 clusters
8 clusters
```

---

### 如果 error 主要来自极少量 outliers

做：

```text
Outlier-aware VQK
```

先：

```text
0.1%
0.5%
1%
```

high precision side channel。

---

### 如果两者都有

做：

```text
4-cluster VQK
+
0.1% outlier side channel
```

但不要第一步直接组合。

---

# 13. Stage 5 — Model-Level Expansion

只有在 `o_proj` 的 numerical representation 定下来以后，才换 module。

顺序：

```text
o_proj
↓
v_proj
↓
q_proj
↓
k_proj
```

每一个 module 先单独测。

然后：

```text
q+k+v+o
```

做整个 attention projection replacement。

不要再一层一层随便换不同设计。

必须保持：

> 同一个 VQK representation。

---

# 14. Stage 6 — FFN Projection

attention projection成立以后，再测试：

```text
down_proj
up_proj
gate_proj
```

因为 FFN 和 attention 的 activation distribution 可能完全不同。

不要假设同一个 block size 可以直接复用。

---

# 15. Stage 7 — Multi-Layer

module level稳定以后，再做：

```text
Layer 39
↓
Layer 38–39
↓
Layer 37–39
↓
Last 8 layers
```

每增加一组层记录：

```text
PPL accumulation
logit KL accumulation
generation stability
effective model memory
integer compute %
```

这里可以参考 CBQ 的一个重要观察：低 bit error 会跨 block / layer 累积，因此多层结果不能简单从单层误差外推。

---

# 16. Stage 8 — Arithmetic Implementation

只有前面的 numerical representation稳定以后，才做真正 low-bit kernel。

目标：

```text
INT4 × INT8
INT4 × INT6
INT4 × INT4
```

实现三个 backend：

```text
1. reference dequant GEMM

2. integer accumulation simulation

3. bit-sliced / LUT arithmetic simulation
```

这样可以严格分开：

> numerical error

和：

> kernel implementation error / performance。

---

# 17. Do Not Do

接下来暂时不要做以下事情。

### 不要继续只跑 VQK-4 weight-only

因为：

```text
INT4 weight
→ dequant BF16
→ BF16 GEMM
```

只能证明 weight representation，不是最终 VQK compute path。

---

### 不要拿 RTN INT4 当项目目标

RTN可以保留作为普通 reference。

但项目问题不是：

> VQK 是否比 RTN 准？

而是：

> 哪种 block low-bit representation + arithmetic architecture 最适合 Transformer？

---

### 不要现在引入 AWQ/GPTQ 竞争

它们解决的是成熟的 weight-only PTQ 问题。

当前阶段重点不是重新做一个 weight-only quantizer。

---

### 不要同时碰 KV Cache

KV Cache 是另一条 runtime-state compression 路线。

VQK 是：

```text
weight + activation + arithmetic
```

两者分别开发。

后续可以组合，但现在实验完全独立。

---

# 18. Immediate Tasks

接下来实际需要写的代码只有四件事。

## Task 1 — Activation Quantizer

新增：

```text
v8_activation_quantizer.py
```

支持：

```text
INT8
INT6
INT4

per-token
per-token-per-block
```

输出：

```text
integer tensor
scale
```

---

## Task 2 — Integer VQK Linear Reference

新增：

```text
v8_integer_vqk_linear.py
```

实现逻辑：

```text
x
↓
activation quantization
↓
Qx

W
↓
VQK quantization
↓
Qw

Qw @ Qx
↓
integer accumulator
↓
weight scale × activation scale
↓
output
```

第一版允许 PyTorch integer / FP simulation，不追求速度。

重点是数学路径正确。

---

## Task 3 — Stage-1 Runner

新增：

```text
run_v8_stage1.py
```

自动跑：

```text
block=32
block=64
block=128
```

固定：

```text
W4A8
layer39.o_proj
```

---

## Task 4 — Unified Evaluation

输出：

```text
PPL
PPL delta
logit KL
top1
top5
generation
weight storage
scale overhead
activation storage
effective bits
```

必须一轮直接输出完整结果，不再人工从多个log里拼。

---

# 19. First Milestone

第一阶段完成标准不是：

> “VQK成功。”

而是得到一个明确的 design point，例如：

```text
Layer39 o_proj
W4A8
weight block = 64
activation = per-token-per-block
```

并知道：

```text
quality loss
memory reduction
scale overhead
integer arithmetic structure
```

然后进入下一阶段：

```text
W4A6 / W4A4
+
distribution analysis
```

---

# 20. Expected Branching

最终开发树应该是：

```text
                         Transformer VQK
                               |
                        Static W4A8
                               |
                  -------------------------
                  |                       |
           Activation OK            Activation unstable
                  |                       |
             reduce bits          dynamic block scale
                  |                       |
             W4A6 / W4A4                 |
                  |                       |
                  -----------+------------
                              |
                     distribution analysis
                              |
               -----------------------------
               |                           |
        block mismatch                 outliers
               |                           |
       Clustered VQK              Outlier-aware VQK
               |                           |
               -----------+----------------
                           |
                       multi-module
                           |
                       multi-layer
                           |
                    low-bit kernel
                           |
                 bit-sliced / LUT arithmetic
```

这才是接下来完整的路线。

不是：

```text
跑一个VQK
↓
看好不好
↓
不好就停
```

而是：

> **每一轮实验都用于判断下一步应该改变哪一个设计维度。**

这也是现在最需要的开发方式。
