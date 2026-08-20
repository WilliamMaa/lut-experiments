# Transformer VQK Experimental Development Plan

## 1. Goal

在现有 LUT 和 KV Cache Compression 路线之外，增加一条 **VQK-based Transformer Quantization** 实验线。

目标是验证：

> **VQK + block-wise distribution shift 是否能够在低 bit 权重量化下，比普通 INT quantization 更好地保持 Transformer 的 PPL 和生成质量。**

VQK 来源于 DSConv。原方法将权重拆成：

* **VQK (Variable Quantized Kernel)**：低 bit integer 权重；
* **KDS (Kernel Distribution Shift)**：每个 block 一个高精度 scale，用于恢复原始权重分布。

DSConv 原论文在 CNN 上使用这种结构，并报告在无需 retraining 的情况下，4-bit quantization 可以保持较高精度。

---

# 2. Basic Form

对于 Transformer Linear：

[
y=Wx
]

将权重近似为：

[
W\approx S\odot W_q
]

其中：

* (W_q)：2/3/4/6/8-bit integer VQK；
* (S)：每个 block 对应的 FP16/BF16 scale；
* block 沿 input dimension 划分。

DSConv 中 VQK 与原始 weight tensor 保持相同 shape，而 KDS 只保存每个 block 的 floating-point scaling parameter，因此 block size 决定精度与额外浮点参数之间的 trade-off。

---

# 3. First Target Modules

第一轮不要直接量化所有 Transformer Linear。

优先测试：

```text
o_proj
v_proj
down_proj
```

随后再测试：

```text
q_proj
k_proj
gate_proj
up_proj
```

原因：

* `q_proj / k_proj` 的误差会直接改变 attention score；
* `gate_proj / up_proj` 位于 FFN 非线性和门控之前，可能更加敏感；
* `o_proj / v_proj / down_proj` 更适合作为第一轮可行性测试。

---

# 4. Phase 1 — Single Module Sensitivity

先固定一层，例如：

```text
layer 39
```

依次替换：

```text
layer39.o_proj
layer39.v_proj
layer39.down_proj
```

测试：

```text
bits:
8
6
4
3
2
```

以及：

```text
block size:
32
64
128
256
```

---

# 5. VQK Initialization

第一版直接使用 DSConv 风格的 L2 initialization。

对于一个 block：

[
W_B
]

首先得到 integer kernel：

[
W_{q,B}
]

然后求：

[
S_B
===

\frac{
\sum_i W_iW_{q,i}
}{
\sum_i W_{q,i}^2
}
]

该 closed-form scaling 是 DSConv 原论文实际采用的方法，因为其效果与 KL-based optimization 接近，同时计算更加简单。

---

# 6. Baselines

必须至少比较：

```text
BF16
INT8 standard quantization
INT4 standard quantization
VQK-8
VQK-6
VQK-4
VQK-3
VQK-2
```

最关键的比较不是：

```text
VQK vs BF16
```

而是：

```text
VQK-4 vs normal INT4
VQK-3 vs normal INT3
```

如果相同 bit-width 下没有明显优势，就不需要继续复杂化 VQK。

---

# 7. Metrics

继续沿用当前 LUT 实验经验，不以 weight MSE 或 output cosine 作为最终判断标准。

## Local Metrics

```text
Weight MSE
Linear output cosine
Linear output MSE
Post-residual cosine
```

## Functional Metrics

```text
PPL
Logit KL
Top-1 agreement
Top-5 agreement
Generation quality
EOS success rate
Repetition rate
```

## System Metrics

```text
Weight memory
Bytes read / token
MAC type
Decode latency
Throughput
```

---

# 8. Phase 2 — Activation-Aware VQK

原始 DSConv 主要解决 plug-and-play quantization，并不依赖训练数据。

我们已经有真实 rollout 数据，因此可以进一步优化 scale。

不再只求：

[
|W-SW_q|^2
]

而是使用真实 activation：

[
\min_S
\mathbb E_{x\sim D}
\left[
|Wx-SW_qx|^2
\right]
]

其中：

```text
D = real rollout activations
```

这样 scale 优化的是实际模型访问到的输入分布。

---

# 9. Phase 3 — Logit-Aware Calibration

如果 activation-aware VQK 有效果，再加入 downstream objective：

[
L
=

\lambda_1L_{\text{linear}}
+
\lambda_2L_{\text{post-residual}}
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
p_{\text{VQK}}
)
]

原因是当前 LUT 实验已经说明：

> 更高的 local cosine 不一定意味着更低的 PPL 或更好的 generation。

因此 VQK 的目标最终也应该从 weight reconstruction 转向 functional preservation。

---

# 10. Phase 4 — Multi-Layer Scaling

单层稳定以后再扩展：

```text
L39
→ L38–39
→ L37–39
```

分别记录：

```text
PPL
logit KL
generation
memory reduction
```

观察退化是否近似线性累积。

如果最后三层仍然稳定，再扩展到更多层。

---

# 11. Optional — VQK + Activation Quantization

如果 weight-only VQK 成立，再尝试 activation quantization：

```text
VQK-4 weight
+
INT8 activation
```

以及：

```text
VQK-4
+
INT4 / BFP activation
```

进一步观察是否能够把主要计算转换成 integer MAC。

---

# 12. Optional — VQK + Bit-Sliced / LUT Arithmetic

后续可以将低 bit VQK 与当前研究的 bit-slicing 思路结合。

例如：

```text
4-bit VQK weight
+
8-bit activation
```

计算：

[
w_qx_q
]

可以进一步尝试：

```text
small product LUT
```

或：

```text
bit-sliced multiply/add
```

形成：

```text
BF16 Linear
↓
VQK weight compression
↓
activation quantization
↓
integer / LUT arithmetic
```

但这一阶段只有在 VQK 本身已经证明有价值后再做。

---

# 13. First Experiment Matrix

| ID | Method        | Bits | Block |
| -- | ------------- | ---: | ----: |
| B0 | BF16          |   16 |     — |
| B1 | Standard INT8 |    8 |     — |
| B2 | Standard INT4 |    4 |     — |
| V1 | VQK           |    8 |    64 |
| V2 | VQK           |    6 |    64 |
| V3 | VQK           |    4 |    64 |
| V4 | VQK           |    3 |    64 |
| V5 | VQK           |    4 |    32 |
| V6 | VQK           |    4 |   128 |
| V7 | VQK           |    4 |   256 |

第一轮优先跑：

```text
layer39.o_proj
```

只有结果合理再扩展到其他 module。

---

# 14. Decision Criteria

继续 VQK 路线需要满足至少一个条件：

### Same-bit advantage

```text
VQK-4
明显优于
normal INT4
```

### Same-quality advantage

在相同 PPL / generation quality 下：

```text
VQK
使用更少 bit
```

### Multi-layer robustness

多层量化时：

```text
VQK degradation
<
standard quantization degradation
```

如果只是 local cosine 更高，但：

```text
PPL ↑
generation ↓
```

则直接停止该配置。

---

# 15. Relation to Other Current Directions

当前可以把 attention-side optimization 分成三条独立路线：

```text
A. Weight / Projection
   → VQK

B. State Memory
   → KV Cache Compression

C. Attention Computation
   → LUT routing / approximate QK / block selection
```

这样即使完整 Attention LUT 最终不可行，也仍然可以分别优化：

```text
weights
+
KV memory
+
attention routing
```

---

# 16. Current Development Order

建议按以下顺序：

```text
1. layer39.o_proj VQK baseline

2. 8/6/4/3/2-bit sweep

3. block size 32/64/128/256

4. compare with standard INT4/INT8

5. activation-aware scale calibration

6. test v_proj / down_proj

7. test q_proj / k_proj

8. L37–39 multi-layer test

9. optional logit-aware calibration

10. optional VQK + bit-sliced arithmetic
```

当前第一目标非常简单：

> **先确认 VQK 在 Transformer Linear 上是否真的比普通低 bit quantization 更有价值。**

只有这一点成立，后续 on-policy、logit-aware 和硬件化才值得继续。
