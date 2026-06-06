# LLM-LUT v2 Proposal: Functional Replacement

> 停止 table design ablation，接受 2-head uniform joint bucket 为当前最优方法，推进到 functional replacement 阶段。

---

## 0. 前置结论

v0 → v1.0 → v1.1 → v1.2 的所有实验表明：

| 方法 | KL | Recovery |
|------|-----|----------|
| 1D bucket | 0.835 | 76.1% |
| **2D bucket** | **0.607** | **82.7%** |
| Trainable 2D LUT | 0.606 | 82.7% (+0.03%) |
| Learned codebook | 0.826 | 76.4% ❌ |
| Additive (ANOVA) | ~0.65 | ~81% ❌ |
| Additive + 8×8 interaction | ~0.63 | ~82% ❌ |

**2D fixed bucket 是当前最强、最稳定、最可解释的方法。**

后续任何 table design 的边际收益 < 1%，不值得继续投入。

---

## 1. 核心目标

不再问"能不能设计更好的 LUT"，而是问：

> **2-head bucket 能否在真实 forward 中稳定替换 MLP residual contribution group？**

这是从 **perturbation experiment** 到 **functional replacement model** 的跨越。

---

## 2. R1: 0.5B Replacement Model

### 2.1 替换规格

```
Model:      Qwen2.5-0.5B-Instruct
Layer:      6
Target:     mlp_delta group 4
Method:     2-head uniform joint bucket
Bins:       64 × 64
Table size: 64 × 64 × 64 = 256KB
```

### 2.2 替换逻辑

```
original:
  x_next = x + attn_delta + mlp_delta

replacement:
  x_next = x + attn_delta + mlp_delta'

where:
  mlp_delta'[:, group_4] = LUT_2D[bin(a1), bin(a2)]
  mlp_delta'[:, other_groups] = original mlp_delta[:, other_groups]
```

### 2.3 评估维度

| 维度 | 方法 | 成功标准 |
|------|------|----------|
| **PPL** | eval set | ≤ 42 (≈ bucket) |
| **Acc** | next-token | ≥ 0.48 (≈ bucket) |
| **KL** | vs original | ≤ 0.61 (≈ bucket) |
| **Generation** | 固定 prompt 生成 | 无 collapse、无重复、语义连贯 |

### 2.4 输出

- 一个可保存/加载的 replacement checkpoint（table + address params）
- 一份 generation sanity 报告

---

## 3. R2: 1.5B Scaling Check

### 3.1 目标

验证 2D bucket replacement 不是 0.5B 的特例。

### 3.2 轻量 Scan

```
Model: Qwen2.5-1.5B-Instruct (28 layers)
Layers: [7, 14, 21]  (25%, 50%, 75% depth)
Candidate: mlp_delta groups only
Method: zero / mean / 2-head bucket
```

不再做 binning sweep / codebook / additive。只找 strongest group。

### 3.3 替换

找到 strongest group 后，直接构建 R2 replacement（同 R1 逻辑）。

---

## 4. R3: 3B Confirmation（条件触发）

如果 R2 有信号，再上 Qwen2.5-3B-Instruct（36 layers）。

---

## 5. 研究叙事

不是：
> "我们加速了大模型。"

而是：
> "我们证明了一个 selected MLP residual contribution group 可以被 2-head LUT functionally replaced with controlled degradation."

后续如果想谈计算收益：
> "这为把 selected residual contribution 的生成路径从 dense computation 转向 lookup-dominated execution 提供了基础。"

---

## 6. 路线图

```
v2.0  (现在): R1 — 0.5B replacement model + generation eval
v2.1  (下一步): R2 — 1.5B light scan + replacement
v2.2  (条件触发): R3 — 3B confirmation
v3    (远期): Compute-removal path
```

---

## 7. 停止列表

| 停止 | 理由 |
|------|------|
| Codebook | 已验证不如 fixed bucket |
| Additive decomposition | 已验证不如 fixed bucket |
| Interaction table | 已验证不如 fixed bucket |
| Trainable LUT | 边际收益 < 0.1% |
| Multi-group replacement | R1 先做单 group 稳定替换 |
| 3-head / 4-head | 2-head 已足够 |
