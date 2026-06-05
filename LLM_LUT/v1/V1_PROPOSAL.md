# LLM-LUT v1 Proposal: Trainable LUT Prefit

**Status**: Approved — triggered by v0.5 results.

---

## 1. v0.5 结论回顾

v0.5 三实验（binning sweep / multi-group combo / head ablation）验证了 Layer 6 `mlp_delta` 的 LUT 潜力：

| 指标 | 数值 | 含义 |
|------|------|------|
| Group 4, uniform 64 bins | **Recovery = 69.0%** | 最强信号，稳定且显著 |
| KL Zero → Bucket | 2.950 → **0.914** | bucket 替换大幅改善 |
| KL Mean | 2.938 | mean 替换几乎无效，非静态 bias |
| Coverage | 48.4% | 2-head 下表利用率足够 |
| Group 8 | Recovery < 3% | **排除**，组合时引发 nonlinear collapse |
| Uniform vs Quantile | Uniform 碾压 | quantile 当前实现有 bug，不上升为理论结论 |

**核心判断**：Layer 6 `mlp_delta` group 4 是一个 **activation-conditioned、addressable** 的候选，其行为能被 bucket lookup 部分解释，且不是偶然的静态偏置。这正好符合 LUT 化的前提。

---

## 2. v1 目标与收窄范围

### 2.1 核心科学问题

> **Trainable LUT 能否优于 non-trained bucket average？**

v0 和 v0.5 已经证明：bucket average 能把 group 4 的 KL 从 2.95 降到 0.91（69% recovery）。v1 要验证的是：如果这个 bucket 表变成 **可学习的参数**，是否还能进一步降低 approximation error，并最终反映在 model-level PPL / accuracy 上。

### 2.2 范围收窄（关键决策）

| 决策 | 说明 |
|------|------|
| **Layer**: 6 only | 先不扩展，避免引入变量 |
| **Candidate**: `mlp_delta` only | down_proj / attn_out 在 v0 中无信号 |
| **Group**: 4 only (v1.0) | 最强候选；group 3 留到 v1.1 |
| **Groups 排除**: 8, 9 | 单点 recovery 极低，组合时 collapse |
| **Binning**: uniform | quantile 需 debug 后再评估 |
| **Bins**: 64 | recovery 最高，coverage 可接受 |

**不做的事（v1 红线）**：
- 不碰 multi-group replacement（留到 v1.1）
- 不用 CE / full model loss 做端到端训练（先做 local prefit）
- 不扩展 layer scan（留到 v2）

---

## 3. 技术方案

### 3.1 Address 机制（继承 v0.5）

- **Source**: existing scalar activation from Layer 6 MLP gate / up projection
- **Calibration**: 固定 calibration set 上收集 address statistics，bin boundaries 固定
- **Heads**: **2-head** 为主，同时保留 **1-head** 作为 ablation control
- **Coverage**: 2-head 48.4%（v0.5），对 trainable LUT 更友好

### 3.2 LUT 表设计

```
Table shape: [num_bins^heads, group_dim]
             = [64^2, 64]  for 2-head
             = [64, 64]    for 1-head

Initializtion: bucket average (from v0.5 calibration)
Trainable:    yes, all table entries
Frozen:       base model weights, bin boundaries, address source
```

### 3.3 Hook 与替换逻辑

```python
# Forward pass
address = compute_address(hidden_state)        # [B, heads], scalar
index   = bin(address, boundaries)             # [B, heads] -> int
delta   = lookup(LUT_table, index)             # [B, group_dim]

# Replace original mlp_delta_group_4 with LUT output
mlp_delta[:, group_4_slice] = delta
```

---

## 4. 训练策略（两阶段）

### 4.1 Stage 1: Local Prefit（核心）

**目标**：证明 LUT table 本身能学到比 bucket average 更好的 mapping。

```
Teacher: original mlp_delta_group_4 (frozen model forward)
Student: LUT(address)
Loss:    MSE + α * cosine_distance
```

$$\mathcal{L} = \text{MSE}(\text{LUT}(a), \delta^*) + \alpha \cdot (1 - \cos(\text{LUT}(a), \delta^*))$$

- 只训练 LUT table，**模型完全 frozen**
- 数据：calibration set（1024 samples，可扩展到 2048）
- 评估：local MSE / cosine + model-level KL / PPL / accuracy

**为什么先做 local prefit？**
- 如果 LUT 连 local 近似都做不到，说明 address → target 的映射有结构性瓶颈
- 避免被 full model loss 的优化噪声淹没

### 4.2 Stage 2: Logits Distillation（可选，成功后做）

如果 Stage 1 的 model-level KL 未达预期：
- 在 frozen base model + trainable LUT 上，加一层 logits-level KL
- 此时目标是让 **下游 logits 分布** 更接近原始模型，而非仅仅 local delta 接近

---

## 5. 成功标准

### 5.1 基线定义

| 基线 | 来源 | KL | PPL |
|------|------|-----|-----|
| B0: Original | 原始模型 | ~0 | 31.92 |
| B1: Zero | zero ablation | 2.950 | — |
| B2: Mean | mean ablation | 2.938 | — |
| B3: Bucket | bucket average (v0.5 best) | **0.914** | **41.4** |
| B4: Trainable LUT | v1 目标 | < 0.914 | < 41.4 |

### 5.2 评判维度

不能只比 KL。v1 必须同时满足：

| 维度 | Minimum Success | Strong Success |
|------|-----------------|----------------|
| **KL** | < 0.914 (比 bucket 低) | < 0.82 (比 bucket 低 10%+) |
| **PPL** | ≤ 41.4 | 明显低于 41.4 |
| **Local MSE** | 低于 bucket MSE | 显著低于 |
| **Next-token Acc** | 不比 bucket 差 | 恢复接近原始 |
| **Generation** | 无 collapse / 重复 / 乱码 | 流畅可读 |

**失败判定**：KL 或 PPL 比 bucket 还差 → 说明 trainable LUT 没有优势，需重新评估 address 设计。

---

## 6. Head Ablation 设计

v1 主实验 + control：

| 配置 | Head | Bins | 目的 |
|------|------|------|------|
| V1-main | 2-head | 64 | 主实验，coverage 高 |
| V1-control | 1-head | 64 | 验证 coverage 是否转化为 KL 优势 |

**Q2 需要回答**：2-head 的高 coverage（48.4% vs 26.6%）能否在训练后转化为更低的 KL？

如果 2-head trainable LUT 仍然不如 1-head，说明多 head 带来的是 address 噪声而非有效信息增益。

---

## 7. 路线图

```
v1.0  (当前): group 4 single-group, uniform/64, 2-head vs 1-head
              → local prefit + model-level eval
              → 核心问题：trainable LUT 能否 beat bucket?

v1.1  (下一步): group 4 + group 3 multi-group LUT
              → 验证 multi-group 的线性/非线性叠加特性
              → 需先确认 v1.0 单 group 成功

v1.2  (后续): debug quantile binning
              → 测试 alternative address 机制
              → 尝试扩大 bins / 调整 group size

v2    (远期): 跳过对应 dense computation
              → 实现真实计算节省
              → 从 proof-of-concept 到 inference optimization
```

---

## 8. 与已有工作的定位

| 工作 | 层级 | 方法 | 与 LLM-LUT 的区别 |
|------|------|------|-------------------|
| LUT-NN | operator-level | centroid lookup | 近似已有算子；LLM-LUT 是在 LLM 内部 residual group 上找 addressable local function |
| Geva et al. | model-level | key-value memory | 解释 FFN 结构；LLM-LUT 尝试用 LUT 替代部分 FFN residual |

**研究叙事**：

> We first conduct a sensitivity and addressability scan to identify activation-conditioned residual components inside an instruction-tuned LLM. The v0.5 results show that Layer 6 MLP residual delta group 4 is not removable or bias-like, but can be substantially recovered by a simple bucket lookup. v1 replaces this non-trained bucket average with a trainable LUT to test whether the signal can be amplified into a stable approximation.

---

## 9. 执行 Checklist

- [ ] 复用 v0.5 的 `bucket.py` binning 逻辑（uniform 64 bins）
- [ ] 实现 `lut_table.py`: `nn.Parameter` table + `F.embedding` lookup
- [ ] 初始化 LUT table 为 bucket average
- [ ] 实现 `run_v1.py`: local prefit loop (MSE + cosine)
- [ ] 实现 model-level eval: KL / PPL / accuracy / generation sanity
- [ ] 跑 V1-main (2-head) 和 V1-control (1-head)
- [ ] 对比 B0-B4 基线，判断是否达到成功标准
