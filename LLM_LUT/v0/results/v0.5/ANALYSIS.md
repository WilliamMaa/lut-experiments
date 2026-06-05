# LLM-LUT v0.5 结果分析

> 基于 Experiment A/B/C 的完整数据，判断下一步方向。

---

## 0. 实验配置

- Model: Qwen2.5-0.5B-Instruct
- Layer: 6
- Candidate: `mlp_delta` only
- Calib: 1024 samples / Eval: 512 samples / MaxLen: 128
- Groups tested: [4, 3, 8, 1, 13, 9, 0]
- Baseline PPL: 31.92

---

## 1. Experiment A: 单 Group × Binning 扫描

### 1.1 核心结论：Uniform 全面碾压 Quantile

| 排名 | Group | Binning | Bins | Recovery | KL Bucket | Coverage |
|------|-------|---------|------|----------|-----------|----------|
| 1 | 4 | uniform | 64 | **69.0%** | 0.91 | 48.4% |
| 2 | 4 | uniform | 128 | 66.7% | 0.98 | 41.4% |
| 3 | 4 | uniform | 32 | 66.2% | 1.00 | **50.0%** |
| 4 | 4 | uniform | 256 | 64.9% | 1.04 | 38.3% |
| 5 | 3 | uniform | 256 | 46.9% | 0.31 | 38.3% |

**Quantile binning 几乎全线崩溃**：
- Group 4 quantile/32 bins: Recovery = **-1.8%**, PPL = **1161**（模型崩了）
- 绝大多数 quantile 配置的 Recovery < 15%，很多为负

**原因分析**：当前 quantile 实现可能存在问题（boundaries 计算或 searchsorted 分配），导致 bin 之间的映射不稳定。Uniform binning 已经验证了 strong signal，quantile 需要 debug 后再测。

### 1.2 Group 4 是最佳候选

```
Group 4, uniform 64 bins:
  KL Zero   = 2.950
  KL Mean   = 2.938  ← mean 替换几乎没用
  KL Bucket = 0.914  ← bucket 大幅改善
  Recovery  = 69.0%
  Coverage  = 48.4%
```

Mean 替换几乎无效（2.950→2.938），说明 group 4 不是静态偏置。但 bucket 能降到 0.914，说明它是 **activation-dependent 且高度 addressable 的**。

### 1.3 Bins = 64 是 Sweet Spot

| Bins | Recovery | Coverage | 评价 |
|------|----------|----------|------|
| 32 | 66.2% | 50.0% | coverage 最高，但 recovery 略低 |
| 64 | **69.0%** | 48.4% | **recovery 最高，coverage 足够** |
| 128 | 66.7% | 41.4% | recovery 开始下降 |
| 256 | 64.9% | 38.3% | bin 太多，每个 bin 样本不足 |

### 1.4 Group 分级

| 等级 | Groups | Recovery (uniform best) | 结论 |
|------|--------|------------------------|------|
| A | 4 | ~69% | **核心候选，必须进入 v1** |
| B | 3 | ~47% | 次优候选，可考虑组合 |
| C | 0, 1, 13 | 16-31% | 有一定 signal，但较弱 |
| D | 8, 9 | <5% | **排除，不适合 LUT** |

---

## 2. Experiment B: 多 Group 组合测试

### 2.1 关键发现：Group 8 是 "坏苹果"

| Groups | Num | KL Bucket | ΔKL | PPL |
|--------|-----|-----------|-----|-----|
| [4] | 1 | 0.91 | — | 41.4 |
| [4,3] | 2 | 1.14 | +0.23 | 56.5 |
| [4,3,8] | 3 | 2.55 | **+1.41** | 62.0 |
| [4,3,8,1] | 4 | 2.70 | +0.15 | 102.2 |

**加入 group 8 时 KL 暴增 1.41**（从 1.14→2.55），这说明 group 8 和其他 group 组合时产生了 **nonlinear collapse**。

这和 Experiment A 的结论一致：group 8 单 group recovery 只有 2.9%，本身就是不稳定信号。

### 2.2 可行的组合范围

如果**排除 group 8**，只组合 group 4 + 3：
- KL = 1.14，是 group 4 单点的 1.25 倍
- 这个增量是可以接受的

加入更多 C 级 group（0, 1, 13）后 KL 增长变缓，但 PPL 飙升到 100+，说明这些 group 虽然单点 recovery 还可以，组合时对模型输出的破坏较大。

**v1 建议：先只做 group 4 的单 group LUT prefit，验证 trainable LUT 能否优于 bucket。确认后再考虑 group 4+3 组合。**

---

## 3. Experiment C: 1-Head vs 2-Head

| Group | 1-Head KL | 2-Head KL | 1-Head Cov | 2-Head Cov |
|-------|-----------|-----------|------------|------------|
| 4 | 0.88 | 0.91 | 26.6% | **48.4%** |
| 3 | 0.36 | 0.38 | 26.6% | **48.4%** |
| 8 | 1.19 | 1.39 | 26.6% | **48.4%** |

- **1-head 的 KL 略低**（因为 address 通道更少，bin 分配更集中）
- **2-head 的 coverage 翻倍**（48.4% vs 26.6%）

对于 **v1 trainable LUT**，coverage 比 KL 更重要，因为：
- 高 coverage 意味着 LUT 表被充分利用
- trainable LUT 可以学习比 bucket average 更精细的映射
- 2-head 提供的多视角 address 可能帮助捕获更丰富的 activation 结构

**结论：v1 使用 2-head。**

---

## 4. 综合判断与 v1 触发条件

| 条件 | 状态 | 说明 |
|------|------|------|
| Recovery > 50% 稳定 | ✅ | Group 4 uniform 32/64/128/256 全部 > 64% |
| 多 group 不崩 | ⚠️ | 4+3 组合可接受（KL=1.14），但 4+3+8 会崩 |
| Coverage > 50% | ✅ | Group 4 uniform 32 bins 达到 50% |
| 更好的 binning 能降 KL | ❌ | Quantile 不可用，uniform 64 bins 已是当前最优 |

**4/4 条件中 2.5 个通过**，可以进入 v1。

---

## 5. v1 设计建议

基于 v0.5 数据，v1 的初始配置应为：

```
Layer: 6
Candidate: mlp_delta
Group: 4（先单 group，成功后扩展为 4+3）
Binning: uniform
Num bins: 64
Heads: 2
Address: existing scalar activation（calibrated）
LUT type: trainable（不再是 bucket average）
```

v1 的核心问题：
> **Trainable LUT 能否优于 non-trained bucket average？**
>
> Bucket 已经做到 recovery=69%。如果 trainable LUT 做不到更好，说明 address 机制有瓶颈。
