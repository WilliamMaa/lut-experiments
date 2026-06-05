# LLM-LUT v0 → v0.5 下一步工作计划

> 基于 `rank_report.md`（Layer 6 单轮扫描结果）和 `v0_analysis_v1.md` 的分析，明确下一步验证路线。

---

## 1. v0 结果核心发现

### 1.1 扫描范围
- 只跑了 **Layer 6**，3 个 candidate type × 14 groups = 42 个候选
- Calib: 128 samples / Eval: 64 samples / MaxLen: 128（数据量偏小）

### 1.2 关键结论

| 结论 | 证据 |
|------|------|
| **`mlp_delta` 最适合 LUT** | group 4 的 bucket recovery ≈ **70%**（KL 从 4.89 → 1.44），且多个 groups（3, 8, 1, 13）均有 >20% recovery |
| **`down_proj` 不适合 v0 bucket** | coverage 仅 14%，bucket 相对 mean 没有优势，说明 uniform binning 无法捕捉 down_proj 输出的 addressable 结构 |
| **`attn_out` 敏感度低但 addressability 差** | KL Zero 本身很小（<0.02），说明这些 group 本来就不重要，但 bucket 替换也没有带来额外收益 |
| **coverage 普遍偏低** | mlp_delta 34.38%, attn_out 56.25%, down_proj 14.06%。说明 uniform binning 在当前 setting 下不是最优的 |

### 1.3 最重要信号

```
Layer 6, mlp_delta, group 4:
  KL Zero   = 4.8949
  KL Mean   = 4.9046
  KL Bucket = 1.4408
  Recovery  = (4.8949 - 1.4408) / 4.8949 ≈ 70.6%
```

Mean 替换几乎没用（4.89→4.90），但 bucket 大幅降到 1.44。这说明：
- 该 group 不是静态偏置（mean 无效）
- 但它是 **activation-dependent 的，且可以被 scalar address 索引**

这是理想的 LUT 候选。

---

## 2. 当前问题（承自 v0_analysis_v1.md）

### 2.1 Final Score 不可靠
- Addressability 为 0 的 group 仍有正 Final Score，说明 scoring 公式有缺陷
- **下一版 ranking 改用明确指标**：
  - `Recovery = (KL_Zero - KL_Bucket) / KL_Zero`
  - `Bucket_Advantage = KL_Mean - KL_Bucket`

### 2.2 Coverage 偏低
- mlp_delta 34.38% 意味着 64 个 bin 中只有约 22 个被用到
- 可能原因：address activation 分布集中、uniform binning 不合适、calib 数据不够
- **必须加入 quantile binning 和 occupancy entropy 分析**

### 2.3 单 group ≠ 多 group
- YOLO v10 里已经验证过：6 only / 8 only / 6+8 的行为不是线性叠加
- LLM 里同样可能出现 **nonlinear collapse**：多个低敏感度 group 同时替换时，误差可能非线性累积

---

## 3. 下一步实验计划（v0.5）

### Stage 1：复现性确认 + 数据扩大

**目标**：确认 Layer 6 mlp_delta top groups 的信号在更大数据量下稳定。

```
Layer: 6
Type: mlp_delta
Groups: [4, 3, 8, 1, 13, 9, 0]  (按 recovery 排序的前 7 个)
Calib: 512 → 1024 samples
Eval:  256 → 512 samples
MaxLen: 128 → 256
```

**新增指标**：
- Recovery
- Coverage
- Bin occupancy entropy
- Empty-bin ratio

### Stage 2：多 group 组合测试（最关键）

**目标**：验证多 group 同时替换是否会出现 nonlinear collapse。

```
Test configurations:
  1. group 4 only
  2. group 4 + 3
  3. group 4 + 3 + 8
  4. group 4 + 3 + 8 + 1
  5. group 4 + 3 + 8 + 1 + 13
```

每种配置跑：Zero / Mean / Bucket，看 KL 是否近似线性叠加，还是出现崩坏。

这直接对应 YOLO v10 里的 `target_mode = 6 / 8 / 68` 逻辑。

### Stage 3：Address / Binning Ablation

**目标**：找到更好的 binning 策略和 address 机制，提高 coverage 和降低 KL。

对 group 4、3、8 跑：

| 维度 | 测试值 |
|------|--------|
| Binning | uniform vs **quantile** |
| num_bins | 32 / 64 / 128 / 256 |
| Address heads | 1 (single scalar) vs 2 (two scalar → average) |

Quantile binning 实现：
```python
# 用 calibration data 的 address 分布计算 quantile boundaries
# 而不是固定的 uniform clip
```

### Stage 4：跨层验证（可选，视 Stage 1-3 结果而定）

如果 Layer 6 的信号稳定，扩展到：
```
Layers: [3, 6, 10, 12, 14, 18, 21]
Type: mlp_delta only（已验证是最优 candidate）
```

看敏感度是否随深度变化（通常 middle/late layers 更适合近似）。

---

## 4. v1 的触发条件

只有在 v0.5 全部验证通过后，才进入 v1（trainable LUT 预训练）：

| # | 条件 | 为什么 |
|---|------|--------|
| 1 | Top group 的 recovery > 50% 在扩大数据量后仍然稳定 | 排除小样本噪声 |
| 2 | 多 group 组合（至少 3-4 个）不会导致 nonlinear collapse | 证明可扩展性 |
| 3 | Quantile binning 或更好的 binning 能显著降低 KL | 证明 address 机制有改进空间 |
| 4 | Coverage > 50% 且 occupancy entropy 合理 | 证明 LUT 表是"可用"的，不是稀疏的 |

v1 的核心问题：
```
trainable LUT 能否优于 non-trained bucket average？
```

如果 bucket 已经能做到 recovery 70%，trainable LUT 理论上应该能做到更好（因为 bucket 只是均值，LUT 可以学习更精细的映射）。但如果 trainable LUT 做不到更好，说明 address 机制本身有瓶颈。

---

## 5. 不做的事（明确边界）

- **不做 down_proj**：v0 已证明 coverage 太低，不值得继续
- **不做 attn_out**：虽然 KL 低，但 bucket 没有带来额外收益，说明不适合 LUT
- **不做端到端训练**：v1 仍然是单 component 的 LUT prefit，不是 multi-layer 联合训练
- **不引入任何 MLP/HyperNetwork**：严守 AGENTS.md 红线 #1

---

## 6. 预期产出

v0.5 完成后，应该能回答：

1. Layer 6 mlp_delta 的 LUT 信号是否可复现？
2. 同时替换多个 groups 的累积行为如何？
3. 更好的 binning 能提升多少？
4. 最有希望的 entry point 是什么（layer / group / binning config）？

这些答案将直接决定 v1 的 LUT 模块接口设计。
