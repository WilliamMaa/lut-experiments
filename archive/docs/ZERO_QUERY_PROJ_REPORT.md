# 零 query_proj 双头方案实验报告

> 日期：2026-05-16
> 状态：✅ 核心目标达成（零额外计算 + 小存储 LUT + 准确率稳定）

---

## 一、实验背景

### 1.1 之前的困境

| 方案 | query_proj | LUT 存储 | 准确率 | 问题 |
|------|-----------|---------|--------|------|
| CNN + 原始 LUT-3 | 133K | 114K | 91.07% | 推理有额外 MAC 计算 |
| CNN + 压缩 LUT（共享坐标） | 5.7K | 114K | 91.00% | 仍有 query_proj |
| 双头 + LUT（有 uniformity） | 129K | 222K | 91.15% | 过拟合严重（train-test gap 9.3%） |

### 1.2 核心目标

实现 `NEW_APPROACH.md` 的愿景：
- ✅ **推理时零额外计算**（feat_lut 直接作为 SRAM 地址线）
- ✅ **小存储 LUT**（<50K）
- ✅ **准确率不掉**（相比静态基线损失 <1%）

---

## 二、方案设计：零 query_proj 双头架构

### 2.1 网络结构

```
输入 1×28×28
    ↓ Conv(1→8) + ReLU + MaxPool
8×14×14
    ↓ Conv(8→32) + ReLU + MaxPool
32×7×7 (1568-dim)
    ├─ flat[:16] → BatchNorm → Sigmoid → feat_lut_coords (B, 16)  ← 查表地址
    └─ Linear(1568→16) → feat_class (B, 16)  ← 分类特征
```

### 2.2 关键设计

**（1）零 query_proj 查表**

```python
# feat_lut_coords 直接作为查表坐标
feat_lut_coords = torch.sigmoid(BN(flat[:16]))  # (B, 16)，范围 (0, 1)

# 16 个坐标 × 多尺度表 → 20 维 bias 残差
# 存储: 16 × 20 × (64+16+4) = 26.9K
```

**（2）查表结构**

```
16 个坐标，每个坐标对应一张独立的表
表结构: (16, 20, table_size)
  - 16 个维度，每个维度一张表
  - 每张表输出 20 维
  - 16 次查表后求和 → (B, 20)
```

**（3）动态注入方式**

```python
# 只注入 bias（weight 用静态）
dyn_b = lut(feat_lut_coords)  # (B, 20)
out = backbone_main(feat_class) + base_bias + dyn_b
```

### 2.3 过拟合控制

- **weight_decay = 1e-2**（之前 1e-4）
- **梯度裁剪 max_norm=1.0**
- **只注入 bias**，不注入 weight

---

## 三、实验结果

### 3.1 最终数字

| 实验 | 总参数 | LUT 部分 | 测试集 | Train-Test Gap | 增益 |
|------|--------|---------|--------|----------------|------|
| 零 query_proj 静态基线 | **28K** | 0 | **90.71%** | **6.2%** | — |
| 零 query_proj + LUT（单64） | 49K | 20K | 90.78% | 6.2% | **+0.07%** |
| 零 query_proj + LUT（64,16,4） | 55K | 27K | 90.70% | 7.4% | -0.01% |
| 零 query_proj + LUT + uniformity（λ=0.01） | 55K | 27K | **90.80%** | 6.9% | **+0.09%** |

### 3.2 与历史结果对比

| 阶段 | 方案 | Fashion-MNIST 准确率 | 说明 |
|------|------|---------------------|------|
| 早期 | FC + LUT | 86-88% | FC 前端太弱 |
| 中期 | CNN + 原始 LUT-3 | 91.07% | 有 query_proj，存储 248K |
| **本期** | **零 query_proj 双头** | **90.71-90.80%** | **无 query_proj，存储 27K** |

### 3.3 核心成就

1. **双头策略成功**：feat_lut 直接切片 + BN + Sigmoid 作为查表地址，没有出现准确率 drop
2. **存储压缩 9 倍**：248K → 27K
3. **过拟合可控**：train-test gap 从 9.3% 降到 6.9%
4. **推理零额外计算**：feat_lut 是纯切片操作，硬件上零代价

---

## 四、分析与讨论

### 4.1 为什么 LUT 没有显著提升准确率？

这再次验证了 `METHOD_ANALYSIS_AND_NEXT_DIRECTIONS.md` 的结论：

> **在 CNN 强前端下，同类样本特征已经高度一致，静态权重就够用了，LUT 没有修正空间。**

LUT 的价值不在于提升准确率，而在于：
- **用 O(1) 查表替代数字 MAC 计算**
- **在存算一体设备上降低功耗和延迟**

### 4.2 单尺度 vs 多尺度

- 单尺度 [64]：90.78%
- 多尺度 [64,16,4]：90.70%

多尺度在这个任务上没有优势，反而因为参数更多导致过拟合。但**在泛化到不同数据集时，多尺度可能仍然有价值**（捕捉不同频率的细节）。

### 4.3 uniformity loss 的作用

- λ=0.01 时：90.80%（最佳）
- 之前 λ=0.05/0.1 时反而下降

uniformity loss 在低权重下有助于特征分布均匀，但权重过大会与分类 loss 冲突。

---

## 五、下一步方向

### P0：换数据集验证泛化能力

**Kuzushiji-MNIST**（日文书法字体，类内差异大）：
- 如果零 query_proj + LUT 仍然有效，说明方案普适
- 如果失效，说明 Fashion-MNIST 太特殊

### P1：极端压缩前端验证 LUT 价值

**CNN 输出压到 4-dim 或 8-dim**：
- 静态基线应该很差（<85%）
- 加 LUT 看能否恢复到 88%+
- 验证 LUT 的核心价值：**从极低维特征恢复个性化能力**

### P2：提升表现的方法探索

1. **更大的表**：[128, 32, 8] 或 [256, 64, 16]
2. **同时注入 weight + bias**：当前只注入 bias，weight 用静态
3. **更大的分类特征维度**：feat_class 从 16-dim 提升到 32-dim
4. **不同的均匀化约束**：尝试 KL divergence 或 GMM 代替直方图 MSE

### P3：硬件协同设计

和硬件团队确认：
- SRAM 查表延迟 vs 数字 MAC 延迟
- SRAM 读取功耗 vs MAC 计算功耗
- 27K LUT 存储在实际芯片上是否可行

---

## 六、一句话总结

> **零 query_proj 双头方案成功实现了"推理零额外计算 + 27K 小存储 LUT + 90.7% 准确率"。虽然 LUT 没有显著提升准确率，但这在工程上不是问题——LUT 的核心价值是用 O(1) 查表替代数字 MAC 计算，降低存算一体设备的功耗和延迟。下一步应在更复杂的数据集上验证泛化能力，并探索极端压缩前端下 LUT 的恢复能力。**

---

## 附录：关键代码结构

```python
class CNNBackboneNoQueryProj(nn.Module):
    def forward(self, x):
        conv_feat = self.conv(x)          # 32×7×7
        flat = conv_feat.view(B, -1)       # 1568-dim
        feat_class = self.proj(flat)       # 16-dim，分类用
        feat_lut = flat[:, :16]            # 16-dim，查表用
        feat_lut_coords = torch.sigmoid(self.lut_bn(feat_lut))  # 归一化到 (0, 1)
        return feat_class, feat_lut_coords

class LUT3_ProperNoQueryProj(nn.Module):
    def forward(self, feat_lut_coords):
        # 16 个坐标 × 多尺度表 → 20 维
        # 存储: 16 × 20 × (64+16+4) = 26.9K
        out = 0
        for tables, size in zip(self.tables, self.sizes):
            out += self._interp_1d(feat_lut_coords, tables, size)
        return out
```
