# 双头解耦表示 + LUT 动态注入：完整实验方案

> 目标：在推理阶段零额外计算的前提下，实现特征查表地址与分类特征的解耦，
> 同时保留 LUT O(1) 查表在存算一体设备上的核心价值。

---

## 一、核心思路

### 1.1 问题根源

当前方案（旁路 query_proj）的矛盾：

```
feat_16 → query_proj(133K参数) → 查表坐标
```

- query_proj 比 LUT 表本身还重（133K vs 114K）
- query_proj 在推理时需要额外 MAC 计算，不符合 CIM 硬件目标
- feat_16 同时承担"分类判别"和"查表地址"两个互相冲突的目标

### 1.2 解决思路：训练时解耦，推理时零计算

```
训练时：
  主分支 feat_32  → 分类 loss（交叉熵）
  查表分支 feat_lut → uniformity loss（覆盖表空间）
  两个 loss 联合优化，互不干扰

推理时：
  feat_32  → 分类头
  feat_lut → 直接作为 SRAM 地址线输入
  ← 没有任何额外计算，query_proj 彻底消失
```

### 1.3 关键设计：从 conv 输出直接切片

CNN 的 `32×7×7 = 1568-dim` conv 输出天然是高维特征，
直接切取其中 16 维作为查表地址，**切片操作在硬件上零代价**。

```
32×7×7 (1568-dim)
    ↓                    ↓
Linear(1568→32)      flat[:, :16]  ← 零计算切片
feat_32（分类）       feat_lut（查表）
```

---

## 二、网络结构

### 2.1 CNNBackbone（双头版）

```python
class CNNBackboneDualHead(nn.Module):
    def __init__(self, class_dim=32, lut_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1×28×28 → 8×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8×28×28 → 8×14×14
            nn.Conv2d(8, 32, kernel_size=3, padding=1),  # 8×14×14 → 32×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32×14×14 → 32×7×7
        )
        self.proj = nn.Linear(7 * 7 * 32, class_dim)    # 分类用，1568→32
        self.lut_dim = lut_dim

    def forward(self, x):
        conv_feat = self.conv(x)
        flat = conv_feat.view(conv_feat.size(0), -1)     # 1568-dim

        feat_class = self.proj(flat)                      # 32-dim，分类用
        feat_lut = flat[:, :self.lut_dim]                 # 16-dim，查表用，零计算

        return feat_class, feat_lut
```

### 2.2 完整模型结构

```python
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackboneDualHead(class_dim=32, lut_dim=16)
        self.lut = LUT3Main(query_dim=16, inject_dim=340)  # 现有LUT结构
        self.classifier = nn.Linear(32, 10)                # 分类头

    def forward(self, x):
        feat_class, feat_lut = self.backbone(x)

        # LUT动态注入（作用在分类头的权重上）
        dyn_weight, dyn_bias = self.lut(feat_lut)
        logits = dynamic_linear(feat_class, dyn_weight, dyn_bias)

        return logits
```

---

## 三、训练策略

### 3.1 Loss 设计

```python
def uniformity_loss(feat):
    """
    Wang & Isola, ICML 2020
    让特征均匀分布在超球面上，最大化表空间利用率
    """
    feat = F.normalize(feat, dim=-1)
    sq_dists = torch.pdist(feat, p=2).pow(2)
    return sq_dists.mul(-2).exp().mean().log()

# 训练 loop
feat_class, feat_lut = model.backbone(x)
logits = model(x)

loss_cls = F.cross_entropy(logits, y)
loss_uni = uniformity_loss(feat_lut)
loss = loss_cls + lambda_uni * loss_uni
```

### 3.2 lambda_uni 调参策略

分三阶段逐步验证，避免 uniformity loss 破坏分类性能：

| 阶段 | lambda_uni | 目的 |
|------|-----------|------|
| 第一阶段 | 0.0 | 纯静态基线，确认双头结构本身不掉点 |
| 第二阶段 | 0.01 | 轻度约束，观察准确率变化 |
| 第三阶段 | 0.05 / 0.1 | 加强约束，观察表利用率提升 |

**停止条件**：若准确率相比 lambda=0 下降超过 0.5%，回退到上一档。

### 3.3 为什么准确率理论上不受影响

uniformity loss 只作用在 `feat_lut`（1568 维 conv 输出的前 16 维切片），
完全不经过 `feat_class` 的梯度路径。

```
loss_uni → feat_lut = flat[:, :16] → conv 参数（轻微影响）
loss_cls → feat_class = proj(flat) → conv 参数 + proj 参数（主要路径）
```

两条路径在 conv 层有轻微耦合，但 `loss_cls` 的梯度量级远大于 `loss_uni`，
实际影响可控。

---

## 四、实验计划

### 实验 0：双头静态基线（无 LUT，无 uniformity loss）

**目的**：确认双头结构（feat_32 分类 + feat_lut 切片）不引入额外损失。

```
CNN双头（32-dim分类）静态 vs CNN单头（16-dim）静态
预期：CNN双头 ≥ 91.28%（不应该比原来差）
```

### 实验 1：双头 + LUT，无 uniformity loss（lambda=0）

**目的**：确认 feat_lut 直接切片查表的基础可行性。

```
CNN双头（32-dim分类）+ LUT（feat_lut查表）
对比基线：CNN单头 + LUT旁路query_proj（91.07%）
预期：≥ 90.5%（允许轻微下降，因为feat_lut分布未优化）
```

### 实验 2：双头 + LUT + uniformity loss（lambda 扫描）

**目的**：验证 uniformity loss 能否在不损失准确率的前提下提升表利用率。

```
lambda = 0.0 / 0.01 / 0.05 / 0.1
观察指标：
  - 测试集准确率
  - feat_lut 的分布均匀度（用 uniformity loss 值本身衡量）
  - LUT 表的访问热力图（哪些区域被频繁访问）
```

### 实验 3：LUT 压缩（在最优 lambda 下）

**目的**：在双头方案确认有效后，压缩 LUT 存储到工业可用范围。

| 方案 | 表大小 | 预期存储 | 目标准确率 |
|------|--------|---------|-----------|
| A | 单尺度 256 | ~87K | ≥ 90.5% |
| B | 单尺度 128 | ~43K | ≥ 90.0% |
| C | 单尺度 64  | ~22K | ≥ 89.5% |

---

## 五、成功标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 推理额外计算 | **零** | feat_lut 为纯切片操作 |
| query_proj 参数 | **0** | 彻底消除 |
| 测试集准确率 | **≥ 90.5%** | 相比静态CNN（91.28%）损失 <1% |
| LUT 总存储 | **≤ 50K** | 压缩后工业可接受范围 |
| 训练-测试差距 | **≤ 4%** | 过拟合可控 |

---

## 六、与现有方案的对比

| 维度 | 旧方案（query_proj旁路） | 新方案（双头切片） |
|------|------------------------|-----------------|
| 推理额外计算 | query_proj（133K MAC） | **零** |
| CIM 适配性 | 中（还有旁路网络） | **高（纯地址线输入）** |
| query_proj 参数 | 133K | **0** |
| 分类特征质量 | 16-dim（弱） | 32-dim（强） |
| 查表地址分布 | 由 query_proj 决定 | 由 uniformity loss 优化 |
| 准确率（当前） | 91.07% | 待验证（预期 ≥ 90.5%） |

---

## 七、风险与预案

**风险1：feat_lut 切片分布太差，LUT 完全无效**

- 预案：换切片位置（不一定取前16维，可以取中间或随机选16个通道）
- 或：用 PCA 投影代替切片，仍然是线性操作，硬件上可固化

**风险2：uniformity loss 与分类 loss 冲突，准确率掉超过 1%**

- 预案：减小 lambda，或只对 feat_lut 加 BatchNorm 而不加 uniformity loss
- BatchNorm 本身有一定的均匀化效果，且对准确率影响极小

**风险3：压缩 LUT 后准确率掉太多**

- 预案：接受方案 A（87K），总存储 = CNN 28K + LUT 87K = 115K，
  仍比旧方案（304K）压缩 2.6 倍

---

## 八、核心价值主张（面向硬件厂商）

> 在存算一体设备上，CNN 前端提取的 conv 特征直接驱动 SRAM 地址线，
> 无需任何额外计算，实现 per-sample 的动态权重注入。
> 相比静态网络，准确率损失 < 1%，但获得了 O(1) 查表加速和
> 动态个性化推理能力。

这是旧方案（有 query_proj）做不到的，是本方案最核心的硬件创新点。

(py310-dev) guoshoucai@ps:~/copilot/jewel-backend$ python fashion_dual_head_lut.py
Using device: cuda

============================================================
>>> 实验0：双头静态基线（无LUT，无uniformity loss）
============================================================
参数量: 52,954
  [Exp0-Static Epoch  10] Train: 91.60% | Test: 90.12% | Best: 90.12%                       
  [Exp0-Static Epoch  20] Train: 93.38% | Test: 91.09% | Best: 91.09%                       
  [Exp0-Static Epoch  30] Train: 94.35% | Test: 91.12% | Best: 91.12%                       
  [Exp0-Static Epoch  40] Train: 95.11% | Test: 90.67% | Best: 91.12%                       
  [Exp0-Static Epoch  50] Train: 95.71% | Test: 90.58% | Best: 91.12%                       
  [Exp0-Static Epoch  60] Train: 96.16% | Test: 90.58% | Best: 91.12%                       
  [Exp0-Static Epoch  70] Train: 96.57% | Test: 90.50% | Best: 91.12%                       
  [Exp0-Static Epoch  80] Train: 96.93% | Test: 90.34% | Best: 91.12%                       
  [Exp0-Static Epoch  90] Train: 97.04% | Test: 90.30% | Best: 91.12%                       
  [Exp0-Static Epoch 100] Train: 97.15% | Test: 90.29% | Best: 91.12%                       

  >>> 实验0 最佳测试准确率: 91.12%

============================================================
>>> 实验1：双头 + LUT（无uniformity loss）
============================================================
参数量: 405,042
  [Exp1-LUT Epoch  10] Train: 92.09% | Test: 89.92% | Best: 89.92%                          
  [Exp1-LUT Epoch  20] Train: 95.24% | Test: 90.07% | Best: 90.66%                          
  [Exp1-LUT Epoch  30] Train: 97.68% | Test: 89.46% | Best: 90.66%                          
  [Exp1-LUT Epoch  40] Train: 99.17% | Test: 89.96% | Best: 90.66%                          
  [Exp1-LUT Epoch  50] Train: 99.91% | Test: 90.22% | Best: 90.66%                          
  [Exp1-LUT Epoch  60] Train: 100.00% | Test: 90.49% | Best: 90.66%                         
  [Exp1-LUT Epoch  70] Train: 100.00% | Test: 90.48% | Best: 90.66%                         
  [Exp1-LUT Epoch  80] Train: 100.00% | Test: 90.45% | Best: 90.66%                         
  [Exp1-LUT Epoch  90] Train: 100.00% | Test: 90.44% | Best: 90.66%                         
  [Exp1-LUT Epoch 100] Train: 100.00% | Test: 90.45% | Best: 90.66%                         

  >>> 实验1 最佳测试准确率: 90.66%

============================================================
>>> 实验2：双头 + LUT + uniformity loss (lambda=0.01)
============================================================
参数量: 405,042
  [Exp2-LUT-lam0.01 Epoch  10] Train: 92.34% | Test: 89.64% | Best: 90.24% | UniLoss: -1.6325
  [Exp2-LUT-lam0.01 Epoch  20] Train: 95.15% | Test: 90.84% | Best: 91.14% | UniLoss: -1.6677
  [Exp2-LUT-lam0.01 Epoch  30] Train: 97.47% | Test: 90.56% | Best: 91.15% | UniLoss: -1.6701
  [Exp2-LUT-lam0.01 Epoch  40] Train: 99.16% | Test: 90.60% | Best: 91.15% | UniLoss: -1.6708
  [Exp2-LUT-lam0.01 Epoch  50] Train: 99.75% | Test: 90.58% | Best: 91.15% | UniLoss: -1.6802
  [Exp2-LUT-lam0.01 Epoch  60] Train: 99.92% | Test: 90.55% | Best: 91.15% | UniLoss: -1.6876
  [Exp2-LUT-lam0.01 Epoch  70] Train: 100.00% | Test: 90.69% | Best: 91.15% | UniLoss: -1.7239
  [Exp2-LUT-lam0.01 Epoch  80] Train: 100.00% | Test: 90.67% | Best: 91.15% | UniLoss: -1.7534
  [Exp2-LUT-lam0.01 Epoch  90] Train: 100.00% | Test: 90.68% | Best: 91.15% | UniLoss: -1.7765
  [Exp2-LUT-lam0.01 Epoch 100] Train: 100.00% | Test: 90.60% | Best: 91.15% | UniLoss: -1.7856

  >>> 实验2 (λ=0.01) 最佳测试准确率: 91.15%

============================================================
>>> 实验2：双头 + LUT + uniformity loss (lambda=0.05)
============================================================
参数量: 405,042
  [Exp2-LUT-lam0.05 Epoch  10] Train: 92.17% | Test: 89.69% | Best: 89.69% | UniLoss: -2.0940
  [Exp2-LUT-lam0.05 Epoch  20] Train: 95.37% | Test: 90.55% | Best: 90.55% | UniLoss: -2.1129
  [Exp2-LUT-lam0.05 Epoch  30] Train: 97.81% | Test: 90.02% | Best: 90.55% | UniLoss: -2.1196
  [Exp2-LUT-lam0.05 Epoch  40] Train: 99.30% | Test: 89.64% | Best: 90.55% | UniLoss: -2.1224
  [Exp2-LUT-lam0.05 Epoch  50] Train: 99.90% | Test: 89.98% | Best: 90.55% | UniLoss: -2.1204
  [Exp2-LUT-lam0.05 Epoch  60] Train: 99.97% | Test: 89.95% | Best: 90.55% | UniLoss: -2.1161
  [Exp2-LUT-lam0.05 Epoch  70] Train: 99.99% | Test: 90.06% | Best: 90.55% | UniLoss: -2.0972
  [Exp2-LUT-lam0.05 Epoch  80] Train: 100.00% | Test: 90.12% | Best: 90.55% | UniLoss: -2.0859
  [Exp2-LUT-lam0.05 Epoch  90] Train: 100.00% | Test: 89.97% | Best: 90.55% | UniLoss: -2.0754
  [Exp2-LUT-lam0.05 Epoch 100] Train: 100.00% | Test: 89.93% | Best: 90.55% | UniLoss: -2.0720

  >>> 实验2 (λ=0.05) 最佳测试准确率: 90.55%

============================================================
>>> 实验2：双头 + LUT + uniformity loss (lambda=0.1)
============================================================
参数量: 405,042
  [Exp2-LUT-lam0.1 Epoch  10] Train: 92.12% | Test: 90.01% | Best: 90.01% | UniLoss: -1.6838
  [Exp2-LUT-lam0.1 Epoch  20] Train: 94.96% | Test: 90.67% | Best: 90.67% | UniLoss: -1.7407
  [Exp2-LUT-lam0.1 Epoch  30] Train: 97.46% | Test: 90.17% | Best: 90.88% | UniLoss: -1.7725
  [Exp2-LUT-lam0.1 Epoch  40] Train: 99.18% | Test: 90.01% | Best: 90.88% | UniLoss: -1.7828
  [Exp2-LUT-lam0.1 Epoch  50] Train: 99.71% | Test: 89.93% | Best: 90.88% | UniLoss: -1.7761
  [Exp2-LUT-lam0.1 Epoch  60] Train: 99.91% | Test: 90.64% | Best: 90.88% | UniLoss: -1.7648
  [Exp2-LUT-lam0.1 Epoch  70] Train: 99.99% | Test: 90.50% | Best: 90.88% | UniLoss: -1.7512
  [Exp2-LUT-lam0.1 Epoch  80] Train: 100.00% | Test: 90.65% | Best: 90.88% | UniLoss: -1.7436
  [Exp2-LUT-lam0.1 Epoch  90] Train: 100.00% | Test: 90.60% | Best: 90.88% | UniLoss: -1.7418
  [Exp2-LUT-lam0.1 Epoch 100] Train: 100.00% | Test: 90.58% | Best: 90.88% | UniLoss: -1.7398

  >>> 实验2 (λ=0.1) 最佳测试准确率: 90.88%

======================================================================
【实验结果汇总】
======================================================================
实验名称                                总参数        lambda     测试集        增益      
----------------------------------------------------------------------
双头静态基线                                52,954       N/A   91.12%         —
双头+LUT(无uniformity)                  405,042       0.0   90.66%    -0.46%
双头+LUT+uniformity(λ=0.01)            405,042      0.01   91.15%    +0.03%
双头+LUT+uniformity(λ=0.05)            405,042      0.05   90.55%    -0.57%
双头+LUT+uniformity(λ=0.1)             405,042       0.1   90.88%    -0.24%
======================================================================