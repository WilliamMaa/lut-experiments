# 方案 3：Leaf-Level 低秩修正 / PCA 诊断

## 目标

判断 LUT 当前 piecewise-constant 输出的表达上限是否可以用低秩局部线性项突破。对每个 leaf 内的 residual 做 PCA，如果低秩能解释大部分方差，就说明值得加 `c_ell + A_ell * z(x)` 修正。

---

## 核心原则

- 不是给每个 leaf 加一个完整低秩矩阵（存储爆炸）。
- 先诊断：同一 leaf 内的 residual 是否主要分布在一个低维子空间上。
- 优先用**共享输入投影 V + leaf 相关小系数**的形式，控制存储。

---

## 输入输出

### 输入

1. **v3 base checkpoint**：
   - `shared_coarse.pt`
   - `residual_g*.pt`
2. **calibration 数据**：`calib_x`, `calib_y`（FFN input / output）。
3. **可选**：当前 hard correction checkpoint（判断 hard correction 后的残差是否仍有低秩结构）。

### 输出

`./diagnostics/leaf_pca/` 目录：

```text
group_pca_summary.json     # 每个 group 的 leaf-level PCA 统计
  ├── group_0/
  │     ├── leaf_rank_ev.csv      # 每个 leaf 的 rank-k explained variance
  │     ├── leaf_residual_norm.csv
  │     └── top_eigenvectors.pt   # 共享或 per-leaf 主方向（可选）
  ├── group_1/
  ...
hard_correction_after_pca/   # 如果基于 hard-corrected 残差
```

---

## 理论形式

当前 LUT 输出是 piecewise constant：

```
hat y = c_ell
```

其中 `ell = address(x)` 是查表得到的 leaf。

低秩局部修正形式：

```
hat y = c_ell + A_ell * z(x)
```

但为了避免每个 leaf 存一个 `hidden x rank` 矩阵，采用：

```
z(x) = V^T * x              # 全局共享输入投影，rank = r (e.g., 4 or 8)
hat y = c_ell + a_ell * z(x)  # 每个 leaf 只存一个 [output_dim, r] 系数
```

或者更省：

```
hat y = c_ell + diag(b_ell) * V^T * x
```

每个 leaf 只存 `output_dim` 个缩放系数。

---

## 诊断步骤

### Step 1：计算 base residual

对 calibration 数据，先算 LUT base 预测：

```python
base_pred = predict_base(coarse_address, coarse_lut,
                       residual_addresses, residual_luts,
                       group_ids, group_size, calib_x, device)
```

然后：

```python
residual = calib_y - base_pred  # [N, hidden_size]
```

如果需要 hard correction 后的残差，就再加 `hard_correction`：

```python
residual = calib_y - (base_pred + hard_corr)
```

### Step 2：按 leaf 分组 residual

对每个 group，对其中每个 residual leaf：

```python
for gid in group_ids:
    leaf_indices = residual_address_g[gid].compute_indices(calib_x.unsqueeze(0)).view(-1)
    for leaf_id in unique(leaf_indices):
        mask = leaf_indices == leaf_id
        leaf_residual = residual[mask, g_start:g_end]  # [n_leaf_samples, group_size]
```

### Step 3：对每个 leaf 做 SVD/PCA

```python
U, S, Vt = torch.svd(leaf_residual.float())
# explained variance ratio
explained = S ** 2 / (S ** 2).sum()
cumulative = explained.cumsum(0)
```

记录：

```text
rank-1 ev, rank-2 ev, rank-4 ev, rank-8 ev
leaf sample count
leaf residual norm mean / std
leaf residual variance
```

### Step 4：聚合统计

对每个 group 输出：

```text
mean rank-1 explained variance
mean rank-4 explained variance
mean rank-8 explained variance
p10 / p50 / p90 of rank-4 ev
leaf count with rank-4 ev > 0.8
leaf count with rank-4 ev < 0.3
```

---

## 判断标准

| 状态 | rank-4 累计解释方差 | 含义 | 行动 |
|-----|------------------|-----|------|
| 强低秩 | > 0.8 | residual 主要在一个 4D 子空间 | 加 rank-4 局部线性修正 |
| 中等 | 0.5–0.8 | 有结构但不集中 | 试 rank-8 或共享 V |
| 弱低秩 | < 0.5 | residual 很分散 | 低秩修正无效，回退到增容量或改地址 |

---

## 如果诊断有效，下一步实现

### 形式 A：全局共享 V，per-leaf 系数 a_ell

```python
# 全局 V: [hidden_size, rank]
V = compute_shared_pca_topk(calib_residual, rank=8)

# 对每个 leaf，学习 a_ell: [group_size, rank]
for leaf_id in leaves:
    z = calib_x[mask] @ V  # [n, rank]
    # 最小二乘
    a_ell = (z.T @ z + lambda * I).inverse() @ z.T @ residual[mask]
```

部署时：

```python
z = x @ V
output = c_ell + a_ell @ z
```

### 形式 B：per-leaf top-k 主方向

如果不同 leaf 的子空间差异很大，给每个 leaf 存自己的 `V_ell` 和 `a_ell`。存储更高，但更灵活。

### 形式 C：只加 leaf-level bias

如果 residual 没有明显低秩结构，但同一 leaf 内均值不为 0，可以先试更简单的：

```python
output = c_ell + m_ell
```

其中 `m_ell` 是该 leaf 内 residual 均值。这其实是当前 piecewise-constant 的自然扩展，相当于把每个 leaf 的表值从常数改成“常数 + 输入线性项”。

---

## 存储估算

假设：
- hidden_size = 2048，group_size = 64
- rank = 4
- coarse 14-bit = 16384 leaves，residual 16-bit = 65536 leaves

### 共享 V
- `V`: `[2048, 4]` × FP16 = 16 KiB（可忽略）
- per-leaf `a_ell`：每个 residual leaf 存 `[64, 4]` FP16 = 512 B
- 总增量：65536 × 512 B ≈ **32 MiB per group**
- 32 groups：≈ **1 GiB**

这超过了当前预算，所以要么：
- 只用 rank = 1（8 MiB per group，256 MiB total）
- 只对困难 leaf 加低秩修正
- 用共享 V 但 per-group 而不是 per-leaf 系数

### 共享 V + per-group 系数
- 每个 group 存 `[64, 4]` 矩阵，32 groups：32 × 512 B = 16 KiB（可忽略）
- 但表达能力弱很多，等价于给每个 group 加一个全局线性修正。

---

## 与现有代码的接口

- 复用 `build_tail_aware_hard_correction.py` 里的 `predict_base()` 计算 base residual。
- 新建 `diagnose_leaf_pca.py`，输入 v3 checkpoint 目录和数据目录。
- 如果诊断有效，修改 `v6_replacement_engine.py` 的 `_hook`：
  - 加载 `V` 和 `a_ell`
  - 在查表后加 `+ a_ell @ (x @ V)`

---

## 验证指标

1. **诊断阶段**：
   - 每个 group 的 rank-1/4/8 explained variance 分布
   - 困难 leaf（低 cosine）是否有更高/更低的低秩性

2. **实现阶段**：
   - 加低秩修正后的 full-output cosine
   - 与单纯增大 LUT 容量（如 hard_num_bits +1）的对比
   - 模型级生成 PPL / 文本质量

---

## 风险

1. **存储爆炸**：per-leaf 低秩矩阵容易让 LUT 从 80 MiB 涨到 1 GiB，需要严格控制 rank 和覆盖范围。
2. ** leaf 样本太少**：如果某个 leaf 只有几个样本，PCA 不稳定，需要只统计样本数 > threshold 的 leaf。
3. **与 hard correction 重复**：如果 hard correction 已经把系统性的低秩偏差吃掉，剩下的 residual 可能是纯噪声，PCA 会显示弱低秩。所以诊断要同时做 base residual 和 hard-corrected residual。
