# LLM_LUT v5 设计草案：Multi-Head High-Order Address LUT (MHOAL)

> 目标：把领导 `new_lut.py` / `index_lut.py` 中**符合项目红线**的思路整合进 v4，优先解决 v4 的表存储瓶颈，扩大可替换的 MAC 比例。

---

## 1. v4 当前瓶颈

| 指标 | Stage 2（当前最佳） | Stage 3（崩） |
|---|---|---|
| 替换层/group | L15-L27 混合 | L15-L27 group=12 |
| MAC 削减 | 2.21% | 2.78% |
| LUT 存储 | 40.5 MiB | 49.25 MiB |
| PPL | 27.67 | 35+ |

**核心矛盾**：每多替换一个 group，就多一张 `[64, 64, 64]` 的 2D 表（262,144 元素）。存储线性增长，很快碰到显存/带宽上限，导致无法继续扩大 MAC 削减。

领导代码给我们的最大启发：
- `new_lut.py`：**不要把所有信息塞进一张 2D 大表**，用多张 1D 子表 + sum 聚合，可以指数级降低存储。
- `index_lut.py`：**address 本身可以是非线性的、高阶的**，但必须在 LUT 框架内做，不能用 MLP/Conv 生成参数。

---

## 2. v5 核心思想：MHOAL

### 2.1 基本定义

对 v4 中每一个被替换的 output group（大小仍为 `group_size=64`），v5 不再使用一张 2D 表，而是使用 **H 个 address head**，每个 head 配一张 1D 表。

```text
v4:  output_delta = Table_2D[b1, b2]          # Table shape: [64, 64, 64]

v5:  output_delta = sum_{h=1}^{H} Table_h[b_h]  # Each Table_h: [num_bins, 64]
```

其中每个 head 的 address `b_h` 来自一个**高阶 address 函数** `a_h(x)`：

```text
Type 0: single(i)        -> a = x_i
Type 1: pair_sum(i,j)    -> a = x_i + x_j
Type 2: pair_diff(i,j)   -> a = x_i - x_j
Type 3: quad_cmp(i,j,k,l)-> a = sign(x_i + x_j - x_k - x_l)
Type 4: random_proj(w)   -> a = sum_d w_d * x_d   (w 固定随机)
```

所有函数都是 O(1) 的固定变换，**不涉及 MLP/Conv/可学习参数**，符合项目红线。

### 2.2 存储对比

| 方案 | 单 group 表大小 | 相对 v4 |
|---|---|---|
| v4 2D `[64,64,64]` | 262,144 el | 1× |
| v5 H=8, bins=64 | 8 × 64 × 64 = 32,768 el | **0.125×** |
| v5 H=16, bins=64 | 16 × 64 × 64 = 65,536 el | **0.25×** |
| v5 H=8, bins=128 | 8 × 128 × 64 = 65,536 el | 0.25×，更细粒度 |
| v5 H=4, bins=64 | 4 × 64 × 64 = 16,384 el | **0.0625×** |

**影响**：如果 Stage 2 的 40.5 MiB 降到 1/8，同样的存储预算可以替换 **8 倍 group 数**。MAC 削减可能从 2.21% 提升到 **15% 以上**。

### 2.3 表达能力分析

- **可加性假设**：如果目标函数 `delta(x)` 可以分解为多个低维函数的加和，MHOAL 是高效的。
- **2D 表的交互项**：v4 的 2D 表能建模 `a1` 和 `a2` 的联合交互。MHOAL 用多个 1D 表丢失了 head 间交互，但可以通过以下方式补偿：
  - 增加 head 数 H；
  - 让某些 head 本身就是高阶组合（如 `pair_sum`、`quad_cmp`），隐式编码交互；
  - 引入少量 2D head（如 1 个 2D 表 + 多个 1D 表）。

`new_lut.py` 的成功说明：对很多复杂函数，**随机高阶哈希 + 多表 sum 的近似能力已经足够强**。

---

## 3. 从领导代码里“挑”什么进来

### ✅ 优先整合（符合红线、帮助最大）

| 优先级 | 来源 | 想法 | 收益 | 风险 |
|---|---|---|---|---|
| **P0** | `new_lut.py` | 多 1D 子表聚合（MHOAL） | 存储降 4-16 倍，MAC 可大幅扩大 | 需要验证近似能力 |
| **P0** | `new_lut.py` | 高阶 address 函数（A+B>C+D） | 同表大小下 PPL 可能更低 | address 搜索空间变大 |
| **P1** | `index_lut.py` | 给每个 head 加可学习 scale | 小幅提升表达能力，参数量极小 | 训练略复杂 |
| **P1** | `index_lut.py` | base>2 的离散 address | 可能用更少 bins 达到同样精度 | 硬件支持需确认 |

### ❌ 不整合（违反红线或当前不适用）

| 来源 | 想法 | 问题 |
|---|---|---|
| `index_lut.py` | `Conv+MLP+Gumbel` 学习 Index | 这是 HyperNetwork，违反“动态参数必须通过 LUT 查表”红线 |
| `index_lut.py` | 每个像素存完整 W 矩阵 | 存储爆炸，本质上是动态卷积，不是 LUT |
| `new_lut.py` | 照搬 512 tables × 1024 rows × 10 committees | 存储不可接受，且为 toy 任务设计 |
| `new_lut.py` | 直接拟合最终 logits | LLM 多层结构需要保留中间层语义，不能跳层拟合 logits |

---

## 4. v5 系统架构

### 4.1 文件结构

```text
LLM_LUT/v5/
├── README.md
├── docs/
│   └── v5_design.md              # 本文档
├── calibrate_v5.py               # v5 address 函数搜索与校准
├── table_builder_v5.py           # 构建多 head 1D 表
├── partial_linear_v5.py          # V5PartialEngine（核心推理）
├── trainable_engine_v5.py        # 可训练版本（联合微调）
├── finetune_multi_layer_v5.py    # 多层联合微调入口
├── quantize_lut_v5.py            # INT8/FP16 量化
├── search_layer_configs_v5.py    # 非均匀 group 分配搜索
├── data.py                       # 从 v4 复制/复用
├── metrics.py                    # 从 v4 复制/复用
└── triton_kernels_v5.py          # 多 head 1D LUT 的 Triton/PyTorch kernel
```

> 遵循 v4 的“独立目录”原则，v5 不改动 v4/v3/v0 文件，只复用数据路径。

### 4.2 数据结构变化

#### v4 group checkpoint（单张 2D 表）

```python
{
    "addr_idx": [2],          # 2 个 address channel
    "addr_mean": [2],
    "addr_std": [2],
    "table": [64, 64, 64],
}
```

#### v5 group checkpoint（多 head 1D 表）

```python
{
    "num_heads": 8,
    "num_bins": 64,
    "addr_funcs": [
        {"type": "single", "idx": [7]},
        {"type": "pair_sum", "idx": [12, 88]},
        {"type": "pair_diff", "idx": [45, 201]},
        {"type": "quad_cmp", "idx": [3, 9, 120, 334]},
        # ... 共 H 个
    ],
    "addr_mean": [H],         # 每个 head address 的均值
    "addr_std": [H],          # 每个 head address 的标准差
    "tables": [H, 64, 64],    # H 张 1D 表
}
```

### 4.3 推理流程

```python
# 1. 对当前 token 的 normed_x，计算每个 head 的 address scalar
addr_vals = []
for func in addr_funcs:
    addr_vals.append(apply_addr_func(normed_x, func))
addr_vals = stack(addr_vals)  # [H]

# 2. 归一化、分桶
z = (addr_vals - addr_mean) / addr_std
z = clamp(z, -clip, clip)
bins = round((z + clip) / (2*clip) * (num_bins-1))

# 3. 查 H 张 1D 表并聚合
delta = sum(tables[h, bins[h]] for h in range(H))

# 4. 加回输入
output = normed_x_group + delta
```

聚合方式可选：
- **sum**（默认，对应 `new_lut.py` 的 EmbeddingBag sum）
- **weighted sum**：`delta = sum_h scale_h * tables[h, bins[h]]`，scale 可学习

### 4.4 Address 函数搜索策略

在 v3/v4 的 calibrate 阶段增加 address 搜索：

1. **候选池生成**：对每个 group，从输入维度中随机/启发式生成 K 个 address 函数候选（single/pair_sum/pair_diff/quad_cmp）。
2. **贪婪选择**：
   - 初始残差 = 目标 delta；
   - 每次选使当前残差 MSE 下降最大的候选；
   - 更新残差，重复 H 次。
3. **建表**：对选出的 H 个函数，分别建 1D 表。

这一步是离线做的，只跑一次，不增加推理成本。

---

## 5. 预期效果与验证计划

### 5.1 第一阶段验证（单 layer，L21）

先在 L21 做单层对比实验：

| 实验 | 配置 | 目标 |
|---|---|---|
| v4 baseline | L21, 8 groups, 2D `[64,64,64]` | 复现 v4 结果 |
| v5-MHOAL-8 | L21, 8 groups, H=8, bins=64 | 验证同质量下存储降 8 倍 |
| v5-MHOAL-16 | L21, 8 groups, H=16, bins=64 | 验证更多 head 能否追上 2D 质量 |
| v5-HO2D | L21, 8 groups, 2D 但 address 高阶 | 验证只改 address 是否有收益 |

评估指标：PPL、Acc、KL、LUT 存储、覆盖率。

### 5.2 第二阶段验证（多层 MAC 扩展）

如果单层 OK，尝试用 v5 替换更多 group：

- 目标：在保持 PPL<30 的前提下，把 MAC 削减从 2.21% 推到 **5% 以上**。
- 方法：利用存储优势，把 Stage 2 的 group 数扩大 3-4 倍，或用更大 group_size。

### 5.3 成功标准

| 标准 | 说明 |
|---|---|
| 表大小 | 同 group 数下，v5 存储 ≤ v4 的 1/4 |
| 质量 | 同 group 数下，v5 PPL ≤ v4 + 5% |
| MAC 扩展 | 同 PPL 预算下，v5 可替换 group 数 ≥ v4 的 3 倍 |
| 红线 | 不引入 MLP/HyperNetwork/Conv 作为参数/address 生成器 |

---

## 6. 实现路线图

### Phase 1：最小可验证原型（1-2 天）

只改 L21 单层：
1. 实现 `calibrate_v5.py` 的 address 搜索；
2. 实现 `table_builder_v5.py` 的多 head 1D 表构建；
3. 实现 `partial_linear_v5.py` 的 V5PartialEngine；
4. 写一个 `run_l21_ablation_v5.py` 脚本，对比 v4/v5。

### Phase 2：完整 v5 流水线（3-5 天）

1. 补齐 `trainable_engine_v5.py` 和 `finetune_multi_layer_v5.py`；
2. 补齐 `quantize_lut_v5.py`；
3. 复用/扩展 `search_layer_configs_v5.py`；
4. 跑 L15-L27 的多层实验。

### Phase 3：优化与硬件对齐（后续）

1. 写 Triton kernel 做多 head 1D LUT 聚合；
2. 尝试 INT4 量化进一步压存储；
3. 测速/测 MAC 削减端到端。

---

## 7. 风险与应对

| 风险 | 可能性 | 应对 |
|---|---|---|
| MHOAL 近似能力不够，PPL 崩 | 中 | 先做单层 ablation；若 H=16 不够，加少量 2D head |
| Address 搜索太慢 | 中 | 限制候选池大小；用并行/缓存 |
| Triton kernel 不好写 | 低 | 先用 PyTorch fallback，验证效果后再优化 |
| 量化后精度损失大 | 中 | 1D 表更小，量化误差可能反而更低；可尝试 per-head scale |
| 联合微调不稳定 | 中 | 沿用 v4 的 fp32 训练、低 lr、KL 蒸馏 |

---

## 8. 与 v4 改进路线图的对应

| 路线图方向 | v5 如何回应 |
|---|---|
| 方向 1：从 group12 往回剪 | v5 让存储不再是瓶颈，可以反向操作：从低存储基线往上加 group |
| 方向 4：改进微调目标 | 暂不改动，沿用 v4 KL 蒸馏；后续可加 logits ensemble |
| 方向 5：非均匀 group_size | 可自然扩展为“非均匀 head 数/ bins 数” |
| 方向 6：改进 LUT address | **核心改动**：高阶 address 函数 + 可搜索 |
| 方向 7：跨层共享 LUT 表 | MHOAL 的 1D 表本身更小；后续可尝试跨 group 共享 head |
| 方向 8：工程落地 | 1D 表比 2D 表更适合硬件并行查表 |

---

## 9. 结论

v5 的核心价值是：**用领导 `new_lut.py` 的多表聚合思想，把 v4 的 2D 大表拆成多张 1D 小表，并用高阶 address 函数提升表达能力**。在不违反项目红线的前提下，这能让 LUT 存储降 4-16 倍，从而把可替换 group 数扩大数倍，MAC 削减有望从 2% 级别进入 10% 级别。

**建议先落地 Phase 1（L21 单层 ablation）**，验证 MHOAL 的近似能力是否足够。如果 OK，再继续完整流水线。
