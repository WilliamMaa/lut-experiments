# v3 Multi-Layer LUT Replacement Scan 分析

> 文件：`results/multi_layer_scan.json`  
> 时间：2026-06-04  
> 模型：Qwen/Qwen2.5-7B-Instruct  
> 扫描层：L19–L23  
> 每组 group size = 64，每个 count = 4 / 8 / 12 / 16 groups  

---

## 1. 实验目标

验证 v3 的“partial down_proj + LUT 查表”方案在多层同时替换时的质量-功耗 trade-off。

核心问题：

1. 5 层 MLP 各替换 4/8/12/16 个输出 group，能带来多少全模型 MAC 削减？
2. 质量退化（KL / PPL / Acc）是否可接受？
3. 能不能线性外推到 5–10% 全模型 MAC 削减？

---

## 2. 关键数字

### 2.1 基线

| metric | value |
|---|---|
| PPL | 19.55 |
| Next-token Acc | 0.5133 |

### 2.2 三个累积配置

| 配置 | 替换策略 | MAC 削减 | LUT 存储 | KL | PPL | Acc |
|---|---|---|---|---|---|---|
| `all_layers_half` | L19–L23 各 8 group | **0.68%** | 40 MiB | 0.554 | 21.23 | **0.5460** |
| `middle_half_max` | L21+L22 各 16 group | **0.54%** | 32 MiB | 0.363 | 21.05 | 0.5367 |
| `all_layers_max` | L19–L23 各 16 group | **1.36%** | 80 MiB | 0.992 | **25.95** | 0.5023 |

### 2.3 单层 16 group 敏感度排名（KL 从小到大）

| 层 | KL | PPL | ΔPPL | Acc | ΔAcc |
|---|---|---|---|---|---|
| L23 | 0.115 | 20.17 | +0.61 | 0.5148 | +0.00 |
| L22 | 0.152 | 19.70 | +0.15 | 0.5179 | +0.00 |
| L21 | 0.180 | 19.98 | +0.42 | 0.5351 | +0.02 |
| L20 | 0.188 | 19.59 | +0.03 | 0.5133 | +0.00 |
| L19 | 0.251 | 19.81 | +0.26 | 0.5133 | +0.00 |

**结论：L23/L22 是最耐替换的层，L19 最敏感。**

---

## 附录 A：LUT 查表 key 的构建与每次替换的实际操作

下面描述从 calibration 到 inference 的完整流程，对应代码中的 `v0/calibrate.py`、`v3/table_builder.py`、`v3/partial_linear.py` / `v3/triton_kernels.py`。

### A.1 离线 Calibration：确定每组的“地址”channel

目标模块是 `layer.mlp`（candidate_type=`mlp_delta`）。对每一层：

1. 跑 calibration 数据，capture 每个 token 的：
   - `x`：进入 MLP 的 residual（`[B, seq, hidden_size]`）
   - `y`：MLP 的输出（`[B, seq, hidden_size]`）
2. 把 `hidden_size=3584` 的输出维度切成 `num_groups = 3584 / 64 = 56` 个 group，每个 group 64 维。
3. 对 **输入维度**（`mlp_delta` 下输入维度等于 `hidden_size`）做统计：
   - 每个 channel 的 `mean_in`、`std_in`
   - 每个 channel 与输出残差幅度的相关度 proxy
4. 为每个输出 group 选 2 个 address channel：
   - head 0：从 `var_in` 全局排名最高的 channel 里挑
   - head 1：从 correlation proxy 全局排名最高的 channel 里挑
   - 同一 group 内的两个 head 不重复
5. 记录每组的 `addr_idx[g]`（2 个 channel id）、`addr_mean[g]`、`addr_std[g]`。

> 所以 LUT key 不是原始 hidden state 的全向量，而是**从输入维度中挑选出的 2 个最具判别性的 channel 上的标准化值**。

### A.2 离线 Table 构建：2D joint bucket 平均

对每一个要替换的输出 group：

1. 对 calibration 中每个 token，取出该 group 对应的 2 个 address channel 的激活值：
   ```
   a = x[:, addr_idx[g]]   # [B, seq, 2]
   ```
2. 标准化并量化到 2D bin index：
   ```
   z = (a - addr_mean) / addr_std
   z = clamp(z, -3.0, 3.0)
   q = round((z + 3) / 6 * 63)   # 每个 head -> [0, 63]
   bin_idx = (q0, q1)             # 2D key
   ```
3. 该 token 的 teacher target 是该 group 的 MLP 输出相对于 residual 的差：
   ```
   target = (y - x)[:, g_start:g_start+64]   # [B, seq, 64]
   ```
4. 把所有 token 按 `bin_idx` 扔进 64×64 的 2D bucket，对每个 bucket 内的 target 做平均：
   ```
   table[b0, b1, :] = mean(target[bin_idx == (b0, b1)])
   ```
   空 bucket 填 0。

最终得到 per-group table：形状 `[64, 64, 64]`（64 bins × 64 bins × group_size 64），约 1 MiB（FP32）。

### A.3 在线推理：partial matmul + LUT fill

运行时通过两个 hook 实现替换：

1. **MLP pre-hook**：在 `layer.mlp` 输入处捕获 `normed_x`（即 residual `x`），并用 A.1 的 `addr_mean/std` 预计算所有被替换 group 的 2D bin index。
2. **down_proj forward patch**：把 `mlp.down_proj` 的 forward 替换为自定义函数：
   - **active groups**（未被替换的 group）：从 `down_proj.weight` 中取出对应行，做正常的 `F.linear(hidden, active_weight, active_bias)`。
   - **replaced groups**（被 LUT 替换的 group）：对每个 token，用 `(b0, b1)` 去查该 group 的 table，得到 `table[b0, b1, :]`（64 维 delta），然后加上该 token 的 residual group：
     ```
     output_group = residual_group + table[b0, b1, :]
     ```
   - 用 `index_copy_` 把 active 输出和 LUT 输出拼回 `[B, S, hidden_size]`。

> 这里 `table` 存的是 **MLP 输出相对于 residual 的残差**，所以 `residual_group + table` 就还原了该 group 原本应由 down_proj 产生的输出。

### A.4 一个具体 token 的完整例子

假设某层 hidden_size=3584，group_size=64，要替换 group 5：

| 步骤 | 张量形状 | 说明 |
|---|---|---|
| residual `x` | `[1, 512, 3584]` | 进入 MLP 的 hidden state |
| address 激活 | `[1, 512, 2]` | 从 `x` 中取 group 5 对应的 2 个 channel |
| 标准化/量化 | `[1, 512, 2]` | 每个 head -> `[0, 63]` |
| bin index | `[1, 512, 2]` | 例如 `(12, 47)` |
| LUT 查表 | `[1, 512, 64]` | `table[12, 47, :]` |
| residual group | `[1, 512, 64]` | `x[:, :, 5*64:6*64]` |
| 输出 group | `[1, 512, 64]` | `residual_group + table[12,47]` |

### A.5 Checkpoint 里存了什么（per group）

```python
{
    "layer_id": 21,
    "group_id": 5,
    "group_size": 64,
    "addr_idx":  [c0, c1],           # 2 个 address channel
    "addr_mean": [m0, m1],           # 2 个 channel 的 mean
    "addr_std":  [s0, s1],           # 2 个 channel 的 std
    "table":     [64, 64, 64],       # FP32 2D bucket table
    "num_bins":  64,
    "addr_clip": 3.0,
}
```

一个 group 的 LUT 存储 ≈ 64×64×64×4 B ≈ 1 MiB。16 group × 5 层 ≈ 80 MiB，与 `multi_layer_scan.json` 中的 `lut_storage_human` 一致。

---

## 3. 现象分析

### 3.1 小替换有时反而降 PPL

L19 4g、L22 4g/8g 的 PPL 比 baseline 还低。这说明：

- 对特定 group 做 LUT 均值/查表替换，可能起到“去噪”或“平滑 outlier”的作用；
- 但多层叠加后，这种正面效果被累积误差淹没。

### 3.2 多层误差不是线性相加

| 配置 | 单层 KL 之和 | 实际 KL | 膨胀系数 |
|---|---|---|---|
| `all_layers_max` (5×16g) | 0.886 | 0.992 | ×1.12 |
| `all_layers_half` (5×8g) | 0.226 | 0.554 | ×2.45 |

`all_layers_half` 的实际 KL 是单层之和的 2.45 倍，说明**层数越多，误差耦合越严重**。因此：

> **不能直接用单层敏感度做线性外推来设计 multi-layer 配置。**

### 3.3 PPL 与 Acc 不完全一致

`all_layers_half` 的 PPL 最高（21.23），但 Acc 也最高（0.5460）。可能原因：

- eval set 只有 128 条，噪声较大；
- next-token Acc 对这种“平滑化”不敏感；
- 需要 downstream task 验证真实能力退化。

### 3.4 MAC 削减非常有限

每层 16 group 只贡献约 **0.27%** 的全模型 MAC 削减。原因是：

- 只替换了 down_proj 的部分输出维度；
- up_proj / gate_proj / attention  still 全算；
- 28 层里只动了 5 层。

---

## 4. 5–10% 全模型 MAC 削减可行吗？

按当前方法简单线性估算：

| 目标 MAC 削减 | 大概需要 |
|---|---|
| 5% | 约 18–19 层各 16 group |
| 10% | 全部 28 层各约 12–14 group |

但 L19–L23 全 16 group 已经把 PPL 干到 25.95。如果继续扩层数/扩 group，不做任何微调，质量会进一步崩塌。

**当前判断：**

> **不加联合微调的纯 LUT 替换，很难直接做到 5–10% 全模型 MAC 削减且保持可接受质量。**

要接近这个目标，必须同时做两件事：

1. **质量恢复**：对选中的层做 end-to-end fine-tune，让 down_proj 主动适应 LUT；
2. **更聪明的分配**：优先替换不敏感层（L23/L22），减少敏感层（L19/L20）的替换量。

---

## 5. 还需要哪些数据才能继续分析

### 5.1 必须有的基础数据

1. **每层 `expand_ratio_l{layer}.json` summary**
   - 看具体选了哪些 group；
   - 看 4/8/12/16 group 的 progressive KL/PPL/Acc 曲线；
   - 判断好 group 池是否已用完。

2. **每层 `zero_scan.json` / `bucket_eval.json`**
   - 看 zero ablation 排名和 bucket recovery 率；
   - 确认当前 16 group 是否已经是该层的“天花板”。

3. **候选配置的生成样本**
   - 尤其是 `all_layers_half` 和 `middle_half_max`；
   - PPL/Acc 是数字，文本漂移才能直观判断可用性。

### 5.2 验证真实质量的数据

4. **Downstream task 评测**
   - 至少跑 1–2 个常识/推理任务（如 ARC-Easy、HellaSwag、CMMLU）；
   - 确认 PPL 21+ 在实际任务上是否可接受。

### 5.3 方法扩展的数据

5. **联合微调后的 multi-layer 结果**
   - 单 L21 微调已经能把 KL 0.18 → 0.17、PPL 20.0 → 19.7；
   - 对 multi-layer 配置做 sequential 或 joint fine-tune 是最关键实验。

6. **LUT 量化实验**
   - FP16 / INT8 对 KL/PPL 的影响；
   - 量化后 LUT 存储减半，决定能不能铺更多层。

7. **更多 multi-layer 组合的实际评测**
   - 例如 L22+L23 16g、L21+L22+L23 12g、非均匀分配等；
   - 现在只有 3 个 preset，无法画出真正的 Pareto 前沿。

---

## 6. 下一步方向

### 6.1 最优先：对 `all_layers_half` 做联合微调

原因：

- MAC 削减 0.68%，PPL 只涨 8.6%，Acc 还略升；
- 是最有希望扩展的起点；
- 如果 fine-tune 能把 PPL 压回 20 以内，说明这条路可以继续扩层数/扩 group。

实现思路：

1. 在 5 层都安装 8-group v3 partial engine；
2. 冻结其他所有参数，只让这 5 层的 `down_proj.weight` 可训练；
3. 用 calibration 数据做 KL 微调（lr 1e-6，3 epochs，参考 L21 单点经验）；
4. 评测 KL/PPL/Acc/generation/downstream task。

如果 fine-tune 后质量恢复良好，再尝试：

- `all_layers_12`（5 层各 12 group）；
- 加入 L18 / L24 等相邻层，扩到 7–9 层。

### 6.2 次优先：非均匀 group 分配搜索

基于单层敏感度，尝试给不同层分配不同 group 数：

| 层 | 建议 group 数 |
|---|---|
| L19 | 4–8（最敏感） |
| L20 | 8–12 |
| L21 | 12–16 |
| L22 | 16 |
| L23 | 16（最不敏感） |

组合若干候选配置做实际评测，画出 Pareto 前沿。

### 6.3 并行探索

- **LUT 量化**：INT8/FP16 对质量和存储的影响；
- **更多层**：L17–L25 全扫一遍，看 middle layers 是否整体更耐替换；
- **任务评测**：把 PPL 指标和真实能力挂钩，避免只看数字。

---

## 7. Train of Thought

1. **先看总量**：3 个累积配置的 MAC 削减都很小（0.5–1.4%），说明当前“5 层 × 16 group”的规模远远不够。

2. **再看单层敏感度**：L23/L22 替换后 KL 最低，L19 最高。第一反应是“应该对不敏感层多换、敏感层少换”。

3. **但多层不是相加**：`all_layers_half` 的实际 KL 是单层之和的 2.45 倍，说明层间耦合严重。非均匀分配的收益可能被低估。

4. **看质量可接受性**：`all_layers_half` PPL 21.23、Acc 0.546，退化不算灾难级；`all_layers_max` PPL 25.95 已经不可接受。

5. **推断 milestone**：要达 5–10% MAC，需要 18–28 层参与。没有微调的话，质量大概率不可接受。

6. **提出关键实验**：对 `all_layers_half` 做联合微调。这是验证“LUT + fine-tune 能否 scale”的最小可行实验。

7. **列出数据缺口**：需要 per-layer summary、generation、downstream task、quantization、更多组合评测，才能做完整 Pareto 判断。

---

## 8. 一句话总结

> **5 层 LUT 替换最多只能换 1.36% MAC，且质量已崩；要冲 5–10% MAC，必须靠联合微调 + 非均匀层分配，下一步优先对 `all_layers_half` 做 5 层 down_proj 联合微调，看质量能不能回来。**
