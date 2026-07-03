# 10% 全模型 MAC 削减路径分析

## 当前基线

- 当前最佳：`L15–L27` 共 13 层，down_proj 替换 167 个 group。
- 全模型 MAC 削减：**2.84%**。
- 质量：PPL=29.25，Acc=0.470。

模型参数：
- hidden_size = 3584
- intermediate_size = 18944
- num_layers = 28
- group_size = 64
- 全模型总 major-linear MAC ≈ **7.14B / token**
- 每个 down_proj group 削减 MAC = `intermediate_size × 64` ≈ **1.212M / token**

## 纯 down_proj 要到 10% 需要多少？

| 目标 MAC 削减 | 需要 down_proj group 数 | 占全部 down_proj 输出通道比例 |
|---|---|---|
| 5% | 294 | 18.8% |
| **10%** | **589** | **37.6%** |
| 15% | 883 | 56.4% |
| 20% | 1178 | 75.1% |

当前只有 167 个 group，缺口 **422 个 group**。

### 几种扩法对应的 MAC

| 方案 | 总 group 数 | MAC 削减 | 点评 |
|---|---|---|---|
| 当前 | 167 | 2.84% | 已验证可用 |
| 全部 28 层 group=12 | 336 | 5.70% | 预计 PPL 会明显恶化 |
| 全部 28 层 group=16 | 448 | 7.61% | 很可能不可用 |
| 全部 28 层 group=20 | 560 | 9.51% | 接近 10%，但质量悬崖 |
| 全部 28 层 group=24 | 672 | 11.41% | 远超当前 LUT 容量 |

### 要调多少轮？

假设每轮实验增加 2 层 + 现有层 group 提 2：
- 每轮新增 group ≈ `2×12 + 13×2 = 50`
- 缺口 422 → **约 8–9 轮**

如果更激进，每轮增加 4 层 + 现有层 group 提 4：
- 每轮新增 group ≈ `4×12 + 13×4 = 100`
- 缺口 422 → **约 4–5 轮**

但这只是数量上的估算。实际上：
- 每轮都要重新 fine-tune，实验周期很长。
- 随着替换比例接近 40%，层间误差耦合会急剧放大，PPL 可能断崖式上升。
- 当前 LUT 是固定校准表， quality ceiling 有限。

**结论：纯 down_proj 线性扩展到 10% 既慢又危险。**

## 更快到达 10% 的三条路

### 路 1：提升 LUT 本身质量，把 down_proj 天花板顶上去

核心问题不是“能不能换更多 group”，而是“换了之后还能不能训回来”。

可行杠杆：

1. **可训练 LUT 表**（最大杠杆）
   - 当前 table 是 calibration 均值，固定不动。
   - 把 LUT 表也变成可训练参数，联合微调。
   - 这样可以在同样 group 数下显著降低 PPL，从而敢上更高 group 数。
   - 存储不变，推理仍是 O(1) 查表。

2. **更好的 address 选择**
   - per-group 选 address channel。
   - 3D/4D address（小 bins 控制存储）。
   - 用 hidden state 统计量做 address。

3. **更强的蒸馏目标**
   - KL + hidden-state MSE。
   - 难 token reweighting。

预期：如果可训练 LUT 能让 group=20 全层回到 PPL<35，就能接近 10%。

### 路 2：换 / 加替换目标模块

当前只替换了 down_proj。但 MLP 里 gate_proj + up_proj 的总 MAC 是 down_proj 的 **2 倍**。

| 模块 | 每 group 削减 MAC | 每层总 group 数 | 全替换一层削减 |
|---|---|---|---|
| down_proj | 1.212M | 56 | 1.90% |
| gate_proj | 0.229M | 296 | 1.90% |
| up_proj | 0.229M | 296 | 1.90% |
| gate+up 合计 | 0.459M | 592 | **3.80%** |

也就是说：
- 全替换 3 层的 gate+up ≈ **5.7%** 全模型 MAC。
- 全替换 5–6 层的 gate+up ≈ **10%** 全模型 MAC。

这比死磕 down_proj 快得多。但风险：
- gate/up 没有残差连接，输出不接近输入，LUT 重建可能很差。
- 需要先快速扫描 gate/up 的可替换性。

### 路 3：不同层用不同策略（layer-adaptive）

不是每层都 down_proj group20，而是按层敏感度组合：

| 层类型 | 策略 | 理由 |
|---|---|---|
| 耐替换层（如 L22/L23）| down_proj 高 group + 甚至 gate/up | 能承受更多替换 |
| 敏感层（如 L15/L19）| down_proj 低 group 或保持 | 保住质量 |
| 中间层 | down_proj 中等 group | 平衡 |

这样可以避免“均匀加 group”导致的质量悬崖。

## 推荐路径

要达到 10%，必须**同时**走多条路：

1. **立即扫描 gate_proj / up_proj 的可替换性**（最快判断路 2 是否可行）。
2. **并行实现可训练 LUT 表**（提升 down_proj 天花板）。
3. **用扫描结果设计 layer-adaptive 配置**，而不是均匀扩 group。

如果 gate/up 扫描结果显示任意一层可替换，优先级立刻转到 gate/up；如果 gate/up 也不行，那就只能靠可训练 LUT + 混合策略硬顶 down_proj。

## 下一步建议

1. 跑 `gate_proj` / `up_proj` 独立检查脚本（类似 o_proj inspection）。
2. 同时把 `finetune_multi_layer.py` 改成支持 **trainable LUT**。
3. 根据 (1)(2) 的结果，设计一个 layer-adaptive 的 10% 实验。
