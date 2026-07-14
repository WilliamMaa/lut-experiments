# LLM_LUT v5 有效进展总结

> 截至 2026-07-13，把目前跑通且有价值的实验、方法论、以及下一步可选方向汇总成一份文档。

---

## 1. 目前有效的实验结果

### 1.1 小规模验证（已确认方法有效）

| 实验 | 替换层/组 | 模式 | MAC 削减 | LUT 存储 | 最佳 PPL | 最佳 Acc |
|---|---|---|---|---|---|---|
| down L21–23 tree | L21–23，各 8 group | down_proj | 0.41% | 3 MiB | 20.84 | 0.523 |
| o L17 direct | L17，8 group | o_proj direct | ~0.03% | 1 MiB | 17.90 | 0.515 |
| joint small | down L21–23 + o L17 | sequential | 0.43% | 4 MiB | 20.38 | 0.516 |
| sequential small | down L18–23 + o L15–17 | sequential | 0.89% | 9 MiB | 17.59 | 0.530 |

### 1.2 大规模验证（当前最佳）

| 实验 | 替换层/组 | MAC 削减 | LUT 存储 | 最佳 PPL | 最佳 Acc |
|---|---|---|---|---|---|
| **sequential large** | down L15–L27（164 group）+ o L15/L16/L17 direct + L27 delta | **2.89%** | **24.50 MiB** | **28.08** | **0.482** |

作为对比：
- 原模型：PPL 19.55，Acc 0.513
- v4 2D：PPL 29.25，Acc 0.470，MAC 削减 2.78%，LUT 49 MiB
- independent tree + o：PPL 51.35，Acc 0.443（同配置，非 sequential build）

**结论**：sequential deployment-aware build 把同配置的 PPL 从 51 拉到 28，基本追平 v4，但 LUT 存储减半。

---

## 2. 已验证有效的方法论

### 2.1 Deployment-Aware Sequential Build

核心原则：

> 每个 LUT 都在所有会影响其输入的前序替换已经部署后的 student 分布上构建。

具体执行顺序：

```
for l = 0 to L-1:
    build o_proj^(l) on current student
    install o_proj^(l)
    build down_proj^(l) on current student
    install down_proj^(l)
```

同一层内先 o_proj 后 down_proj，因为 MLP 的输入来自 attention 输出 + 残差。

### 2.2 Tree Address

- 比 v4 的 2D address 单点 build MSE 低 8.5%。
- 单 group LUT entry 数 1024（2D 是 4096），存储只有 1/4。
- 目前默认 `num_bits=10`，`channels_per_bit=4`，`tree_candidates=32`。

### 2.3 o_proj 直接/残差混合模式

- 浅层（L15–L17）用 **direct**：直接预测 `o_proj(x)`，rel_mse 低。
- 深层 L27 用 **delta**：预测 `o_proj(x) - x`，rel_mse 仅 0.18。

### 2.4 Joint Fine-Tune

- 冻结原模型其他参数。
- 只训练被替换的 `down_proj.weight`、`o_proj.weight` 和 LUT table。
- 目标：原模型 logits 的 KL 散度。

---

## 3. 已经跑过但效果不行的

| 方法/配置 | 结果 | 结论 |
|---|---|---|
| Independent build + joint fine-tune（13 层 down + 4 层 o） | PPL 51.35 | build-deployment mismatch 在大规模下不可忽略 |
| Random high-order address | 比 2D 差 4–5 倍 | 地址 split 必须面向 target residual |
| 同事的 `build_mlp_lut.py` | 小数据有效，大数据不可扩展 | 违反红线，MLP 插值器不是 LUT |

---

## 4. 还可以试的方法

### 4.1 优化训练过程（低成本）

- **更多 epoch / lr decay**：large 实验 10 epoch 仍有波动，epoch 8/9 PPL 反弹到 37/39，epoch 10 才回到 28。加 cosine decay 或更长的 epoch 可能更稳。
- **Hidden-state / module-output distillation**：不只约束 logits KL，还约束被替换层输出的归一化 MSE。既可以帮助训练，也可以诊断误差从哪一层开始爆炸。
- **更大的 `tree_candidates` 和 `max_samples`**：当前 32/16384 是速度妥协，调回 128/65536 可能提升 build 质量。

### 4.2 改进 LUT 本身（不违反红线）

- **LUT entry 间固定权重插值**：借鉴 `build_mlp_lut.py` 的插值思想，但不用 MLP。对 2D/tree 做线性/双线性/叶子平滑插值。
- **INT8 量化 LUT table**：把 LUT 存储再压一半，便于扩更多层。

### 4.3 扩展替换规模（主要方向）

- **增加每层的 group 数**：从当前 12/16 提高到 20/28/全部 56，直接提升 MAC 削减。
- **扩展 o_proj 到更多层**：目前只替换了 L15/L16/L17/L27。v4 预研显示 L20/L23/L24/L25 也可尝试。
- **扩展到 L0–L14**：浅层可能更敏感，但也可能是新的 MAC 来源。
- ** Progressive build + short recovery**：每 build 完一层或几层后做一次短 fine-tune，避免误差累积。

### 4.4 更激进的轴（如果 down_proj + o_proj 不够）

- **gate_proj / up_proj**：占全模型 58% MAC，但输入是 hidden_size，输出是 intermediate_size，结构和 down_proj 对偶，可以研究。
- **Attention Q/K/V projection**：单个体量小，但合起来 7.1%，而且和 o_proj 在同一层，可能可以联合替换。

---

## 5. 推荐下一步扩展路线

### 路线 A：继续加大当前规模（最快验证上限）

在 sequential large 成功的基础上，把 down_proj group 数翻倍：

```
down L15–L27：每层 24/32 group
o_proj：L15/L16/L17/L20/L23/L27
```

目标：MAC 削减 **4–5%**，看 PPL 能否维持在 30–35。

### 路线 B：全模型 down_proj + 关键 o_proj

```
down L0–L27：非均匀 group（浅层少、深层多）
o_proj：L15/L16/L17/L27
```

目标：MAC 削减 **6–8%**，接受 PPL 35–45。

### 路线 C：引入 hidden-state distillation

在做路线 A 或 B 的同时，加入被替换模块输出的归一化 MSE loss，辅助中间层稳定。

---

## 6. 当前红线检查

| 红线 | 状态 |
|---|---|
| 动态参数必须通过 LUT 查表 | ✅ 未违反 |
| 比较基准同等计算量 | ⚠️ 后续扩量时需要和同等 group 数的静态/2D 对照 |
| 准确率只是验证指标 | ✅ 未违反 |
| 新增方法必须对 LUT 查表有帮助 | ✅ sequential build 直接改进 LUT 部署质量 |
| 禁止自动多卡分配 | ✅ 始终单卡 |

---

## 7. 一句话总结

> **Deployment-aware sequential build 是 v5 目前最大的有效进展。** 它把大规模 down_proj + o_proj joint replacement 从“完全不可用”拉到“勉强可用/可用”区间（2.89% MAC，PPL 28，Acc 0.48）。下一步应该在保持该方法的基础上，继续扩 group 数量和层数，同时尝试 hidden-state distillation 和 lr 优化来进一步压低 PPL。

---

*更新时间：2026-07-13*
