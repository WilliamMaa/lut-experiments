# LLM_LUT v5 Expansion Roadmap

> 目标：把当前 ~0.4%–2.8% 的全模型 MAC 削减，系统性地推向 10%。
> 本文件汇总已掌握的 down_proj / o_proj 可扩展性数据，作为后续实验的路线图。

---

## 1. 当前位置

| 实验 | 替换轴 | 层/组配置 | MAC 削减 | LUT 存储 | 最佳 PPL | 最佳 Acc |
|---|---|---|---|---|---|---|
| v4 2D INT8 | down_proj | L15–L27, 12–16 groups | **2.78%** | 49.25 MiB | 29.25 | 0.470 |
| v5 Tree FP16 | down_proj | L21–L23, 8 groups | 0.41% | 3.00 MiB | **20.84** | 0.523 |

**关键观察**：v5 tree + 可训练 LUT 在质量恢复上显著优于 v4 2D；但替换量还很小。下一步核心问题是：**用 tree 跑 v4 同规模（13 层、~160 group）配置，能否把 PPL 压到 25 甚至 20 以下？**

---

## 2. 全模型 MAC 预算拆解

模型：Qwen/Qwen2.5-7B-Instruct

| 算子 | 每 token MACs | 占单层比例 | 占全模型比例 | 全部替换可获 MAC 削减 |
|---|---|---|---|---|
| q_proj | 12.85M | 5.5% | 5.5% | 5.5% |
| k_proj | 1.84M | 0.8% | 0.8% | 0.8% |
| v_proj | 1.84M | 0.8% | 0.8% | 0.8% |
| **o_proj** | **12.85M** | **5.5%** | **5.5%** | **5.5%** |
| gate_proj | 67.90M | 29.1% | 29.1% | 29.1% |
| up_proj | 67.90M | 29.1% | 29.1% | 29.1% |
| **down_proj** | **67.90M** | **29.1%** | **29.1%** | **29.1%** |
| 单层合计 | 232.98M | 100% | — | — |
| 全模型合计 | 6.52B | — | 100% | 100% |

> 说明：以上按一次 forward 中各线性层矩阵乘法计算。attention Q/K/V/O 合计约 7.1%，MLP 占约 87.5%（其中 down_proj 29.1%，gate+up 58.3%）。

**结论**：
- 10% 全模型 MAC 削减 ≈ 替换 **34% 的 down_proj**，或 **100% down_proj + 少量 o_proj**。
- 若只盯 down_proj，至少要替换 1/3 的 down_proj 权重列（group）。
- o_proj 虽是“小头”（5.5%），但可以作为 down_proj 的补充轴，帮助突破 down_proj 单独的天花板。

---

## 3. down_proj 扩量空间

### 3.1 v4 已覆盖的层

v4 最佳配置已替换 **L15–L27 共 13 层**，非均匀 group 数：

| 层 | groups | 备注 |
|---|---|---|
| 15–21 | 12 | 标准覆盖 |
| 22, 23 | 16 | 对 group 削减更敏感，保留更多 group |
| 24–27 | 12 | 标准覆盖 |

> 来源：`LLM_LUT/v4/results/finetune_l15_l27_13layer_group12_calib2048_summary.json`

### 3.2 单层敏感度（减少 group 时的 raw LUT 质量变化）

从 `LLM_LUT/v4/results/group_sensitivity_decrease.json` 看，把某些层 group 数从 baseline 减少 2 个时：

| 变动 | ΔPPL（raw LUT） | 解读 |
|---|---|---|
| L15: 12→10 | **-482**（变好） | L15 对减少 group 不敏感，甚至可以少给 group |
| L16: 12→10 | **-91**（变好） | L16 同样不敏感 |
| L24: 12→10 | +135（变差） | L24 有一定敏感度 |
| L22: 16→12 | +1285（明显变差） | **L22 是高敏感度层，应多给 group** |

**启示**：
- **L22、L23** 是多给 group 的好候选（质量收益高）。
- **L15、L16** 可以少给 group，把预算让给敏感层。
- **L24** 需保持当前水平或谨慎增加。

### 3.3 尚未覆盖的层

v4/v5 目前主要替换 **深层（L15–L27）**。浅层 L0–L14 尚未系统扫描。按经验：
- 浅层通常对输出分布影响更大（早期特征提取），可能更敏感。
- 但也不排除部分浅层可以用较少 group 替换，进一步扩量。

**建议**：用 v5 tree address 对 L0–L14 做快速 raw LUT 扫描，找出质量≥L21 水平的层。

### 3.4 down_proj 扩量路径

| 阶段 | 动作 | 预计 MAC 削减 | 预计 LUT 存储（FP16 tree） |
|---|---|---|---|
| A | 复刻 v4 13 层配置，改用 tree | 2.78% | ~13 MiB |
| B | 在 L22/L23 加 group，L15/L16 减 group | 3.0–3.5% | ~15 MiB |
| C | 扩展到 L10–L27（18 层） | 4.5–5.5% | ~25 MiB |
| D | 扩展到 L0–L27（28 层），非均匀 group | 6–8% | ~40 MiB |
| E | 结合量化（INT8）或 group 共享 | 8–10%+ | <50 MiB |

---

## 4. o_proj 扩量空间

### 4.1 为什么看 o_proj？

- o_proj 占全模型 **5.5%** MAC，全部替换可直接贡献 5.5% MAC 削减。
- 但 o_proj 的残差结构与 down_proj 不同，不能直接照搬 down_proj 的 LUT 设计。

### 4.2 已有扫描结果

v4 做过两层 o_proj LUT 预研：

**直接预测 `o_proj(x)`（`o_proj_lut_inspection.json`）**

| 层 | relative_mse | 评估 |
|---|---|---|
| L17 | 0.39 | ✅ 优秀 |
| L15 | 0.46 | ✅ 优秀 |
| L16 | 0.50 | ✅ 优秀 |
| L20 | 0.52 | ✅ 较好 |
| L23 | 0.58 | ✅ 较好 |
| L18 | 0.60 | 可尝试 |
| L25 | 0.63 | 可尝试 |
| L24 | 0.64 | 可尝试 |
| L22 | 0.65 | 可尝试 |
| L26 | 0.67 | 可尝试 |
| L19 | 0.67 | 可尝试 |
| L21 | 0.74 | 可尝试 |
| L27 | ∞ | ❌ 失败（数值爆炸） |

**残差模式 `o_proj(x) - x`（`o_proj_lut_inspection_residual.json`）**

| 层 | relative_mse | 评估 |
|---|---|---|
| L27 | 0.18 | ✅ 极好 |
| L24 | 1.00 | ⚠️ 临界 |
| L17 | 1.06 | ❌ 略差 |
| L15 | 1.11 | ❌ 差 |
| L22 | 1.13 | ❌ 差 |
| L23 | 1.14 | ❌ 差 |
| L25 | 1.19 | ❌ 差 |
| L20 | 1.33 | ❌ 差 |
| L16, L18, L26 | 1.33–1.34 | ❌ 差 |
| L21 | 1.66 | ❌ 差 |
| L19 | 1.71 | ❌ 差 |

### 4.3 解读

- **L27 是唯一适合“残差模式”的层**：其 attention 输出 delta 很小、规律，用 LUT 预测 `o_proj(x) - x` 质量很高。
- **早期层（L15–L18、L20）适合“直接模式”**：直接预测 `o_proj(x)` 本身即可。
- **L21–L23、L25、L26** 两种模式都一般，暂不建议作为 o_proj 首发。

### 4.4 o_proj 首发建议

| 目标层 | 模式 | 理由 | 预计贡献 |
|---|---|---|---|
| **L27** | 残差 | rel_mse 仅 0.18，质量最高 | +0.2% MAC（单点验证） |
| **L17** | 直接 | rel_mse 0.39，早期层中最好 | +0.2% MAC |
| **L15/L16** | 直接 | rel_mse 0.46–0.50 | +0.4% MAC |

若 L27 + L15 + L16 + L17 同时成功，可额外获得约 **1.0–1.2%** 全模型 MAC 削减，且与 down_proj 不冲突。

### 4.5 工程注意

当前 `HybridPartialEngine` 只支持 `down_proj`。要替换 `o_proj` 需要：
1. 在 `self_attn.o_proj` 上安装 hook/替换 forward。
2. 处理 attention 输出维度（与 down_proj 同样是 hidden_size，可直接复用 group_size=64 概念）。
3. 支持“直接模式”和“残差模式”两种 reconstruction。

---

## 5. LUT 存储估算

### 5.1 单 group 存储

| Address | entries | tables | group_size | FP16 | INT8 |
|---|---|---|---|---|---|
| 2D (64×64) | 4096 | 1 | 64 | 512 KiB | 256 KiB |
| Tree (10-bit) | 1024 | 1 | 64 | 128 KiB | 64 KiB |

Tree 的单 group 存储是 2D 的 **1/4**，因为地址空间更小（1024 vs 4096）。

### 5.2 规模化估算

| 场景 | group 总数 | 2D INT8 | Tree FP16 | Tree INT8 |
|---|---|---|---|---|
| 当前 v5（3 层×8） | 24 | 6 MiB | 3 MiB | 1.5 MiB |
| v4 等效（13 层×~12） | ~160 | 40 MiB | 20 MiB | 10 MiB |
| 全 down_proj（28 层×56） | 1568 | 392 MiB | 196 MiB | 98 MiB |
| down_proj 1/3 + o_proj 20% | ~600 | 150 MiB | 75 MiB | 37.5 MiB |

**结论**：
- Tree 把存储压到可接受范围；若上 INT8，10% MAC 削减场景 LUT 存储可控制在 **<50 MiB**。
- v4 的 49 MiB INT8 对应 2.78% MAC；tree 用 FP16 就能在同等存储下做到更大替换量。

---

## 6. 推荐实验序列

**方向修正**：当前优先把 tree address 扩展到 **o_proj**，而不是继续在 down_proj 上做更多对照。

| # | 实验 | 目的 | 预期产出 |
|---|---|---|---|
| 1 | **o_proj L27 delta 单点 fine-tune** | 验证 tree + delta 模式在 o_proj 上能否端到端收敛 | 0.05% MAC，PPL/Acc 基准 |
| 2 | **o_proj L17 direct 单点 fine-tune** | 验证 tree + direct 模式在早期层效果 | 0.05% MAC，PPL/Acc 基准 |
| 3 | **o_proj 多点扩展**（L27 + L17 + L15/L16） | 看 o_proj 替换量能否线性累积 | 0.2–0.4% MAC |
| 4 | **down_proj + o_proj 联合 fine-tune** | 验证两轴同时替换不会互相拖垮 | 0.6–1.0% MAC，质量仍可用 |
| 5 | **tree 复刻 v4 13 层 down_proj** | 若 o_proj 扩展顺利后再回头做同规模对照 | 2.78% MAC，tree vs 2D |
| 6 | **浅层扫描 + 非均匀 group** | 在 down_proj 上进一步扩量 | 3.5–5% MAC |
| 7 | **gate/up_proj 预研** | 若 down_proj + o_proj 仍到不了 10%，评估 MLP 输入侧 | 理论最大 58% MAC |

---

## 7. 风险与红线检查

| 红线 | 当前规划是否违反？ | 说明 |
|---|---|---|
| 动态参数必须通过 LUT 查表 | ✅ 未违反 | tree/2D 都是固定 O(1) 地址 |
| 比较基准必须同等计算量 | ⚠️ 待执行 | 后续 o_proj 实验需与无 o_proj 替换的 down_proj-only 配置对比；down_proj tree vs 2D 对照延至步骤 5 |
| 准确率只是验证指标 | ✅ 未违反 | 目标始终是 MAC 削减与 LUT 表现力 |
| 新增方法必须对 LUT 查表有帮助 | ✅ 未违反 | o_proj 扫描/tree 改进都服务于 LUT 适配 |
| 禁止自动多卡分配 | ✅ 未违反 | 继续单卡 `CUDA_VISIBLE_DEVICES` |

---

## 8. 待补数据

- [ ] v4 13 层 tree 对照实验结果。
- [ ] 浅层 L0–L14 tree/raw LUT 质量扫描。
- [ ] o_proj engine 改造后的端到端 fine-tune 结果。
- [ ] gate/up_proj raw LUT 质量扫描（若需）。

---

*创建时间：2026-07-04*
