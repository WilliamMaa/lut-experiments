# LLM-LUT v1 开发计划

> 基于 V1_PROPOSAL.md，v1.0 的目标是：**验证 trainable LUT 能否优于 non-trained bucket average**。

---

## 1. 交付物清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `config.py` | V1Config | 待实现 |
| `lut_table.py` | TrainableLUTTable (nn.Module) | 待实现 |
| `lut_hook.py` | TrainableLUTHook (forward hook) | 待实现 |
| `train.py` | Local prefit + model-level eval | 待实现 |
| `run_v1.py` | 主入口，跑 V1-main + V1-control | 待实现 |
| `results/v1.0_report.md` | 实验报告 | 待生成 |

---

## 2. 实现阶段

### Phase 1: 基础设施 (config + LUT table)
- [x] V1Config: 继承 v0 配置，固定 group=4, bins=64, uniform, 2-head/1-head
- [x] TrainableLUTTable: `nn.Parameter` table + `F.embedding` lookup
- [x] 初始化：从 v0.5 bucket average 加载

### Phase 2: Hook 与替换逻辑
- [x] TrainableLUTHook: 复用 v0 address 计算，但 lookup 走 trainable table
- [x] 冻结 base model，只替换 target group

### Phase 3: 训练循环
- [x] Local prefit: MSE + cosine distance
- [x] Model-level eval: KL / PPL / next-token accuracy
- [x] Baseline pre-computation: original / zero / mean / bucket

### Phase 4: 主实验
- [x] V1-main: 2-head, 64 bins
- [x] V1-control: 1-head, 64 bins
- [x] 对比所有基线，生成 report

---

## 3. 关键技术决策

### 3.1 LUT Table 形状

```
1-head: [num_bins, group_size]         = [64, 64]
2-head: [num_bins, num_bins, group_size] = [64, 64, 64]
```

- 1-head: address 经 binning 后单索引 → `table[bin_idx]`
- 2-head: 两个 head 各产生一个 bin_idx → `table[bin_h0, bin_h1]`

### 3.2 训练设置

| 参数 | 值 | 理由 |
|------|-----|------|
| Optimizer | AdamW | 标准选择 |
| LR | 1e-3 | table 参数数量小，可用较大 LR |
| Scheduler | cosine | 简单 |
| Epochs | 20-50 | 观察收敛 |
| Loss | MSE + 0.1*cosine | local prefit |
| Batch size | 与 calib_loader 一致 | 复用已有 data loader |

### 3.3 评估流程

```
1. 原始模型跑 eval → B0 (KL=0, PPL=31.92)
2. zero/mean/bucket 替换 → B1/B2/B3
3. trainable LUT 替换 (frozen model) → B4
4. 比较 B3 (bucket) vs B4 (trainable)
```

---

## 4. 风险与预案

| 风险 | 概率 | 预案 |
|------|------|------|
| Trainable LUT < bucket | 中 | 尝试增大 calibration data / 改 loss weight / 加 L2 |
| 2-head 不如 1-head | 低 | V1-control 已设计，直接比较 |
| 训练 divergence | 低 | 从 bucket init 开始，LR 从 1e-4 试起 |
| Memory issue | 低 | 单 group 64-dim，table 极小 (64*64*64=256KB) |

---

## 5. 成功判定

跑完后看数据，按 V1_PROPOSAL.md §5.2 的 Minimum / Strong 标准判定。

```
Minimum: KL < 0.914, PPL ≤ 41.4
Strong:  KL < 0.82,  PPL 明显下降
```

---

**开始时间**: 现在  
**预计完成**: 1-2 小时内（含运行）
