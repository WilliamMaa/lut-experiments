# 方案 4：Logit-Aware 与 Group-Aware 定向修正

## 目标

解决训练目标错位问题：当前 LUT 用 hidden-space MSE/cosine 训练，但模型真正关心的是下游 logit / next-token 行为。通过把 RMSNorm + lm_head top-k 接进训练目标，让 LUT 优先保护 teacher 的 top-1、top-k 集合和关键 token 排序。

同时根据各 group 的实际误差分布做非均匀容量分配，避免 32 组平均用力。

---

## 拆分成两个子分支

### 4A：Logit-aware 训练目标

训练时不直接最小化 `||F_T(x) - F_LUT(x)||`，而是最小化 teacher 和 LUT 在下一层 logit 上的差异。

### 4B：Group-aware 非均匀修正

根据每个 group 的误差、对 logit 的敏感度、难度稳定性，决定额外修正资源的分配。

---

## 4A：Logit-Aware 训练目标

### 核心观察

Layer 39 的 FFN 输出经过 RMSNorm 和 lm_head 后变成 logits。对 LLM 来说：

- 不是每个 hidden 维度同等重要
- 真正重要的是 top-k token 的 logit 值和排序
- EOS、newline、高频结构 token 的排序往往比低频词更重要

### 目标函数

对输入 `x`（layer 39 FFN input），定义：

```python
# 真实路径
h_teacher = x + FFN_T(x)
h_teacher_norm = RMSNorm_39_to_40(h_teacher)  # 假设只有一层 RMSNorm，需查模型结构
logits_teacher = lm_head(h_teacher_norm)      # 或 lm_head 的一部分

# LUT 路径
h_lut = x + FFN_LUT(x)
h_lut_norm = RMSNorm_39_to_40(h_lut)
logits_lut = lm_head(h_lut_norm)
```

但完整 lm_head 是 [vocab_size, hidden_size]，35B 模型 vocab 可能 15 万，直接 forward 太贵。

### Top-k 近似

只计算 teacher top-k（如 k=32 或 64）和当前 LUT top-k：

```python
topk_teacher = logits_teacher.topk(k=64).indices
union = unique(concat(topk_teacher, topk_lut))
# 只在这些 token 上算 loss
loss = F.kl_div(
    F.log_softmax(logits_lut[union], dim=-1),
    F.softmax(logits_teacher[union], dim=-1),
    reduction='batchmean'
)
```

但 `topk_lut` 需要完整 logits，除非我们有一个 cheap 的 proxy。更实际的做法：

### 更便宜的方案：lm_head 子集

预先把 lm_head 分成小块，或者只保留高频 token 对应行：

```python
# 预先选 top-4096 高频 token
important_tokens = get_top_frequent_tokens(tokenizer, corpus, k=4096)
head_subset = lm_head.weight[important_tokens]  # [4096, hidden_size]
```

训练时：

```python
logits_teacher_subset = h_teacher_norm @ head_subset.T
logits_lut_subset = h_lut_norm @ head_subset.T
loss = F.kl_div(
    F.log_softmax(logits_lut_subset, dim=-1),
    F.softmax(logits_teacher_subset, dim=-1),
)
```

这样可以不 forward 完整 lm_head，只用 4096 行矩阵乘。

### 组合损失

```python
loss = mse_hidden + alpha * cosine_hidden + beta * kl_logits
```

`beta` 可以随训练增大，先让 hidden 基本对齐，再优化 logit。

---

## 4A 实现步骤

### Step 1：定位 RMSNorm 和 lm_head

对 Qwen3.6-35B-A3B，需要确认：

- layer 39 FFN output 之后经过哪些 RMSNorm / Layernorm
- lm_head 的 weight shape 和 dtype
- 是否 tied embedding

用 `model.named_modules()` 打印路径。

### Step 2：选择重要 token 子集

```python
important_tokens = set()
# 1. 高频 token（从训练/验证语料统计）
important_tokens.update(top_frequent_tokens)
# 2. 结构 token：\n, EOS, 空格, 标点
important_tokens.update([newline_id, eos_id, space_id, comma_id, period_id])
# 3. 当前 teacher top-k 里出现的 token
important_tokens.update(topk_teacher.flatten().tolist())
```

最终保留 2048–8192 个 token。

### Step 3：修改 finetune 函数

在 `build_tail_aware_hard_correction.py` 或新的 `finetune_logit_aware.py` 中：

```python
for batch in dataloader:
    xb = calib_x[idx]
    true_yb = calib_y[idx]

    pred_yb = predict_lut(xb)

    # hidden loss
    mse = F.mse_loss(pred_yb, true_yb)
    cos = F.cosine_similarity(pred_yb, true_yb, dim=-1).mean()

    # logit-aware loss
    h_teacher = xb + true_yb
    h_lut = xb + pred_yb
    h_teacher_norm = rmsnorm(h_teacher)
    h_lut_norm = rmsnorm(h_lut)

    logits_teacher = h_teacher_norm @ head_subset.T
    logits_lut = h_lut_norm @ head_subset.T

    kl = F.kl_div(
        F.log_softmax(logits_lut, dim=-1),
        F.softmax(logits_teacher, dim=-1),
        reduction='batchmean'
    )

    loss = mse + (1 - cos) + beta * kl
```

### Step 4：可选 top-k 指标监控

训练时打印：

```python
teacher_top1 = logits_teacher.argmax(dim=-1)
lut_top1 = logits_lut.argmax(dim=-1)
top1_match = (teacher_top1 == lut_top1).float().mean()

teacher_topk = logits_teacher.topk(k=10).indices
lut_topk = logits_lut.topk(k=10).indices
# top-k overlap
overlap = (teacher_topk.unsqueeze(-1) == lut_topk.unsqueeze(-2)).any(dim=-1).float().mean()
```

---

## 4B：Group-Aware 非均匀修正

### 核心思想

32 个 output group 不是同等难。有些 group 的 residual 已经被 coarse + residual 拟合得很好，有些一直很差。应该：

- 对**持续困难**的 group：额外加一张 residual 表或低秩修正
- 对**偶发困难**的 group：联合 128/256 维修正
- 对**容易**的 group：保持当前结构

### 诊断指标

对每个 group：

```text
1. base cosine mean / p10
2. residual norm mean / p90
3. 跨样本 cosine 方差
4. group 对 full-output cosine 的边际贡献（ablation）
5. group 对 logit top-1 的敏感度（扰动实验）
```

### Group 难度分级

```python
for gid in group_ids:
    group_metrics = evaluate_group_ablation(gid)  # 把该组替换成基线，看 full-output 掉多少
    group_logit_sensitivity = measure_logit_sensitivity(gid)  # 扰动该组输出，看 logits 变化

# 按综合难度排序
hard_groups = top_k(difficulty, k=8)
medium_groups = next_k(difficulty, k=8)
easy_groups = rest
```

### 非均匀修正策略

| group 类型 | 修正方式 | 存储增量 |
|-----------|---------|---------|
| hard (8 groups) | 各加一张 16-bit residual 或低秩修正 | ~8 × 8 MiB = 64 MiB |
| medium (8 groups) | 共享一张 256-dim 联合修正 | 32 MiB |
| easy (16 groups) | 保持现有结构 | 0 |

### 与 4A 结合

logit-aware loss 本身就能自动告诉模型哪些 group 重要：如果某个 group 对 top-k logit 影响大，loss 梯度会自然更多地调整它。所以 4A 的训练目标可以部分替代 4B 的手工分组。

但 4B 仍然有价值：
- 决定在哪里**加表**（结构层面）
- 4A 只是调整已有表的值

---

## 与现有代码的接口

### Logit-aware finetune

- 新建 `finetune_logit_aware.py`，复用：
  - `build_tail_aware_hard_correction.py` 的 base 预测
  - `v6_replacement_engine.py` 的 checkpoint 加载
  - 从完整模型提取 RMSNorm 和 lm_head
- 输入：v4 checkpoint + 完整模型 + calibration 数据
- 输出：fine-tuned LUT checkpoint

### Group-aware 分配

- 新建 `diagnose_group_importance.py`：
  - 输入 v3/v4 checkpoint + 数据
  - 输出每个 group 的 difficulty / logit sensitivity
  - 推荐额外修正的 group 列表

---

## 实现顺序

建议：

1. **先做 4A 的 top-k logit 指标监控**（不加训练，只观察）
   - 用当前 LUT forward 一批样本
   - 记录 teacher top-1 match、top-k overlap
   - 判断 hidden cosine 0.81 时 logit 实际差多少

2. **再做 4A 的 finetune**
   - 初始 `beta=0.01`，慢慢加大
   - 同时监控 hidden cosine 不要崩

3. **最后做 4B 的分组**
   - 基于 4A 结果，看哪些 group 对 logit 最敏感
   - 只对困难且敏感的 group 加容量

---

## 风险

1. **lm_head 子集选取有偏**：只选高频 token 可能漏掉重要低频 token。可以动态扩展：把 teacher top-k 也加入子集。
2. **logit-aware loss 破坏 hidden 对齐**：如果 `beta` 太大，LUT 可能只优化重要 token 的方向，牺牲整体 hidden 质量。需要联合监控。
3. **RMSNorm 层数搞错**：如果 layer 39 后不只一层 norm，需要正确截取路径。
4. **成本高**：即使只用 top-4096 token，batch 里做矩阵乘也会增加显存。建议 batch_size 从 1024 降到 256。

---

## 与方案 1、方案 3 的关系

| 方案 | 解决什么问题 | 依赖 |
|-----|------------|------|
| 方案 1 | 数据分布不对 | 无（最基础） |
| 方案 3 | piecewise-constant 表达上限 | v3 checkpoint |
| 方案 4A | 优化目标错位 | 完整模型可 forward |
| 方案 4B | 容量分配不均 | 方案 4A 的敏感度分析 |

建议落地顺序：**方案 1 → 方案 3 诊断 → 方案 4A → 方案 4B**。
