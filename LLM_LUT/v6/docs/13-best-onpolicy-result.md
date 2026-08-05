# Layer 39 shared_expert 最佳结果记录

> 记录截至目前的最佳 single-layer LUT：`outputs_ffn_lut_layer39_shared_expert_v3_onpolicy`
> 该结果在 eval 上 cos=0.8375，模型级 PPL baseline 8.00 → LUT 11.68，长文本生成基本连贯。

---

## 1. 复现命令

### 1.1 训练 v3 on-policy base

```bash
python -u build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path qwen_35b_shared_expert_l39.pt \
  --dataset_dir datasets/layer39_mixed_v3/input \
  --output_dataset_dir datasets/layer39_mixed_v3/output \
  --output_root outputs_ffn_lut_layer39_shared_expert_v3_onpolicy \
  --group_size 64 \
  --group_ids "0-31" \
  --coarse_num_bits 14 \
  --residual_num_bits 16 \
  --tree_candidates 256 \
  --tree_min_samples 4 \
  --tree_max_samples 400000 \
  --calib_size 550000 \
  --eval_size 68000 \
  --coarse_finetune_epochs 0 \
  --residual_finetune_epochs 10 \
  --finetune_epochs 50 \
  --finetune_loss_mode multi \
  --device cuda:0 \
  --resume \
  > train_v3_onpolicy.log 2>&1 &
```

关键参数说明：

| 参数 | 取值 | 含义 |
|---|---|---|
| `coarse_num_bits` | 14 | shared global coarse tree，16,384 个 leaf |
| `residual_num_bits` | 16 | 每组 residual tree，65,536 个 leaf |
| `group_size` / `group_ids` | 64 / 0-31 | 把 2048 维输出切成 32 个 64 维 group |
| `tree_candidates` | 256 | 每 bit 候选 256 组通道 |
| `tree_min_samples` | 4 | 建树时叶子最小样本数 |
| `tree_max_samples` | 400000 | 建树时最多用的样本数 |
| `calib_size` / `eval_size` | 550000 / 68000 | 训练/验证样本数 |
| `residual_finetune_epochs` | 10 | 每组 residual 单独 finetune |
| `finetune_epochs` | 50 | coarse + 全部 residual 联合 finetune |
| `finetune_loss_mode` | `multi` | MSE + cosine + residual cosine + log norm ratio |
| `resume` | — | 从已有 v3 rollout checkpoint 继续，不重建 tree |

### 1.2 转成 v4 engine 格式

```bash
python -u convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/checkpoints \
  --output_root outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_as_v4 \
  --device cuda:0
```

### 1.3 跑模型级 PPL / 生成

```bash
python -u run_model_eval.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --checkpoint_dir outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --layer_idx 39 \
  --hook_path "model.model.layers[39].mlp.shared_expert" \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --max_eval_samples 128 \
  --max_new_tokens 4096 \
  --output_json generation_v3_onpolicy_2048_3prompts.json
```

---

## 2. 原理

### 2.1 结构：shared global coarse + per-group residual

- **完整 FFN 输出是 2048 维**，直接作为 coarse 目标。
- 一棵 **shared global coarse tree**（14 bit，16K leaf）覆盖完整 2048 维。
- 用 coarse 预测减去真实输出得到 **residual**。
- 把 residual 切成 32 个 64 维 group，每个 group 独立建一棵 **16 bit residual tree**（65K leaf）。
- 最终输出：

```
y_LUT = coarse_table[coarse_leaf] + sum_g residual_table_g[residual_leaf_g]
```

### 2.2 为什么用 multi-loss

joint finetune 时，`multi` 模式同时优化：

- MSE：让 LUT 输出逼近 teacher；
- output cosine：保证方向一致；
- **residual cosine**：让 `x + y_LUT` 和 `x + y_teacher` 方向一致，对自回归下一层输入至关重要；
- **log norm ratio**：让输出范数接近 teacher，避免过度放大/缩小。

这就是为什么它在生成上比只用 MSE 的低秩/配对实验更稳定。

### 2.3 数据：mixed_v3

`datasets/layer39_mixed_v3/` 由两部分合并：

1. **rollout 数据**（~60 万 token-level 样本）
   - 用 1000 条 candidate prompts；
   - 每条生成 512 个新 token；
   - 采集 layer 39 `shared_expert` 的真实输入/输出。
2. **on-policy 数据**（17,988 条高价值样本）
   - 用训练好的 v3 LUT 替换 layer 39 shared_expert；
   - 在 100 条 prompt 上各生成 1024 token；
   - 记录 LUT 实际访问到的 FFN 输入；
   - 用独立 teacher 给这些状态重新打标签；
   - 按位置 + 困难度 + leaf 新颖性采样保留。

关键教训：必须让模型真正生成长文本，prompt-only 只能覆盖 encoding 状态，看不到 layer 39 在长生成中的真实访问分布。

---

## 3. 指标

### 3.1 FFN-level eval

来源：`outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/summary.json`

| 指标 | 值 |
|---|---|
| cosine_similarity | 0.8375 |
| cosine_similarity_p10 | 0.6766 |
| cosine_similarity_p50 | 0.8590 |
| cosine_similarity_p90 | 0.9851 |
| relative_l2 | 0.5055 |
| norm_ratio | 0.9492 |
| coarse table | 64 MiB |
| residual tables | 256 MiB |
| total table | 320 MiB |

### 3.2 模型级 PPL

来源：`results/generation_v3_onpolicy_2048_3prompts.json`

| | PPL |
|---|---|
| Baseline | 8.0004 |
| LUT | 11.6844 |
| Δ | +3.6840 |

---

## 4. 生成示例

### Prompt 1（中文：一战原因）

**Baseline** 开头：

> 1914年爆发的第一次世界大战，被历史学家普遍视为现代世界历史的转折点……

**LUT** 开头：

> **火药桶的引信：第一次世界大战爆发的多维深度解析**
>
> 1914年6月28日，奥匈帝国皇储弗朗茨·斐迪南大公在萨拉热窝遇刺，这一看似偶然的悲剧性事件，迅速引爆了早已干涸积满火药的国际关系体系……

结论：LUT 版本结构完整、结尾自然，没有明显复读或突兀截断。

### Prompt 2（英文：2008 金融危机）

**LUT** 给出的结构：

1. Deconstruct the Prompt
2. Outline Structure
3. Draft - Section by Section
4. 最终展开 causes / consequences / policy responses

虽然会以 "thinking process" 开头，但整体逻辑清晰。

### Prompt 3（中文：Transformer 注意力机制）

**LUT** 输出包含：

- 引言
- 自注意力机制（Self-Attention）：线性投影、注意力分数、加权求和
- 多头注意力机制（Multi-head Attention）
- 位置编码（Positional Encoding）
- 前馈神经网络（Feed-Forward Network）
- 总结

公式和实现细节基本正确，技术表达准确。

---

## 5. 产物位置

| 产物 | 路径 |
|---|---|
| teacher 权重 | `/home/u/mmy/qwen_35b_shared_expert_l39.pt` |
| v3 base checkpoint | `outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/checkpoints/` |
| v4 engine checkpoint | `outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_as_v4/checkpoints/` |
| mixed 数据 | `datasets/layer39_mixed_v3/` |
| 生成/PPL 报告 | `results/generation_v3_onpolicy_2048_3prompts.json` |
| 摘要 | `results/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_summary.json` |

---

## 6. 注意事项 / 红线

1. **不要再往 `outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/checkpoints/` 里写新文件**
   - 当前这个目录里有我们误存进去的 `pairwise.pt`。
   - 用该 base 做 v4 转换前，建议先删掉 `pairwise.pt`，避免 engine 误加载：
     ```bash
     rm -f outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/checkpoints/pairwise.pt
     ```
2. **低秩和配对实验的权重不要混到这个目录**
   - 后续低秩/配对结果应放到独立的 output_root 里。
3. **这是 single-layer 结果**
   - 下一步是 multi-layer（layer 37/38/39 shared_expert），单层的 cos/PPL 已经接近天花板，不值得继续刷。

---

## 7. 简要结论

- 这是目前 layer 39 shared_expert 上 **生成质量最稳定、PPL 最好** 的 LUT 配置。
- 核心成功因素：**mixed 数据 + multi-loss + 联合 finetune**。
- 后续所有 multi-layer 实验都应以这个 checkpoint 为起点。
