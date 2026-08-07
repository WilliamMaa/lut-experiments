# 多层 shared_expert 替换计划（Layer 37/38/39）

> 目标：在单点最佳 layer 39 基础上，把 LUT 替换扩展到相邻三层（37/38/39），
> 验证“浅层替换 → 影响深层分布 → 必须逐层 on-policy 采集”这一核心假设。

---

## 0. 当前状态

| 项目 | 状态 | 路径/产物 |
|---|---|---|
| 最佳单层结果 | 完成 | `outputs_ffn_lut_layer39_shared_expert_v3_onpolicy` |
| l39 teacher 权重 | 完成 | `/home/u/mmy/qwen_35b_shared_expert_l39.pt` |
| l37/l38/l39 teacher 权重 | 完成 | `qwen_35b_shared_expert_l{37,38,39}.pt` |
| l37 数据（无替换） | 完成 | `datasets/layer37_shared_expert_v3_onpolicy/`：1000 文件，669,646 tokens |
| l37 LUT 训练 | **下一步** | 未启动 |
| l38 数据（l37 替换后） | 待做 | 依赖 l37 v4 checkpoint |
| l39 数据（l37+l38 替换后） | 待做 | 依赖 l37/l38 v4 checkpoint |
| 三层联合评估 | 待做 | 依赖三层 v4 checkpoint |

---

## 1. 核心原则

1. **从浅到深**：先训 37，再训 38，最后训 39。
2. **每层必须见过前面层的 LUT 输出**：采集某层数据时，已训好的上层/浅层要安装 LUT hook。
3. **hook 必须挂在 `shared_expert`，不能挂在 `mlp` 整体**：LUT 只近似 shared_expert，挂 mlp 会覆盖 routed experts 的输出。
4. **单层最佳 checkpoint 不动**：`outputs_ffn_lut_layer39_shared_expert_v3_onpolicy` 只作为参考，多层实验用新目录。

---

## 2. 已修复的关键 bug

`collect_shared_expert_data.py` 在多层采集时，替换引擎的 hook_path 默认挂到了完整 `mlp`，
这会让 LUT 输出覆盖整个 mlp 输出（包含 routed experts）。
已改为显式指向 `model.model.layers[idx].mlp.shared_expert`。

---

## 3. 完整执行顺序

### 3.1 训练 Layer 37（无前置替换）

```bash
python -u /home/u/mmy/build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path /home/u/mmy/qwen_35b_shared_expert_l37.pt \
  --dataset_dir /home/u/mmy/datasets/layer37_shared_expert_v3_onpolicy/input \
  --output_dataset_dir /home/u/mmy/datasets/layer37_shared_expert_v3_onpolicy/output \
  --output_root /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy \
  --group_size 64 --group_ids "0-31" \
  --coarse_num_bits 14 --residual_num_bits 16 \
  --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
  --calib_size 600000 --eval_size 69000 \
  --coarse_finetune_epochs 10 --residual_finetune_epochs 10 --finetune_epochs 50 \
  --finetune_loss_mode multi --device cuda:0 \
  > train_l37_v3_onpolicy.log 2>&1 &
```

注意：
- 数据在 `/home/u/mmy/datasets/layer37_shared_expert_v3_onpolicy/`，共 669,646 tokens。
- `calib_size=600000`、`eval_size=69000`，覆盖绝大部分数据。
- 如果实际 token 数不足，命令会报错，届时把 `calib_size` 和 `eval_size` 调小。

### 3.2 转换 Layer 37 到 v4 engine

```bash
python -u /home/u/mmy/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy/checkpoints \
  --output_root /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4 \
  --device cuda:0
```

### 3.3 采集 Layer 38 数据（l37 已替换）

```bash
python -u /home/u/mmy/collect_shared_expert_data.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --layer_idx 38 \
  --calib_file /home/u/mmy/candidate_prompts.jsonl \
  --output_dir /home/u/mmy/datasets/layer38_shared_expert_v3_onpolicy \
  --max_prompts 1000 --max_tokens_per_prompt 512 --max_new_tokens 512 \
  --max_total_tokens 1200000 \
  --replace_layer_idx 37 \
  --replace_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  > collect_l38.log 2>&1 &
```

### 3.4 训练 Layer 38

根据实际采集到的 token 数调整 `calib_size`/`eval_size`：

```bash
python -u /home/u/mmy/build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path /home/u/mmy/qwen_35b_shared_expert_l38.pt \
  --dataset_dir /home/u/mmy/datasets/layer38_shared_expert_v3_onpolicy/input \
  --output_dataset_dir /home/u/mmy/datasets/layer38_shared_expert_v3_onpolicy/output \
  --output_root /home/u/mmy/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy \
  --group_size 64 --group_ids "0-31" \
  --coarse_num_bits 14 --residual_num_bits 16 \
  --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
  --calib_size 1000000 --eval_size 100000 \
  --coarse_finetune_epochs 10 --residual_finetune_epochs 10 --finetune_epochs 50 \
  --finetune_loss_mode multi --device cuda:0 \
  > train_l38_v3_onpolicy.log 2>&1 &
```

### 3.5 转换 Layer 38 到 v4

```bash
python -u /home/u/mmy/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy/checkpoints \
  --output_root /home/u/mmy/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy_as_v4 \
  --device cuda:0
```

### 3.6 采集 Layer 39 数据（l37 + l38 已替换）

```bash
python -u /home/u/mmy/collect_shared_expert_data.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --layer_idx 39 \
  --calib_file /home/u/mmy/candidate_prompts.jsonl \
  --output_dir /home/u/mmy/datasets/layer39_shared_expert_v3_onpolicy_multilayer \
  --max_prompts 1000 --max_tokens_per_prompt 512 --max_new_tokens 512 \
  --max_total_tokens 1200000 \
  --replace_layer_idx 37 \
  --replace_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --replace_layer_idx 38 \
  --replace_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  > collect_l39_multilayer.log 2>&1 &
```

### 3.7 重新训练 Layer 39

**不要**用旧的 `outputs_ffn_lut_layer39_shared_expert_v3_onpolicy`，
要用新的多层 on-policy 数据训练：

```bash
python -u /home/u/mmy/build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path /home/u/mmy/qwen_35b_shared_expert_l39.pt \
  --dataset_dir /home/u/mmy/datasets/layer39_shared_expert_v3_onpolicy_multilayer/input \
  --output_dataset_dir /home/u/mmy/datasets/layer39_shared_expert_v3_onpolicy_multilayer/output \
  --output_root /home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer \
  --group_size 64 --group_ids "0-31" \
  --coarse_num_bits 14 --residual_num_bits 16 \
  --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
  --calib_size 1000000 --eval_size 100000 \
  --coarse_finetune_epochs 10 --residual_finetune_epochs 10 --finetune_epochs 50 \
  --finetune_loss_mode multi --device cuda:0 \
  > train_l39_multilayer.log 2>&1 &
```

### 3.8 转换 Layer 39 到 v4

```bash
python -u /home/u/mmy/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer/checkpoints \
  --output_root /home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer_as_v4 \
  --device cuda:0
```

### 3.9 三层联合评估

```bash
python -u /home/u/mmy/run_multilayer_model_eval.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --layer_idx 37 --checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --layer_idx 38 --checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --layer_idx 39 --checkpoint_dir /home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer_as_v4/checkpoints \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --max_eval_samples 128 --max_new_tokens 4096 \
  --output_json /home/u/mmy/multilayer_l37_39_eval.json \
  > multilayer_eval.log 2>&1 &
```

---

## 4. 资源与并发安排

- 数据采集用 `device_map=balanced_low_0`，占满所有 GPU，训练用 `device cuda:0` 占单卡。
- **不要并行**：采集和训练同时跑会抢 GPU。
- 每个 output_root 预计 1-2 GB（v3 checkpoint + v4 checkpoint + log）。

---

## 5. 预期结果与回退方案

| 情况 | 判断 | 行动 |
|---|---|---|
| 三层 PPL < 20，生成连贯 | 成功 | 继续扩展到更多层或更大 group |
| PPL 20-40，生成有轻微退化 | 可用 | 尝试加 on-policy 精选数据、增加训练 epoch |
| PPL > 40 或生成崩溃 | 不可用 | 退回到两层（37+38 或 38+39），保留单层 best |

---

## 6. 现在立刻要做的

1. 确认 `collect_shared_expert_data.py` 已更新（已修复 hook_path bug）。
2. 启动 **Layer 37 训练**（命令见 3.1）。
3. 训练完成后转 v4，然后启动 **Layer 38 数据采集**（命令见 3.3）。
