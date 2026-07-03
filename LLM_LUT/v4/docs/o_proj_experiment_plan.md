# o_proj 实验计划（与 down_proj 联合）

目标：在已有的 `down_proj` LUT 替换基础上，额外替换部分 `o_proj` 输出通道，进一步降低 MAC，同时尽量控制 PPL/Acc 损失。

## 当前结论

- 完整输出 LUT（非 residual）的 `o_proj` 重建误差较大（relative_mse 0.39~0.74），直接替换不现实。
- 最有潜力的层（按 full-output 结果）：**L17、L15、L16、L20**。
- 需要先用 **residual 模式** 重跑 inspection，确认真实可替换性。
- 实现方案已就绪：`build_o_proj_lut.py` + `TrainableOProjPartialEngine` + 扩展后的 `finetune_multi_layer.py`。

## 实验步骤

### Step 1: residual 检查（ranking）

```bash
cd LLM_LUT/v3
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inspect_o_proj_lut.py \
    --residual \
    --layers "15,16,17,18,19,20,21,22,23,24,25,26,27" \
    --calib_size 256 --eval_size 128 \
    --output_path results/o_proj_lut_inspection_residual.json
```

输出会包含 `relative_mse`、`group_rmse` 等指标。挑选原则：
- 层级 `relative_mse` 尽量低（建议 < 0.25 再考虑）。
- per-group RMSE 没有明显尖峰（max 接近 mean）。
- 优先试 **4 层 × 4 group**，不要一次铺太多。

### Step 2: 生成 o_proj LUT checkpoints

假设 residual 结果表明 L17、L15、L16、L20 各 4 group 可试：

```bash
cd LLM_LUT/v4
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_o_proj_lut.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "17:4,15:4,16:4,20:4" \
    --calib_size 256 --eval_size 128 \
    --output_root ../v3/o_proj_outputs
```

默认使用同一 address pair 替换一层内所有 group；如果效果不好可加 `--per_group_addr`。

### Step 3: 第一阶段 —— 只训 o_proj，freeze down_proj

以当前 best down_proj checkpoint（`finetune_l15_l27_13layer_group12_calib2048` epoch 14）为起点：

```bash
cd LLM_LUT/v4
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_multi_layer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:12,16:12,17:15,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --checkpoint_root ../v3/outputs_int8 \
    --o_proj_configs "17:4,15:4,16:4,20:4" \
    --o_proj_checkpoint_root ../v3/o_proj_outputs \
    --resume results/finetune_l15_l27_13layer_group12_calib2048 \
    --resume_best_epoch \
    --train_o_proj_only \
    --freeze_layers "15,16,17,18,19,20,21,22,23,24,25,26,27" \
    --epochs 5 --lr 5e-6 --calib_size 2048 --eval_size 128 \
    --output_dir results/finetune_down12_o4_first_stage
```

关键参数：
- `--train_o_proj_only`: 冻结所有 `down_proj`。
- `--freeze_layers`: 同时保证这些层不参与训练。
- `--resume_best_epoch`: 从 best PPL epoch 恢复 down_proj 权重。

### Step 4: 第二阶段 —— 低学习率 joint polish（可选）

如果第一阶段 PPL/Acc 进入可用区间（PPL < 35, Acc > 0.45），可以尝试把 down_proj 也放开一起低学习率 polish：

```bash
python finetune_multi_layer.py \
    ... \
    --resume results/finetune_down12_o4_first_stage \
    --resume_best_epoch \
    --epochs 3 --lr 1e-6 --calib_size 2048 \
    --output_dir results/finetune_down12_o4_polish
```

注意：**不要加 `--train_o_proj_only` 和 `--freeze_layers`**，让两层权重一起微调。

## 预期 MAC 增益

- `o_proj` 每层每 group 的 MAC 约为 `down_proj` 每 group 的 `hidden_size / intermediate_size ≈ 3584 / 18944 ≈ 18.9%`。
- 当前 down_proj group12 × 13 层 ≈ **2.78%** 全模型 MAC 削减。
- 加上 4 层 × 4 group 的 o_proj 约额外 **0.2~0.3%** 绝对 MAC 削减。
- 收益不大，但如果质量损失接近 0，还是值得往前推。

## 判断标准

| 结果 | PPL | Acc | 行动 |
|---|---|---|---|
| 优秀 | < 30 | > 0.45 | 继续扩 o_proj group 或层数 |
| 可用 | < 35 | > 0.40 | 尝试 joint polish 或更多 group |
| 不可用 | ≥ 35 或 Acc < 0.40 | — | o_proj 这条路在当前地址/表大小下走不通，回退到纯 down_proj 优化 |

## 代码改动清单

- `LLM_LUT/v3/inspect_o_proj_lut.py`: 增加 `--residual` 和 fp32 累加。
- `LLM_LUT/v4/build_o_proj_lut.py`: 新建，生成 per-group residual LUT checkpoints。
- `LLM_LUT/v4/o_proj_engine.py`: 新建 `TrainableOProjPartialEngine`。
- `LLM_LUT/v4/finetune_multi_layer.py`: 支持 `--o_proj_configs`、`--o_proj_checkpoint_root`、`--train_o_proj_only`、o_proj resume/save。
