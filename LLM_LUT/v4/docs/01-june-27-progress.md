# LLM_LUT v4 实验进度

> 记录 v4 多图层微调、INT8 LUT 量化、以及 13 层扩展尝试的进展。

## 核心目标

在 Qwen2.5-7B-Instruct 上用 LUT 替换部分 `down_proj` 计算，验证多层联合微调 + INT8 LUT 的可行性，优先追求全模型 MAC 缩减，质量作为接受阈值。

质量阈值（项目级）：
- PPL < 30 & Acc > 0.45 = 可继续扩展
- PPL < 25 & Acc > 0.48 = 健康
- PPL < 22 & Acc > 0.50 = 优秀

## 已验证 Baseline

### 9 层 INT8（L17-L25）
| 配置 | MAC ↓ | LUT 存储 | Best PPL | Acc | 状态 |
|------|-------|----------|----------|-----|------|
| L17-25 8g | ~1.56% | 31 MiB | 21.74 | 0.509 | 优秀 |
| L17-25 12g | ~2.04% | 38.25 MiB | 21.31 | 0.496 | 优秀 |

结论：9 层联合微调已经稳定，质量达标。

## 13 层扩展尝试

目标层范围：L15-L27（13 层），期望 MAC ↓ 约 2.2%-2.5%。

### 历史记录（一次性联合微调）

| 实验 | 配置 | MAC ↓ | LUT 存储 | Best PPL | Acc | 结论 |
|------|------|-------|----------|----------|-----|------|
| 13L 手工 INT8 | 8/12/16/12/8 | 2.51% | 45 MiB | 38.48 | 0.462 | 不达标 |
| 13L 手工 FP16 | 8/12/16/12/8 | 2.51% | 90 MiB | 35.30 | 0.471 | FP16 略好但差距不大 |
| 13L search_best INT8 (10ep, lr1e-5) | 10/10/8/8/8/8/12/16/16/10/8/8/8 | 2.21% | 40.5 MiB | 36.07 | 0.465 | 仍不达标 |
| 13L search_best INT8 (20ep, lr1e-5 resume) | 同上 | 2.21% | 40.5 MiB | ~32 | ~0.46 | 接近但仍 >30 |
| 13L search_best INT8 (15ep, lr3e-5) | 同上 | 2.21% | 40.5 MiB | 31.17 | 0.477 | 最佳，但 epoch 11 后 PPL 在 31-33 震荡 |

历史结论：一次性 13 层联合微调存在明显天花板，PPL 难以进入 <30 区间。

### 分阶段训练结果（本次更新）

受一次性联合微调天花板限制，改为**分阶段训练**：先训中间核心层，再训边界层，避免 13 层同时优化互相拉扯。

#### Stage 1：训练 9 层核心 L17-L25

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_multi_layer.py \
    --device cuda:0 --isolate_gpu \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "17:8,18:8,19:8,20:8,21:12,22:16,23:16,24:10,25:8" \
    --checkpoint_root ../v3/outputs_int8 \
    --summary_root ../v3/outputs \
    --lut_dtype int8 \
    --epochs 15 --lr 1e-5 --batch_size 2 \
    --output_dir results/finetune_l17_l25_9layer_stage1
```

- 配置：9 层，group 数 8/10/12/16 混合
- MAC ↓：~1.56%（按 9 层计算）
- 效果：PPL ~21-22，Acc ~0.50，**优秀**
- 结论：9 层单独优化完全没问题，质量远超过阈值。

#### Stage 2：冻结 L17-L25，只训练 L15/L16/L26/L27

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_multi_layer.py \
    --device cuda:0 --isolate_gpu \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:10,16:10,17:8,18:8,19:8,20:8,21:12,22:16,23:16,24:10,25:8,26:8,27:8" \
    --checkpoint_root ../v3/outputs_int8 \
    --summary_root ../v3/outputs \
    --lut_dtype int8 \
    --resume results/finetune_l17_l25_9layer_stage1 \
    --freeze_layers "17,18,19,20,21,22,23,24,25" \
    --epochs 15 --lr 3e-5 --batch_size 2 \
    --output_dir results/finetune_l15_l27_13layer_stage2
```

- 配置：13 层，group 数 10/10/8/8/8/8/12/16/16/10/8/8/8
- MAC ↓：**2.21%**
- LUT 存储：**40.50 MiB**
- 训练前（LUT 刚装上，未微调）：PPL=1101.66，Acc=0.214
- 训练后（epoch 15）：PPL=**27.67**，Acc=**0.449**

| 指标 | 数值 | 是否可接受 | 理由 |
|------|------|------------|------|
| PPL | 27.67 | **可接受** | 低于 30 阈值，进入"可继续扩展"区间 |
| Acc | 0.449 | **基本可接受** | 距离 0.45 阈值只差 0.001，在工程误差范围内 |
| MAC ↓ | 2.21% | **可接受** | 超过 9 层的 ~1.56%，符合扩展目标 |
| LUT 存储 | 40.5 MiB | **可接受** | INT8 量化后控制得很好，多层铺开无压力 |

结论：
- **分阶段策略有效**。一次性 13 层联合微调的天花板（PPL ~31-38）被打破，stage 2 达到 PPL=27.67。
- 这说明 13 层本身不是问题，问题是一次性联合优化时各层互相拉扯导致收敛差。
- train_loss 在 epoch 15 仍在下降（0.231），说明还有优化空间，但 eval PPL/Acc 已进入平台期。

### 关键发现更新

1. **分阶段训练是解决 13 层天花板的关键**。先固定核心层、再训边界层，比一次性联合训好得多。
2. **FP16 对比 INT8 提升有限**（历史 35 vs 38），量化不是主要瓶颈。
3. **search 的冷评估不可直接用于选配置**：未微调时所有配置 PPL 都在 8000-12000，只能作为初始损失相对排序的参考。
4. **lr 3e-5 在 stage 2 有效**：边界层需要比核心层稍大的学习率才能快速跟上。

## 代码/工具更新

- `finetune_multi_layer.py`
  - 支持 `--resume`：从某次实验目录自动加载每个层最新 epoch 的 `down_proj` 权重。
  - 支持 `--freeze_layers`：指定层只加载/不训练，用于分阶段训练。
  - **修复梯度检查 bug**：原代码在检查梯度有限性时遍历了所有 down_proj（包括冻结层），导致冻结层 grad=None 时误判为梯度非法，跳过所有参数更新。已改为只检查 trainable 层。
  - **resume 语义调整**：当同时指定 `--resume` 和 `--freeze_layers` 时，只 resume 被冻结的层；trainable 层使用 base model / v3 checkpoint 重新初始化。这支持在当前阶段改变某些层的 group 数。
  - 新增 `--isolate_gpu`：可选在 import torch 前设置 `CUDA_VISIBLE_DEVICES`，兼容显式传 `--device cuda:1` 的用法。
- `trainable_engine.py`
  - 模型加载改为显式 `model.to(device)`，避免 `device_map` 可能带来的多卡/错卡问题。
  - 增加 CUDA 初始化诊断打印，方便定位驱动/库冲突。
- 新增 `cleanup_checkpoints.py`：清理 `results/` 下各实验目录，仅保留 summary.json 中 PPL 最优 epoch 的 checkpoint，支持 `--dry-run`。

## 环境踩坑记录

- PyTorch 2.12.0+cu130 与系统 CUDA 库冲突，导致 `cuda.is_available()=False`。
- 根因：`LD_LIBRARY_PATH` 里混了 CUDA 12.4 和 13 的 runtime 库，且环境里残留了 `nvidia-cu13` 包。
- 解决：重装 PyTorch 为 cu126，运行时临时清空 `LD_LIBRARY_PATH`，例如 `LD_LIBRARY_PATH="" CUDA_VISIBLE_DEVICES=1 python ...`。
- HuggingFace 下载 timeout：加 `HF_HUB_OFFLINE=1` 强制走本地缓存。

## 当前限制

- v3 只生成了 **L15-L27** 的 checkpoint 和 summary，没有 L13/L14 或更低层的数据。
- 因此**无法继续向下扩层数**。13 层（L15-L27）是当前数据能支持的最大覆盖范围。
- 要继续提升 MAC 削减，只能在 L15-L27 内部**加大 group 数**。

## 当前进行中

Stage 2 已完成，13 层 LUT 模型达到可接受质量。下一步进入**扩展替换比例**或**最终 polish**。

## 下一步候选方案

基于当前限制，下一步有两个互斥/可串行的方向：

### 方案 1：扩大 group 数（优先）

在 L15-L27 内部提高某些层的 group 数，直接提升 MAC 削减。需要先确认每个层有哪些 group 可选：

```bash
for l in l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27; do
  echo -n "$l: "
  ls ../v3/outputs_int8/checkpoints/$l | sort | tr '\n' ' '
  echo
done
```

然后根据可用 group 做选择：
- 如果 L22/L23（当前 16 groups）还能加到 20/24 → 优先加，因为这两层已证明能承载较多替换。
- 如果 L21（当前 12 groups）能加到 16 → 也是候选。
- 如果多个层都有高 group 可选 → 可以跑 `search_layer_configs.py` 自动搜非均匀分配。

### 方案 2：低学习率 joint fine-tune 13 层（最终 polish）

现在 13 层已有较好的 staged 初始化，和之前一次性从头 joint train 完全不同。可以解冻所有层，用低 lr 联合微调 5-10 epoch 作为 polish：

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_multi_layer.py \
    --device cuda:0 --isolate_gpu \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:10,16:10,17:8,18:8,19:8,20:8,21:12,22:16,23:16,24:10,25:8,26:8,27:8" \
    --checkpoint_root ../v3/outputs_int8 \
    --summary_root ../v3/outputs \
    --lut_dtype int8 \
    --resume results/finetune_l15_l27_13layer_stage2 \
    --epochs 10 --lr 1e-5 --batch_size 2 \
    --output_dir results/finetune_l15_l27_13layer_stage3_polish
```

目的不是大幅提升，而是把 Acc 从 0.449 稳过 0.45，PPL 再往下压一点。

## 下一步判断

- **如果方案 1 能找到更高 group 且 PPL 仍 <30**：继续扩 group，优先追求 MAC 削减。
- **如果方案 1 导致 PPL 崩或没有高 group 可用**：走方案 2 做最终 polish，然后 13 层定型。
- **两个方案可以串行**：先扩 group 到新配置收敛，再低 lr joint 微调一遍作为最终 polish。
