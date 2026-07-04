# LLM_LUT v5

> 最新进展与详细思考记录见 [`PROGRESS.md`](./PROGRESS.md)。

v5 在 v4 的基础上引入**多元化 LUT** 策略，核心是两个新能力：

1. **可训练 LUT 表值**：不再固定使用 calibration 均值，而是把 table 当成可训练参数，在 fine-tune 中联合优化。
2. **多元化 address**：支持原版 2D、高阶随机投影，以及**离线贪心决策树 address**。决策树在 calibration 数据上选择能最大降低残差方差的随机投影做 split，构建固定树索引，无训练参数，仍属 O(1) 查表。

目标：通过提升 LUT 本身的表现力，突破当前 down_proj 替换的质量天花板，更快地向 10% 全模型 MAC 削减推进。

## 文件结构

```
v5/
  address.py              # Address2D + AddressHighOrderRandom + AddressGreedyTree
  lut.py                  # 可训练集成 LUTGroup
  engine.py               # HybridPartialEngine（目前支持 down_proj）
  build_lut.py            # 生成 v5 LUT checkpoints
  inspect_address_modes.py # 对比 2D / high_order / tree 重建误差
  finetune.py             # 支持可训练 LUT 的多层微调
  data.py / metrics.py    # 从 v4 复用
```

## 快速开始

### 1. 对比三种 address 模式

```bash
cd LLM_LUT/v5
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inspect_address_modes.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layers "21,22,23" --num_groups 8 \
    --num_bits 10 --tree_candidates 128 --tree_min_samples 32 \
    --calib_size 256 --eval_size 128 \
    --output_path results/address_compare.json
```

### 2. 生成 tree LUT checkpoints

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_lut.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8" \
    --address_mode tree --num_bits 10 --tree_candidates 128 --tree_min_samples 32 \
    --calib_size 256 --eval_size 128 \
    --output_root ../v5/outputs_tree
```

### 3. Fine-tune（LUT 表值可训练）

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8" \
    --checkpoint_root ../v5/outputs_tree \
    --epochs 5 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_v5_tree
```

## 设计原则

- **地址生成无训练参数**：2D / 高阶随机 / 决策树都是离线构造、推理时固定，没有可学习 MLP/CNN，不违反红线。
- **决策树地址是数据依赖的**：split 选择基于 calibration 数据上的残差方差下降，比纯随机投影更有信息量。
- **表值可训练**：`LUTGroup.table` 是 `nn.Parameter`，fine-tune 时会更新。
- **集成**：每组 M 张独立小表，输出相加。

## 后续扩展

- [ ] 把 tree address 应用到 o_proj / gate_proj / up_proj 扫描
- [ ] 混合精度 LUT（FP16/INT8）
- [ ] 不同层用不同 address_mode / num_tables 的 layer-adaptive 配置
- [ ] hidden-state 蒸馏目标
