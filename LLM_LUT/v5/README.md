# LLM_LUT v5

v5 在 v4 的基础上引入**多元化 LUT** 策略，核心是两个新能力：

1. **可训练 LUT 表值**：不再固定使用 calibration 均值，而是把 table 当成可训练参数，在 fine-tune 中联合优化。
2. **高阶随机 address**：不再只依赖 2 个 residual channel，而是用固定随机的高阶比较/投影生成多 bit 索引，组成 1D LUT 集成。

目标：通过提升 LUT 本身的表现力，突破当前 down_proj 替换的质量天花板，更快地向 10% 全模型 MAC 削减推进。

## 文件结构

```
v5/
  address.py              # Address2D + AddressHighOrderRandom
  lut.py                  # 可训练集成 LUTGroup
  engine.py               # HybridPartialEngine（目前支持 down_proj）
  build_lut.py            # 生成 v5 LUT checkpoints
  inspect_address_modes.py # 对比 2D vs high_order 重建误差
  finetune.py             # 支持可训练 LUT 的多层微调
  data.py / metrics.py    # 从 v4 复用
```

## 快速开始

### 1. 对比 address 模式

```bash
cd LLM_LUT/v5
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inspect_address_modes.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layers "21,22,23" --num_groups 8 \
    --calib_size 256 --eval_size 128 \
    --output_path results/address_compare.json
```

### 2. 生成 high_order LUT checkpoints

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python build_lut.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8" \
    --address_mode high_order --num_tables 4 --num_bits 10 \
    --calib_size 256 --eval_size 128 \
    --output_root ../v5/outputs
```

### 3. Fine-tune（LUT 表值可训练）

```bash
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8" \
    --checkpoint_root ../v5/outputs \
    --epochs 5 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_v5_highorder
```

## 设计原则

- **地址生成固定随机**：不引入可学习 MLP/CNN，不违反项目红线。
- **表值可训练**：这是提升 LUT 拟合能力的关键。
- **集成**：多张小表独立寻址、输出相加，降低单地址冲突带来的方差。

## 后续扩展

- [ ] 把 high_order address 应用到 o_proj / gate_proj / up_proj 扫描
- [ ] 混合精度 LUT（FP16/INT8）
- [ ] 不同层用不同 address_mode / num_tables 的 layer-adaptive 配置
- [ ] hidden-state 蒸馏目标
