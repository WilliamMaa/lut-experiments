# VQK-based Transformer Quantization

验证 DSConv 风格的 VQK + block-wise KDS 在 Transformer Linear 上是否优于普通 INT quantization。

## 目标问题

> **VQK + block-wise distribution shift 是否能够在低 bit 权重量化下，比普通 INT quantization 更好地保持 Transformer 的 PPL 和生成质量？**

## 基本形式

```text
W ≈ S ⊙ W_q
```

- `W_q`：2/3/4/6/8-bit 整数权重。
- `S`：每个 block 一个 FP16/BF16 scale，block 沿 input dimension 划分。

## 实现计划

### Phase 1：Single Module Sensitivity

先固定 `layer39`，依次替换：

1. `o_proj`
2. `v_proj`
3. `down_proj`

测试 bit / block 组合：

```text
bits:   8, 6, 4, 3, 2
block:  32, 64, 128, 256
```

### Phase 2：Activation-Aware VQK

用真实 rollout 数据优化 scale：

```text
min_S  E_x~D || Wx - S W_q x ||^2
```

### Phase 3：Logit-Aware Calibration

加入 downstream logit KL 目标。

### Phase 4：Multi-Layer Scaling

```text
L39 → L38-39 → L37-39
```

## 待实现文件

- `vqk_patch.py`：`VQKPatch(EvalPatch)` 实现
- `eval_vqk.py`：bit/block sweep 入口
- `vqk_linear.py`：VQK Linear layer / wrapper

## 首个实验

```bash
python -u LLM_LUT/v8/vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --bits 4 \
  --block_size 64 \
  --eval_file eval.jsonl \
  --device_map balanced_low_0 \
  --output_json results/vqk_l39_o_proj_b4_block64.json
```

## 决策标准

如果以下任一条件不成立，停止 VQK 线：

1. **Same-bit advantage**：VQK-4 明显优于普通 INT4。
2. **Same-quality advantage**：相同 PPL 下 VQK 使用更少 bit。
3. **Multi-layer robustness**：多层量化时 VQK 退化小于普通量化。
