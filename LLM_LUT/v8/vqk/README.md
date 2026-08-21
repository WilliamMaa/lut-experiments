# VQK-based Transformer Quantization

验证 DSConv 风格的 VQK + block-wise KDS 在 Transformer Linear 上是否优于 RTN / standard INT quantization。

当前阶段是 **weight representation 实验**：运行时先把 `W_q, S` 反量化回 BF16 再做普通 `Linear`，所以先验证表示质量，不直接承诺 INT4 GEMM 收益。

## 目标问题

> 对 `layer39.self_attn.o_proj`，VQK-style block-wise integer kernel + scale 是否能在相同 bit budget 下，比 RTN / standard INT4 更好地保持 PPL / logit KL / 生成质量？
>
> 如果 VQK-4 明显优于 RTN INT4，下一阶段再与 AWQ / GPTQ 等现代 LLM PTQ 方法对比。

## 我们替换的是什么

VQK **不改动层的计算图**，只把目标 `nn.Linear` 的权重矩阵 `W` 从 BF16 换成低 bit 表示，运行时反量化：

```text
原前向：y = x @ W.T
替换后：y = x @ (S ⊙ W_q).T   # S ⊙ W_q 先反量化成 BF16 再 GEMM
```

- `W_q`：2/3/4/6/8-bit 整数权重，shape 同 `W`。
- `S`：每个 block 一个 FP16/BF16 scale，block 沿 input dimension 划分。

### 与 v6 FFN LUT 的区别

| | v6 FFN LUT | v8 VQK |
|---|---|---|
| 替换对象 | `mlp.shared_expert` 子层的**输出计算** | `self_attn.o_proj` 的**权重矩阵** |
| 计算图 | 改变：用 LUT 直接生成输出 | 不变：仍做矩阵乘法 |
| 优化目标 | 减少 FFN **MAC** | 减少 Linear **权重表示精度 / 存储** |
| 当前阶段 | 验证端到端 LUT 可行性 | 验证 block-wise scale 表示是否优于 RTN |

## 基本形式

```text
W ≈ S ⊙ W_q
```

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
cd LLM_LUT/v8
python -u vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --quant_method vqk \
  --bits 4 \
  --block_size 64 \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/vqk_l39_o_proj_b4_block64.json
```

## 决策标准

如果以下任一条件不成立，停止 VQK 线：

1. **Same-bit advantage**：VQK-4 明显优于 RTN INT4。
2. **Same-quality advantage**：相同 PPL 下 VQK 使用更少 bit。
3. **Multi-layer robustness**：多层量化时 VQK 退化小于 RTN。

下一阶段（若 VQK-4 优于 RTN）：引入 AWQ / GPTQ 作为更强 baseline。

