# VQK 评测计划（Layer 39 o_proj 首发）

> 目标：验证 DSConv 风格的 VQK + block-wise KDS 在 Transformer Linear 上是否优于普通 INT quantization。
> 首轮只跑 `layer39.self_attn.o_proj` 的 bit/block sweep，快速拿到 go / no-go 决策。

---

## 0. 当前状态

| 项目 | 状态 | 路径/产物 |
|---|---|---|
| v8 统一 eval 框架 | 完成 | `LLM_LUT/v8/common/evaluator.py` |
| VQK patch / wrapper | 待实现 | `LLM_LUT/v8/vqk/vqk_patch.py`、`vqk_linear.py` |
| VQK sweep 脚本 | 待实现 | `LLM_LUT/v8/vqk/eval_vqk.py` |
| 首轮实验（layer39.o_proj） | **下一步** | 未启动 |
| 跨 module 扩展（v_proj / down_proj） | 待决策 | 依赖 o_proj 结果 |
| 多层扩展 | 待决策 | 依赖单层结果 |

---

## 1. 核心原则

1. **先 o_proj，再 v_proj / down_proj**：`o_proj` 对 attention score 的影响是间接的，最适合做第一轮可行性测试。
2. **关键是同 bit 对比**：核心问题不是 VQK vs BF16，而是 **VQK-4 vs INT4、VQK-3 vs INT3**。
3. **不看 local cosine，只看 PPL / generation / logit KL**：权重 MSE 和输出 cosine 只作辅助参考。
4. **block size 必须 sweep**：32/64/128/256 都跑，不能只看一个 block。
5. **patch 必须可逆**：`install()` 替换 Linear，`uninstall()` 恢复原模块，保证多次 eval 不互相污染。

---

## 2. VQK 基本形式

对权重矩阵 `W` 沿 input dimension 分 block：

```text
W ≈ S ⊙ W_q
```

- `W_q`：2/3/4/6/8-bit 整数权重，与 `W` 同 shape。
- `S`：每个 block 一个 FP16/BF16 scale，shape 为 `(num_blocks, 1)` 或 broadcastable。
- scale 初始化采用 DSConv 的 closed-form L2 解：

```text
S_B = (Σ_i W_i * W_{q,i}) / (Σ_i W_{q,i}^2)
```

运行时：

```text
y = (S * W_q) @ x
```

等价于：先做低 bit 整数矩阵乘，再按 block 缩放。

---

## 3. 待实现模块

### 3.1 `vqk_linear.py`

- `VQKLinear`：封装 `nn.Linear`，替换为 `weight_q`（integer）+ `scales`（FP/BF16）。
- 支持 `bits={2,3,4,6,8}`，`block_size={32,64,128,256}`。
- 支持 `quantize_fn`：`round`（symmetric）或 `clamp-round`（asymmetric）。
- 第一轮用 **symmetric round-to-nearest** 即可。

### 3.2 `standard_quant.py`

- 用于生成 baseline：
  - `BF16`
  - `INT8`（per-token / per-channel 都行，先选简单 per-token）
  - `INT4`（同样 per-token）

### 3.3 `vqk_patch.py`

- `VQKPatch(EvalPatch)`：
  - `__init__(layer_idx, module_path, bits, block_size)`
  - `install(model)`：定位 module，替换为 `VQKLinear`
  - `uninstall(model)`：恢复原 `nn.Linear`
  - `name()` / `config()`

### 3.4 `eval_vqk.py`

- 从命令行接收 `--layer_idx`、`--module_path`、`--bits`、`--block_size`。
- 加载模型，构造 `VQKPatch`，调用 `Evaluator.evaluate()`。
- 支持 `--quant_method {vqk, int}` 用于跑 standard INT baseline。

---

## 4. 首轮实验矩阵

| ID | Method | Bits | Block | 目标模块 |
|----|--------|------|-------|----------|
| B0 | BF16 | 16 | — | layer39.o_proj |
| B1 | Standard INT8 | 8 | — | layer39.o_proj |
| B2 | Standard INT4 | 4 | — | layer39.o_proj |
| V1 | VQK | 8 | 64 | layer39.o_proj |
| V2 | VQK | 6 | 64 | layer39.o_proj |
| V3 | VQK | 4 | 64 | layer39.o_proj |
| V4 | VQK | 3 | 64 | layer39.o_proj |
| V5 | VQK | 4 | 32 | layer39.o_proj |
| V6 | VQK | 4 | 128 | layer39.o_proj |
| V7 | VQK | 4 | 256 | layer39.o_proj |

> 第一轮先不跑 VQK-2，避免 2-bit 直接崩掉浪费时间。

---

## 5. 完整执行顺序

### 5.1 实现 VQK 模块

```bash
cd LLM_LUT/v8/vqk
# 编写 vqk_linear.py、standard_quant.py、vqk_patch.py、eval_vqk.py
```

### 5.2 跑 BF16 baseline

```bash
python -u run_baseline_eval.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/eval.jsonl \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/vqk_b0_bf16.json
```

### 5.3 跑 Standard INT8 / INT4 baseline

```bash
python -u vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/eval.jsonl \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --quant_method int \
  --bits 8 \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/vqk_b1_int8.json

python -u vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/eval.jsonl \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --quant_method int \
  --bits 4 \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/vqk_b2_int4.json
```

### 5.4 跑 VQK bit sweep（block=64）

```bash
for bits in 8 6 4 3; do
  python -u vqk/eval_vqk.py \
    --model_path /data/models/Qwen3.6-35B-A3B \
    --eval_file /data/eval.jsonl \
    --layer_idx 39 \
    --module_path self_attn.o_proj \
    --quant_method vqk \
    --bits $bits \
    --block_size 64 \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --logit_metrics \
    --output_json results/vqk_v${bits}_b64.json
done
```

### 5.5 跑 VQK block sweep（bits=4）

```bash
for block in 32 128 256; do
  python -u vqk/eval_vqk.py \
    --model_path /data/models/Qwen3.6-35B-A3B \
    --eval_file /data/eval.jsonl \
    --layer_idx 39 \
    --module_path self_attn.o_proj \
    --quant_method vqk \
    --bits 4 \
    --block_size $block \
    --max_eval_samples 128 \
    --max_new_tokens 256 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --logit_metrics \
    --output_json results/vqk_v4_b${block}.json
done
```

### 5.6 汇总结果

```bash
python -u vqk/summarize_vqk_results.py \
  --result_dir results \
  --output_json results/vqk_summary_l39_o_proj.json
```

输出表格：

| ID | Method | Bits | Block | PPL | ΔPPL | Top-1 Agree | Top-5 Agree | Avg KL | EOS Rate | Repetition |
|----|--------|------|-------|-----|------|-------------|-------------|--------|----------|------------|

---

## 6. 资源与并发安排

- 每次 eval 会加载一次（或两次，若 `--logit_metrics`）Qwen3.6-35B-A3B，占用全部 GPU。
- **不要并行跑多个 eval**：会 OOM 或导致 CUDA 死锁。
- 建议用 shell loop 串行执行，或用 `wait` 保证上一个结束再跑下一个。
- 每个 eval JSON 预计 < 5 MB。

---

## 7. 预期结果与决策标准

| 情况 | 判断 | 行动 |
|---|---|---|
| VQK-4 block=64 PPL 明显优于 INT4 | **通过** | 扩展 block sweep，然后试 v_proj / down_proj |
| VQK-4 与 INT4 持平或略差 | 无价值 | 停止 VQK 线，转向 KV Cache Compression |
| VQK-4 优于 INT4 但 3-bit 崩 | 有价值但有限 | 只保留 4-bit，看是否能在更小 block 下进一步压缩 |
| 所有 VQK PPL 都崩 | 不可用 | 停止 VQK 线 |

继续 VQK 的**必要条件**（至少满足一条）：

1. **同 bit 优势**：VQK-4 明显优于 INT4。
2. **同质量优势**：相同 PPL 下 VQK 使用更少 bit。
3. **多层鲁棒性**：多层量化时 VQK 退化小于普通量化。

---

## 8. 现在立刻要做的

1. 实现 `LLM_LUT/v8/vqk/vqk_linear.py`、`standard_quant.py`、`vqk_patch.py`、`eval_vqk.py`。
2. 先跑 **B0（BF16 baseline）** 确认 eval 框架通顺。
3. 跑 **B1/B2（INT8/INT4 baseline）** 拿到对比基准。
4. 跑 **V1–V4（VQK bit sweep, block=64）**。
5. 如果 VQK-4 优于 INT4，继续 **V5–V7（block sweep）**。
