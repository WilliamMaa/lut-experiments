# v8 Native Factorized GDN Update — Compressed Plan

## 诊断阶段结束

已经有足够证据：

- Recurrent state 是 `(B, 32, 128, 128)` FP32，每层 2 MB。
- 每 head 的 128×128 state 矩阵强烈低秩：rank 32 重建误差约 1%。
- State 随时间变化大，不是 sparse delta。
- Head 能量不均。

**不再做更细的 diagnose。** 现在只验证最后一个最小问题：

> rank-32 的 snapshot 是不是立刻就不能用？

如果不是，马上进入方法实现：native factorized GDN update。

---

## 最小 smoke test（30 分钟级）

### 目标

回答：rank-32 压缩所有 GDN 层的 recurrent state 后，模型生成 64 个 token 会不会立刻崩溃。

### 范围

- 4 prompts
- prefix 256 tokens
- continuation 64 tokens
- rank 32 vs untouched
- 看：生成文本、EOS 成功率、重复率

### 命令

```bash
cd /data/mamingyu/v8
python -u attention_compact/eval_low_rank_state.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --rank 32 \
  --max_eval_samples 4 \
  --max_new_tokens 64 \
  --max_length 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --output_json results/gdn_low_rank_r32_smoke.json
```

### 判读

- 没崩、没严重重复、EOS 率没断崖 → 进入方法实现。
- 立刻崩 → rank-32 snapshot 不可用，跳过 factorized state，直接做 transition/LUT。

---

## 真正主线：Native Factorized GDN Update

### 为什么不是每 token SVD

`low_rank_state_patch.py` 是事后压缩，开销巨大。它只能验证 feasibility，不是方法。  
真正的方法要把低秩结构写进 GDN 的递推里。

### GDN 核心更新

Gated DeltaNet 的 recurrent update 是：

```
S_t = γ_t * S_{t-1} + u_t v_t^T
```

如果 `S_{t-1} = U_{t-1} V_{t-1}^T`，更新后天然得到：

```
S_t = [γ_t U_{t-1},  u_t] [V_{t-1}, v_t]^T
```

也就是说，**rank 每次 update 只增加 1**。我们真正要做的是：

```text
rank-r factors U_t, V_t
↓
delta update → rank (r + Δ)
↓
cheap truncation → rank r
↓
next U_{t+1}, V_{t+1}
```

不是：

```text
dense S
↓
full SVD
↓
dense S
```

### 进入 LUT 的路径

1. **Factorized recurrent state**（连续）
   - 维护 `S_t ≈ U_t V_t^T`。
   - 验证 dense update 和低秩 update 的 next-state / output 误差。

2. **Quantized / codebook factors**
   - 对 `U_t, V_t` 的 block 做 vector quantization。
   - 只存 `(code_u, code_v)`，必要时从 codebook 恢复。

3. **Transition LUT**
   - 最终目标：`(state_code, input_code) → next_state_code`。
   - 和 FFN LUT 统一：FFN 是 stateless `x_code → output`，GDN-LUT 是 stateful `(state_code, input_code) → next_state_code`。

---

## 2–3 天执行计划

### Day 1 上午

跑上面的 smoke test。只看一个结果：rank32 会不会立刻炸。

### Day 1 下午

读 DeltaNet recurrent update kernel。从 `modeling_qwen3_5_moe.py` 的 `torch_recurrent_gated_delta_rule` 和 `torch_chunk_gated_delta_rule` 里把 update 公式拆出来。目标：确认 `q, k, v, g, beta, S_{t-1}` 到 `S_t` 和 `output_t` 的具体算子。

### Day 2

实现单 head 的 factorized update prototype：

- 输入：dense `S_{t-1}`、`q, k, v, g, beta`
- 用公式做 rank-r 低秩 update + truncation
- 输出：`S_t` 和 `q @ S_t`
- 对比 dense teacher：
  - `||S_t_lowrank - S_t_dense||_F / ||S_t_dense||_F`
  - `||q @ S_t_lowrank - q @ S_t_dense||_F / ||q @ S_t_dense||_F`

先不在完整模型里跑，只在采好的 state 数据上跑。

### Day 3

把 factorized update 接回一个 GDN layer：

- 写 patch 替换某个 GDN 层的 decode/update 路径。
- 跑 32 / 64 / 128 token decode。
- 看 logit KL、top-1、生成是否稳定。

---

## 停止做这些事

- 不再跑 32+ samples 的完整 PPL sweep。
- 不再优化每 token SVD 的速度。
- 不再补 position-wise KL、完整 continuation curve 等 diagnose。
- 不回退到 VQK 或 KV cache。
- 不再“把结构搞得更明白”才开始方法。

---

## 核心原则

```text
提出结构假设 → 最小实验验证 → 马上做方法
```

72 小时内目标：

> 一个 native factorized GDN update prototype，能跑起来，能测 next-state 误差。

差也比再跑一天 diagnose 有价值。
