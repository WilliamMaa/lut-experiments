# v8 VQK 到底在替换什么？

## v8 定位

```text
不是新的计算图
不是 LUT
不是 attention approximation

而是：
Transformer-specific weight representation experiment
```

## 一句话

> VQK **不替换任何层的计算图**，只是把某个 `nn.Linear` 的权重矩阵从 BF16 压缩成 **低 bit 整数 + block-wise scale**。
>
> 当前实现运行时先把 `W_q, S` 反量化成 BF16 再做普通 `Linear`；
> 所以第一阶段验证的是**权重表示是否保质量**，而不是 INT4 GEMM 的硬件收益。

## 目标层：layer 39 `self_attn.o_proj`

在 Transformer 中，`self_attn.o_proj` 是 attention 子层的最后一个线性投影：

```text
Q, K, V  →  attention 计算  →  o_proj  →  残差连接  →  下一层
```

`o_proj` 的输入是 attention 加权后的上下文向量，输出是 hidden_size 维的向量，加到 residual 上。

我们要替换的是 `o_proj.weight` 这个矩阵：

```text
原权重：W  (out_features, in_features)  BF16
替换后：W ≈ S ⊙ W_q
         W_q: int4 整数，shape 同 W
         S:   FP16 scale，每个 block 一个
```

前向传播时：**先把 `W_q, S` 反量化为 BF16 权重 `Ŵ = S ⊙ W_q`，再跑普通 GEMM**。因此第一阶段只看表示质量，不看计算收益。

## v8 VQK 与 v6 FFN LUT 的根本区别

### v6 FFN LUT：替换计算结果

```text
输入 x ──→ [mlp.shared_expert] ──→ y_teacher
                ↓
           用 LUT 查表直接得到 y_LUT
                ↓
           不再执行矩阵乘法和激活函数
```

- 替换对象：`mlp.shared_expert` 这个**子层本身**。
- 效果：绕过 dense 矩阵乘法，目标是**减少 MAC**。
- 数据流改变：x 直接进 LUT，输出 y_LUT。

### v8 VQK：替换权重精度

```text
输入 x ──→ [o_proj] ──→ y
                ↑
           W 从 BF16 变成 S ⊙ W_q
                ↑
           运行时反量化回 BF16 再做普通 GEMM
```

- 替换对象：`o_proj.weight` 这个**权重张量**。
- 效果：权重**存储**从 BF16 降到 4-bit，但前向仍是 FP16/BF16 GEMM。
- 数据流不变：x 仍然和反量化后的权重做矩阵乘法。

## 为什么先选 `o_proj`？

v8 文档里的顺序是 `o_proj` → `v_proj` → `down_proj` → 再看 `q_proj/k_proj/gate_proj/up_proj`。

原因：

- `q_proj / k_proj` 的误差会直接改变 attention score（QK^T），最敏感。
- `gate_proj / up_proj` 在 FFN 门控之前，误差会被 SiLU/门控放大，也较敏感。
- `o_proj / v_proj / down_proj` 是 attention 和 FFN 的**输出投影**，不直接修改 attention score。

但注意：**layer 39 的 `o_proj` 并不“安全”**。它紧邻 logits，任何权重扰动都会直接传导到输出分布。选它做第一轮不是因为它安全，而是因为它**既能测试权重表示的敏感度，又比 Q/K 更容易分析**。

## 实验到底在回答什么问题

> 对于 `layer39.self_attn.o_proj` 这个权重矩阵，
> VQK-style block-wise integer kernel + scale 是否能在相同 bit budget 下，
> 比 RTN / standard INT4 更好地保持 PPL / logit KL / 生成质量？

如果答案是 **No**（VQK-4 不如 RTN INT4 或差不多），整个 VQK 路线就没必要继续复杂化。

如果答案是 **Yes**，下一阶段再引入更强的 LLM PTQ baseline：

```text
RTN INT4
→ VQK INT4
→ AWQ / GPTQ reference
→ activation-aware VQK (block scale + activation-aware channel scaling)
→ multi-layer
```

## 怎么看结果

只看模型级指标：

- **PPL delta**：VQK 相比 baseline 涨了多少；再和 RTN INT4 比谁的涨幅小。
- **Logit KL**：VQK 的输出分布离 baseline 多远；和 RTN INT4 比谁更近。
- **Top-1 / Top-5 agreement**：token 级预测一致性。

**不要看** weight MSE 或 output cosine，这些 local 指标和最终 PPL 不一定相关。

## 当前进度

- ✅ v8 eval 框架
- ✅ VQK 实现（weight-only，运行时反量化）
- ✅ layer39.o_proj VQK-4 block=64 已跑通
- ⏳ 等待 RTN INT4 baseline 对比
- ⏳ 等待 bit/block sweep
