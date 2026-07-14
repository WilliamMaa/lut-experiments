# 从 3% 到 10% MAC 削减的路线图

> 记录当前判断：为什么线性扩展会崩，以及一条更现实的通往 10% 的路径。

---

## 1. 当前位置

截至 2026-07-13，最有效的结果是：

| 实验 | MAC 削减 | LUT 存储 | 最佳 PPL | 最佳 Acc |
|---|---|---|---|---|
| sequential large | **2.89%** | 24.50 MiB | 28.08 | 0.482 |

这个结果是通过 **deployment-aware sequential build** 得到的：
- down_proj L15–L27（164 group）
- o_proj L15/L16/L17 direct + L27 delta

它证明了一个核心假设：

> **大规模 joint replacement 失败的主要原因是 build–deployment distribution mismatch，而不是 down_proj 和 o_proj 本身不能共存。**

---

## 2. 为什么线性扩展到 10% 会再次崩

Sequential build 解决的是最严重的问题，但如果继续用“每层加 group、每层加层”的方式线性扩展，会遇到第二层问题：

1. **局部近似误差沿残差流累积**
   - 即使每个 LUT 都在正确分布上 build，大量局部 nearest-neighbor 近似仍会叠加。
   - 残差连接会放大某些方向的误差。

2. **被替换模块之间互相放大**
   - o_proj 误差影响 down_proj 输入；
   - down_proj 误差影响下一层 attention；
   - 大规模下这种耦合是非线性的。

3. **资源分配过于均匀**
   - 当前给每层固定 12/16 group，没有考虑不同层、不同 group 的敏感度差异。
   - 有些层可能可以承受 30+ group，有些层 4 个都困难。

4. **Fine-tune 会再次改变分布**
   - Sequential build 只保证 build 时分布一致；
   - 但 joint fine-tune 后，student 分布又变了；
   - 没有 post-finetune rebuild。

因此，10% 不能理解为“把 2.89% 扩大 3.5 倍”，而应该理解为：

> **把 10% MAC 分散到模型真正能承受的位置，并用分阶段恢复维持一致性。**

---

## 3. 核心方法论转向

### 3.1 从“按层数/group 数扩展”转向“按单位质量损失分配 MAC”

最重要的指标变成：

```
replacement score = Δquality / ΔMAC
```

对每个候选 group，做一次快速单点替换评估，记录：
- module-output normalized MSE
- block-output MSE
- logits KL
- PPL delta
- 该 group 的 MAC
- LUT storage

然后做全局 budget selection：

```
min Σ C_g   s.t.   Σ MAC_g ≥ 10%
```

这会自然形成非均匀分布：
- 某些层替换 30–40 group；
- 某些层只替换 4–8 group；
- 极敏感层完全不动。

### 3.2 从一次性 sequential build 转向“build—recover—rebuild”

当前流程：

```text
build → install → build → install → ... → final fine-tune
```

建议升级成：

```text
build 2–3 layers
→ short recovery
→ recollect activations
→ rebuild affected downstream LUTs
→ continue expansion
```

最终阶段再加：

```text
final fine-tune
→ post-finetune rebuild of worst LUTs
→ final short recovery
```

这样解决两个问题：
- sequential build 解决前序 LUT 导致的漂移；
- iterative rebuild 解决 fine-tune 自身导致的二次漂移。

### 3.3 从 module-output MSE 转向 block-output anchor

不要约束孤立模块输出，而是约束 **经过残差相加后进入下一层的 block output**：

```text
L = λ_KL * L_KL + λ_block * L_block + λ_module * L_module
```

其中：
- `module loss`：只约束刚替换的 LUT 模块；
- `block loss`：约束完整 decoder block 输出（residual stream）；
- `KL loss`：负责最终语义行为。

block loss 不需要每层都加，可以只在几个 teacher anchor 上加：
- L7 / L14 / L21 / L27

或者每完成一个 replacement stage，只约束该 stage 的末端输出。

### 3.4 从“LUT 拟合完整输出”转向“LUT 拟合残差”

L27 o_proj 的 delta 模式已经验证了残差拟合更好。可以推广到所有模块：

```text
y ≈ y_cheap + LUT(x)
```

`y_cheap` 可以是：
- 保留一小部分 exact channels；
- 低秩固定近似；
- group mean / centroid projection；
- 极低精度线性基线。

关键约束：reference path 的 MAC 必须真实核算，不能隐藏计算。

### 3.5 从统一 LUT 容量转向敏感度自适应容量

当前统一使用：

```text
num_bits = 10
tree_candidates = 32
max_samples = 16384
```

10% 规模下应该按敏感度分配：

| group 类型 | bits | candidates | precision |
|---|---:|---:|---|
| 低敏感、容易拟合 | 8–9 | 16–32 | INT8 |
| 中等 | 10 | 32–64 | INT8/FP16 |
| 高收益但敏感 | 11–12 | 128 | FP16 |
| 极敏感 | 不替换 | — | — |

同时监控每个 tree leaf：
- 命中次数；
- 输出方差；
- teacher residual 方差；
- fine-tune 梯度大小。

对热门 leaf 拆分更多 bit，对低频 leaf 做邻居平滑或合并。

### 3.6 地址结构第二轮优化

Tree address 当前固定后不再改变。可以做轻量级交替优化：

```text
A. 固定 tree address，优化 LUT table
B. 固定 LUT/teacher，重新搜索误差最大的 split
C. 重建 table
D. short recovery
```

不需要地址本身可微，也不需要 MLP。只替换那些：
- leaf 内方差最大；
- 命中高度不均衡；
- downstream KL 贡献最大的 split。

---

## 4. 下一个模块的优先级

按 Qwen2.5-7B 配置（hidden=3584, intermediate=18944, 28 attn heads, 4 KV heads, 28 layers），单个模块占线性投影 MAC 如下：

| 模块 | 占比 |
|---|---:|
| gate_proj | 29.1% |
| up_proj | 29.1% |
| down_proj | 29.1% |
| q_proj | 5.5% |
| o_proj | 5.5% |
| k_proj | 0.8% |
| v_proj | 0.8% |

### 第一选择：gate_proj

最值得尝试。

原因：
- 单个模块占 29%，补 MAC 最快；
- 有研究显示 down_proj 相对耐受，gate_proj 敏感度层间差异大，适合选择性替换。

关键细节：
- gate_proj 误差经过 SiLU 后会与 up_proj 输出相乘；
- 不能只优化 `|ĝ - g|²`；
- 必须以 **post-SwiGLU output** 为 build target：

```text
|SiLU(ĝ) ⊙ u - SiLU(g) ⊙ u|²
```

### 第二选择：q_proj

占 5.5%，结构规整。

建议：
- 按 head 或 query head 分组；
- 优先中后层少量 heads；
- 评估指标不能只看 output MSE，还要看 attention logits MSE、softmax attention KL、attention output MSE。

### 第三选择：up_proj

同样占 29%，但敏感度通常比 gate 高。

建议：
- 放在 gate_proj 之后；
- 不要同时大规模替换同一层的 gate 和 up；
- 以 post-SwiGLU output 为 target。

### 不推荐优先：k_proj、v_proj

Qwen2.5-7B 用 GQA，28 query heads 对应 4 KV heads，k/v 矩阵很小。

- 每个只占约 0.8% MAC；
- 全部替换收益有限；
- 但会影响所有共享 KV heads 的 query heads，质量风险不低。

---

## 5. 通往 10% 的四阶段路线

### Stage 1：稳定 3%

目标：把当前 2.89% 的 PPL 28 稳定到 25–27。

做法：
- cosine lr decay；
- block-output anchor loss；
- post-finetune rebuild；
- 选出 best checkpoint。

### Stage 2：down_proj 扩到 5%

目标：MAC 削减 5%，PPL 不超过 30。

做法：
- 全局 group sensitivity ranking；
- 非均匀增加 group；
- 每增加约 1% MAC 做一次 short recovery；
- 用 sequential build + rebuild。

### Stage 3：加入选择性 gate_proj 到 7–8%

目标：MAC 削减 7–8%。

做法：
- 做单 group sensitivity map；
- 以 post-SwiGLU error 排序；
- 只选低敏感层；
- 暂不碰相同 group 的 up_proj。

### Stage 4：用 q_proj 或少量 up_proj 补到 10%

目标：MAC 削减 9–11%。

做法：
- q_proj 按 head 替换，优先低 attention-KL heads；
- 或少量 up_proj group；
- 最后做完整 progressive recovery 和 rebuild。

### 更现实的 10% 构成预估

```text
down_proj     5.5%–6.5%
gate_proj     1.5%–2.5%
o_proj        0.8%–1.2%
q_proj        0.5%–1.0%
-----------------------
total         9%–11%
```

---

## 6. 下一步最该做的两件事

### A. 实现 group sensitivity scanner

对 down_proj L15–L27 的每个 group：
1. install 一个临时 LUT；
2. 跑 eval（不 fine-tune）；
3. 记录 PPL、KL、block MSE、MAC；
4. uninstall。

输出一个排序表，指导 Stage 2 的非均匀扩展。

### B. 给当前 2.89% 加 block-output anchor + cosine decay

在当前 sequential large 配置上：
- 加 cosine lr decay；
- 加 L7/L14/L21/L27 block-output normalized MSE；
- 跑 10–15 epoch；
- 做一次 post-finetune rebuild。

目标：把 PPL 从 28 压到 25 左右，作为后续扩量的稳定起点。

---

## 7. 关键判断

> **10% 是可行的，但实现方式不是“把当前配置扩大 3.5 倍”，而是“sensitivity-aware progressive LUT deployment”。**

核心要素：
1. 按 Δquality/ΔMAC 选择 group；
2. 分阶段 build、恢复、重建；
3. 约束 residual stream 而不是孤立模块；
4. 对不同敏感度分配不同 LUT 容量；
5. 适时扩展 gate_proj / q_proj，而不是只盯着 down_proj + o_proj。

---

*记录时间：2026-07-13*
