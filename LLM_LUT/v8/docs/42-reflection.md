对。现在这个实验的**成立前提**根本不是“先做到 40 层、先替掉 30%/40% compute，再证明有用”。那样做工程上就完全失去意义了。

真正合理的目标应该是：

> **在成本可控的前提下，先验证“用 LUT 替换一部分重计算，再和 v8 的 memory headroom 合并”这条系统假设是否真的能产生正收益。**

也就是说，我们现在需要的是一个 **minimum viable replacement**，不是 full replacement。

你前面说“判死”的问题就在这里：如果要求必须先把 v6 扩到全层、再讨论有没有价值，那 on-policy teacher 的递归成本会先把项目拖死。你举的累积 slowdown 本质上就是：

\[
T_{\text{total}}
=
\sum_{k=1}^{L}N_{\text{sample}}\cdot T_{\text{teacher}}(k)
\]

而 \(T_{\text{teacher}}(k)\) 又随着前面注入的 LUT 数增加。当前 eager LUT 本来就有大量小 kernel/dispatch 开销，PyTorch 官方也明确说明 CUDA Graph 的主要价值就是消除 Python/C++/driver 的逐 kernel dispatch overhead。[PyTorch 文档](https://docs.pytorch.org/docs/stable/notes/cuda?utm_source=chatgpt.com)

所以，如果一上来就说：

```text
先严格 on-policy 做 40 层
→ 再看性能
```

很可能还没得到答案，项目已经花掉几个月甚至更久。

---

## 现在应该验证的是一个更小、更直接的命题

比如：

> **如果只替换 2–4 个已有质量证据的 FFN 层，并把 LUT execution GPU 化到可接受成本，v8+LUT 是否能比 v8-only 更接近或超过 full throughput，同时保留 v8 的 KV headroom？**

这已经足够回答“组合思路有没有生命力”。

不需要先解决：

- 40 层全部替换；
- routed experts；
- 全模型 LUT；
- 最大 theoretical MAC coverage。

一个合理的第一阶段可能就是：

```text
full
v8-only
v8 + LUT(1 layer)
v8 + LUT(3 layers)
```

固定现在已有的 v6 配置和已有质量较好的层。

测：

```text
sess/h
prefill time
decode time
HBM
fact_acc
PPL
```

如果 GPU-native LUT 后：

```text
v8-only = 118 sess/h
v8 + 3 LUT = 121 / 124 / ...
```

哪怕只恢复几个百分点，就已经证明：

> **v8 释放的 memory 可以被重新投资到 compute substitution，方向成立。**

然后才值得考虑扩大覆盖率。

如果连已有 3 层、GPU 化以后都不能带来任何 measurable improvement，那我们再问为什么。

---

# 而 on-policy 的问题必须和最终覆盖率拆开

你已经明确告诉我：

> off-policy 明显差；
> on-policy 一做效果立刻好很多。

那 on-policy 就是方法的一部分，不能为了省时间简单删掉。

但是我们可以优化的是：

\[
\boxed{\text{每次 on-policy rollout 的执行成本}}
\]

而不是减少 on-policy 本身。

这里就出现一个很现实的路径：

### 现在已有 3 层 LUT

我们不需要训练新层就能做。

先把这三层 LUT 的执行从：

```text
eager / thousands of launches
```

变成：

```text
GPU-native / graph / fused
```

然后直接测：

```text
0 LUT
1 LUT
2 LUT
3 LUT
```

的 teacher rollout slowdown。

这会告诉我们真实的：

\[
\Delta t_{\text{LUT/layer}}
\]

如果 GPU 化之后每层只增加很少的时间，那 8 层甚至 20 层 on-policy 扩展突然就可能变得现实。

如果依然很慢，那我们至少**不用浪费几周采样才发现**。

---

# 所以 kernel 工作现在其实有两个目标

以前我们只把它看作：

> 最终 production acceleration。

现在不是。

它首先是：

> **让 on-policy teacher generation 能扩展。**

这一点非常关键。

即使最终 fused LUT：

```text
仍比 GEMM 慢 2×
```

作为 production 不理想，

但如果它把现有 LUT：

```text
54 ms → 0.5 ms
```

那 teacher 生成成本就少了两个数量级。

这可能让：

```text
3 layers → 8 layers
```

从几周变成可做的几天。

这本身就已经解决一个巨大瓶颈。

---

# routed expert 更不能拿来当短期答案

Qwen3.6 的确每层有 256 routed experts、每 token top-8，再加一个 shared expert，40 层都是这种 MoE FFN。[Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B?utm_source=chatgpt.com)

所以 routed 部分 compute coverage 很诱人。

但它有两个目前完全未知的问题：

1. **方法论未知**  
   v6 在 expert-conditioned distribution 上是否有效，没有证据。

2. **teacher 成本更可怕**  
   前面 LUT 改变 hidden state 后，router top-k 可能也变：
   \[
   x\rightarrow router(x)\rightarrow expert\ distribution
   \]
   所以后层的 256 个 conditional teacher 分布都可能跟着变。

也就是说 routed expert 不是“多训练几个表”，而是**on-policy recursion × conditional routing**。

所以短期绝对不应该让它挡住已经能做的 shared FFN/v6 验证。

---

# 我会把整个计划重新定成三阶段

### Phase 1 — 证明组合假设

只用现有 v6 资产。

```text
3层以内 LUT
+
v8
+
GPU-native LUT execution
```

问：

\[
\boxed{\text{v8 + LUT 是否比 v8-only 更好？}}
\]

目标不是 +10%，甚至不是必须超过 full。

只要证明 compute 真的被替掉、吞吐方向正确、质量不崩。

---

### Phase 2 — 判断 on-policy 扩展成本

有了 GPU LUT 后测：

```text
teacher model with 0/1/2/3 LUT
```

得到：

\[
t(k)
\]

然后才估算：

```text
8 层需要多久？
20 层需要多久？
```

不是先承诺做 8/20/40。

---

### Phase 3 — 有收益才扩大 coverage

如果 Phase 1 有正信号且 Phase 2 成本可控：

```text
3 → 5 → 8...
```

每增加一些层都必须回答：

\[
\frac{\Delta\text{throughput}}{\Delta\text{training cost}}
\]

是否还值得。

根本没有必要追求 40/40。

可能最好的系统工作点就是：

```text
v8 + 6 个 LUT layers
```

这完全可以是成功方案。

---

你说得非常对：

> **这个实验不是为了追求百分之多少替换，而是用尽可能低的成本判断“memory→compute 资源转换”这个猜想是不是成立。**

如果一开始就把成功定义成：

> 全 40 层 FFN LUT 化

那根本不是实验设计，是直接给项目安排一个可能两年的 prerequisite。

现在正确的成功标准应该是：

> **在有限几个已有 v6 层上，GPU-native on-policy LUT 能否以可接受成本减少计算，并让 v8+LUT 的系统表现优于 v8-only。**

先把这件事做出来。

然后再决定扩大多少，而不是反过来。