对，我觉得**闭环必须保留，而且100个 prompt 是一个合理的第一轮规模**。不是再退回16条做玩具实验，也不是硬跑1000条把几百小时烧掉。

但这100条要承担的是：

> 建立第一版真实的 on-policy 长尾数据集，然后反复用于方案3、方案4以及后续结构优化。

而不是继续作为1000条筛选流程的前置步骤。

目前 v4 已经证明短程内容能力还在，问题主要集中在回答结束后持续续写、模板循环、角色标签泄漏以及后段逐渐恶化。 所以我们现在真正要优化的“长尾”至少包含两种：

1. **映射误差长尾**：少量 token 的 FFN cosine、relative L2 特别差；
2. **轨迹长尾**：前面正常，某个位置以后进入重复、错误格式或异常吸引域。

这两种不能再用一张静态 hard correction table统一解决。正确方向应该是：

## 先把这100条闭环数据跑对

前一版100条结果出现：

```text
平均每条约397个样本
但 short_max_new_tokens = 256
```

而且 norm ratio从离线约1.07掉到0.395。这首先需要修复生成切片、teacher对象和runtime LUT一致性。采集脚本里 left-padding后使用 `prompt_lens[b]`切生成结果，以及把聚合leaf histogram当作逐token leaf访问的问题，都会污染数据。

所以不需要立刻重跑全部100条。先做：

```text
3条 prompt × 16 token
10条 prompt × 64 token
确认无误后
100条 prompt × 2048 token
```

前两步不是缩小研究规模，只是确保几百小时不会再次浪费。

必须通过三个硬性检查：

```text
生成token数 <= max_new_tokens
original teacher vs loaded teacher cosine ≈ 1
offline LUT vs runtime LUT cosine ≈ 1
```

只有这样，100条闭环数据才可信。

## 100条 prompt 不必再做昂贵的 activation selection

既然100条已经能接受，就直接人为构建一个**有明确结构覆盖的固定集合**。这里的目标不是证明这是全世界最优prompt集合，而是覆盖几类不同的长期轨迹。

我建议大致分成：

```text
20 条：长篇分析与解释
15 条：代码、技术报告、系统设计
15 条：数学、逻辑、逐步推导
15 条：短问题，本应尽快停止
10 条：多章节/列表/表格/复杂格式
10 条：中文
5  条：日文
5  条：混合语言或翻译
5  条：开放式创作/故事/对话
```

这里类别可以重叠，最终总数保持100。

尤其需要保留大约15条**本应短答并尽快EOS**的prompt，因为目前 v4 最明确的问题之一，就是问题已经答完却开始自动出下一道题。这些prompt对 termination tail非常关键。

## 100条全部跑闭环，但不要所有token一视同仁

每条生成到2048，保存 LUT实际访问到的 (x_t)，然后用真实 teacher对这些状态重新标注：

[
x_t^{LUT}\rightarrow y_t^{teacher}
]

这一步必须有。

但训练数据应分成四类保存，而不是最后混成一个无标签的大张量：

### A. 正常状态

生成前段正常、teacher–LUT误差也不大的token。

作用是防止新优化破坏目前已经不错的短程能力。

### B. 表示困难状态

例如：

```text
cosine < 0.6
relative L2 > 1.0
norm ratio异常
连续低cosine区间
```

作用是解决hidden映射的低分位。

### C. 轨迹临界状态

第一次明显出现以下行为之前的64–128 token：

```text
开始重复问题
出现Human:/Assistant:/user标签
重复同一格式
代码语法开始持续恶化
换行或空格循环
```

这是最有价值的一批。因为崩溃后已经进入异常吸引域的数据未必容易修，而崩溃前的临界状态更可能决定能否避免进入吸引域。

### D. 崩溃后状态

少量保存，单独标记，不要成为主要训练数据。

否则模型会花大量容量拟合已经极度偏离的状态，反而挤掉正常区和临界区。

## 然后100条数据并行支持三个优化分支

### 分支一：On-policy重建base或residual

先做最直接的闭环验证：

* 原 calibration 数据保留；
* 加入100条 rollout里的正常状态和临界状态；
* 不只加入极端低cosine样本；
* 重新训练当前 shared coarse + grouped residual。

这里的核心问题是：

> 仅仅让LUT看到自己实际会访问的状态，能否把长生成崩溃推迟或消除？

这和此前失败的 shared hard correction不同。此前改变的是额外表结构；这里改变的是训练分布。

### 分支二：Leaf-local low-rank correction

方案3可以直接使用同一批数据。

先不用完整训练，先做诊断：对每个高频困难leaf内部的teacher residual做PCA，统计rank 1、4、8解释率。

如果困难leaf内部：

```text
rank-4 / rank-8解释率高
```

说明纯常量leaf确实是表达瓶颈，可以加入局部线性项。

如果谱非常平，说明低秩也不适合，就不必浪费训练成本。

一个较经济的结构可以是：

[
\hat y=c_\ell+A_\ell V^\top(x-\mu_\ell)
]

其中 (V) 全局共享，rank先试4或8。这样改变了piecewise constant假设，但部署开销仍然远小于原FFN。

### 分支三：Logit-aware finetune

方案4也应该直接加入，而且这可能最贴合现在的生成问题。

训练时，对teacher与LUT的最终hidden经过：

```text
residual add
→ final RMSNorm
→ 选定的lm_head rows
```

重点保护：

* teacher top-k token；
* EOS；
* 换行；
* `Human:` / `Assistant:` / `user`相关token；
* 当前轨迹中即将被错误选中的top token。

损失可以是：

[
L=L_{\text{hidden}}+\lambda_1L_{\text{top-k logits}}
+\lambda_2L_{\text{margin}}+\lambda_3L_{\text{tail}}
]

这里的 `tail loss` 不要只做全局bottom 10%。可以分别对：

```text
表示困难token
轨迹临界token
termination相关token
```

赋更高权重。

这会比继续追平均FFN cosine更有意义，因为目前的异常明显涉及“该结束时不结束”和“错误进入下一轮模板”。

## 训练与验证必须按prompt拆分

100条不要全部拿来训练。

建议固定：

```text
70条：训练闭环数据
15条：validation，调loss和gate
15条：最终held-out生成评估
```

而且必须按prompt拆，不能把同一条轨迹的不同token随机拆到train/eval。否则相邻状态高度相似，结果会严重乐观。

这100条以后可以反复用于：

* base on-policy重训；
* low-rank correction；
* logit-aware训练；
* group sensitivity分析；
* 不同loss比较。

所以一次采集的成本可以被多条实验线复用。

## 我建议的实际推进顺序

第一轮不要同时把所有东西堆进去，否则无法知道谁有效。

### Experiment 1：On-policy data only

结构不改，只加入100条闭环数据重训。

回答“分布闭环本身有没有效果”。

### Experiment 2：On-policy + logit-aware

结构仍不改，加入top-k/EOS/角色标签敏感loss。

回答“生成问题是否主要来自优化目标错位”。

### Experiment 3：On-policy + local low-rank

使用相同数据，在困难leaf上增加rank 4/8局部修正。

回答“piecewise constant是不是主要表达瓶颈”。

### Experiment 4：三者组合

只有前面有两个分支各自表现出明确收益，才做组合。

评估不能只看cosine，要固定报告：

```text
hidden cosine mean / p5 / p10 / p50
relative L2
PPL
首段答案正确率
正确停止率
角色标签泄漏率
重复循环率
首次退化token位置
2048-token正常完成率
```

我的结论就是：

> **闭环不但要做，而且这100条值得跑。**
> 但先用极小sanity set把采集链路修正确，再一次性收集100条可信的长轨迹。之后不要急着扩到1000条，而是把这100条做成固定的闭环研究集，用来系统验证 on-policy数据、局部低秩修正和logit-aware优化。

这100条如果能把长生成稳定性明显推上去，就足以证明方向成立；后面再考虑第二轮补充新的失败prompt，而不是一开始追求数量。
