对，方案2可以先放掉。它最多是一个机制诊断，而且不同 prompt 的可恢复窗口很可能差异很大，最后也未必能沉淀成一个稳定方案。现在更值得投入的是：

1. 把方案1的 prompt 选择做好；
2. 同时推进方案3的局部连续修正；
3. 推进方案4的 logit-aware / group-aware 定向修正。

## 方案1：prompt不能只追求“语义看起来很多样”

你真正想覆盖的不是自然语言主题本身，而是：

> 当前 LUT 在长生成过程中可能访问到的 Layer 39 FFN 输入状态。

因此仅按“历史、数学、代码、医学、中文、英文”分几个类别是不够的。语义不同的 prompt 可能产生相似 activation；看起来相似的 prompt，也可能因为格式、推理深度和输出长度产生完全不同的轨迹。

已有量化研究也发现，calibration 数据的效果不仅取决于文本领域，更取决于 activation-space 的代表性与多样性；只覆盖目标任务而缺少一般数据，也可能牺牲鲁棒性。([arXiv][1])

所以我建议用一个**两阶段选择策略**。

### 第一阶段：先构造一个较大的候选 prompt 池

不直接精心挑几十条，而是先准备大约 500–1000 条便宜的候选 prompt。这里不需要全部跑 2048 token，可以先让当前 LUT 每条只生成 128–256 token，用来提取轨迹特征。

候选池按这些轴组合：

| 维度   | 建议覆盖                           |
| ---- | ------------------------------ |
| 语言   | 中文、英文、日文，少量混合语言                |
| 任务   | 长文分析、解释、代码、数学推导、摘要、翻译、对话、规划、创作 |
| 输出结构 | 连续散文、Markdown标题、列表、代码块、公式、多轮格式 |
| 推理要求 | 直接回答、比较分析、多阶段推理、自我检查           |
| 输入长度 | 短、中、长 prompt                   |
| 目标长度 | 512、1024、2048+                 |
| 生成约束 | 指定字数、指定章节、指定格式、开放式继续写          |
| 知识领域 | 历史、技术、自然科学、社会科学、日常知识           |

多语言不能省。最新针对量化 calibration 的研究发现，英语单一数据与多语言混合数据可能产生明显不同的量化结果，多语言组合往往有更好的跨语言稳定性。([arXiv][2])

但这只是候选池，不是最终选择标准。

---

### 第二阶段：在 activation trajectory 空间选几十条

对每个候选 prompt，先用当前 LUT rollout 128–256 token，提取一组轻量特征。

我建议每条 prompt 不要只取一个平均向量，而是取四类摘要：

```text
Layer 39 FFN input：
mean hidden
hidden covariance / PCA projection
token-to-token displacement
访问到的 tree leaf histogram
```

再加入当前模型的行为特征：

```text
base cosine mean / p10
correction residual norm
logit entropy
top-1 margin
重复率
异常字符率
```

尤其是 **leaf histogram** 很适合你这个问题，因为你的目的本来就是让数据覆盖 LUT 的地址空间。两个 prompt 即使语义不同，如果最终访问的 coarse/residual leaves高度重叠，它们提供的新信息也很有限。

然后对 prompt 建一个 embedding：

[
e_p =
[
\text{semantic embedding},
\text{activation summary},
\text{leaf histogram},
\text{trajectory statistics}
]
]

最终选择不能只做 k-means 后随便取中心，最好采用 **facility location** 或类似的子模覆盖目标：

[
F(S)=\sum_{p\in P}\max_{s\in S}\operatorname{sim}(p,s)
]

它的作用是让选出的几十条 prompt 尽可能代表整个候选池，同时避免重复。Facility location、log-determinant 和 graph-cut 都已被用于 instruction/data subset selection，其中 facility location偏向代表性覆盖，log-det偏向多样性。([arXiv][3])

对你这里，我更建议：

> 先按任务大类设最低配额，再在每类内部用 activation-space facility location 选样本。

而不是直接用纯 DPP。DPP很擅长避免重复，但纯粹追求相互距离，容易选出很多极端离群 prompt，而不一定覆盖高密度常见区域。相关工作确实用 DPP衡量和选择 instruction diversity，但你的目标更接近“覆盖实际访问状态”，所以 facility location更合适。([arXiv][4])

---

## 建议的最终几十条组成

第一轮不用上万条，先选 **64条长 prompt** 就够做方向判断。

可以这样分：

* 32条：activation-space代表性最强；
* 16条：当前 LUT 最容易出现低 cosine或异常的困难轨迹；
* 8条：极端长格式，例如多章节、代码、公式和列表；
* 8条：多语言及混合语言补充。

这里要同时保留“代表性”和“困难性”。只选困难 prompt，会让数据过度集中在异常区域；只选代表性 prompt，又可能看不到真正会崩的轨迹。相关数据选择研究通常也强调 utility 与 diversity需要共同考虑，而不是只优化一个维度。([arXiv][5])

每条生成2048 token，大约是13万 token；如果做4096 token，大约26万 token。对 teacher采集来说不是特别夸张。

## rollout数据怎么采

每条 prompt 最好同时保留两套轨迹：

### A. 原始模型轨迹

用于保证正常状态空间仍有覆盖：

[
x_t^{T}\rightarrow F_T(x_t^{T})
]

### B. 当前 LUT 自由运行轨迹

这是最重要的：

[
x_t^{LUT}\rightarrow F_T(x_t^{LUT})
]

也就是让原始 FFN对 LUT实际访问到的状态重新打标签。On-policy distillation正是通过在 student自己生成的轨迹上获取 teacher监督，解决训练状态和推理状态不一致。([arXiv][6])

不过不能简单保留全部token。长轨迹中后段监督可能更不稳定，而且失败通常集中在更晚位置，因此应分桶保存并提高后段采样率。([arXiv][7])

例如每条轨迹：

```text
0–256：随机保留10%
256–512：保留20%
512–1024：保留40%
1024+：保留60%
崩溃前128 token：全部保留
崩溃后：只保留很少一部分，单独标记
```

这里不应该把大量已经只剩换行的崩溃后状态当主要训练数据，因为它们可能已经不可恢复，反而会污染映射。

## prompt选择还应该迭代一次

不要指望第一次64条就覆盖完。

比较合理的是：

```text
Round 1：
500–1000候选prompt
→ 短rollout
→ 选64条长rollout
→ 训练新方案

Round 2：
新LUT重新跑候选池
→ 找 activation覆盖缺口和新失败区域
→ 再补16–32条
```

因为模型改变后，它访问的状态分布也会改变。动态更新 coreset通常比固定一次选择更能跟上模型当前状态。([arXiv][8])

这样方案1就不是“凭感觉找几十条长prompt”，而是一个完整闭环：

> 大候选池 → 短rollout提取activation → 覆盖选择 → 长rollout teacher标注 → 训练 → 重新覆盖检查。

---

## 方案3：低秩局部修正值得直接并行

这个方向和方案1可以共用数据。

当前叶子输出是常量：

[
\hat y=c_\ell
]

可以改为：

[
\hat y=c_\ell + U_\ell z(x)
]

但不要每个leaf都存一个完整低秩矩阵，否则存储会爆炸。

更实际的形式是：

[
z(x)=V^\top x,\qquad
\hat y=c_\ell+A_\ell z(x)
]

其中：

* (V) 是所有leaf共享的输入投影，rank可以先取4或8；
* 每个leaf存一个小的输出系数；
* 或者 (A) 也共享，仅让leaf存缩放系数。

第一轮真正应该验证的问题是：

> 同一leaf内的residual是否能被少数连续方向解释？

直接对每个leaf residual做PCA，统计：

```text
rank-1 explained variance
rank-4 explained variance
rank-8 explained variance
```

如果困难leaf中 rank-4/8能解释很大比例，那局部线性项非常有戏；如果残差谱很平，低秩修正也不会有效。

---

## 方案4：应该拆成两个子分支

### 4A. Logit-aware训练目标

训练时接上真实的：

```text
residual add
→ final RMSNorm
→ teacher top-k lm_head rows
```

只计算 teacher top-32或top-64 logits，不需要完整词表。

这个目标不是为了把所有hidden维度拟合得更好，而是保护：

* teacher top-1；
* top-1/top-2 margin；
* top-k集合；
* newline、EOS和高频结构token的排序。

它不会增加部署时计算，只影响建表和finetune。

### 4B. 非均匀group修正

先做group-error decomposition，再根据实际难度配置容量：

```text
持续困难group：额外表或低秩修正
偶发困难group：联合128/256维修正
容易group：保持当前结构
```

而不是机械地32组全加一张同样大小的表。

---

所以我觉得下一步的优先级可以很明确：

1. 先实现候选 prompt 短 rollout 和 activation/leaf coverage提取；
2. 从候选池中选64条长序列，建立 on-policy teacher数据；
3. 同时对现有 residual做 leaf-level PCA，判断方案3有没有低秩结构；
4. 在同一批数据上构建 top-k logit-aware loss，推进方案4。

这样方案1解决**数据分布**，方案3解决**piecewise constant表达上限**，方案4解决**优化目标错位**。三条不是互相替代，而是可以逐步叠加。

[1]: https://arxiv.org/html/2311.09755v2?utm_source=chatgpt.com "On the Impact of Calibration Data in Post-training ..."
[2]: https://arxiv.org/html/2601.18306v1?utm_source=chatgpt.com "Language Diversity for Better Quantized Multilingual LLMs"
[3]: https://arxiv.org/html/2403.08370v1?utm_source=chatgpt.com "Submodular Data Mixture Strategy for Instruction Tuning"
[4]: https://arxiv.org/pdf/2402.02318?utm_source=chatgpt.com "Diversity Measurement and Subset Selection for Instruction ..."
[5]: https://arxiv.org/html/2505.01523v1?utm_source=chatgpt.com "Subset Selection for Fine-Tuning: A Utility-Diversity ..."
[6]: https://arxiv.org/abs/2306.13649?utm_source=chatgpt.com "[2306.13649] On-Policy Distillation of Language Models"
[7]: https://arxiv.org/html/2604.00626v3?utm_source=chatgpt.com "A Survey of On-Policy Distillation for Large Language Models"
[8]: https://arxiv.org/html/2606.10706v1?utm_source=chatgpt.com "Unifying Data, Memory, and Compute Efficiency in LLM ..."
