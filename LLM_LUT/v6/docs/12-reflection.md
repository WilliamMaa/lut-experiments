我总体赞成：**先做 per-leaf 低秩修正，比直接上完整 pairwise table 更合理。**

但在实现之前，我会先加一句非常重要的判断：

> “所有 992 对平均都能解释 53.6% 残差”未必意味着存在 992 个彼此独立、都很强的 pairwise interaction；它也可能说明残差本身存在一个很强的共享低维结构，任何两个 group 都能部分预测这个结构。

尤其你们已经发现 rank-16 PCA 能解释 85%，这两件事其实很可能指向同一个结论：**残差主要集中在低维子空间，而不是必须靠巨大二维叶子表记住。**

## 先做 per-leaf low-rank，我觉得是对的

你们提出的形式本质上是一个由 coarse leaf 选择的局部低秩线性模型：

[
\hat y
======

y_{\mathrm{LUT}}
+
A_{\ell(x)}V^\top x
]

其中：

* (V\in\mathbb{R}^{2048\times r}) 提取全局重要输入方向；
* (A_\ell\in\mathbb{R}^{2048\times r}) 将这些方向映射到该 leaf 的输出修正；
* coarse leaf 决定使用哪一个局部映射。

这里按列向量习惯，(A_\ell) 应写成 `[2048, rank]`；你文中写的 `[rank, 2048]` 也可以，但 forward 时需要按相反方向乘。

这比一张完整 pair table 好在：

1. 它在 leaf 内不是常数，而是连续变化；
2. (V^\top x) 同时看到了所有 group，因此可以隐式捕获跨组交互；
3. coarse leaf 又提供了状态条件，使同一低维方向在不同区域有不同响应；
4. 在线计算是规则的小型投影和累加，部署上远比随机二维表查找自然。

所以它不是单纯绕开 pairwise，而是在用一种更紧凑的方式表达 pairwise 乃至更高阶作用。

## 不过存储量要重新精确核算

每个 leaf 的 (A_\ell) 大小是：

[
2048\times r\times2\ \text{bytes}
]

总容量是：

[
N_{\mathrm{leaf}}\times2048\times r\times2
]

因此：

* 16,384 leaves，rank-8：约512 MiB；
* 16,384 leaves，rank-16：约1 GiB；
* 65,536 leaves，rank-8：约2 GiB；
* 65,536 leaves，rank-16：约4 GiB。

所以你写的512 MB / 1 GB，前提应该是 coarse leaf 数量约16K，而不是65,536。这个要在汇报和实现里写清楚。

另外，不一定所有leaf都值得拥有独立的 (A_\ell)。可以先做：

* 高频且高残差leaf：独立 (A_\ell)；
* 普通leaf：共享cluster-level (A_c)；
* 低频leaf：只使用全局 (A_0)。

这样容量可以大幅下降。

---

# Pairwise有没有更便宜的实现

有，而且我觉得最值得尝试的不是“缩小版二维表”，而是**因子化 pairwise interaction**。

## 方案一：低秩因子化pairwise表

完整 pair table 是：

[
T_{gh}[a_g,a_h]\in\mathbb{R}^{2048}
]

不要直接存全部组合，而是分解成：

[
T_{gh}[a,b]
\approx
U_{gh}
\left(
e_g[a]\odot e_h[b]
\right)
]

其中：

* (e_g[a]\in\mathbb{R}^{r})：group (g) 的leaf embedding；
* (e_h[b]\in\mathbb{R}^{r})：group (h) 的leaf embedding；
* (\odot)：逐元素乘；
* (U_{gh}\in\mathbb{R}^{2048\times r})：将交互码映射为输出修正。

forward就是：

```text
eg = embedding_g[leaf_g]
eh = embedding_h[leaf_h]
interaction = eg * eh
correction = U_gh @ interaction
```

它的存储从：

[
L_gL_h\times2048
]

降成：

[
(L_g+L_h)r+2048r
]

假设两个group都是65,536 leaves、rank-8、FP16，一对大约只需要：

[
(65536+65536)\times8\times2
+
2048\times8\times2
\approx2.03\ \text{MiB}
]

Top-20 pairs大约40多MiB，而不是10 TB，更不是每对512 GB。

这个形式本质上是对三维张量：

[
[\text{leaf}_g,\text{leaf}_h,\text{output}]
]

做CP式低秩分解。

## 还能进一步共享embedding

不需要每一个pair都重新保存一套 (e_g)。

可以让每个group只保存一张leaf embedding表：

[
E_g\in\mathbb{R}^{L_g\times r}
]

所有涉及group (g) 的pair共享它。然后每一对只增加：

[
U_{gh}\in\mathbb{R}^{2048\times r}
]

或者再加一个很小的pair变换：

[
M_{gh}\in\mathbb{R}^{r\times r}
]

形式为：

[
\Delta y_{gh}
=============

U_{gh}
\left(
e_g[a_g]\odot M_{gh}e_h[a_h]
\right)
]

假设32个group、每组65,536 leaves、rank-8，所有group embedding加起来约：

[
32\times65536\times8\times2
\approx32\ \text{MiB}
]

Top-20 pairs各自的 (U_{gh}) 总共还不到1 MiB。这个容量非常可控。

代价是它不再能够任意表示每一对leaf的完整2048维输出，而是假设pairwise interaction也具有低秩结构。但你们当前PCA结果恰好说明这个假设很值得试。

---

## 方案二：输出先投影到低维残差空间

你们已经有全局residual PCA，可以设输出basis：

[
U\in\mathbb{R}^{2048\times r_o}
]

pairwise结构不再直接预测2048维输出，只预测低维系数：

[
\Delta y_{gh}
=============

U,c_{gh}(a_g,a_h)
]

然后 (c_{gh}) 可以通过：

* 小型factorized embedding；
* hashed pair table；
* 小MLP；
* bilinear product；

生成。

这会让所有修正共享同一个输出子空间。既然rank-16已经解释85%残差，这会非常自然：

[
\Delta y
========

U
\left[
c_{\mathrm{leaf}}+
\sum_{(g,h)\in\mathcal P}c_{gh}
\right]
]

这样你甚至不需要每个pair单独保存 `[2048, rank]` 的输出矩阵。

---

## 方案三：哈希pair table

最简单粗暴的便宜版本是：

[
b=\operatorname{hash}(a_g,a_h)\bmod M
]

然后查：

[
H_{gh}[b]
]

但不要让bucket直接存2048维输出，仍然只存rank-8或rank-16系数，再乘共享输出basis。

例如：

[
H_{gh}\in\mathbb{R}^{16384\times16}
]

FP16每对只有：

[
16384\times16\times2=512\ \text{KiB}
]

Top-20约10 MiB。

优点是实现极其简单；缺点是hash collision，而且两个从没见过的leaf pair可能撞在一起。它适合作为低成本验证，不太适合作为最终最优结构。

---

## 方案四：只让少数interaction动态启用

你们发现group 19、16、12、25是中心，那可以不做992对，而是只建一个小型interaction graph：

```text
19 ↔ 16
19 ↔ 12
19 ↔ 25
16 ↔ 12
……
```

再通过每个token的置信度或leaf residual统计，只执行最重要的2到4对，而不是固定执行Top-20。

这样计算量也更可控。

例如rank-8，每个pair需要大约：

[
2048\times8
]

级别的输出累加。20对就是约327K次小型乘加/token；4对只有约65K。相对完整FFN仍然不大，但不能无限叠。

---

# 我会怎么安排实验顺序

我不会马上同时实现per-leaf low-rank和复杂pairwise。建议分三步。

## 第一步：验证per-leaf low-rank是否真的泛化

先做rank：

* 4；
* 8；
* 16。

同时测：

* train explained variance；
* held-out explained variance；
* on-policy held-out cosine；
* PPL；
* logit KL；
  -端到端生成；
* 增量存储和延迟。

特别要防止“85%解释率”只是在拟合训练残差。每个leaf样本数可能有限，(A_\ell) 很容易过拟合。

最好采用：

* ridge regression；
* minimum sample threshold；
* leaf与parent/cluster之间的参数收缩；
* 低频leaf共享参数。

## 第二步：做一个极简factorized pairwise

不要上Top-20完整版本，先做Top-1或Top-4：

[
\Delta y_{gh}
=============

U(e_g[a_g]\odot e_h[a_h])
]

rank-8即可。

然后比较：

* low-rank leaf only；
* pairwise only；
* low-rank leaf + pairwise。

如果pairwise在加了per-leaf correction之后几乎没有额外增益，就说明之前那55%主要是共享低维残差，而不是独立pair interaction。

## 第三步：只保留真正互补的pair

关键不是pair单独能解释多少残差，而是：

> 在已有per-leaf low-rank修正之后，这个pair还能解释多少剩余残差？

应该测条件增益：

[
\Delta R^2_{gh}
===============

## R^2(\text{leaf low-rank}+gh)

R^2(\text{leaf low-rank})
]

如果原来每对都是53%，但加入low-rank以后Top pair只多解释2%，就完全不值得实现。

---

# 还有一个统计结果值得警惕

你说：

* Top pair：55%；
* 所有992对平均：53.6%。

Top和平均只差1.4个百分点，这其实不太像“某些pair格外重要”。

如果真正是稀疏pairwise结构，通常应该看到：

* 少数pair很高；
* 大部分pair接近0；
* 分布呈明显长尾。

现在几乎所有pair都在53%左右，更可能是：

1. 每个pair都间接编码了相似的全局状态；
2. group leaf ID和全局activation分布高度相关；
3. 测试中有数据泄漏；
4. pairwise模型容量太强，在相同样本上拟合；
5. 残差本身集中在共享低维子空间。

因此在下结论“group之间交互很强且普遍”之前，我建议加三个control：

* **held-out token evaluation**：不能在拟合pair table的同一批样本上算解释率；
* **shuffle control**：随机打乱其中一个group的leaf ID，看看解释率降到多少；
* **single-group baseline**：分别测group (g)、group (h) 单独能解释多少，再测pair的增量。

真正的交互解释率应该是：

[
R^2(g,h)-\max(R^2(g),R^2(h))
]

甚至应该和加性模型：

[
f_g(a_g)+f_h(a_h)
]

相比，而不是只看pair模型自己的总解释率。

否则“pair能解释55%”可能只是group 19自己已经能解释50%，另一个group只提供了很小增量。

## 最终判断

你的主建议是对的：

> **先做per-leaf低秩修正，再视剩余误差决定是否加入少量pairwise。**

但我会把pairwise的廉价版本明确设计成：

[
\boxed{
\Delta y_{gh}
=============

U_{gh}
\left(
E_g[a_g]\odot E_h[a_h]
\right)
}
]

而不是二维leaf笛卡尔表。

这既保留了真正的乘性交互，又把容量从 (O(L^2d)) 降到 (O(Lr+dr))。更重要的是，它和你们现有“分组、分表、可组合”的方法论完全一致，不会突然变成一个巨大的记忆库。

当前最值得先确认的，其实不是能否把pair table压小，而是：**在per-leaf rank-8/16修正之后，pairwise到底还剩多少独立增益。**如果剩余增益不大，就不用为它增加任何工程复杂度。
