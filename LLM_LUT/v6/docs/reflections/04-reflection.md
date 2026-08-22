我查了一圈，结合你现在这个分布：

```text
mean  = 0.807
p50   = 0.861
p90   = 0.931
p10   = 0.590
norm  = 1.067
```

我觉得问题已经不是“整个 LUT 表达能力都差”，而是**大量普通样本已经够好，少量状态近似得极差**。你现在这套 shared coarse + grouped residual 仍然使用均匀样本、均匀 MSE/cosine，以及相互独立的固定 residual tree，天然容易把容量花在多数普通状态上。

查到的相关工作里，真正能迁移到你这里的不是某一篇现成方案，而是下面四类思路。

# 一、最值得先试：Tail-aware 第三张残差表

这是我现在最推荐的。

你已有：

[
\hat y_{\mathrm{base}}(x)
=========================

y_{\mathrm{coarse}}(x)
+
y_{\mathrm{residual}}(x)
]

再加一张小表：

[
\hat y(x)
=========

\hat y_{\mathrm{base}}(x)
+
\Delta y_{\mathrm{hard}}(x)
]

第三张表只拟合：

[
r_{\mathrm{hard}}(x)
====================

y(x)-\hat y_{\mathrm{base}}(x)
]

但不再均匀训练，而是按 base cosine 分配权重：

[
w_i=
\operatorname{sigmoid}
\left(
\frac{\tau-\cos_i}{T}
\right)
]

例如：

```python
weight = torch.sigmoid((0.80 - base_cos.detach()) / 0.05)
```

它的含义是：

```text
cos > 0.90   几乎不修
cos 约 0.80  中等修正
cos < 0.70   重点修正
p10 区域     获得主要容量
```

这与 Focal Loss 的核心思想一致：降低大量容易样本的影响，把训练能力集中到少量困难样本；Focal Loss本身是分类损失，但“防止多数容易样本淹没困难样本”的原则可以直接迁移到你的回归问题。([arXiv][1])

多级 additive/residual quantization 也有明确依据：用多个 codebook相加通常能比单一或简单分块 codebook获得更小的逼近误差；AQLM便采用可学习的 additive codebooks，并联合优化不同 codebook，而 residual quantization则逐级拟合前一步留下的误差。([arXiv][2])

### 这里不要再做32张普通 correction table

第一版我建议：

> **一棵12-bit或13-bit共享 hard tree + 一张完整2048维 correction table。**

原因是你现在要抢救的是 full-output direction，而不是再次把2048维拆成32个互不协调的64维问题。

额外存储：

* 12-bit：约16 MiB；
* 13-bit：约32 MiB。

相对于现在约320 MiB的表不算大。

而且它始终启用，不需要运行时风险检测：

```text
普通状态 → correction接近0
危险状态 → correction显著
```

这不会引入一个新的动态检测系统。

---

# 二、比普通 residual 更有意思：条件式 residual table

你现在第二阶段虽然叫 residual tree，但它实际上是：

```text
coarse根据x寻址
residual仍然单独根据x重新寻址
```

两者之间没有条件关系。

这可能是一个重要问题。因为同一个输入区域的残差分布，往往依赖 coarse到底选择了哪个原型。Residual Quantization with Implicit Neural Codebooks专门指出：后续残差的分布取决于前一步已选择的 codeword，而传统 residual quantization用固定、独立的后续 codebook，没有利用这种依赖；其解决方式是让后续 codebook受到前一步近似结果的条件约束。([arXiv][3])

迁移到你的结构，不一定要上神经网络，可以做得很 LUT：

[
i_c=h_c(x)
]

[
i_r=h_r(x,\operatorname{embed}(i_c))
]

简单实现甚至不需要真正输入 coarse 2048维结果，可以让 residual tree的早期几位直接来自 coarse leaf：

```text
residual address =
[coarse leaf 的若干高位]
+
[residual tree 自己计算的低位]
```

例如16-bit residual address：

```text
前4 bits：coarse region
后12 bits：当前 residual tree
```

它表达的是：

> 相同的局部输入特征，在不同 coarse region里，应该查不同的 residual correction。

这比再加一棵完全独立的树更有针对性，而且运行时只是地址拼接，不需要额外矩阵计算。

我认为这个方向甚至可能比单纯增加 tree candidates更有效，因为它改变了 residual 表的结构，而不是增加随机搜索次数。

---

# 三、建树也必须向困难样本倾斜

现在的树仍然通过普通目标方差下降选择切分：

[
\Delta V
========

## V(parent)

\frac{n_LV(L)+n_RV(R)}{n}
]

因此10%的困难样本很容易被90%的普通样本覆盖。

第三张 hard table应该改成加权方差：

[
\mu_w
=====

\frac{\sum_i w_i y_i}{\sum_i w_i}
]

[
V_w
===

\frac{\sum_iw_i|y_i-\mu_w|^2}{\sum_iw_i}
]

然后切分收益为：

[
\Delta V_w
==========

## V_w(parent)

\frac{W_LV_w(L)+W_RV_w(R)}{W_L+W_R}
]

这样候选切分真正会寻找：

> 哪些输入特征能够把低-cosine残差分开？

而不是再次优先照顾样本数量最多的普通区域。

此外，LUT-Q并不把 assignment固定到底，而是交替更新 dictionary和 assignment；这说明“只初始化一次地址，然后永远冻结，只训练表值”并不是 LUT训练的唯一方式。([arXiv][4])

你这里不必立刻做可微分树，但可以做两轮重建：

```text
Round 1：构建 base
Round 2：计算逐样本 hard weight和 residual
Round 3：基于 weighted residual 重建 correction tree
Round 4：优化 correction table
Round 5：重新计算困难样本，再重建一次 correction tree
```

只做一到两轮就够，不需要无限迭代。

---

# 四、不要只优化 FFN cosine：做轻量 logit-sensitive correction

你的最后一层 FFN后面基本就是 residual、final RMSNorm和lm_head。所以两个 FFN输出即使 cosine相同，对最终 logits 的破坏也可能完全不同。

量化领域已有一个很稳定的观察：并非所有参数或通道同等重要，保护少量 activation-salient部分，就可能大幅降低最终量化误差；AWQ报告仅保护约1%的显著权重便能明显减少量化误差，其显著性来自 activation分布。([arXiv][5])

你可以把这个原则迁移成：

> 不同的2048维输出方向，对lm_head的影响不同，不应同权处理。

最完整的目标是：

[
L_{\mathrm{logit}}
==================

D\left(
\operatorname{LMHead}(
\operatorname{RMSNorm}(x+y)
),
\operatorname{LMHead}(
\operatorname{RMSNorm}(x+\hat y)
)
\right)
]

但是整个词表太大。第一版只计算 teacher top-k，例如 `k=32` 或 `64`：

```python
with torch.no_grad():
    true_hidden = final_norm(x + y)
    true_logits = lm_head(true_hidden)
    topk_idx = true_logits.topk(k=64, dim=-1).indices

pred_hidden = final_norm(x + pred_y)
true_topk = true_logits.gather(-1, topk_idx)
pred_topk = lm_head_selected(pred_hidden, topk_idx)

loss_logit = F.mse_loss(pred_topk, true_topk)
```

只在 correction table训练阶段计算，不增加部署时成本。

目标可以变成：

[
L
=

L_{\mathrm{output}}
+
\lambda_1L_{\mathrm{tail}}
+
\lambda_2L_{\mathrm{topk\text{-}logit}}
]

Token-scaled logit distillation的研究也指出，decoder语言模型中量化误差并非均匀分布，不同 token和生成位置应给予不同的蒸馏强度。([arXiv][6])

这个路线的价值在于：你不一定需要把 p10 FFN cosine从0.59全部拉到0.90，只要让剩余误差少改变关键 token排序，就可能显著改善生成。

---

# 五、长序列 teacher数据：值得补，但要正确使用

几十条长序列是值得做的，不过它们不应该只被混进原来的400k token里。否则几万条尾部 token仍可能被大量普通样本淹没。

自回归蒸馏研究显示，不同 token的教学难度和合适的蒸馏方式并不相同，统一处理会造成性能下降，因此采用 token-adaptive teaching更有效。([arXiv][7])

建议几十条长 teacher序列这样用：

```text
原始模型轨迹：
0–500
500–1000
1000–2000
2000+

当前 LUT free-running轨迹：
正常段
崩溃前128 token
崩溃前64 token
崩溃前32 token
```

最关键的是第二类：

[
x_{\mathrm{LUT}}
\longrightarrow
F_{\mathrm{teacher}}(x_{\mathrm{LUT}})
]

因为模型部署时遇到的是 LUT自己走出来的状态，不只是 teacher的正常状态。

这些数据最好用于 hard correction table，而不是重训整个 base。否则很可能把当前已经表现不错的 p50和p90破坏掉。

---

# 我建议下一轮只跑三个方案

## Experiment A：Tail-aware joint loss

不增加表，只改 joint finetune：

[
L=L_{\mathrm{mean}}+\lambda L_{\mathrm{worst20%}}
]

它成本最低，用来判断固定容量下能不能把 p10拉起来。

但我预计提升有限，因为地址没变，容量也没增加。

## Experiment B：共享 hard correction table

当前 base冻结：

```text
shared coarse
+ grouped residual
+ 12-bit shared 2048d hard correction
```

用低-cosine加权 residual建树和训练。

这是我认为**最可能有效且实现风险最低**的方案。

## Experiment C：conditional residual address

把 residual address改成：

```text
coarse leaf prefix + residual local suffix
```

它测试的是当前 p10问题是否来自 residual与coarse相互独立。

---

# 暂时不建议继续做的事情

我暂时不会优先做：

* 把 candidates从256继续加到512或1024；
* 单纯增加 coarse/residual bits；
* 再堆更多普通 independent residual表；
* 继续提高普通 mean cosine loss权重；
* norm clipping；
* 复杂在线风险检测；
* 为最后一层增加一个较大的神经 correction网络。

因为这些要么只是扩大现有结构，要么会破坏你 LUT方案原本的简洁性。

最明确的路线是：

> **保留当前能把p50做到0.86、p90做到0.93的 base，不再推倒；增加一小块专门服务于p10的容量，并让它在建树、初始化和loss三个阶段都真正看到“哪些样本危险”。**

下一步最值得做的单个实验就是 **12-bit shared hard correction + weighted residual tree**。它如果能让 mean只涨一点，但把 p10从0.59推到0.72–0.78，同时不损伤p50，这条路就成立。

[1]: https://arxiv.org/abs/1708.02002?utm_source=chatgpt.com "Focal Loss for Dense Object Detection"
[2]: https://arxiv.org/pdf/2401.06118?utm_source=chatgpt.com "Extreme Compression of Large Language Models via Additive Quantization"
[3]: https://arxiv.org/abs/2401.14732?utm_source=chatgpt.com "Residual Quantization with Implicit Neural Codebooks"
[4]: https://arxiv.org/abs/1911.04951?utm_source=chatgpt.com "Iteratively Training Look-Up Tables for Network Quantization"
[5]: https://arxiv.org/abs/2306.00978?utm_source=chatgpt.com "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
[6]: https://arxiv.org/pdf/2308.06744?utm_source=chatgpt.com "Token-Scaled Logit Distillation for Ternary Weight ..."
[7]: https://arxiv.org/abs/2402.11890?utm_source=chatgpt.com "Revisiting Knowledge Distillation for Autoregressive Language Models"
