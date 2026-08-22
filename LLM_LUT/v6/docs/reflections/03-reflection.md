# 最后一层 FFN 累计误差的后续思路

## 1. 背景与当前判断

目前对最后一层 FFN 做完整 LUT 替换时，几个方案的 full-output cosine 大致能稳定在 0.8 左右。端到端生成通常在前几百个 token 内保持正常，但无法稳定保证超过 500 个 token，部分样本会在中途突然崩溃，进入连续换行、重复 token 或乱码状态。

这一现象说明，最后一层 FFN 对误差极其敏感。它后面只剩 residual、final RMSNorm 和 lm_head，已经没有后续 Transformer 层帮助恢复表示。因此，后续不应只继续优化普通 MSE 或平均 cosine，而应从“如何让不可避免的误差尽量不影响最终 token 选择”以及“如何专门修正长生成尾部误差”两个方向推进。

---

## 2. 思路一：让误差落在模型仍能容忍的方向上

### 2.1 核心想法

最后一层 FFN 的误差无法完全消除时，不再要求所有输出维度都同等精确，而是重点避免误差经过 residual、final RMSNorm 和 lm_head 后改变关键 logits 和 token 排名。

目标从：

> 尽量精确地拟合最后一层 FFN 的完整输出

调整为：

> 尽量保持 final hidden、top-k logits 以及关键 token 排名不变。

### 2.2 可以关注的指标

后续实验中逐 token 记录：

* FFN full-output cosine；
* residual add 后的 hidden cosine；
* final RMSNorm 后的 hidden cosine；
* logit KL；
* top-1 一致率；
* top-k overlap；
* top-1 与 top-2 的 margin；
* 实际开始退化或崩溃的位置。

这样可以判断，真正导致轨迹分叉的是 FFN 输出整体误差，还是少量关键 logit 的排序变化。

### 2.3 后续优化方向

在现有 LUT loss 上增加 logit-aware 约束，例如：

* 保持原模型 top-k logits；
* 保护 top-1 与 top-2 的相对顺序；
* 重点修复经过 lm_head 后最敏感的 hidden 方向；
* 必要时增加一个很小的低秩 correction branch，专门修正关键方向。

这条路线不一定要求 full-output cosine 显著提高，但希望让相同程度的误差对 token 选择更无害，从而延长稳定生成长度。

---

## 3. 思路二：建立闭环长序列数据，并学习阶段性矫正表

### 3.1 核心想法

准备大约一万个有代表性的 prompt，覆盖不同语言、领域和任务类型，并要求模型生成足够长的输出，例如 2K–4K token。

生成过程中采集最后一层 FFN 的输入、原始输出以及相关 logits 信息，尤其保留长序列后半段和崩溃前的状态。

除了采集原模型轨迹，还要让当前 LUT 模型进行 free-running generation。对于 LUT 实际走到的 hidden state，再调用原始 FFN得到 teacher 输出，从而形成：

[
x_{\text{LUT trajectory}}
\rightarrow
F_{\text{original}}(x_{\text{LUT trajectory}})
]

这样能够学习 LUT 自己在长生成中真实遇到的尾部误差，而不是只学习原模型正常轨迹。

### 3.2 阶段性矫正表

保留一套通用主表，再额外学习若干小型 correction table。例如：

* 0–500 token：早期表；
* 500–1000 token：中期矫正表；
* 1000 token 以上：尾部矫正表。

最终输出为：

[
\hat y_t
========

\hat y_{\text{base}}(x_t)
+
\Delta \hat y_{b(t)}(x_t)
]

矫正表不重新学习完整 FFN，只学习主表留下的残差：

[
r_t
===

## F_{\text{original}}(x_t)

\hat y_{\text{base}}(x_t)
]

第一版也可以只做两张尾部表，保持前 500 token 完全使用当前主表，避免破坏已经较稳定的前段生成。

### 3.3 重要前提

尾部矫正成立的前提是前段必须足够准确。

如果模型从早期开始持续漂移，那么到 1000 token 时，hidden state 已经偏离正常分布，尾部表很难把语义重新拉回来。

目前部分样本表现为：

> 前面长期正常，在某个位置突然进入连续换行或重复 token。

这种突然崩溃比持续渐进漂移更可能被修复，因为它可能说明模型直到临界点前仍处于可恢复区域。

因此，在训练矫正表前，应先做回放实验：

* 在崩溃前 1、5、10、25、50 个 token 切回原始 FFN；
* 观察生成是否能够恢复正常；
* 确定是否存在明确的 recoverable window。

如果提前较少 token 切回即可恢复，说明尾部 correction table 有较大价值；如果提前几十个 token 仍无法恢复，则说明表面上虽然突然崩溃，内部轨迹可能早已发生偏移。

---

## 4. 下一步安排

先完成当前两组 LUT 实验。

如果结果仍然维持在 full-output cosine 约 0.8，且长生成表现没有明显突破，则不再继续只围绕普通 MSE、分裂方式或平均 cosine 做细节优化，而是进入两个方向：

1. 分析最后一层 FFN 误差经过 final RMSNorm 和 lm_head 后，哪些方向最容易改变 token 排名，并设计 logit-aware 或关键方向修正。
2. 构建大规模长生成闭环数据集，验证尾部误差是否具有稳定结构，并训练 500–1000 与 1000+ token 的阶段性 correction table。

最终判断标准不只是单步 cosine，而是：

* 512、1024 和 2048 token 稳定率；
* 崩溃位置的 p10 与 p50；
* top-1 一致率；
* logit KL；
* 是否能够明显延后突然崩溃点。
