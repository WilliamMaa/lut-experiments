时间有限的话，我建议不要同时铺十条线。**优先做最少改代码、最容易判断成败、而且最可能改善最后一层全替换的三步。**

## 优先级 1：先换成完整 MoE block 的真实输入输出数据

这是最重要的，甚至比扩表更优先。

你现在的构建脚本默认 teacher 是单个 `QwenMoEExpert`，但模型级测试 hook 覆盖的是完整 `layer.mlp` 输出。也就是说，你可能在用“单 expert 的映射”去替代“router + routed experts + shared expert 的最终聚合输出”。第一层能扛住这种错配，不代表最后一层也能扛住。

所以第一件事是直接收集最后一层：

```text
完整 mlp 输入 x
完整 mlp 最终输出 y
```

然后通过现有的 `--output_dataset_dir` 路径训练 LUT。你代码已经支持预计算输出，不需要重写主要结构。

**为什么排第一：**

* 代码改动最小；
* 直接消除最大的 target mismatch；
* 结果无论好坏都很有解释力；
* 如果这一步明显改善，马上就能形成结果；
* 如果不改善，再扩表才有意义。

我会先用你现在已经表现不错的 tree coarse+residual，不要同时换地址算法，避免变量混在一起。

---

## 优先级 2：扩成三级 residual LUT，而不是单纯增加一张超大表

你现有代码已经支持：

[
\hat y=T_{\text{coarse}}+T_{\text{residual}}
]

最自然的扩展是：

[
\hat y
======

T_0[A_0(x)]
+
T_1[A_1(x)]
+
T_2[A_2(x)]
]

第三张表专门拟合前两张剩下的误差。

Residual Quantization 和 additive multi-codebook 的核心优势，就是用多个码本逐级或相加地增加表示能力，而不是要求一个单独码本承担全部输出变化。QINCo 的结果也说明，让后续 residual codebook根据前面已经完成的近似进一步处理剩余误差，会比彼此独立的固定码本更有效；AQLM则表明，多码本相加并联合校准在大模型压缩中可以提供很强的表示能力。([arXiv][1])

建议先跑一个明确的大版本：

```text
coarse: 14 bits
residual-1: 16 bits
residual-2: 18 bits
group size: 64
target: direct 或 residual_mean
```

如果 FP16、32组，理论表大小约为：

* 14-bit：64 MiB；
* 16-bit：256 MiB；
* 18-bit：1024 MiB。

18 bit 全部铺给32组会到 1 GiB，所以时间和资源有限时，可以先用：

```text
14 + 16 + 16
```

总计约：

[
64+256+256=576\text{ MiB}
]

这个规模符合你说的“几百 MiB可以接受”，而且比把单棵 tree 直接加到18或20 bit更容易训练和分析。

**这一轮最可能快速出一个“容量继续增加，最后一层明显恢复”的曲线。**

---

## 优先级 3：直接保护 residual 后 hidden state，而不是只优化 FFN output

你现在的 finetune 已经能够把所有 group 拼回完整输出，并优化：

[
L_{\text{MSE}}+\alpha(1-\cos)
]

这比之前逐group训练好很多。

下一步不必立刻接完整 vocabulary 的 logit KL，那会增加数据采集、显存和代码复杂度。可以先优化最后一层真正传给 final norm 的状态：

[
h_{\text{teacher}}=x+y
]

[
h_{\text{LUT}}=x+\hat y
]

训练：

[
L=
L_{\text{FFN}}
+
\lambda_h
\left[
\operatorname{MSE}(x+\hat y,x+y)
+
\alpha_h\left(1-\cos(x+\hat y,x+y)\right)
\right]
]

虽然从纯代数上看，残差后的 MSE 和 FFN output MSE相同，但 cosine、norm及最终几何关系并不相同。最后一层后面紧接归一化和输出头，保持 residual stream 的方向和尺度更贴近最终目标。

同时加一个完整输出 norm loss：

[
L_{\text{norm}}
===============

\left(
\log
\frac{|\hat y|+\epsilon}
{|y|+\epsilon}
\right)^2
]

特别是你之前已经看到 `norm_ratio=3.1`，这个必须先解决。否则无论地址多精确，幅度失控都会直接污染最终 logits。

---

# 暂时不要优先做的

## 2D 地址

我会先放弃。它虽然稳定，但只使用两个输入通道作为地址信息。最后一层的语义结构复杂，扩大到256×256个表项也不等于获得了足够的输入判别能力。

它适合当一个很便宜的 baseline，不像最有可能救回最后一层的主线。

## High-order 8表

值得跑，但优先级低于“完整 MoE数据”和“三级 residual”。

因为你当前的 high-order 地址是随机通道与随机符号，校准的只是投影均值和方差，不会根据目标输出挑选地址方向。

多表可以增加组合表达能力，但如果每张表的地址本身与最后一层目标关系弱，增加表数未必比目标感知的 tree residual更有效。

## 输出旋转 / PCA basis

这是一个很有潜力的第二阶段方向。QuaRot和SpinQuant都说明，适当旋转隐藏空间可以消除 outlier、改善后续低精度表示；而且SpinQuant发现不同旋转之间的效果差异很大，学习到的旋转明显优于随便选择的旋转。([arXiv][2])

但你们这里需要新增：

* basis学习；
* 数据保存格式；
* LUT target变换；
* 推理时逆变换；
* 额外计算核。

时间有限时，这一步工程量太大，不适合先做。

## 完整 logit KL distillation

理论上最后一层非常适合，但它需要把 final norm、lm_head和可能很大的词表投影接进 LUT finetune。这个方向很可能有效，但不应该是第一轮。

建议先看完整 MoE target + 扩表 + norm控制能不能把结果拉回来。仍然明显失败，再上 sub-logit 或 full-logit loss。

## Group sensitivity和非均匀分配

这很有研究价值，但需要先做32组逐组或多档替换 sweep。如果时间紧，它会消耗大量模型级评估时间。

除非全替换失败得特别严重，否则先平均扩容更快。

---

# 我建议只跑四个实验

保持同一批完整 MoE block 数据和同一套模型级 prompt。

### E0：最后一层当前版本

现有配置，建立真实 baseline。

### E1：只修正 teacher 数据

地址、表大小和loss都不变，只把训练目标换成完整最后一层 MLP 输出。

这是最高信息量实验。

### E2：完整 MoE数据 + 大表

```text
tree
coarse 14
residual 16
direct
mse+cosine
50 epochs
```

约320 MiB。

### E3：完整 MoE数据 + 三级表 + norm loss

```text
tree
14 + 16 + 16
direct
mse+cosine+norm
50 epochs
```

约576 MiB。

只有E3仍然明显崩，才做：

### E4：加 residual-hidden / sub-logit distillation

不要现在就铺开。

---

# 最有可能形成成果的顺序

[
\boxed{
\text{完整MoE target}
\rightarrow
\text{扩大coarse+residual容量}
\rightarrow
\text{三级residual}
\rightarrow
\text{完整输出norm约束}
\rightarrow
\text{logit-aware训练}
}
]

我最看好的不是2D，也不是随机多表，而是：

> **先把训练对象修对，再用三级目标感知 residual tree 吃掉几百 MiB容量。**

原因很简单：你们已经证明80 MiB能顶住第一层。现在最后一层最大的未知不是“表还不够花哨”，而是**teacher错配和容量是否足够**。这两件事最快、最直接，也最容易跑出一条漂亮的结果曲线。

[1]: https://arxiv.org/html/2401.14732v2?utm_source=chatgpt.com "Residual Quantization with Implicit Neural Codebooks"
[2]: https://arxiv.org/abs/2404.00456?utm_source=chatgpt.com "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
