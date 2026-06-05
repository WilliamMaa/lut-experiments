问题 A：Final Score 现在可能不可靠

因为你表里很多 group 的 Addressability 是 0，但 Final Score 有正有负。这说明 Final Score 可能混合了某些 normalization 或 penalty。现在不要用 Final Score 决策。

现阶段最稳的指标应该是：

Absolute improvement = KL Mean - KL Bucket
Relative recovery = (KL Zero - KL Bucket) / KL Zero
Bucket advantage = KL Mean - KL Bucket

我建议你下一版 ranking 直接输出这三列：

Group	KL Zero	KL Mean	KL Bucket	Bucket Advantage	Recovery

比如你补的 group 0：

KL Zero   = 0.8941
KL Bucket = 0.7311
Recovery ≈ 18.2%

group 9：

KL Zero   = 0.4935
KL Bucket = 0.3640
Recovery ≈ 26.2%

group 11：

KL Zero   = 0.6059
KL Bucket = 0.4601
Recovery ≈ 24.1%

这些都算有信号，但和 group 4 那种 70% 左右的 recovery 不是一个等级。

问题 B：coverage 只有 34.38%

这个不是坏事，但一定要解释。

这意味着当前 bucket table 并不是所有 bins 都被有效使用。可能原因有三个：

1. address activation 分布很集中；
2. uniform binning 不适合 LLM activation；
3. calibration data 太少；
4. bin 数过多或 range 设置太宽。

下一轮必须加：

quantile binning
bucket occupancy entropy
empty-bin ratio
top-k bin mass

如果 top few bins 占了绝大多数样本，那么 34.38% coverage 其实可能还偏乐观。反过来，如果 occupancy 虽不满但分布还算均匀，那就问题不大。


第一组：复现性确认

只测：

layer = 6
type = mlp_delta
groups = [4, 3, 8, 1, 13, 9, 0]

数据量扩大一档，比如 eval samples 翻 2–4 倍。

指标：

KL Zero
KL Mean
KL Bucket
PPL
Next-token accuracy
Recovery
Coverage
Entropy
第二组：多 group 组合测试

这个非常关键，因为单 group 好，不代表多 group 能叠加。

测试：

group 4
group 4 + 3
group 4 + 3 + 8
group 4 + 3 + 8 + 1
group 4 + 3 + 8 + 1 + 13

看 bucket replacement 的误差是否线性累积，还是会出现 nonlinear collapse。

这一步对应你 YOLO 里的：

6 only
8 only
6+8

在 LLM 里就是：

single group
multi group
accumulation behavior
第三组：address / binning ablation

对 group 4、3、8 跑：

uniform bins vs quantile bins
num_bins = 32 / 64 / 128 / 256
single-head vs two-head LUT

two-head 版本仍然符合“只读 existing scalar + lookup + add”：

delta_g = LUT_1[bin(a)] + LUT_2[bin(b)]

没有额外矩阵乘法。


现在最有希望的路线是：

Layer 6 MLP residual delta partial LUT approximation

下一步不要急着说省计算，先证明：

1. 这个现象可复现；
2. 多 group 组合不会崩；
3. 更好的 address/binning 能进一步降低 KL；
4. trainable LUT 能优于 non-trained bucket average。