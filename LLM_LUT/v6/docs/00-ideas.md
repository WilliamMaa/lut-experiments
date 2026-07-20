# 单层 FFN 专家级 LUT 扩展方案

> 目标：在不引入大规模最近邻检索、per-anchor Jacobian 或在线 JVP 的前提下，基于现有 LLM_LUT v5 框架，把 LUT 从“projection group 级替换”扩展到“单层 FFN 专家级替换”，并在可控存储预算下评估能否实现更高精度与更大 MAC 削减。

---

## 1. 核心判断

当前最值得验证的不是“三千万 anchor + 最近邻 + Jacobian 修正”，而是：

> 在单个真实 FFN 专家内，把 LUT 存储预算从原来的几十 MiB 放宽到几百 MiB乃至 1 GiB，集中用于一个专家的有限比例计算替换，是否可以在保持 O(1) 地址生成与固定查表路径的同时，实现约 10% MAC 削减，并显著提高输出精度。

这条路线的优势在于：

1. 表大小明确，可控制在 128 MiB、512 MiB、1 GiB 等工程可讨论范围内。
2. 地址生成器固定、无训练参数，推理仍然是 O(1) 查表。
3. 不需要三千万条在线最近邻搜索。
4. 不需要为每个 anchor 存 Jacobian。
5. 不需要在线运行 JVP。
6. 仍然能够直接计算节省了多少 MAC、增加了多少存储和访存。
7. 可以直接复用现有 tree address、sequential build、residual target 和 LUT-only fine-tune。

---

## 2. 研究对象

选择一个真实 Qwen MoE 单层专家：

\[
f(x)=W_d\left(\operatorname{SiLU}(W_gx)\odot W_ux\right)
\]

其中：

- 输入维度：`hidden_size`
- 中间维度：`intermediate_size`
- 输出维度：`hidden_size`
- 模块：
  - `gate_proj`
  - `up_proj`
  - `down_proj`

第一阶段只研究：

> 单层、单专家、单独输入分布下的 FFN 输出近似。

暂时不扩展到整模型，也不做跨层联合部署。先验证单专家上的容量—精度—MAC 曲线。

---

## 3. 总体实验目标

### 3.1 主目标

在单层 FFN 专家中，实现：

- LUT 存储预算：128 MiB / 512 MiB / 1 GiB
- MAC 削减目标：5% / 10% / 20%
- 地址生成：固定 tree / high-order / multi-stage address
- 推理复杂度：固定次数查表，不允许 O(N) 搜索
- 评估：
  - 输出 cosine similarity
  - relative error
  - normalized MSE
  - 实际 MAC reduction
  - LUT storage
  - 单 batch latency
  - 单 token访存量

### 3.2 关键验证问题

1. 40 MiB 版本表现一般，究竟是方法上限，还是容量不足？
2. 当存储预算扩大 10–25 倍后，误差是否显著下降？
3. 单层专家中，10% MAC 替换是否可以做到很高相似度？
4. 直接 FFN-output LUT、分层 LUT、residual LUT 哪一种最好？
5. 精度是否随着表容量快速提升，并在某个规模后饱和？
6. 相同存储预算下，树深、表数、group size、residual 结构如何影响结果？

---

## 4. 三条可执行路线

## 4.1 方案 A：直接 FFN Output Group LUT

### 4.1.1 思路

直接以 FFN 输入 \(x\) 为地址输入，以完整 FFN 输出的某一组通道为 target：

\[
y=f(x)
\]

对于输出 group \(g\)：

\[
\hat y_g=\operatorname{LUT}_g(a_g(x))
\]

其中：

- \(a_g(x)\) 是固定离线地址生成器；
- LUT 输出 64 个或 128 个通道；
- 未替换 group 仍由原 FFN 计算；
- 被替换 group 直接使用 LUT 输出。

### 4.1.2 优点

- 在线路径最简单。
- 真正跳过 gate/up/down 对被替换输出 group 的贡献。
- 容易计算实际 MAC 节省。
- 不依赖中间激活。
- 不需要额外 Jacobian 修正。

### 4.1.3 缺点

- 从输入直接映射到完整 FFN 输出，函数复杂度较高。
- 单级 LUT 可能需要更深的树或更多表。
- 大 group 可能较难拟合。

### 4.1.4 第一阶段推荐配置

- group_size：64
- address_mode：tree
- num_bits：12 / 14 / 16
- num_tables：1 / 2 / 4
- target：
  - direct output
  - output residual against group mean
- 预算：
  - 128 MiB
  - 512 MiB
  - 1 GiB

---

## 4.2 方案 B：Coarse LUT + Residual LUT

### 4.2.1 思路

把一次查表拆成两级：

\[
\hat y_g
=
\operatorname{LUT}^{\text{coarse}}_g(a_c(x))
+
\operatorname{LUT}^{\text{res}}_g(a_r(x))
\]

第一张表预测主要输出，第二张小表预测第一张表的残差。

在线流程：

```text
x
→ coarse address
→ coarse LUT
→ residual address
→ residual LUT
→ 两者相加
```

### 4.2.2 为什么值得做

你现在的 tree LUT 本质上是“每个叶子取 target 均值”。当叶子仍然覆盖较大区域时，平均值会损失局部差异。

residual LUT 可以专门学习：

\[
r(x)=y(x)-\hat y_{\text{coarse}}(x)
\]

它不需要重建完整输出，只需要补偿剩余误差。

### 4.2.3 地址选择

可以尝试：

1. coarse 和 residual 使用独立 tree；
2. residual tree 在每个 coarse bucket 内单独构建；
3. residual address 使用另一组随机投影；
4. residual table 比 coarse table 小。

### 4.2.4 推荐配置

- coarse num_bits：12 / 14
- residual num_bits：8 / 10 / 12
- group_size：64
- coarse/residual 存储比例：
  - 75% / 25%
  - 50% / 50%
- 总预算：
  - 128 MiB
  - 512 MiB
  - 1 GiB

---

## 4.3 方案 C：Multi-Table / Top-k LUT 组合

### 4.3.1 思路

不再让一个输入只能命中一个表项，而是使用多个固定地址生成器：

\[
\hat y_g
=
\sum_{m=1}^{M}w_m\operatorname{LUT}_{g,m}(a_{g,m}(x))
\]

其中：

- \(M=2\) 或 \(4\)
- 地址生成器全部离线固定
- 权重可以固定平均，也可以离线拟合为常数
- 不引入动态 MLP 或 O(N) 计算

### 4.3.2 价值

原来的“平均混合”效果不佳，不代表 multi-table 一定不行。问题可能在于：

- 多张表地址过于相似；
- 所有表都在拟合同一个 target；
- 缺乏残差分工；
- 表间没有互补性。

因此推荐改为：

```text
Table 1：预测主值
Table 2：预测 Table 1 残差
Table 3：预测特定高误差区域
Table 4：预测边界区域
```

而不是简单平均几个相似表项。

---

## 5. 推荐优先级

### 第一优先：直接 FFN Output Group LUT

原因：

- 改动最小；
- 最容易复用现有构建代码；
- 最快回答“单专家、几百 MiB 到 1 GiB 能否做到高精度”；
- 不需要引入复杂中间路径。

### 第二优先：Coarse + Residual LUT

如果直接输出 LUT 达到一定精度但仍有明显残差，这条路线最自然。

### 第三优先：Multi-Table

只有在单表与 residual 已经建立基线后，再研究多表组合是否进一步提高精度。

---

## 6. 代码改造方案

## 6.1 新增 FFN 输出捕获

新增：

```python
capture_ffn_output(model, layer_id, expert_id, dataloader)
```

输出：

```python
{
    "inputs":  [N, hidden_size],
    "targets": [N, hidden_size]
}
```

其中：

```python
target = expert(input)
```

第一阶段只保存单专家输入与输出，不保存 Jacobian。

---

## 6.2 新增 FFN Group Builder

新增脚本：

```text
build_lut_ffn_output.py
```

职责：

1. 加载真实专家。
2. capture FFN input/output。
3. 按输出 group 划分 target。
4. 为每个 group 构建 tree address。
5. 初始化 LUT entries。
6. 输出 group-level build metrics。
7. 保存 checkpoint。

建议接口：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path ... \
  --dataset_dir ... \
  --group_size 64 \
  --group_ids "..." \
  --address_mode tree \
  --num_bits 14 \
  --tree_candidates 64 \
  --tree_min_samples 16 \
  --tree_max_samples 65536 \
  --output_root outputs_ffn_lut_512mb
```

---

## 6.3 新增 FFN Hybrid Engine

新增：

```text
HybridFFNOutputEngine
```

逻辑：

1. forward pre-hook 捕获 FFN 输入。
2. 为所有被替换 group 计算地址。
3. 正常运行 FFN，或仅运行保留部分。
4. 用 LUT 输出覆盖被替换 group。
5. 拼回完整 FFN 输出。

第一版可以先保留完整 FFN forward，只做输出覆盖，用于验证精度。

第二版再做真正 MAC 削减：

- 把 `down_proj` 输出通道按 group 分开；
- 被替换输出 group 不再执行对应 `down_proj` 权重 slice；
- 保留 group 继续做矩阵乘；
- LUT 直接生成被替换输出 group。

这样能够真正减少：

\[
\text{saved MAC}
=
\text{replaced output channels}
\times
\text{intermediate size}
\]

---

## 6.4 Residual LUT Builder

在 coarse LUT 构建完成后：

```python
residual_target = teacher_group_output - coarse_lut_output
```

再对 residual target 构建第二张 tree LUT。

保存：

```text
coarse_address
coarse_table
residual_address
residual_table
```

推理：

```python
output = coarse_table[idx1] + residual_table[idx2]
```

---

## 7. 预算设计

## 7.1 单 entry 大小

如果：

- group_size = 64
- dtype = FP16

则单 entry：

\[
64\times2=128\text{ bytes}
\]

理论 entry 数：

| 总预算 | 理论最大 entry 数 |
|---:|---:|
| 128 MiB | 1,048,576 |
| 512 MiB | 4,194,304 |
| 1 GiB | 8,388,608 |

实际需要扣除：

- address tree
- metadata
- 多表
- residual 表
- 对齐开销

但仍然是百万级 entry。

---

## 7.2 MAC 目标

对于 down-like 输出 group：

\[
\text{group MAC}
=
\text{group size}
\times
\text{intermediate size}
\]

以 Qwen2.5-7B 为例：

\[
64\times18944=1,212,416
\]

MAC/token/group。

单层 FFN 的总 MAC 近似为：

\[
2\times hidden\times intermediate
+
intermediate\times hidden
=
3\times hidden\times intermediate
\]

因此可以直接计算替换多少 output groups 对应 5%、10%、20% 单专家 FFN MAC。

---

## 8. 实验矩阵

## 8.1 Phase A：容量扫描

固定：

- 单专家
- group_size=64
- 目标 MAC=10%
- tree address
- direct FFN output target

扫描：

| 配置 | LUT Budget | num_bits | num_tables |
|---|---:|---:|---:|
| A1 | 128 MiB | 12 | 1 |
| A2 | 512 MiB | 14 | 1 |
| A3 | 1 GiB | 16 | 1 |
| A4 | 512 MiB | 13 | 2 |
| A5 | 1 GiB | 14 | 4 |

观察：

- 容量增加是否带来稳定误差下降；
- 单表和多表谁更有效；
- 1 GiB 是否接近精度饱和。

---

## 8.2 Phase B：结构扫描

固定预算 512 MiB、目标 MAC 10%。

比较：

1. direct LUT
2. coarse + residual LUT
3. two-table residual
4. top-2 independent address
5. deeper tree single LUT

目标是判断：

> 容量应该用于加深地址、增加表数，还是增加 residual 分支。

---

## 8.3 Phase C：MAC 扫描

固定最佳结构，扫描：

- 5%
- 10%
- 20%
- 30%

观察：

- 精度随替换比例如何下降；
- 哪些 group 最敏感；
- 10% 是否是明显可行点；
- 是否存在少数 group 占据大部分误差。

---

## 8.4 Phase D：联合微调

先在单专家输出级做：

- LUT-only
- 只训练表值
- 冻结原始 FFN 权重

损失：

\[
\mathcal{L}
=
\lambda_1\operatorname{MSE}(\hat y,y)
+
\lambda_2(1-\cos(\hat y,y))
\]

可选加入：

\[
\lambda_3\operatorname{MSE}
(\operatorname{RMSNorm}(\hat y),
 \operatorname{RMSNorm}(y))
\]

第一阶段不需要模型级 KL，先把 FFN 局部输出拟合做好。

---

## 9. 指标

每个配置必须记录：

### 精度

- MSE
- normalized MSE
- relative L2 error
- cosine similarity
- per-group error
- p50 / p90 / p99 sample error

### 成本

- LUT total bytes
- address metadata bytes
- lookup count/token
- bytes read/token
- MAC saved/token
- extra additions/token
- online latency
- build time

### 分布稳定性

- calibration set
- held-out in-domain set
- out-of-domain set
- 不同 batch / sequence position
- 不同输入范数区间
- 高频与低频区域分别统计

---

## 10. 成功标准

### 最低可行

- 10% 单专家 FFN MAC reduction
- cosine similarity ≥ 0.95
- relative error ≤ 10%
- LUT ≤ 1 GiB
- 固定 O(1) 地址
- 无 ANN
- 无 Jacobian
- 无在线 JVP

### 较好结果

- cosine similarity ≥ 0.98
- relative error ≤ 5%
- LUT ≤ 512 MiB
- 10% MAC reduction

### 很强结果

- cosine similarity ≥ 0.99
- relative error ≤ 3%
- LUT ≤ 1 GiB
- 20% MAC reduction

---

## 11. 最先跑的最小实验

建议不要一开始就做完整 1 GiB 版本。

先跑：

```text
单专家
→ 只替换 4 个或 8 个 FFN output groups
→ group_size=64
→ tree address
→ num_bits={10,12,14}
→ budget={40MiB,128MiB,512MiB}
```

对每个配置输出：

```text
storage
MAC reduction
cosine similarity
relative error
normalized MSE
lookup latency
```

如果容量从 40 MiB 增到 128/512 MiB 后，误差明显下降，就说明：

> 原来表现一般主要是资源预算不足，而不是 LUT 方法本身失效。

然后再把替换比例扩到单专家 10%。

---

## 12. 结论

这条路线的核心不是枚举三千万个输入，而是：

> 用有限但显著放宽的 LUT 容量，把大量真实输入共享压缩到少量固定表项中。

与大规模 anchor + Jacobian 相比，它仍然满足：

- 存储可控；
- 在线路径固定；
- 查找 O(1)；
- 无高维最近邻搜索；
- 无 per-anchor 局部模型；
- 可直接计算 MAC reduction；
- 可以逐步扩展到硬件实现。

第一阶段只需要回答一个问题：

> 在单层真实 FFN 专家上，把 LUT 预算从几十 MiB扩大到几百 MiB或 1 GiB，是否能在 10% MAC reduction 下把输出精度显著推高？

只要这条容量—精度曲线成立，就值得继续做整层、跨层和模型级扩展。
