# Jacobian 引导的大规模直接寻址与轻量局部修正  
## 模拟阶段开发计划

> 版本：v0.1  
> 当前起始规模：300,000 anchors  
> 长期目标规模：3,000,000–9,000,000 anchors  
> 当前阶段：完整算法闭环模拟，不实现真实硬件物理地址与专用 CUDA kernel

---

## 1. 项目背景

原始 Jacobian-anchor 方案存在两条实现路径，但两条路径都难以形成有意义的在线加速。

第一条路径是在在线阶段，从约 30,000 个 anchor 中暴力检索最相近的 anchor，再在线计算 Jacobian 或 Jacobian-vector product：

\[
a^*(x)=\arg\min_{a_i\in\mathcal A}\|x-a_i\|
\]

\[
\hat F_{\mathrm{old}}(x)
=
F(a^*)+J(a^*)(x-a^*)
\]

这一方案同时承担全库近邻搜索和在线 Jacobian 计算，整体计算量可能高于直接执行一次原始 FFN，因此缺乏部署意义。

第二条路径是提前为 30,000 个 anchor 离线计算并保存完整 Jacobian。在线阶段仍然进行暴力近邻检索，然后读取对应 Jacobian 并执行矩阵向量乘法。该方案虽然移除了在线 Jacobian 构造，但仍然存在：

1. 30,000 个完整 Jacobian 的巨大存储开销；
2. 全库暴力检索；
3. 在线读取完整 Jacobian；
4. 一次接近 dense layer 规模的 Jacobian matvec；
5. 30,000 个 anchor 本身仍然覆盖不足。

本项目的核心转变是：既然原方案认为 30,000 个完整 Jacobian 的存储预算可以接受，那么同等甚至更低的存储预算，应当优先用于保存数十倍到数百倍数量的轻量 anchor，而不是继续为少量 anchor 保存完整局部算子。

目标结构为：

\[
\text{少量重型 anchor}
\quad\longrightarrow\quad
\text{海量轻型 anchor}
\]

即：

\[
\{a_i,F(a_i),J_i\}_{i=1}^{30k}
\]

替换为：

\[
\{a_i,F(a_i),c_i,\text{operator codes}_i\}_{i=1}^{300k\sim9M}
\]

其中完整 Jacobian 的作用不再由单个矩阵承担，而是拆分到：

- anchor 的高密度覆盖；
- 离线构造的结构化地址；
- 在线输入到地址的直接映射；
- 地址携带的共享 operator atoms；
- 每个 anchor 的极小残差 code；
- 命中后的轻量局部计算。

---

## 2. 项目目标

### 2.1 基线目标

本项目当前不要求直接逼近原始 FFN 的绝对最优输出，而是首先达到以下相对目标：

\[
L_{\mathrm{ours}}
\leq
L_{\mathrm{brute\text{-}force\ NN+Jacobian}}+\epsilon
\]

即在 300,000 个及以上 anchor 的条件下，完整模拟链路的效果应当接近或不明显差于：

\[
\text{暴力最近邻检索}
+
\text{完整 Jacobian correction}
\]

同时满足：

- 不在线进行全库检索；
- 不在线计算完整 Jacobian 或 JVP；
- 不读取完整 Jacobian；
- 不执行完整 Jacobian matvec；
- 每次查询只访问固定数量的逻辑地址；
- 每次查询只执行固定规模的轻量 correction。

### 2.2 长期目标

长期目标是构造：

\[
A(x)=(s_1,s_2,\ldots,s_m)
\]

\[
i=\operatorname{Decode}(A(x))
\]

\[
\tilde J_{A(x)}
=
\sum_{k=1}^{m}C^{(k)}_{s_k}
+
C_i^{\mathrm{res}}
\]

\[
\hat F(x)
=
F(a_i)
+
\tilde J_{A(x)}(x-a_i)
\]

其中结构化地址同时决定：

1. 命中的 anchor；
2. 所属 operator regime；
3. 使用的共享 correction atoms；
4. 每个 anchor 的残差参数；
5. 未来的物理存储布局与 kernel 路径。

---

## 3. 五个核心原则

## 原则一：先以海量 anchor 换取局部覆盖，再用轻量修正补剩余误差

原方案依赖少量 anchor 和重型 Jacobian：

\[
N\text{ 小},\qquad J_i\text{ 重}
\]

本方案采用：

\[
N\text{ 大},\qquad C_i\text{ 轻}
\]

anchor 数量从 30,000 提高到 300,000、3,000,000 或 9,000,000 后，目标是使：

\[
\delta_i=x-a_i
\]

整体缩小，从而降低局部修正的表达难度。

第一阶段不需要证明海量 anchor 可以完全恢复原始 FFN，只需要证明其最终效果不明显差于原来的 30,000-anchor 暴力检索加 Jacobian 方案。

---

## 原则二：地址不是普通编号，而是局部函数模型的编码

在线映射不应被定义为普通的几何最近邻分类：

\[
A(x)\approx\arg\min_i\|x-a_i\|
\]

更合理的目标是选择经过局部修正后误差较小的 anchor：

\[
i^*(x)
=
\arg\min_i
\left\|
F(x)-F(a_i)-C_i(x-a_i)
\right\|^2
\]

因此，地址表达应同时编码：

- 输入所属区域；
- 局部 operator family；
- correction atoms；
- 具体 anchor；
- 安全替代邻居。

离线 anchor 编址和在线 query 编址必须尽量使用同一套 encoder、quantizer 和地址规则，避免离线空间与在线空间不一致。

---

## 原则三：不重建完整 Jacobian，只重建真实偏移上的 Jacobian action

本项目不以以下目标为核心：

\[
\|J_i-\tilde J_i\|_F^2
\]

因为部署时真正需要计算的只是：

\[
J_i\delta,\qquad \delta=x-a_i
\]

因此离线训练应优先优化：

\[
\mathcal L_{\mathrm{action}}
=
\mathbb E_{\delta\sim\mathcal N_i}
\left[
\|J_i\delta-C_i(\delta)\|^2
\right]
\]

或直接使用输出差分：

\[
\mathcal L_{\mathrm{local}}
=
\left\|
F(a_i+\delta)-F(a_i)-C_i(\delta)
\right\|^2
\]

完整 Jacobian、在线 JVP 和暴力搜索只允许作为离线 teacher 或评估上界，不允许进入部署模拟路径。

---

## 原则四：地址可以出错，但只能错到功能上安全的邻居

本项目最大的算法风险不是平均地址准确率不足，而是小幅地址偏差可能导致跨 regime 跳转，从而造成灾难性输出误差。

因此不能只优化：

\[
P(A(x)=i^*)
\]

还必须优化 misrouting regret：

\[
R_{\mathrm{route}}(x)
=
L(x,A(x))-L(x,i^*)
\]

地址空间需要满足：

\[
d_H(A_i,A_j)\text{ 小}
\Rightarrow
L_{\mathrm{cross}}(i,j)\text{ 小}
\]

其中：

\[
L_{\mathrm{cross}}(i,j)
=
\mathbb E_{x\in\mathcal C_i}
\left[
\|F(x)-F(a_j)-C_j(x-a_j)\|^2
\right]
\]

也就是说：

- 容易混淆的地址必须功能相近；
- 功能差异大的地址必须在编码空间中隔开；
- 单 bit 或单子码错误不得直接跳到危险区域；
- 低置信度输入可以触发固定的 top-2 地址保险；
- 不允许回退到全库搜索。

---

## 原则五：模拟阶段实现完整算法闭环，但暂时抽象物理硬件交互

模拟阶段必须真实实现：

\[
\text{anchor selection}
\rightarrow
\text{offline address compilation}
\rightarrow
\text{online address mapping}
\rightarrow
\text{fixed logical lookup}
\rightarrow
\text{light correction}
\rightarrow
\hat F(x)
\]

但可以暂时省略：

- CUDA memory bank 分配；
- cache-line alignment；
- warp/thread mapping；
- shared-memory staging；
- kernel fusion；
- physical prefetch；
- 编译期地址分支；
- 真实静态哈希或硬件直接寻址。

模拟阶段使用逻辑地址和 tensor slot 代替物理地址：

```python
address = address_encoder(x)
slot = logical_address_table[address]
record = anchor_records[slot]
output = correction_reference(x, record, shared_atoms)
```

必须禁止：

```python
address = address_encoder(x)
candidate = search_inside_bucket(x)
```

即逻辑 lookup 后不能隐藏第二次近邻搜索。

未来硬件阶段再将：

\[
\text{logical address}
\]

映射为：

\[
\text{bank}+\text{page}+\text{block}+\text{slot}
\]

并将固定 record layout 与 correction kernel 融合。

---

## 4. 五步开发拆分

## 第一步：构造 300,000 个高质量 anchor

### 4.1 输入

- 从目标 FFN 层收集的大规模真实 activation；
- 对应真实 FFN 输出；
- 可选的 JVP probes；
- routing/expert 信息；
- 数据频率与来源信息。

### 4.2 第一版选择策略

第一版不追求一次得到最终最优 anchor 集合，而是优先建立可运行基线。

建议依次实现：

1. 分层 reservoir sampling；
2. k-means++ / mini-batch k-means；
3. 按 routing regime 或 expert 分桶后独立采样；
4. 去除极近重复点；
5. 依据 cell frequency 分配 anchor 配额；
6. 依据局部 residual 对高误差 cell 继续切分。

建议 anchor 数量曲线：

\[
30k\rightarrow100k\rightarrow300k
\]

已有 30k 结果作为原方案基线，100k 和 300k 用于判断数量扩展趋势。

### 4.3 后续 functional-aware 选择

定义候选描述：

\[
\phi_i=
[
\lambda_x P_xa_i,\;
\lambda_yP_yF(a_i),\;
\lambda_jg_i
]
\]

其中 \(g_i\) 是低成本 JVP signature 或局部函数特征。

最终 anchor 不是单纯的输入几何中心，而应优先覆盖：

- 高频区域；
- 局部变化较大区域；
- 路由边界区域；
- 当前轻量 correction 残差较大的区域；
- 地址容易混淆且误差代价较大的区域。

### 4.4 输出

```text
anchors              [N, d_in]
anchor_outputs       [N, d_out]
anchor_bank_ids      [N]
anchor_metadata      [N, ...]
```

---

## 第二步：离线编译为“逻辑地址 + 轻量 record”

对于每个 anchor \(a_i\)，离线生成：

\[
R_i=
[
A_i,\;
a_i,\;
F(a_i),\;
c_i,\;
s_{i1},\ldots,s_{im}
]
\]

其中：

- \(A_i\)：结构化逻辑地址；
- \(c_i\)：极小的 per-anchor residual code；
- \(s_{ik}\)：共享 operator atom 的选择码；
- \(a_i,F(a_i)\)：anchor input/output；
- bank 信息隐含在高位地址中。

第一版 correction 可以采用：

\[
C_i(\delta)
=
U_{b(i)}
\operatorname{Diag}(c_i)
V_{b(i)}^\top\delta
\]

更进一步可采用组合式 operator：

\[
C_i(\delta)
=
\sum_{k=1}^{m}C^{(k)}_{s_{ik}}(\delta)
+
C_i^{\mathrm{res}}(\delta)
\]

### 4.5 地址格式建议

```text
[ regime bits | bank bits | cell bits | anchor/atom bits ]
```

地址需要具备三个性质：

1. 可由离线 anchor encoder 生成；
2. 可由在线 query encoder 生成；
3. 可在未来直接解码成物理 record 偏移。

### 4.6 输出

```text
anchor_records
logical_address_table
shared_operator_atoms
address_codebook
safe_neighbor_table
```

---

## 第三步：在线输入直接映射到地址

在线路径只能使用未来允许保留的计算：

\[
z=E(x)
\]

\[
A(x)=Q(z)
\]

不允许调用：

- exact nearest-neighbor search；
- FAISS search；
- 全库 reranking；
- 完整 Jacobian；
- 在线 JVP。

### 4.7 地址形式

优先考虑弱依赖或并行子码：

\[
A(x)=
[q_1(P_1x),q_2(P_2x),\ldots,q_m(P_mx)]
\]

相较严格残差式量化，这种结构可以降低高位地址错误向后级联的风险。

### 4.8 functional supervision

训练目标不应只模仿几何最近邻，而应模仿最终局部误差较小的地址。

定义：

\[
L(x,i)
=
\|F(x)-F(a_i)-C_i(x-a_i)\|^2
\]

允许多个可接受地址：

\[
\mathcal S_\epsilon(x)
=
\{i:L(x,i)\leq L(x,i^*)+\epsilon\}
\]

soft teacher target：

\[
p_i^*(x)
\propto
\exp(-L(x,i)/\tau)
\]

router 目标是命中安全集合，而不必死记唯一 anchor。

### 4.9 边界保险

默认路径：

```text
high confidence -> top-1 fixed address
low confidence  -> top-2 fixed addresses
```

top-2 只能访问两个已由 encoder 直接输出的固定地址，不允许在候选池内继续搜索。

---

## 第四步：地址命中后的轻量计算

地址命中 anchor \(i\) 后：

\[
\delta=x-a_i
\]

第一版：

\[
q=V_b^\top\delta
\]

\[
r=c_i\odot q
\]

\[
\Delta y=U_br
\]

\[
\hat F(x)=F(a_i)+\Delta y
\]

组合式版本：

\[
\tilde J_i
=
\sum_{k=1}^{m}C^{(k)}_{s_{ik}}
+
C_i^{\mathrm{res}}
\]

\[
\hat F(x)
=
F(a_i)+\tilde J_i(x-a_i)
\]

### 4.10 约束

- correction FLOPs 必须显著低于原 FFN；
- 读取字节数必须可明确计算；
- 不允许读取完整 Jacobian；
- correction reference 必须与未来 kernel 的算子结构一致；
- 第一版使用 PyTorch reference implementation；
- 后续使用 Triton 进行 microbenchmark；
- 最后再决定是否进入 CUDA/CUTLASS 或定制硬件实现。

---

## 第五步：联合逼近“暴力检索 + 完整 Jacobian”

teacher 路径：

\[
a^*_{\mathrm{teacher}}
=
\arg\min_i
L_{\mathrm{teacher}}(x,i)
\]

\[
y_{\mathrm{teacher}}
=
F(a^*)+J(a^*)(x-a^*)
\]

deployment simulation 路径：

\[
i=A(x)
\]

\[
y_{\mathrm{ours}}
=
F(a_i)+C_i(x-a_i)
\]

联合训练目标：

\[
\mathcal L
=
\mathcal L_{\mathrm{output}}
+
\lambda_1\mathcal L_{\mathrm{address}}
+
\lambda_2\mathcal L_{\mathrm{action}}
+
\lambda_3\mathcal L_{\mathrm{neighbor}}
+
\lambda_4\mathcal L_{\mathrm{balance}}
\]

其中：

\[
\mathcal L_{\mathrm{output}}
=
\|F(x)-y_{\mathrm{ours}}\|^2
\]

\[
\mathcal L_{\mathrm{action}}
=
\|J(a_i)\delta-C_i(\delta)\|^2
\]

\[
\mathcal L_{\mathrm{neighbor}}
=
\mathbb E_{\tilde A\sim\mathcal C(A)}
\left[
\|F(x)-\hat F(x;\tilde A)\|^2
\right]
\]

训练中主动注入：

- 单子码翻转；
- 第二候选 bank；
- 相邻安全地址；
- encoder 小扰动；
- 低 margin 候选；
- address code corruption。

目标不是假设地址永不出错，而是让常见的小错误只能落到低代价区域。

---

## 5. 软件架构

```text
src/
├── data/
│   ├── activation_collector.py
│   ├── dataset.py
│   └── sampling.py
├── anchors/
│   ├── anchor_builder.py
│   ├── clustering.py
│   ├── functional_features.py
│   └── residual_split.py
├── teacher/
│   ├── exact_search.py
│   ├── functional_oracle.py
│   ├── jvp_teacher.py
│   └── safe_sets.py
├── addressing/
│   ├── address_encoder.py
│   ├── quantizer.py
│   ├── address_compiler.py
│   ├── logical_memory.py
│   └── safe_neighbors.py
├── correction/
│   ├── shared_basis.py
│   ├── compositional_atoms.py
│   ├── anchor_codes.py
│   └── reference_kernel.py
├── training/
│   ├── train_address.py
│   ├── train_correction.py
│   ├── joint_train.py
│   └── corruption_training.py
├── evaluation/
│   ├── quality.py
│   ├── routing.py
│   ├── storage.py
│   ├── cost_model.py
│   └── model_level.py
└── kernels/
    ├── triton_lookup_reference.py
    └── README.md
```

---

## 6. Teacher Path 与 Deployment Path

## 6.1 Teacher Path

允许使用昂贵操作，仅用于离线监督和上界评估：

```text
query
-> exact / high-recall search
-> functional anchor ranking
-> full JVP or full Jacobian action
-> safe candidate set
-> teacher output
```

可以使用 FAISS 进行大规模 exact/ANN teacher 构建、聚类和参数评估。Faiss 官方定位即为密集向量的高效相似度搜索与聚类工具，并同时提供 C++、Python 和部分 GPU 实现。

## 6.2 Deployment Simulation Path

严格受部署约束：

```text
query
-> address encoder
-> one or two logical addresses
-> fixed record lookup
-> lightweight correction
-> output
```

禁止：

```text
query
-> address
-> bucket search
-> rerank many candidates
-> full Jacobian/JVP
```

---

## 7. 评估指标

## 7.1 输出质量

- FFN output cosine similarity；
- relative L2；
- MSE；
- teacher Jacobian-output gap；
- downstream KL；
- PPL；
- task accuracy；
- generation sample quality。

## 7.2 Anchor 覆盖

\[
\mathbb E\|x-a_i\|
\]

\[
P50/P95/P99(\|x-a_i\|)
\]

\[
\mathbb E
\|F(x)-F(a_i)-J(a_i)(x-a_i)\|
\]

比较：

\[
30k,\quad100k,\quad300k
\]

## 7.3 地址质量

### Exact-address rate

\[
P(A(x)=A^*(x))
\]

### Safe-address rate

\[
P(A(x)\in\mathcal S_\epsilon(x))
\]

### Misrouting regret

\[
R_{\mathrm{route}}(x)
=
L(x,A(x))-L(x,i^*)
\]

报告：

- mean；
- median；
- P95；
- P99；
- max。

### Catastrophic routing rate

\[
P(R_{\mathrm{route}}(x)>\tau)
\]

这是当前最重要的风险指标之一。

### Top-2 trigger rate

记录低置信度路径触发比例，以及 top-2 对灾难错误率的改善。

## 7.4 Correction 质量

对比：

1. true \(J\delta\)；
2. bank-shared low rank；
3. compositional atoms；
4. per-anchor residual；
5. 无 correction 的 bare anchor。

## 7.5 存储

记录：

\[
S_{\mathrm{anchor}}
\]

\[
S_{\mathrm{output}}
\]

\[
S_{\mathrm{codes}}
\]

\[
S_{\mathrm{shared\ atoms}}
\]

\[
S_{\mathrm{address\ tables}}
\]

并与 30,000 个完整 Jacobian 的存储预算比较。

## 7.6 虚拟硬件成本

每个 query 记录：

- address encoder MACs；
- lookup count；
- record bytes；
- shared atom bytes；
- correction MACs；
- top-2 rate；
- branch count；
- arithmetic intensity。

模拟成本：

\[
T_{\mathrm{sim}}
=
\alpha C_{\mathrm{address}}
+
\beta B_{\mathrm{record}}
+
\gamma C_{\mathrm{correction}}
+
\eta N_{\mathrm{lookup}}
\]

初期系数仅用于相对比较；后续通过 Triton/CUDA microbenchmark 校准。

---

## 8. 实验矩阵

| 组别 | Anchor | Address | Correction | 作用 |
|---|---:|---|---|---|
| A0 | 30k | Exact NN | True \(J\delta\) | 原方案基线 |
| A1 | 100k | Exact NN | True \(J\delta\) | 覆盖 scaling |
| A2 | 300k | Exact NN | True \(J\delta\) | 300k oracle |
| B0 | 300k | Exact NN | Bare anchor | anchor-only 下界 |
| B1 | 300k | Exact NN | Shared low-rank | correction 上界 |
| B2 | 300k | Exact NN | Compositional atoms | 组合修正 |
| C0 | 300k | Direct address | True \(J\delta\) | 单独测地址损失 |
| C1 | 300k | Direct address | Shared low-rank | 完整第一版 |
| C2 | 300k | Direct address top-2 | Shared low-rank | 边界保险 |
| C3 | 300k | Robust address | Compositional atoms | 完整增强版 |

重点 gap：

\[
G_{\mathrm{address}}
=
L(\text{direct},J)-L(\text{exact},J)
\]

\[
G_{\mathrm{correction}}
=
L(\text{exact},C)-L(\text{exact},J)
\]

\[
G_{\mathrm{joint}}
=
L(\text{direct},C)-L(\text{exact},J)
\]

---

## 9. 阶段计划

## Phase 0：数据与旧基线复现

目标：

- 固化 activation 数据集；
- 复现 30k 暴力 NN + true Jacobian/JVP；
- 固化评估脚本；
- 记录完整时间、存储与质量基线。

完成条件：

- 同一数据切分下结果可重复；
- teacher path 与 deployment path 的代码入口完全分离。

---

## Phase 1：300k Anchor Scaling

实现：

- 100k 和 300k anchor；
- exact NN teacher；
- bare anchor；
- true \(J\delta\) oracle；
- anchor 覆盖统计。

核心问题：

> 300k anchor 下，原始“暴力检索 + Jacobian”上界是否至少不差于 30k 基线？

停止条件：

- 若 300k true Jacobian oracle 仍显著差于 30k baseline，需要检查 anchor 构造或距离定义；
- 不立即否定整体方案，因为本项目最低目标仍是匹配原方法，但必须先解释 scaling 异常。

---

## Phase 2：Jacobian Action Distillation

实现：

- shared low-rank；
- rank 8/16/32/64；
- per-anchor code；
- compositional atoms；
- output-difference training；
- JVP teacher training。

核心问题：

> 不保存完整 Jacobian 时，能否在 exact anchor 条件下接近 true \(J\delta\)？

通过条件：

- correction gap 可控；
- correction FLOPs 明显低于原 FFN；
- 每-anchor code 存储远小于完整 Jacobian。

---

## Phase 3：Direct Functional Addressing

实现：

- shared offline/online encoder；
- product-style address；
- functional teacher labels；
- safe candidate sets；
- logical address table；
- 禁止 bucket search。

核心问题：

> 单地址映射错误是否集中在低代价邻居，而不是出现大量跨 regime 灾难跳转？

通过条件：

- safe-address rate 高；
- P99 routing regret 可控；
- catastrophic routing rate 足够低。

---

## Phase 4：Robust Addressing

实现：

- safe-neighbor code placement；
- corruption training；
- margin loss；
- top-2 boundary path；
- 独立/弱依赖子码；
- 邻居一致性约束。

核心问题：

> 在不恢复搜索的前提下，能否显著压低尾部 misrouting regret？

---

## Phase 5：Full Simulation Loop

完整链路：

```text
query
-> direct functional address
-> fixed logical lookup
-> compositional Jacobian-action correction
-> approximate FFN output
```

最终比较：

\[
\text{ours}
\quad\text{vs.}\quad
\text{30k brute-force NN + true Jacobian}
\]

同时报告：

- 质量；
- 存储；
- MAC；
- bytes/query；
- lookup count；
- catastrophic routing rate。

---

## Phase 6：Hardware Microbenchmark

在算法成立后实现最小 Triton kernel：

```text
integer address
-> decode slot
-> gather one/two records
-> low-rank or atom correction
-> output
```

Triton 的官方目标是以 Python 风格开发高吞吐自定义 DNN kernel，因此适合作为 PyTorch reference 与 CUDA/CUTLASS 之间的中间验证层。

这一阶段不实现完整 300k 地址系统，只验证：

- 单次 record lookup；
- 连续 record layout；
- gather + correction fusion；
- top-1/top-2 两条固定路径；
- 实际 bandwidth 与 latency。

---

## Phase 7：Physical Address and Kernel Co-design

最后再处理：

- logical-to-physical address encoding；
- bank/page/block/slot 布局；
- fixed stride；
- memory alignment；
- query batch reordering；
- coalesced access；
- prefetch；
- shared memory；
- kernel specialization；
- CUTLASS/CUDA 或定制硬件实现。

CUTLASS 将高性能 GEMM 和相关计算拆分为布局、数据移动、硬件 atom 与执行层次，因此该阶段应被视为独立的硬件映射工作，而不是模拟阶段的前置要求。

---

## 10. 第一版具体建议

第一版使用：

```text
N anchors            = 300,000
num banks             = 256 or 512
address parts         = 3–6
correction rank       = 8, 16, 32, 64
lookup count          = 1 by default
boundary lookup       = top-2
per-anchor code       = FP16/INT8 coefficients
shared atoms          = bank-level or sub-address-level
implementation        = PyTorch + Faiss teacher
hardware prototype    = deferred
```

第一版不做：

- 3M/9M 全量部署；
- 手写 CUDA；
- Cython 物理地址控制；
- kernel 内数百万独立分支；
- 大规模硬件布局优化；
- 在线 bucket reranking。

---

## 11. 项目成功标准

第一阶段成功不要求最终硬件加速已经实现，但必须同时满足：

1. **完整闭环**  
   从输入到地址、固定 lookup、轻量 correction、最终输出全部真实运行。

2. **无隐藏搜索**  
   deployment path 不调用 FAISS、exact NN 或 bucket reranking。

3. **无完整 Jacobian**  
   deployment path 不保存、读取或计算完整 Jacobian。

4. **效果匹配**  
   与原始 30k 暴力 NN + Jacobian 相比，结果相当或只出现可接受的小幅下降。

5. **成本结构成立**  
   模拟统计显示存储、计算和 lookup 次数均具有明确优势。

6. **尾部风险可控**  
   P99 routing regret 与 catastrophic routing rate 不出现不可接受的爆炸。

7. **可映射到硬件**  
   地址是结构化整数码，record 是固定布局，lookup 次数固定，correction 算子结构固定。

---

## 12. 当前最重要的研究问题

当前最优先的问题不是硬件访存，也不是证明百万 anchor 能完美逼近原 FFN，而是：

> 如何构造一个 functional address space，使输入轻微映射偏差只能落到功能相近、修正后误差较小的 anchor，而不会跨越局部 operator regime 造成灾难性结果？

因此，下一步研究重点应集中在：

- functional teacher label；
- safe candidate set；
- safe-neighbor address layout；
- misrouting regret；
- corrupted-address training；
- top-2 boundary insurance；
- compositional operator code。

---

## 13. 参考工具与相关思路

1. **Faiss**  
   用于离线聚类、exact/ANN teacher、索引评估和参数调优。  
   https://faiss.ai/  
   https://github.com/facebookresearch/faiss

2. **Billion-scale Similarity Search with GPUs**  
   用于理解大规模向量搜索中的 GPU 并行、k-selection 和 memory hierarchy 问题。  
   https://arxiv.org/abs/1702.08734

3. **Distill-VQ**  
   其核心启发是：量化目标不必只优化向量重建误差，而可以蒸馏 teacher 的最终检索行为。本项目可将 teacher relevance 替换为 correction 后的 functional loss。  
   https://arxiv.org/abs/2204.00185

4. **Triton**  
   用于算法成立后的 lookup + gather + correction reference kernel。  
   https://triton-lang.org/  
   https://github.com/triton-lang/triton

5. **NVIDIA CUTLASS**  
   用于后期研究固定布局、数据移动、硬件 atom 与层次化 kernel 映射。  
   https://docs.nvidia.com/cutlass/latest/overview.html

---

## 14. 一句话总结

本项目不是对 30,000 个 Jacobian 做简单压缩，而是将原本由“暴力最近邻 + 完整 Jacobian”承担的功能，重新拆分到海量 anchor 覆盖、结构化 functional address、共享 operator atoms 和轻量 Jacobian-action correction 中，并在纯软件模拟阶段先完成一个无隐藏搜索、无完整 Jacobian、可映射到未来硬件的完整闭环。
