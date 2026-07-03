# 从 `new_lut.py` / `index_lut.py` 看多元化 LUT 的启示

> 这两个文件不是为 LLM 设计的，方法本身也不适合直接搬用，但里面有几个思想可以迁移到 v4，帮助我们突破当前 down_proj LUT 的质量天花板。

## 两个文件在做什么

### `new_lut.py`：高阶随机路由 + 可训练 EmbeddingBag 集成

- **地址生成**：`HighOrderRouter` 从输入里随机抽 4 个 channel，做 `(A+B > C+D)` 比较，产生 1 bit。重复 `tables × bits` 次，得到多个 10-bit 索引。
- **表结构**：每个 committee 有 512 张表，每张表 2^10 行，每行存 `NUM_CLASSES` 维 logits。用 `nn.EmbeddingBag(..., mode='sum')` 把多张表的输出相加。
- **训练**：表值（EmbeddingBag weight）端到端训练， committees 之间做 bagging/集成投票。
- **本质**：用随机高阶特征构造 LSH 地址，再用大量可训练小表做集成学习，近似一个教师 MLP。

### `index_lut.py`：可学习索引生成器 + 参数 LUT

- **索引生成器**：用 `Conv2d + MLP` 把图像特征映射成 Gumbel-Softmax 比特，是可学习的。
- **参数 LUT**：LUT 里存的不是输出特征，而是**网络参数**（W 矩阵）。根据比特从多张表里查出参数并加权求和，为每个样本动态拼出一个 Linear 的权重。
- **本质**：HyperNetwork / 细粒度 MoE 的 LUT 化，用查表代替部分前馈计算。

## 对 LLM_LUT v4 的启示

### 1. 可训练 LUT 表值（最大杠杆）

当前 v4 的 table 是 calibration 数据每个 bin 的**均值**，固定不动。

`new_lut.py` 的表值是端到端训练的 Embedding，显著提升了拟合能力。

**迁移**：把 v4 的 per-group 2D table 变成可训练参数，初始化用 calibration 均值，然后在 fine-tune 里联合优化。推理时仍是 O(1) 查表，存储不变。

预期效果：同样 group 数下 PPL 更低，从而敢上更多 group/层。

### 2. 高阶 / 随机化 address，而不只是 2 个 channel

当前 v4 只从 residual 里挑 2 个 channel 作为 2D address。这表达能力有限，尤其是 o_proj 这种输出和输入关系复杂的模块。

`new_lut.py` 用 4 个 channel 的随机组合产生 1 bit，再用多个 bit 构造索引。

**迁移**：
- 对每组/每张表，随机选若干输入 channel，做固定权重的线性组合或比较，生成 B 个 bit。
- 用 B bit 作为 1D LUT 地址（2^B 个 entry）。
- 这是固定随机、无训练参数的地址生成，不违反“LUT 查表”红线（没有可学习 MLP/CNN）。

优点：
- 地址空间更丰富，对复杂函数（如 o_proj/gate_proj）可能更鲁棒。
- 1D 表比 2D 表更紧凑：2^10 = 1024 entry vs 64×64 = 4096 entry。

### 3. 多张小表集成，而不是一张大表

当前 v4 每组一张 `[64, 64, 64]` 表。

`new_lut.py` 用 512 张小表，sum 输出。

**迁移**：
- 每组用 M 张独立小表，每张有自己的随机地址，输出相加。
- 例子：M=4，B=10，group_size=64 → 4 × 1024 × 64 = 262K entry，FP16 约 512KB，**和当前一张 64×64 表存储相同**。
- 集成能降低单张表地址冲突/量化带来的方差。

### 4. 不要直接搬 `index_lut.py` 的参数 LUT 思想

`index_lut.py` 本质是用 LUT 存动态权重矩阵，然后做矩阵乘法。这不再是 O(1) 查表替代矩阵乘，而是把矩阵乘的参数动态化，计算量没有减少，违反项目核心目标。

但里面“多 bit + 加权组合”的思想可以借鉴到**输出值 LUT**：用多个 bit 查多张表，加权求和得到输出 delta。

## 不能违反的红线

从 AGENTS.md：
- ❌ 不能用可学习 MLP/CNN 生成地址或参数。
- ✅ 可以用固定随机的高阶比较/投影生成地址。
- ✅ LUT 表值可以训练。
- ✅ 推理必须是 O(1) 查表，不能引入前馈网络。

所以 `index_lut.py` 里的 `Conv2d+MLP` 可学习索引生成器**不能直接搬**。

## 建议落地到 v4 的实验设计

### 方案 A：可训练 2D table（最小改动，先验证杠杆）

1. 在 `TrainableV3PartialEngine` 里把 `_batched_tables` 设为 `requires_grad=True`。
2. `finetune_multi_layer.py` 把 table 加入 optimizer。
3. 在 best down_proj 配置（group12 13 层）上重跑，看 PPL 能下降多少。
4. 如果 PPL 明显降低，再扩 group/层。

### 方案 B：随机高阶 address + 1D 可训练 table 集成

1. 地址生成：对每组/每张表，随机选 4-8 个 residual channel，固定随机权重，生成 10-bit 索引。
2. 每组 M=4 或 8 张 1D 表，输出相加。
3. 表值可训练。
4. 在同样存储预算下与方案 A 对比。

### 方案 C：多元化模块 + 多元化 LUT

不同层/模块用不同 LUT 策略：
- `down_proj` 质量好 → 用方案 A/B 上更多 group。
- `o_proj` 复杂 → 尝试方案 B 的高阶 address，如果仍然不行就放弃。
- `gate_proj` / `up_proj` → 用方案 B 做独立扫描，判断可替换性。

## 下一步建议

1. **先实施方案 A**（可训练 table）。改动最小，能直接判断“训练 LUT 表值”这个杠杆有多大。
2. **并行实施方案 B 的 address 生成**，在 v3 inspection 脚本里加一个 `--high_order_bits` 模式，对比 2D address 和 10-bit 高阶 address 的重建误差。
3. 根据 (1)(2) 结果，决定要不要把 o_proj / gate_proj / up_proj 重新纳入扫描。
