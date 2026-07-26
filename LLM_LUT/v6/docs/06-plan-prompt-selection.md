# 方案 1：Prompt 选择与 On-Policy 长序列数据 Pipeline

## 目标

解决当前 calibration 数据分布与长生成实际访问状态不匹配的问题。通过 activation-space / leaf coverage 选择 64 条长序列，建立 on-policy teacher 标注数据集，用于训练 / fine-tune LUT。

---

## 核心原则

- 不追求 prompt 语义多样，而是追求 **Layer 39 FFN input 状态空间覆盖**。
- 两个 prompt 即使主题不同，如果访问的 leaf 高度重叠，信息增益就低。
- 必须同时包含：代表性样本 + 困难轨迹 + 极端格式 + 多语言补充。

---

## 输入输出

### 输入

1. **候选 prompt 池**：约 500–1000 条，纯文本文件或 JSONL。
2. **当前 LUT checkpoint**：`outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints` 或任意 v6 checkpoint。
3. **完整模型**：Qwen3.6-35B-A3B（用于 rollout 和 teacher 标注）。

### 输出

`./onpolicy_data/layer39_v4/` 目录：

```text
selected_prompts.json          # 64 条被选中的 prompt 及元信息
candidate_features.pkl         # 所有候选 prompt 的 trajectory features
short_rollout/                 # 500~1000 条短 rollout（128~256 token）
long_rollout/                  # 64 条长 rollout（2048 token）
  ├── prompt_0000/
  │     ├── tokens.pt          # 输入 token ids
  │     ├── hidden_states.pt   # layer 39 FFN input（ teacher 模型前向得到）
  │     ├── teacher_ffn_out.pt # 真实 teacher FFN 输出
  │     ├── lut_ffn_out.pt     # 当前 LUT FFN 输出
  │     ├── leaves_coarse.pt   # coarse tree 访问的 leaf 索引
  │     ├── leaves_residual.pt # residual tree 访问的 leaf 索引
  │     └── metrics.json       # 每步 cosine / 异常率 / entropy 等
```

---

## Pipeline 步骤

### Step 1：候选池短 rollout

对每条候选 prompt，用**当前 LUT 替换 layer 39 FFN** 生成 128~256 token。

```python
# 伪代码
for prompt in candidate_pool:
    engine = V6ReplacementEngine(model, layer_idx=39, checkpoint_dir=...)
    engine.install()
    output_ids = model.generate(prompt, max_new_tokens=256)
    # 同时 hook 保存 layer 39 FFN input 和 LUT FFN output
```

**注意**：短 rollout 不需要 teacher 模型，用当前 LUT 自身生成就行，成本低。

### Step 2：提取 trajectory 特征

对每条 rollout 的每个 token 位置，提取：

```text
Layer 39 FFN input：
  - mean hidden [hidden_size]
  - hidden covariance / top-4 PCA projection [hidden_size * 4]
  - token-to-token displacement ||h_t - h_{t-1}||

LUT 访问统计：
  - coarse leaf histogram [num_coarse_leaves]
  - residual leaf histogram (per group) [32, num_residual_leaves]

行为指标：
  - base cosine mean / p10
  - correction residual norm
  - logit entropy
  - top-1 margin
  - 重复率
  - 异常字符率
```

**实现建议**：
- 在 `V6ReplacementEngine._hook` 里增加可选的 `record_mode=True`：保存每个 token 的 FFN input、预测输出、访问的 leaf。
- 或者写一个独立的短 rollout wrapper，复用 engine hook 但加钩子保存张量。

### Step 3：prompt-level embedding

把每条 prompt 的所有位置特征聚合成一个向量：

```python
feature = concatenate([
    semantic_embedding(prompt_text),        # 例如 sentence-transformers，可选
    mean_hidden_over_traj,
    pca_mean_and_std,
    leaf_histogram_flattened,              # 最核心
    mean_cosine,
    p10_cosine,
    mean_residual_norm,
    repetition_rate,
    anomaly_rate,
])
```

维度控制在 2K~8K，方便后续 facility location。

### Step 4：facility location 选择

目标函数：

```
F(S) = sum_p max_{s in S} sim(p, s)
```

实现建议：

```python
from sklearn.metrics.pairwise import cosine_similarity

def greedy_facility_location(features, k=64):
    # features: [N, D]
    n = features.shape[0]
    sim = cosine_similarity(features)  # [N, N]
    selected = []
    max_sim = np.zeros(n) - np.inf
    for _ in range(k):
        gains = sim.max(axis=1) - max_sim  # 边际增益
        # 第一次全选
        if _ == 0:
            idx = sim.sum(axis=1).argmax()
        else:
            idx = gains.argmax()
        selected.append(idx)
        max_sim = np.maximum(max_sim, sim[idx])
    return selected
```

**配额约束**：
- 先按任务大类（代码/数学/多语言/长文本/对话）设最低配额。
- 在每类内部独立做 facility location。
- 最终 64 条组成：
  - 32 条 activation-space 代表性最强
  - 16 条当前 LUT 低 cosine / 异常轨迹
  - 8 条极端长格式
  - 8 条多语言及混合语言

### Step 5：长 rollout + teacher 标注

对选出的 64 条 prompt，跑 **2048 token** 长生成，同时保存：

1. **LUT 自由运行轨迹**：`x_t^LUT` 和 `F_T(x_t^LUT)`
   - 用当前 LUT 生成，得到真实 FFN 输入状态。
2. **Teacher 标注**：对同一批 `x_t^LUT`，单独 forward teacher FFN，得到 `F_T(x_t^LUT)`。

**采样策略**（后段加权）：

```text
0–256：   随机保留 10%
256–512： 保留 20%
512–1024：保留 40%
1024+：   保留 60%
崩溃前 128 token：全部保留
崩溃后：  只保留少量，单独标记
```

这样避免崩溃后的不可恢复状态污染训练数据。

### Step 6：迭代

训练新 LUT 后，重新跑候选池短 rollout，找：
- 哪些 leaf 从未被访问（coverage gap）
- 哪些 prompt 在新 LUT 下仍然低 cosine / 异常

补 16–32 条新 prompt，进入下一轮。

---

## 与现有代码的接口

- **短 rollout**：复用 `run_model_eval.py` 的模型加载逻辑，但改为批量处理 prompt，并保存 hidden states。
- **hook 记录**：扩展 `V6ReplacementEngine._hook` 增加 `record` 开关，保存 `x`, `pred`, `leaves`。
- **facility location**：新建 `select_prompts.py`，输入 `candidate_features.pkl`，输出 `selected_prompts.json`。
- **长 rollout 标注**：新建 `collect_onpolicy_data.py`，读取 `selected_prompts.json`，输出 `long_rollout/` 目录。

---

## 验证指标

- 选出的 64 条对候选池的 coverage：
  - `mean(max_sim)` 越高越好
  - 未覆盖 prompt 的 `max_sim` 分布
- 每轮迭代后，候选池里的困难 prompt 比例是否下降
- 长 rollout 里 layer 39 FFN input 的分布与短独立样本的 KL/PCA 差异

---

## 风险

1. **候选池本身有偏**：如果 500–1000 条 prompt 就不代表真实使用场景，facility location 也救不回来。
2. **长 rollout 成本高**：64 × 2048 token ≈ 13 万 token，35B 模型多卡生成需要数小时。
3. **崩溃检测**：需要自动判断“崩溃”，可以用重复率、异常字符率、logit entropy 联合阈值。
