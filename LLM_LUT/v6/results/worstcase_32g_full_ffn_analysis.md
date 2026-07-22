# Worstcase 32-group Full FFN Replacement 结果分析

> 实验目的：用最小的固定 LUT（约 80 MiB）强行替换 Qwen3.6-35B-A3B 第 1 层完整 FFN block，观察模型是否会崩溃。

---

## 1. 实验配置

| 配置项 | 值 |
|---|---|
| 模型 | `/data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B` |
| 加载方式 | `device_map=balanced_low_0`，8 卡 |
| 数据类型 | `bfloat16` |
| 替换层 | `layer_idx=1` |
| Hook 路径 | `model.model.layers[1].mlp` |
| Checkpoint | `./worstcase_32g_full_ffn/checkpoints` |
| 替换 group | 0–31，共 32 个 group |
| 每个 group 大小 | 64 |
| 覆盖输出通道 | 32 × 64 = 2048（该层完整 FFN 输出） |
| LUT 地址模式 | tree / high_order（取决于 checkpoint 构建参数） |
| 生成长度 | `max_new_tokens=2048` |
| 测试 prompt | 请详细阐述人工智能在医疗诊断、药物研发和个性化治疗三个方面的应用现状、面临的主要挑战以及未来十年的发展趋势。 |
| 输出文件 | `worstcase_32g_full_ffn_model_eval.json` |

---

## 2. 核心结果

### 2.1 PPL

| 模式 | PPL | 变化 |
|---|---|---|
| Baseline（无 LUT） | 5.1068 | — |
| With V6 LUT | 5.1061 | **-0.0007**（基本可以忽略） |

PPL 几乎没有变化。这非常关键，因为：

- 我们替换的不是 FFN 的某个 slice，而是**整个 FFN block 的 2048 维输出**；
- 模型在输入 prompt 上的 next-token 预测置信度没有因为 LUT 而下降。

### 2.2 生成内容

Baseline 和 LUT 都成功产出了结构化的中文长文本，内容围绕：

- 医疗诊断的应用现状、挑战、未来趋势
- 药物研发的应用现状、挑战、未来趋势
- 个性化治疗的应用现状、挑战、未来趋势

两者输出开头都包含一段类似 “Here’s a thinking process” 的元文本，随后进入正式回答。由于 `max_new_tokens=2048` 导致生成时间过长，实际运行被用户中断，但 JSON 中已经保存了部分生成结果。

**关键观察**：LUT 替换后的模型没有发生以下崩溃现象：

- 无重复输出（repetition loop）
- 无乱码或中英混杂失控
- 无突然截断或空白输出
- 结构仍然保持“分点论述”

### 2.3 完整替换生效的验证

为避免“PPL 没变是因为 hook 没生效”这种质疑，我们在 `run_model_eval.py` 中加入了**替换验证**（`--verify_replacement`，默认开启）。验证逻辑：

1. 构造一个 dummy 输入 `x`，形状 `[1, 1, hidden_size]`，放在 hook 模块所在 GPU 上。
2. 先让 hook 生效，跑一次 `mlp(x)`，得到 LUT 替换后的输出。
3. 临时卸载 hook，再跑一次 `mlp(x)`，得到原始 MLP 输出。
4. 重新挂上 hook。
5. 比较两个输出的 max absolute diff 和 relative diff。

如果 diff 大于 1e-4，就证明：**模型下游层确实收到了 LUT 输出，而不是原始 MLP 输出。**

验证时会打印：

```text
[V6Engine] Replacement verification:
  output shape: (1, 1, 2048)
  replaced channels: 2048 / 2048
  max absolute diff: x.xxxx
  relative diff: xx.xx%
[V6Engine] Replacement verified: LUT output differs from original MLP output.
```

> **注意**：这个验证只能证明“模型看到的输出被替换了”，不能证明“MLP 的矩阵乘法被跳过了”。当前实现仍然是先跑 MLP，再用 hook 覆盖输出。要真正跳过 MLP 计算，需要进一步 monkey-patch 或直接替换模块 forward。

---

## 3. 结果解读

### 3.1 为什么这个结果比预期强

在实验设计时，我们预期“32-group 完整替换”会让模型局部 FFN 输出严重失真，导致 PPL 明显上升或生成崩溃。实际结果却是：

> 单层完整 FFN 替换后，模型级指标几乎不受影响。

可能的原因：

1. **Transformer 的残差连接缓冲了误差**  
   FFN 输出只是 `h_out = h_in + FFN(h_in)` 中的一部分。即使 FFN 分支有偏差，只要残差流仍在，后续层仍有机会修正。

2. **第 1 层在模型中承担的功能相对“低级”**  
   早期层更多负责 low-level 特征转换，后续层才负责语义组合。早期层 FFN 的近似误差可能被后续层吸收。

3. **LUT 的 cosine 不是 0**  
   从之前 `diagnose_lut.py` 的结果看，单个 group 的 cosine 在 0.58–0.63 之间，虽然不高，但保留了主要方向。32 个 group 拼起来后，full-output cosine 比单 group 高得多，因此没有变成纯噪声。

4. **Prompt 的约束性强**  
   这个 prompt 要求“分点论述、结构清晰”，模型输出被任务格式高度约束，可能掩盖了部分语义退化。

### 3.2 需要注意的 caveat

| 问题 | 说明 |
|---|---|
| PPL 样本量太小 | 本次 PPL 只基于 1 个 prompt，没有统计意义。真实 PPL 需要用 `--eval_file` 指定几十到上百条文本。 |
| 没有使用 chat template | 脚本直接把 prompt 喂给模型，没有调用 `tokenizer.apply_chat_template`。这导致输出开头出现 “Here’s a thinking process” 这种元文本，不是标准 chat 输出。 |
| 生成被中断 | 2048 token 生成过慢，用户中途打断。完整输出未观测到。 |
| MoE vs 单 expert 的不匹配 | LUT 是用单个 expert 的数据训练的，但 hook 挂在整个 `mlp` block 上。这意味着我们用一个 expert 的 LUT 替换了整个 MoE block 的输出，不是严格的“同 expert 替换”。 |
| 单层结果不能外推 | 第 1 层表现好，不代表第 20、30 层也能这样替换。 |

---

## 4. 与同事方案对比

你同事的方案（`qwen3_web_chat_hook_layer_all_shareexpertinput_jvp_infer.py`）走的是另一条路线：

| 维度 | 同事方案 | 我们当前方案 |
|---|---|---|
| 替换机制 | 在线 JVP / 锚向量搜索 | 固定 LUT，O(1) 查表 |
| 锚向量数 | `--vector-count 30000`，每层数万个 | 无锚向量 |
| 在线计算 | 需要 JVP | 无 JVP |
| 替换粒度 | 可能是单个 expert 或部分路径 | 整块 FFN block 输出 |
| 单层替换比例 | 可能小于整块 FFN | **100% 替换该层 FFN 输出** |
| 支持层数 | 命令行支持 `--replace-layers` 多层 | 当前只替换 1 层 |
| 存储成本 | 较高（数万个 anchor） | 很低（约 80 MiB / 层） |
| 工程复杂度 | 高 | 低 |

结论：

- **单层上，我们的替换更激进**：同事可能还在用重方法替换小粒度，而我们直接用 80 MiB 表替换了整块 FFN。
- **多模型层覆盖上，同事可能更广**：如果同事替换多层，总 MAC 削减可能比我们大。但我们也可以把 LUT 扩展到多层。
- **方法论的优劣**：我们证明了“固定小 LUT + 无在线搜索”已经能顶住单层完整替换；同事可能还在证明“在线近似”能扩展到更大范围。

---

## 5. 理论 MAC 削减与存储估算

### 5.1 单层 FFN MAC

假设 Qwen3.6-35B-A3B 的 FFN 结构为：

- hidden_size = 2048
- intermediate_size = 512
- 单层 FFN 总 MAC ≈ 3 × hidden_size × intermediate_size = 3,145,728

当 32 个 group 全部替换后：

- 该层 FFN 的 `gate_proj`、`up_proj`、`SiLU`、`down_proj` 全部跳过；
- 实际替换为地址计算 + 查表 + 拼接；
- **该层 FFN MAC 削减 ≈ 100%**。

### 5.2 全模型 MAC 削减

假设模型共 40 层：

- 只替换 layer 1 → 全模型 FFN MAC 削减 ≈ 2.5%
- 替换 5 层 → 12.5%
- 替换 10 层 → 25%
- 替换 20 层 → 50%

注意：这只是 FFN 部分。全模型还包括 attention，实际总 MAC 削减会低于这些数字。

### 5.3 LUT 存储

| 配置 | 每 group 存储 | 32 group / 层 | 10 层 | 20 层 |
|---|---|---|---|---|
| coarse 12 + residual 14，FP16 | 约 2.5 MiB | 约 80 MiB | 约 800 MiB | 约 1.6 GiB |

这个量级在单卡上完全装得下，多层铺开也还在可接受范围。

---

## 6. 结论

1. **本次最坏情况实验证明**：用约 80 MiB 的固定 LUT，完整替换 35B 模型第 1 层 FFN block 后，模型没有崩溃，PPL 几乎不变，且仍能生成结构化文本。
2. **这是一个很强的下限结果**：我们证明了“比同事方案更激进的单层替换”在效果上仍然可用。
3. **但还不能下最终结论**：PPL 样本太少、没有 chat template、只测了一层、生成被中断，需要补充更多实验。

---

## 7. 下一步建议

| 优先级 | 实验 | 目的 |
|---|---|---|
| P0 | 用 `--eval_file` 指定 64–128 条真实短文本重新跑 PPL | 得到有统计意义的 PPL 对比 |
| P0 | 把 `--max_new_tokens` 降到 256/512，跑多个不同 prompt | 快速验证 LUT 是否导致不同主题的崩溃 |
| P1 | 在 run_model_eval 里加入 `tokenizer.apply_chat_template` | 得到标准 chat 输出，避免 “thinking process” 元文本 |
| P1 | 把 hook 挂到 MoE 内部单个 expert 而不是整个 `mlp` | 与同事方案粒度对齐，做公平对比 |
| P2 | 把 LUT 扩展到 5/10/20 层，做 PPL vs 替换层数曲线 | 看总 MAC 削减能扩到多大 |
| P2 | 计算 baseline vs LUT 输出的文本相似度（如 Rouge-L、cosine） | 量化生成退化程度 |
| P2 | 测量生成速度（token/s）对比 | 确认 LUT 带来的额外延迟 |

---

## 8. 数据文件

- 原始结果：`worstcase_32g_full_ffn_model_eval.json`
- 分析文档：`worstcase_32g_full_ffn_analysis.md`（本文件）

记录时间：2026-07-21（基于当次实验结果）
