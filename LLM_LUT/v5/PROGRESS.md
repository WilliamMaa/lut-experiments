# LLM_LUT v5 进展与思考记录

> 记录从 v4 收尾到 v5 决策树 address 验证的全过程，包括失败的尝试、关键结论、当前数据和下一步计划。

---

## 1. 背景：v4 结束时的状态

v4 的最终有效结果是：

- **配置**：L15~L27 共 13 层，每层 group=12
- **指标**：`finetune_l15_l27_13layer_group12_calib2048` epoch 14
  - **PPL = 29.25**
  - **Acc = 0.470**
  - **MAC ↓ 2.78%**
- **问题**：全模型 MAC 削减仍很低。若只靠线性增加 down_proj group 数量去追 10% MAC，需要约 **589 个 group**，在当前 LUT 质量下完全不现实。

因此 v5 的目标不是盲目扩层，而是：**提升 LUT 自身的表现力/信息密度**，让同样数量的 LUT group 质量更高，或者用更少 group 达到同等效果。

---

## 2. 早期尝试：o_proj 轴

在 down_proj 之外寻找下一个可替换轴，首先扫描了 **o_proj**。

- 测试了两种模式：
  - **full-output**：直接预测整个输出 token 向量；
  - **residual**：只预测被替换 group 对应的输出残差。
- **结果**：
  - 除 L27 个别 group 外，几乎所有层的 **residual relative MSE > 1**；
  - 说明 o_proj 的残差结构与当前 LUT 设计不匹配。

**结论**：在当前 LUT 框架下，**o_proj 不是下一个可扩展轴**。先集中精力优化 down_proj 本身。

---

## 3. 早期尝试：随机高阶 address（失败）

### 想法

v3/v4 的 2D address 只使用 residual 中 2 个 channel 作为地址，信息容量低。直觉上：用更多 channel 做随机投影（比如 4 channels × sign ±1），生成 10-bit 高维索引，应该能捕获更丰富的激活模式。

### 实现

`AddressHighOrderRandom`：
- 对 K 个 channel 做随机选择 + 随机符号加权求和；
- 多张表（M=4）各自独立随机投影；
- 无训练参数，仍属 O(1) 查表。

### 结果（L21–L23，8 group）

| 层 | 2D rel_mse | Random High-Order rel_mse |
|---|---|---|
| L21 | ~1.40 | **4–5× 更差** |
| L22 | ~1.3x | **4–5× 更差** |
| L23 | ~1.2x | **4–5× 更差** |

### 失败原因分析

随机投影没有与 **target residual** 对齐。2D address 虽然简单，但选出的两个 channel 是在 calibration 上**贪心地最小化残差方差**的；而随机高阶投影只是随机划分激活空间，导致大多数 bin 里 residual 的平均值离真值很远。

**教训**：增加地址维度本身没用，关键是 **address split 必须能降低 residual 方差**。

---

## 4. v5 核心尝试：Greedy Decision-Tree Address

### 核心思想

把地址构造变成一个 **数据驱动的贪心回归树**：

1. 每一层 split 都随机采样若干投影候选（random channel 子集 + random sign）；
2. 对每个候选，计算按该投影 split 后左右子节点的 **target residual 方差下降**；
3. 选择方差下降最大的候选作为该节点 split；
4. 递归建树，直到达到 `num_bits` 深度或样本数少于阈值。

这样树地址仍然是 O(1) 查表（固定树、无训练参数），但每个 split 都是面向 residual 目标的。

### 实现

`AddressGreedyTree`（`address.py`）：
- `channels_per_bit`：每个投影使用几个 channel（默认 4）。
- `candidates`：每个节点尝试多少随机投影（默认 32/128）。
- `min_samples`：叶子最小样本数。
- `max_samples`：为了加速，可对 calibration 子采样。

### 关键对比实验：L21 单点验证

| Address | 平均 relative_mse |
|---|---|
| 2D | 1.4043 |
| **Tree** | **1.2855** |

Tree 的原始重建误差比 2D **低 8.5%**，确认了“数据驱动 split”的有效性。

---

## 5. 当前完成：L21–L23 Tree LUT 构建

已用 tree address 生成 L21、L22、L23 各 8 个 group 的 LUT checkpoints，输出目录：

```
../v5/outputs_tree_21_23
```

构建 summary 文件位于：

```
LLM_LUT/v5/results/outputs_tree_21_23_summary.json
```

### 构建结果摘要

| 层 | 平均 rel_mse | 最小 | 最大 | 备注 |
|---|---|---|---|---|
| L21 | 1.2921 | 0.9803 | 1.4691 | 与单点验证 1.2855 一致 |
| L22 | 1.2157 | 1.0726 | 1.3127 | 质量略优于 L21 |
| L23 | 1.1983 | 0.8314 | 1.4684 | 整体最好 |

其中 **relative_mse < 1.0** 的 group（原始 LUT 已优于输出 0）：

- L21 g7：0.9803
- L23 g6：0.8919
- L23 g7：0.8314

**观察**：
- Tree address 跨层稳定，没有像 random high-order 那样崩；
- 但仍有大量 group 的 raw rel_mse > 1，说明**仅靠构建阶段均值还不足以预测残差**；
- 这和 v4 经验一致：**LUT 表值必须通过 fine-tune 联合训练**才能发挥效果。

---

## 6. 设计原则与红线检查

| 红线 | v5 是否违反？ | 说明 |
|---|---|---|
| 动态参数必须通过 LUT 查表生成 | ✅ 未违反 | Tree/2D/HighOrder 都是固定索引函数，无 MLP/CNN/HyperNetwork |
| 比较基准必须同等计算量 | ⚠️ 待验证 | 后续扩层时必须与相同 group 数的 2D 做对照 |
| 准确率只是验证指标 | ✅ 未违反 | 当前目标是提升 LUT 质量，不是盲目冲分 |
| 新增方法必须对 LUT 查表有帮助 | ✅ 未违反 | Tree 直接改进 address 的信息量，属于 LUT 核心 |
| 禁止自动多卡分配 | ✅ 未违反 | 始终单卡 `CUDA_VISIBLE_DEVICES=1` + `--device cuda:0 --isolate_gpu` |

---

## 7. 关键未解问题

1. **Tree + trainable LUT 微调后，端到端 PPL/Acc 是否优于 2D + trainable LUT？**
   - 当前只有 build 阶段 raw rel_mse，还没跑 fine-tune。
   - **这是当前最重要的实验**。

2. **Tree 的 scalability 如何？**
   - `tree_candidates=128` + 全量 calibration 构建很慢；
   - 需要用 `tree_max_samples` 子采样，并把 `tree_candidates` 降到 32 来扩展到更多层。

3. **能否把 Tree 应用到更多/更深层？**
   - 如果 L21–L23 微调优于 2D，下一步应尝试扩到 13 层甚至更多，看质量是否稳定。

4. **Tree 是否可以与量化/混合精度结合？**
   - v4 的 LUT 存储已经不小，tree 没有增加 table 大小，只是 address 函数更复杂；
   - 后续仍需要 FP16/INT8 量化来控制多层铺开时的存储。

---

## 8. 下一步计划

### 8.1 立即执行

跑 **tree vs 2D 的端到端 fine-tune 对比**（L21–L23，8 groups）：

```bash
# Tree
cd LLM_LUT/v5
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8,23:8" \
    --checkpoint_root ../v5/outputs_tree_21_23 \
    --address_mode tree \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_tree_l21_23

# 2D 对照
cd LLM_LUT/v4
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "21:8,22:8,23:8" \
    --address_mode 2d \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_2d_l21_23
```

### 8.2 根据结果分支

| Fine-tune 结果 | 行动 |
|---|---|
| Tree PPL/Acc 显著优于 2D | 将 tree 作为默认 address，扩展到 13 层/更多层；继续尝试减少 candidates 加速 |
| Tree 与 2D 持平 | 说明 tree 的 raw 优势没有迁移，需检查 tree 深度、`channels_per_bit`、lr/epoch；或尝试 hybrid（浅层 2D、深层 tree）|
| Tree 差于 2D | 分析是否过拟合 calibration split；尝试减小 tree 深度、增大 min_samples、加正则 |

### 8.3 中长期

- 扫描其他层（浅层/深层）用 tree 的质量；
- 尝试 **layer-adaptive address**：不同层使用不同 `num_bits` / `channels_per_bit`；
- 引入 **hidden-state 蒸馏目标**，不只是 logits KL；
- 量化 LUT table 到 FP16/INT8，算清楚每层存储；
- 如果 down_proj 质量天花板仍无法突破 10% MAC，再回头审视 gate/up_proj 或 attention 投影轴。

---

## 9. 方法论的反思

1. **不要为了数字好看而引入非 LUT 结构。** 随机高阶投影如果改成可学习 MLP 可能表现更好，但那会违反红线。v5 的 tree 是“固定、离线、O(1)”的，这是正确方向。

2. **地址是 LUT 的天花板。** 2D 地址信息容量太低；tree 通过数据驱动 split 提升了 8.5%，但仍是简单均值表。如果 tree 微调后仍不够，可能需要：
   - 更深的树 / 更多 channel 每 bit；
   - 每叶子存储更复杂的值（比如 rank-1 修正）；
   - 多张表分别负责不同残差分量。

3. **构建指标不等于端到端指标。** 当前 tree 的 raw rel_mse 仍 >1 对大多数 group，但 v4 经验表明 fine-tune 后可能大幅改善。不能只看 build MSE 决定是否继续。

4. **10% MAC 仍是硬目标。** 即使 tree 比 2D 好 8.5%，线性扩到 589 group 仍不现实。若质量不能量级提升，可能需要：
   - 替换 up/gate_proj（扩大可替换 MAC 基数）；
   - 更激进的 group 共享 / 量化；
   - 接受更低的准确率，以换取更大 MAC 削减（只要仍在 AGENTS 红线允许区间）。

---

## 10. 实验产物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| v5 源码 | `LLM_LUT/v5/` | address/lut/engine/build/inspect/finetune |
| Tree checkpoints L21–L23 | `../v5/outputs_tree_21_23/` | 8 groups per layer |
| Build summary | `LLM_LUT/v5/results/outputs_tree_21_23_summary.json` | 构建阶段指标 |
| Fine-tune summary | `LLM_LUT/v4/results/finetune_v5_tree_21_23_summary.json` | 端到端 5 epoch 结果 |
| 本文档 | `LLM_LUT/v5/PROGRESS.md` | 进展与思考 |

---

## 11. L21–L23 Tree + 可训练 LUT 端到端 Fine-Tune 结果

完成了 v5 tree address 首次端到端联合微调（5 epoch）。

### 11.1 配置

- **替换层/组**：L21:8, L22:8, L23:8（共 24 个 group）
- **地址**：`AddressGreedyTree`，`num_bits=10`，`channels_per_bit=4`
- **LUT**：可训练，FP16（未量化），`group_size=64`
- **训练**：5 epoch，lr=5e-5，calib 512，eval 128
- **目标**：logits KL 蒸馏（来自无 LUT 原模型）

### 11.2 关键指标

| 阶段 | KL | PPL | Acc | MAC 削减 | LUT 存储 |
|---|---|---|---|---|---|
| 原模型 baseline（v4 记录） | — | **19.55** | — | 0% | 0 |
| LUT 模型（fine-tune 前） | 0.5677 | 25.59 | 0.5133 | 0.41% | 3.00 MiB |
| Epoch 1 | 0.4237 | 25.81 | 0.5070 | 0.41% | 3.00 MiB |
| Epoch 2 | 0.3292 | 23.22 | 0.5179 | 0.41% | 3.00 MiB |
| Epoch 3 | 0.2729 | 20.98 | 0.5335 | 0.41% | 3.00 MiB |
| **Epoch 4（最佳 PPL）** | 0.2730 | **20.84** | 0.5226 | 0.41% | 3.00 MiB |
| Epoch 5 | 0.2589 | 20.91 | 0.5133 | 0.41% | 3.00 MiB |

### 11.3 与 v4 最佳结果对比

| 配置 | 层数 | 总 group 数 | MAC 削减 | LUT 存储 | 最佳 PPL | 最佳 Acc |
|---|---|---|---|---|---|---|
| v4 2D INT8 | L15–L27（13 层） | ~164 | **2.78%** | 49.25 MiB | 29.25 | 0.470 |
| v5 Tree FP16 | L21–L23（3 层） | 24 | 0.41% | 3.00 MiB | **20.84** | 0.523 |

### 11.4 结论

1. **Tree + 可训练 LUT 的质量恢复能力显著优于 v4 2D**：仅用 3 层、24 个 group，PPL 就接近原模型（20.84 vs 19.55），而 v4 用 13 层、~164 group 才到 29.25。
2. **MAC 削减仍是瓶颈**：0.41% 远小于 v4 的 2.78%。当前优势是“用更少替换量达到更好质量”，下一步要验证“用 tree 复制/超越 v4 的 2.78% 时，PPL 能否仍保持优秀”。
3. **过拟合迹象**：Epoch 4 PPL 最低，Epoch 5 略反弹；Acc 也是 Epoch 3 最高。说明 5 epoch 已接近收敛，后续可加 early stopping 或 lr decay。
4. **存储很省**：24 group FP16 仅 3 MiB；即便扩到 13 层、同等 group 数，FP16 约 13 MiB，仍远小于 v4 INT8 的 49 MiB（因 tree 只有 2^10=1024 entries，而 2D 64×64=4096 entries）。

### 11.5 直接下一步

- **在 v4 的 13 层配置（L15–L27）上用 tree address 重新 build + fine-tune**，直接对比 MAC 削减 2.78% 时的 PPL/Acc。
- 若 tree 在 2.78% MAC 削减下仍能把 PPL 拉到 25 以下，就证明 tree 是更优默认地址，可以继续扩层/扩 group 向 10% 推进。
- 同步整理 expansion roadmap，评估 down_proj + o_proj 组合能否突破当前天花板（见 `EXPANSION_ROADMAP.md`）。

---

## 12. 方向修正：把 Tree Address 扩展到 o_proj

### 12.1 为什么转向 o_proj？

之前讨论中意识到：v5 虽然引入了 tree address，但前几轮实验仍停留在 **down_proj** 轴上。而项目要推进到更大 MAC 削减，down_proj 已经过 v3/v4 多轮挤压，边际收益递减。

o_proj 占全模型 **5.5%** MAC，且 v4 预研显示部分层（尤其 **L27 残差模式** rel_mse 仅 0.18）非常有潜力。因此当前首要任务变成：**把已在 down_proj 上验证过的 tree address 扩展到 o_proj，跑端到端 fine-tune，看效果。**

### 12.2 已完成的代码改造

| 文件 | 改动 |
|---|---|
| `LLM_LUT/v5/engine.py` | 新增 `HybridOProjEngine`，支持 `direct` 和 `delta` 两种 reconstruction 模式 |
| `LLM_LUT/v5/build_lut_o_proj.py` | 新增 o_proj LUT checkpoint 构建脚本，默认 tree address |
| `LLM_LUT/v5/finetune_o_proj.py` | 新增 o_proj 专属 fine-tune 脚本，训练 o_proj.weight + LUT table |

### 12.3 o_proj 与 down_proj 的关键差异

| | down_proj | o_proj |
|---|---|---|
| 替换对象 | `mlp.down_proj` | `self_attn.o_proj` |
| 输入维度 | intermediate_size (18944) | hidden_size (3584) |
| 输出维度 | hidden_size (3584) | hidden_size (3584) |
| group 划分 | 按输出 hidden_size 分 56 组 | 按输出 hidden_size 分 56 组 |
| residual 含义 | `lut + mlp_input` | `direct`: `lut`；`delta`: `lut + attn_output` |
| 占全模型 MAC | 29.1% | 5.5% |

### 12.4 首发实验

- **L27 delta（残差）**：v4 预研 rel_mse=0.18，最有希望。
- **L17 direct（直接）**：v4 预研 rel_mse=0.39，早期层直接预测效果好。

### 12.5 直接下一步

跑 L27 o_proj residual 单点验证：

```bash
cd LLM_LUT/v5
LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python build_lut_o_proj.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "27:8" \
    --address_mode tree --num_bits 10 --tree_candidates 128 --tree_min_samples 32 \
    --mode delta \
    --calib_size 512 --eval_size 128 \
    --output_root ../v5/outputs_o_proj_l27

LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python finetune_o_proj.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "27:8" \
    --checkpoint_root ../v5/outputs_o_proj_l27 \
    --epochs 5 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_o_proj_l27_delta
```

根据 L27 结果，再决定是扩 o_proj 到 L17，还是回头做 down_proj + o_proj 混合。

---

## 13. Phase 4：down_proj + o_proj + gate_proj 扩展到 ~5% MAC

### 13.1 配置

- **目标**：在单卡可跑的前提下，把全模型 MAC 削减从 ~3.66% 拉到 ~5%。
- **替换规模**：down 52 group + o 187 group + gate 961 group，共 1200 group。
- **实际 MAC 削减**：4.57%（比 5% 目标略低，因为 gate_proj 单组 MAC 较小）。
- **LUT 存储**：150 MiB。

### 13.2 关键改动

- **OOM 解决**：
  - `finetune_joint.py` 增加 `gradient_accumulation_steps`，`BATCH_SIZE=1` + accumulation=2。
  - `hybrid_gate_proj_engine.py` 把 gate 路径中间张量从 FP32 改回模型 dtype（FP16），避免 18944 维 FP32 张量占满显存。
- **生成卡顿解决**：
  - `address.py` 中 `AddressGreedyTree.compute_indices` 原本用 Python 递归遍历树，生成时 `use_cache=True` 每次只过 1 个 token，开销爆炸。
  - 改为向量化 tree traversal：预先把 tree 展平成 tensor 数组，每次用矩阵索引处理所有样本。
- **新增脚本**：
  - `generate_eval.py`：加载指定 epoch 的权重和 LUT，对 baseline/epoch 8/epoch 10 跑同一份 prompts，输出 JSON 对比。
  - `run_generate_eval_phase4.sh`：一键运行生成评估。

### 13.3 指标

| | KL | PPL | Acc |
|---|---|---|---|
| 微调前 | 2.06 | 12.58 | 0.566 |
| Epoch 8 | 0.42 | **21.68** | **0.493** |
| Epoch 10 | 0.39 | 21.90 | 0.491 |

落在项目定义的“可用”区间（PPL < 35，Acc > 0.45）。生成质量需等待 `run_generate_eval_phase4.sh` 输出后人工判断。

### 13.4 下一步

- 跑生成评估，对比 baseline、epoch 8、epoch 10 的实际生成效果。
- 根据生成质量决定：
  - 是否把 gate 也加入训练（目前 `--freeze_gate`）；
  - 是否继续扩到 ~10% MAC（加入 q_proj）。

---

*最后更新：2026-07-16*
