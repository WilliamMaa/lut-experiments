# LLM_LUT v5 方法论：LUT 形成与微调

> 本文档总结 v5 中从敏感度扫描、LUT 构建、部署到联合微调的完整方法论，以及当前正在跑的 Phase 4 配置示例。

---

## 1. 项目目标与红线

### 1.1 目标

在存算一体（CIM）设备上，用 **O(1) 查表（LUT）** 替代/辅助大语言模型（LLM）中的部分线性投影计算，从而降低推理功耗与延迟。准确率只是验证“替代不会崩得太厉害”的指标，不是目的本身。

### 1.2 红线

| # | 红线 | 说明 |
|---|---|---|
| 1 | 动态参数必须通过 LUT 查表生成 | 禁止用 MLP / HyperNetwork / CNN 生成参数或地址。地址生成器必须是离线固定、无训练参数的。 |
| 2 | 比较基准必须是同等计算量/参数量 | 不能把大动态模型和小静态模型对比。 |
| 3 | 准确率只是验证指标 | 实验设计围绕“O(1) 查表加速”展开，而不是为了刷分引入 O(N) 计算。 |
| 4 | 禁止自动多卡分配 | 模型必须显式加载到指定单卡；严禁 `device_map="auto"` 等自动切片。 |

---

## 2. 核心概念：部分 LUT 替换

### 2.1 替换单元

对某个线性投影（`down_proj` / `o_proj` / `gate_proj` 等）的输出通道，按 `group_size`（默认 64）分成若干组。只替换其中部分组，其余组保留原始矩阵乘法。

- **down_proj**：输入 `intermediate_size` → 输出 `hidden_size`（Qwen2.5-7B: 18944 → 3584）。每组 64 个输出通道，权重 slice 为 `18944 × 64`，替换后节省 `18944 × 64` MAC/ token。
- **o_proj**：输入 `hidden_size` → 输出 `hidden_size`（3584 → 3584）。每组 64 个输出通道，权重 slice 为 `3584 × 64`，节省 `3584 × 64` MAC/ token。
- **gate_proj**：输入 `hidden_size` → 输出 `intermediate_size`（3584 → 18944）。结构与 `o_proj` 对偶，节省 `3584 × 64` MAC/ token。

### 2.2 计算什么被替换了

全模型主要线性投影 MAC（按每层计）：

```text
per_layer_total = 4 × hidden_size² + 3 × hidden_size × intermediate_size
full_model_total = num_layers × per_layer_total
```

单个 group 的 MAC 削减：

```text
down_group  = group_size × intermediate_size
o_group     = group_size × hidden_size
gate_group  = group_size × hidden_size
```

MAC 削减比例：

```text
ratio = Σ(saved_group_mac) / full_model_total
```

---

## 3. 整体流程概览

```text
敏感度扫描 ──→ 选择 group ──→ 部署感知构建 LUT ──→ 安装引擎 ──→ 联合微调 ──→ 评估/生成
```

| 阶段 | 脚本/文件 | 输出 |
|---|---|---|
| 敏感度扫描 | `scan_module_sensitivity.py` | `results/sensitivity_scan.json` |
| 部署感知构建 | `build_lut_sequential.py` | `outputs_*/checkpoints/l*/{down_proj,o_proj,gate_proj}/g*/` |
| 联合微调 | `finetune_joint.py` | `results/finetune_*/summary.json` + 每 epoch checkpoint |
| 生成评估 | `generate_eval.py` | `generation_*.json` |
| 重分配分析 | `analyze_redistribution.py` | 新配置字符串 |

---

## 4. 阶段 1：敏感度扫描与组选择

### 4.1 原理

对每个候选 group，在原模型上用 hook 把该 group 的输出替换为 calibration 均值，跑 eval 集，记录模型级指标变化：

- `delta_ppl`：PPL 上升量（负值表示下降）
- `delta_kl`：与 baseline 的 KL 散度增加
- `delta_acc`：next-token accuracy 下降量
- `mac_saved_per_token`：该 group 替换后节省的 MAC

按单位 MAC 带来的质量损失排序：

```text
score_ppl_per_mac = -delta_ppl / mac_saved_per_token
score_acc_per_mac = -delta_acc / mac_saved_per_token
```

### 4.2 用法

```bash
python scan_module_sensitivity.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --modules down_proj,o_proj,gate_proj,q_proj \
    --layers 15-27 \
    --group_size 64 \
    --calib_size 256 --eval_size 64 \
    --output_json results/sensitivity_scan.json
```

### 4.3 配置选择原则

1. **优先替换单位 MAC 质量损失低的 group**（score 高）。
2. **按 MAC 预算贪婪选择**，直到达到目标 MAC 削减比例。
3. **加每层上限**（例如 down≤15, o≤16, gate≤60），避免所有替换集中在同一层导致分布崩溃。
4. **重分配**：如果某个模块（如 gate）导致 OOM 或生成质量差，可以缩小其规模，从 sensitivity scan 中挑选其他模块（通常是 down_proj）的 group 补回 MAC。

---

## 5. 阶段 2：LUT 构建

### 5.1 LUT 结构

`LUTGroup`（`lut.py`）是一个可训练的查找表：

```text
shape: [num_tables, num_entries, group_size]
forward: 输入 indices [B, S, num_tables] → 输出 [B, S, group_size]
```

- `num_tables`：ensemble 表数，默认 1。
- `num_entries`：2^num_bits，默认 1024（num_bits=10）。
- `group_size`：输出通道组大小，默认 64。
- 初始化：每个 entry 取 calibration 中落入该 entry 的所有 target 向量的均值。

### 5.2 地址生成器

所有地址生成器都 **离线构建、无训练参数**，推理时是 O(1) 查表。

#### 5.2.1 Address2D

从输入中挑选 2 个通道，按均值/标准差归一化后分 bins，形成 2D 地址。`num_bins=64` 时 4096 entries。目前主要用于 down_proj 小规模实验。

#### 5.2.2 AddressHighOrderRandom

固定随机投影：每个 bit 随机选 `channels_per_bit` 个通道并随机赋 ±1 符号，做投影、标准化、阈值化，得到 B 位二进制地址。`num_bits=10` 时 1024 entries。

#### 5.2.3 AddressGreedyTree（默认）

离线贪心决策树：每个 split 从 `tree_candidates` 个随机投影中选一个，使 target 残差的方差下降最大。树深 `num_bits` 决定 2^num_bits 个叶子。这是当前默认，因为它在 build 质量和 LUT 存储之间最平衡：

- 单 group build MSE 比 2D 低约 8.5%；
- 1024 entries，存储只有 2D（4096 entries）的 1/4。

关键参数：

- `tree_candidates`：每节点候选随机投影数（默认 32）。
- `tree_min_samples`：节点最小样本数，小于则不分裂（默认 32）。
- `tree_max_samples`：build 时最多使用的 calibration 样本数（默认 16384），用于控制内存和速度。

### 5.3 部署感知构建（Deployment-Aware Sequential Build）

#### 5.3.1 为什么需要 sequential build？

如果所有 LUT 都在原模型上一次性 build 好，再一起 install，深层模块的 LUT 是在**原始输入分布**上构建的，但部署时看到的是**被前面所有替换扰动过的分布**。这会导致误差逐层累积（distribution cascade），大规模替换时 PPL 从 19 直接崩到 19 万。

#### 5.3.2 构建顺序

在同一层内，按 Transformer 正向顺序：

```text
o_proj → gate_proj → down_proj
```

全模型从浅层到深层：

```text
for l in all_layers:
    build o_proj(l) on current student
    install o_proj(l)
    build gate_proj(l) on current student
    install gate_proj(l)
    build down_proj(l) on current student
    install down_proj(l)
```

这样每个 LUT 都在**所有会影响其输入的前序替换已经部署后的 student 分布**上构建。

#### 5.3.3 脚本

```bash
python build_lut_sequential.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --down_configs "..." \
    --o_configs "..." \
    --gate_configs "..." \
    --address_mode tree --num_bits 10 --tree_candidates 32 --tree_min_samples 32 \
    --calib_size 512 --eval_size 128 \
    --output_root ../v5/outputs_phase4_down_o_gate_400g \
    --capture_batch_size 16
```

- `--resume`：可跳过已完成的 layer，但**配置改变后不要直接 resume**，否则已 build 的层与新配置不匹配。
- `--capture_batch_size`：控制 capture forward 的 batch size，降低峰值显存。

### 5.4 各模块构建细节

#### 5.4.1 down_proj

- **Target**：残差形式 `target = down_proj(x) - x`，即 LUT 预测 down_proj 输出与输入残差之间的差值。重建时 `output = x + lut`。
- **评估指标**：`relative_mse = MSE(lut+x, down_proj(x)) / Var(down_proj(x))`。

实现：`build_lut_sequential.py` 调用 `build_lut.py` 中的 `capture_mlp_residual` 和 `evaluate_group`。

#### 5.4.2 o_proj

- **Target 形式**：
  - `direct`：LUT 直接预测 `o_proj(x)`。
  - `delta`：LUT 预测 `o_proj(x) - x`，重建为 `x + lut`。深层 L27 经验上 delta 更稳定（rel_mse 可低至 0.18）。
- **评估指标**：`relative_mse = MSE(reconstruction, o_proj(x)) / Var(o_proj(x))`。

实现：`build_lut_sequential.py` 调用 `build_lut_o_proj.py` 中的 `capture_o_proj_residual` 和 `evaluate_group`。

#### 5.4.3 gate_proj

- **关键**：不能只优化 `|ĝ - g|²`，因为误差会经过 SiLU 和与 `up_proj` 的 element-wise 乘法。
- **Build target**：post-SiGLU 输出 `SiLU(gate) ⊙ up`。
- **LUT 初始化**：仍然用 pre-activation 的 `gate` 值初始化，因为 forward 中模型会自己施加 SiLU 和乘法。
- **评估指标**：`post-SiGLU MSE = MSE(SiLU(lut) ⊙ up, SiLU(gate) ⊙ up)`。

实现：`build_lut_sequential.py` 调用 `build_lut_gate_proj.py` 中的 `capture_gate_proj_residual` 和 `evaluate_gate_group`。

### 5.5 Checkpoint 格式

每个 group 保存一个 `.pt`：

```text
outputs_*/checkpoints/l{layer}/{module}/g{count}/replacement_l{layer}g{group_id}.pt
```

包含：

- `layer_id`, `group_id`, `group_size`
- `address_type`（2d / high_order / tree）
- 地址生成器状态（channel_idx、signs、threshold、tree_state 等）
- `lut_table`：可训练表值

---

## 6. 阶段 3：部署与推理（Engines）

### 6.1 Engine 机制

- `HybridPartialEngine`：替换 `mlp.down_proj`。
- `HybridOProjEngine`：替换 `self_attn.o_proj`。
- `HybridGateProjEngine`：替换 `mlp.gate_proj`。

每个 engine：

1. 用 `register_forward_pre_hook` 缓存模块输入并预计算所有被替换组的 bin indices。
2. 把模块的 `forward` 替换为 `_patched_forward`。
3. 在 `_patched_forward` 中：
   - 非替换组：用原始权重 slice 做正常矩阵乘。
   - 替换组：用预计算 indices 从 LUT 取值。
   - 把两部分拼回完整输出。

### 6.2 训练模式与推理模式

- 推理时：engine 安装后，被替换组就是 LUT 查表，其余组是矩阵乘，整体仍是固定计算图。
- 训练时：LUT 表值是 `nn.Parameter`，可以反向传播；若 `--lut_only`，则冻结所有原始投影权重，只训练 LUT。

### 6.3 安装与卸载

```python
engine.add_group(group_id, address, lut_group)
engine.install()   # 安装 hook + patched forward
engine.uninstall() # 恢复原 forward
```

---

## 7. 阶段 4：联合微调（Joint Fine-Tune）

### 7.1 目标

在保持原始模型大部分权重冻结的前提下，让被替换的 LUT（以及可选的原始投影权重）适应联合部署后的分布，使 student 的 logits 尽可能接近 teacher（原始模型）。

### 7.2 训练参数

`finetune_joint.py` 支持四种训练模式：

| 模式 | 可训练参数 | 说明 |
|---|---|---|
| 默认 | 被替换的原始投影权重 + 对应 LUT 表值 | 原始 down/o/gate 权重切片也参与训练 |
| `--lut_only` | 仅 LUT 表值 | 原始投影权重全部冻结，显存占用最小，最符合红线 |
| `--freeze_down` / `--freeze_o` / `--freeze_gate` | 指定模块冻结 | 例如只训练 gate 的 LUT 而冻结 down |
| `--gradient_accumulation_steps` | 保持等效 batch size | 减小瞬时激活显存 |

当前 Phase 4 使用：

```bash
python finetune_joint.py \
    ... \
    --lut_only \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --epochs 10 --lr 5e-5
```

### 7.3 损失函数

使用 KL 散度把 student 的 logits 拉近 teacher：

```python
log_probs = F.log_softmax(student_logits, dim=-1)
target_probs = F.softmax(teacher_logits, dim=-1)
loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
```

- 训练前会预计算 calibration 集上的 teacher logits，避免重复 forward teacher。
- eval 时使用 `compute_baseline_probs` 计算 eval 集上的 teacher 概率分布，在线计算 KL。

### 7.4 流程

1. 加载模型与数据。
2. 根据 `--down_configs` / `--o_configs` / `--gate_configs` 从 checkpoint 构建 engines。
3. 收集 eval 集上的 baseline probabilities（teacher）。
4. 安装所有 engines，做一次 pre-train eval。
5. 对每个 epoch：
   - 模型切到 train（被训练模块）/ eval（冻结模块）。
   - 对 calibration 集 forward，计算 KL loss，累积梯度。
   - 每 `gradient_accumulation_steps` 步做一次 `optimizer.step()` + `clip_grad_norm_(max_norm=1.0)`。
   - eval 集上计算 KL / PPL / Acc。
6. 保存每个 epoch 的原始权重和 LUT checkpoint。
7. 卸载 engines，输出 `summary.json`。

### 7.5 输出结构

```text
results/finetune_*/
  summary.json
  l{layer}_epoch{epoch}_down_proj.pt
  l{layer}_epoch{epoch}_down_lut/
  l{layer}_epoch{epoch}_gate_proj.pt
  l{layer}_epoch{epoch}_gate_lut/
  l{layer}_epoch{epoch}_o_proj.pt
  l{layer}_epoch{epoch}_o_lut/
```

---

## 8. 阶段 5：评估与生成

### 8.1 指标

| 指标 | 含义 | 目标区间（参考 AGENTS） |
|---|---|---|
| PPL |  perplexity | < 30 优秀；< 35 可用；< 45 勉强可用 |
| Acc | next-token accuracy | > 0.48 优秀；> 0.45 可用；> 0.40 勉强可用 |
| KL | 与 baseline 的平均 KL | 越低越好 |
| MAC 削减 | 全模型主要线性投影 MAC 削减比例 | 当前目标 ~5% |
| LUT 存储 | 所有 LUT 表占用的字节数 | 希望控制在几十 MiB 内 |

### 8.2 生成评估

```bash
python generate_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --down_configs "..." \
    --o_configs "..." \
    --gate_configs "..." \
    --checkpoint_dir results/finetune_joint_phase4_down_o_gate_400g \
    --epochs "8,10" \
    --prompts results/generation_prompts.txt \
    --max_new_tokens 50 \
    --baseline \
    --output generation_phase4_400g_epoch_8_10.json
```

- 生成 baseline（原模型）+ 指定 epoch 的文本输出，用于人工判断。
- 指标正常不等于生成质量好，必须以生成文本为准。

---

## 9. 当前 Phase 4 配置示例（gate=400）

在 `run_phase4_down_o_gate_400g.sh` 中：

```bash
DOWN_CONFIGS="..."  # 188 groups
O_CONFIGS="..."     # 167 groups
GATE_CONFIGS="..."  # 400 groups
```

构成目标：

| 模块 | 组数 | 每组 MAC | 总 MAC | 占比 |
|---|---|---:|---:|---:|
| down_proj | 188 | 1,212,416 | 228,014,208 | 3.19% |
| o_proj | 167 | 229,376 | 38,305,792 | 0.54% |
| gate_proj | 400 | 229,376 | 91,750,400 | 1.28% |
| **合计** | **755** | — | **358,070,400** | **5.01%** |

选择逻辑：

- gate 从 961 组降到 400 组，缓解中间激活和显存爆炸。
- 用 down_proj（高 MAC 密度）和 o_proj 补回 MAC，使总削减仍到 5%。
- 按 sensitivity scan 的 cost/MAC 排名贪婪选择，且每层上限：down≤15, o≤16, gate≤60。

---

## 10. 故障排查与调参

| 问题 | 可能原因 | 处理 |
|---|---|---|
| 联合微调 OOM | gate 替换过多 / optimizer state 过大 | 缩小 gate 规模；或加 `--freeze_gate`；或降低 batch size / 增加 gradient accumulation |
| 生成乱码/重复 | 替换比例过大 / 分布 cascade | 减少 gate；增加 down_proj 补偿；使用 sequential build；加更多 recovery epoch |
| build 阶段 OOM | capture batch 太大 | 减小 `--capture_batch_size` |
| PPL 仍高 | 训练不够 / 优化器参数不合适 | 尝试 cosine decay、更长 epoch、block-output anchor loss |
| count mismatch 报错 | 配置字符串里 count 和 ID 数量不一致 | 用 `analyze_redistribution.py` 或本地脚本校验 |

---

## 11. 文件速查

| 文件 | 作用 |
|---|---|
| `address.py` | 地址生成器：2D / high_order / tree |
| `lut.py` | 可训练 LUT 表 `LUTGroup` |
| `engine.py` | `HybridPartialEngine` / `HybridOProjEngine` |
| `hybrid_gate_proj_engine.py` | `HybridGateProjEngine` |
| `build_lut_sequential.py` | 部署感知 sequential build（o → gate → down） |
| `build_lut.py` | down_proj 构建辅助函数 |
| `build_lut_o_proj.py` | o_proj 构建辅助函数 |
| `build_lut_gate_proj.py` | gate_proj 构建辅助函数 |
| `finetune_joint.py` | 多模块联合微调 |
| `generate_eval.py` | 生成评估 |
| `scan_module_sensitivity.py` | 全局敏感度扫描 |
| `analyze_redistribution.py` | 在缩小某模块规模时重新分配 MAC |
| `metrics.py` | KL / PPL / Acc 计算 |
| `utils.py` | 模型与数据加载、baseline logits 收集 |

---

*基于 LLM_LUT v5 当前代码状态整理。*
