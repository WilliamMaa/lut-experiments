# LLM_LUT v5 下一阶段实验计划

> 基于 `JOINT_STRATEGY.md` 的复盘，设计 Phase 1 隔离实验和 Phase 2 deployment-aware build 验证实验。

---

## 目标

1. 隔离 expansion v1 失败的主要原因。
2. 用逐层 hidden-state drift 诊断误差传播路径。
3. 验证 deployment-aware sequential build 是否能解决分布 mismatch。
4. 为后续规模化提供可信路径。

---

## Phase 1：隔离问题（使用现有 checkpoint）

全部使用 expansion v1 已经 build 好的 checkpoint：
- down_proj：`../v5/outputs_tree_l15_l27`（L15–L27，tree，candidates=32，max_samples=16384）
- o_proj：`../v5/outputs_o_proj_exp`（L15/L16/L17 direct + L27 delta）

### Exp A：Down-only recovery

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj L15–L27 | down_proj L15–L27 | 大规模 tree down_proj 本身是否能恢复 |

```bash
python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --checkpoint_root ../v5/outputs_tree_l15_l27 \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/phase1_down_only_l15_l27
```

### Exp B：O-only recovery

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| o_proj L15/L16/L17/L27 | o_proj L15/L16/L17/L27 | 该 o_proj 配置本身是否可恢复 |

```bash
python finetune_o_proj.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:8,16:8,17:8,27:8" \
    --checkpoint_root ../v5/outputs_o_proj_exp \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/phase1_o_only_l15_17_27
```

### Exp C：Down train + frozen-o

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj + o_proj | down_proj only | down_proj 能否补偿固定的 o_proj 扰动 |

```bash
python finetune_joint.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --freeze_o \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/phase1_down_train_o_frozen
```

### Exp D：O train + frozen-down

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj + o_proj | o_proj only | o_proj 能否补偿固定的 down_proj 扰动 |

```bash
python finetune_joint.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --freeze_down \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/phase1_o_train_down_frozen
```

### Exp E：Layerwise drift diagnostic

比较原模型与 joint-replaced 模型每层 hidden state 的相对误差：

```bash
python measure_layerwise_drift.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --down_checkpoint_root ../v5/outputs_tree_l15_l27 \
    --o_configs "15:8,16:8,17:8,27:8" \
    --o_checkpoint_root ../v5/outputs_o_proj_exp \
    --eval_size 128 --max_seq_len 512 \
    --output_json results/phase1_drift_joint_raw.json
```

输出每层：

```
E_l = ||h_l^S - h_l^T||_2 / (||h_l^T||_2 + ε)
```

如果 E_l 从 L15/L16 开始突然上升并逐层累积，就强支持 distribution cascade 假设。

### Phase 1 判定标准

| 结果组合 | 推断 |
|---|---|
| A 好，B 好，C 好，D 好 | 问题主要在 joint optimization，sequential build 可能帮助有限 |
| A 好，B 差 | o_proj 配置本身是瓶颈 |
| A 差，B 好 | down_proj tree build 质量是瓶颈 |
| C 差，D 好 | down_proj 难以适应 o_proj 扰动 |
| C 好，D 差 | o_proj 难以适应 down_proj 扰动 |
| E 中 E_l 从浅层逐层爆炸 | distribution cascade 是主因 |

---

## Phase 2：验证 deployment-aware build

### 目标

不要直接重做全部 13 层。选一个小但能暴露 mismatch 的配置：
- down_proj L18–L23
- o_proj L15–L17

比较三种 build 方式：
1. **Independent build**：当前方式，所有 LUT 在原模型上 build。
2. **Partial-aware build**：先 build o_proj L15–L17，install 后再 build down_proj L18–L23。
3. **Sequential build**：从 L15 开始，逐层按 `o_proj^(l) → down_proj^(l) → o_proj^(l+1)` 推进。

如果 sequential build 显著优于前两种，就证明逐层 build mismatch 是关键因素。

### 所需代码

- `build_lut_sequential.py`：支持在已有部分替换的 student 模型上逐层 build。
- 或者先手动实现：
  1. build o_proj L15 → install → forward → build down_proj L15 → install → …

### Phase 2 判定标准

| 结果 | 行动 |
|---|---|
| Sequential 显著优于 Independent | 确认 deployment-aware build 有效，开始规模化 |
| Partial-aware 接近 Sequential | 主要问题是 o_proj 对后续 down_proj 的影响，不必逐层到每个 down_proj |
| 三者接近 | 问题不在 build mismatch，应回到优化/目标函数/训练策略 |

---

## Phase 3：规模化（确认 Phase 2 有效后再执行）

采用 deployment-aware sequential build：

```
for l = 0 to L-1:
    build o_proj^(l) on current student
    install o_proj^(l)
    build down_proj^(l) on current student
    install down_proj^(l)
    optional: short recovery fine-tune
```

可选：
- 每加入若干层后做一次短 recovery fine-tune，避免误差累积。
- 配合局部 hidden-state / module-output distillation。

---

## 一键运行 Phase 1

已提供脚本：`run_phase1.sh`

```bash
cd /data/mingyu/LLM_LUT/v5
bash run_phase1.sh
```

该脚本会依次执行 Exp A–E。

---

## 新增/修改代码清单

| 文件 | 改动 |
|---|---|
| `finetune_joint.py` | 新增 `--freeze_down` / `--freeze_o` 参数 |
| `measure_layerwise_drift.py` | 新增：逐层 hidden-state drift 诊断 |
| `run_phase1.sh` | 新增：Phase 1 一键运行脚本 |
| `EXPERIMENT_PLAN.md` | 本文档 |

---

*创建时间：2026-07-10*
