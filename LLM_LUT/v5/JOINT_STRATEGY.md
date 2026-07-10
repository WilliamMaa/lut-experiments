# Joint down_proj + o_proj 实验复盘与下一步策略

> 记录从“小尺度联合成功”到“大尺度联合崩掉”的全过程，提出部署感知（deployment-aware）build 的核心原则，并规划下一步隔离验证实验。

---

## 1. 已完成的联合实验

### 1.1 小尺度联合：down L21–23 + o L17（成功）

| 指标 | 原模型 | 联合替换未微调 | 联合微调 Epoch 5 |
|---|---|---|---|
| KL | — | 0.575 | **0.255** |
| PPL | 19.55 | 26.33 | **20.38** |
| Acc | 0.513 | 0.515 | 0.516 |
| MAC 削减 | — | 0.43% | 0.43% |
| LUT 存储 | — | 4 MiB | 4 MiB |

**结论**：替换量小的时候，down_proj 和 o_proj 可以共存，联合微调能把质量拉回原模型附近。

### 1.2 Expansion v1：down L15–L27 + o L15/L16/L17/L27（失败）

配置：
- down_proj：L15–L27 共 13 层，164 group（v4 非均匀配置）
- o_proj：L15/L16/L17 direct + L27 delta，共 32 group
- build：tree address，`candidates=32`，`max_samples=16384`
- fine-tune：5 epoch，lr=5e-5，KL 蒸馏

结果：

| 指标 | 原模型 | 联合替换未微调 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|---|---|
| KL | ~0 | 9.465 | 3.356 | 2.247 | 1.851 | 1.699 | 1.613 |
| PPL | 19.55 | **190954** | 433.2 | 82.9 | 51.6 | **51.3** | 52.5 |
| Acc | 0.513 | 0.002 | 0.298 | 0.389 | 0.437 | 0.432 | **0.443** |
| MAC 削减 | — | 2.89% | 2.89% | 2.89% | 2.89% | 2.89% | 2.89% |
| LUT 存储 | — | 24.5 MiB | — | — | — | — | — |

**结论**：
- 未微调时模型几乎完全崩掉（PPL 19 万，Acc 接近 0）。
- 5 epoch 联合微调后最好 PPL 51.35、Acc 0.443，仍未进入“可用”区间（AGENTS 标准：PPL<45 且 Acc>0.40 才勉强可用）。
- 作为对比，v4 用 2D address 在 2.78% MAC 削减下能做到 PPL 29.25、Acc 0.470。
- expansion v1 的 tree + o_proj 联合方案，**在 2.89% MAC 削减下表现不如 v4**。

---

## 2. 失败原因：主要假设与混杂因素

### 2.1 最值得优先验证的假设：逐层分布漂移（cumulative distribution mismatch）

当前流程：
1. 在原模型上分别 build down_proj LUT 和 o_proj LUT。
2. 把两者同时 install 到一个模型里。
3. 联合微调。

问题：
- 当 L15 的 o_proj 被替换后，L15 的 down_proj 输入分布已经变了。
- 当 L15 的 down_proj 再被替换，L16 的输入分布又变了。
- 以此类推，深层模块的 LUT 是在**原始分布**上 build 的，但部署时看到的是**被前面所有替换扰动过的分布**。
- 小尺度实验（L21–23 + L17）没暴露这个问题，因为替换从较深位置开始，中间层缓冲了大部分扰动。
- 大尺度里替换从浅层开始，错误逐层放大，导致 PPL 19 万。

**重要提醒**：这仍然是一个假设，尚未被实验隔离证明。Expansion v1 同时改变了很多变量，不能直接断定“分布漂移是唯一根因”。

### 2.2 尚未隔离的混杂因素

| 变量 | expansion v1 取值 | 小尺度/对照取值 | 影响 |
|---|---|---|---|
| 替换位置数 | 17 | 4 | 优化空间更陡 |
| tree candidates | 32 | 128 | 单点 build 质量可能下降 |
| tree max_samples | 16384 | 65536 | 子采样减少，tree 代表性下降 |
| address 类型 | tree | tree/2D | 未与 v4 2D 同配置对比 |
| 替换层范围 | L15–L27 + L15/L16/L17/L27 | L21–23 + L17 | 浅层替换对后续影响更大 |
| 总 MAC 削减 | 2.89% | 0.43% | 规模本身带来的难度 |

**结论**：目前只能确认“expansion v1 整体配置效果不如 v4”，不能单独证明 mismatch 是主因。

### 2.3 其他可能原因

- **一次性替换太多层，优化起点太烂**：5 epoch、lr=5e-5 从 PPL 19 万出发，很难收敛。
- **联合优化冲突**：o_proj 和 down_proj 的梯度可能互相拉扯。
- **目标函数只约束 logits**：深层 hidden state 漂移没有直接监督。
- **tree build 质量下降**：candidates 和 max_samples 降低可能让初始 LUT 更差。

---

## 3. 核心改进原则：部署感知 build（Deployment-Aware Build）

> **每一个后续 LUT，都应在所有会影响其输入的前序替换已经部署后的 student 分布上构建。**

这不是简单的“先 build o_proj，再 build down_proj”，而是**逐层、按模块顺序**推进。

### 3.1 单层的合理顺序

Transformer block 内：attention → o_proj → residual → MLP（gate/up/down）

因此同一层内如果同时替换 o_proj 和 down_proj，合理顺序是：

```
o_proj^(l) → down_proj^(l) → o_proj^(l+1)
```

### 3.2 全模型 sequential staged build 流程

```
for l = 0 to L-1:
    在当前 student 模型上 build o_proj^(l)
    install o_proj^(l)
    在当前 student 模型上 build down_proj^(l)
    install down_proj^(l)
```

这样每个 LUT 都基于“所有会影响它输入的前序替换已经生效”的分布。

### 3.3 与 naive 策略的对比

| 策略 | 做法 | 问题 |
|---|---|---|
| Naive | 原模型上 build 所有 LUT，再一起 install | 深层 LUT 看到错误分布 |
| Partial-aware | 先 build 所有 o_proj，install 后再 build 所有 down_proj | down_proj L16–L27 仍未看到 L15–L14 down_proj 替换的影响 |
| **Deployment-aware（推荐）** | 逐层按 o→down→o→down 顺序 build + install | 每个 LUT 都基于真实输入分布 |

---

## 4. 候选改进策略

### 4.1 Sequential Staged Build（最核心）

按上述 3.2 流程逐层 build 和 install。需要修改 build 脚本，支持：
- 接收一个已经 install 了部分引擎的 student 模型；
- 只 capture 指定模块在当前 student 分布上的输入/输出；
- build 完成后 install，继续下一层。

**优点**：从根本上消除 build-deployment mismatch。
**缺点**：build 流程变长；如果某一层 build 失败，会影响后续所有层。

### 4.2 控制实验：先隔离问题（最高优先级）

在投入大量工程做 sequential build 之前，先跑低成本对照实验，确认主因。

#### 实验 A：Tree down-only L15–L27

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj L15–L27 | down_proj L15–L27 | 确认大规模 tree down_proj 本身是否能恢复 |

```bash
python finetune.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12" \
    --checkpoint_root ../v5/outputs_tree_l15_l27 \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_down_l15_l27_only
```

#### 实验 B：O-only L15/L16/L17/L27

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| o_proj L15/L16/L17/L27 | o_proj L15/L16/L17/L27 | 确认该 o_proj 配置本身的破坏程度和可恢复性 |

```bash
python finetune_o_proj.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "15:8,16:8,17:8,27:8" \
    --checkpoint_root ../v5/outputs_o_proj_exp \
    --epochs 10 --lr 5e-5 --calib_size 512 --eval_size 128 \
    --output_dir results/finetune_o_l15_17_27_only
```

#### 实验 C：Down + frozen-o

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj + o_proj | down_proj only | down_proj 能否补偿固定的 o_proj 扰动 |

```bash
# 需要 finetune_joint.py 支持 --freeze_o 参数（待实现）
python finetune_joint.py \
    --down_configs "15:12,...,27:12" \
    --o_configs "15:8,16:8,17:8,27:8" \
    --freeze_o \
    --epochs 10 --lr 5e-5 ...
```

#### 实验 D：O + frozen-down

| 安装模块 | 可训练模块 | 目的 |
|---|---|---|
| down_proj + o_proj | o_proj only | o_proj 能否补偿固定的 down_proj 扰动 |

这三个实验能回答：
- 大规模 tree down 本身是不是主要瓶颈；
- o_proj 配置本身是不是主要瓶颈；
- 问题是否主要出在“二者联合优化”而不是单轴能力。

### 4.3 Progressive Build + Progressive Fine-Tune

不是简单地把预 build 好的 LUT 逐层 install，而是：

1. 从已收敛的小配置出发；
2. 在当前 student 分布上 capture 新模块；
3. build 新 LUT；
4. install；
5. 微调；
6. 继续扩展下一组。

**扩展顺序建议**：从浅到深（L15 → L16 → … → L27）。这样每次加入新层时，已经 build 好的深层 LUT 不会再因为更浅层的扰动而失效。

**反例**：从 L21–23 出发，向浅层加入 L20、L19……每次加浅层都会改变后面所有层的输入分布，导致已 build 的深层 LUT 再次 mismatch。

### 4.4 Hidden-State + Module-Output Distillation

在 KL logits 之外，加入被替换模块输出的归一化 MSE：

```
L = L_KL
  + λ_h Σ_{l∈R} ||h_l^S - h_l^T||_2^2 / (||h_l^T||_2^2 + ε)
  + λ_m Σ_{m∈M} ||y_m^S - y_m^T||_2^2 / (||y_m^T||_2^2 + ε)
```

其中：
- `h_l` 是 block-level hidden state；
- `y_m` 是被替换 projection（down_proj / o_proj）的输出；
- 使用归一化 MSE，避免不同层尺度差异；
- 不需要 cache 全部 teacher hidden states，可以 teacher/student 同时在线 forward。

**更重要的作用：诊断**。记录每层：

```
E_l = ||h_l^S - h_l^T||_2 / (||h_l^T||_2 + ε)
```

如果 E_l 从 L15/L16 开始突然上升并逐层累积，就强支持 distribution cascade 假设。

### 4.5 更长训练 / 更好优化器

- 增加 epoch 到 10–20；
- 加 lr decay / warmup；
- 尝试更大或更小的 lr。

这是一个低成本对照，但不应作为首要做法。如果根本问题是 mismatch，单纯加 epoch 救不回来。

### 4.6 LUT Entry 间固定权重插值

从 `build_mlp_lut.py` 借鉴“entry 之间插值”思想，但不用 MLP：
- 2D address → 双线性插值；
- tree address → 叶子边界平滑。

**优先级明确降低**。当前 PPL 19 万不是局部量化误差能解释的，插值更适合主流程恢复后做最后打磨。

---

## 5. 推荐执行顺序

### 第一阶段：隔离问题（1–2 天）

跑三个低成本对照实验：
1. **Tree down-only L15–L27**（实验 A）
2. **O-only L15/L16/L17/L27**（实验 B）
3. **Down + frozen-o**（实验 C）

同时实现并输出 **逐层 hidden-state drift 曲线**，用于诊断误差从哪一层开始爆炸。

### 第二阶段：验证 deployment-aware build（2–3 天）

先不要重做全部 13 层。选一个小但能暴露 mismatch 的配置：
- down L18–L23
- o L15–L17

比较三种 build 方式：
1. independent build（当前方式）
2. 先 o 后一次性 build down
3. **浅到深 sequential build**（推荐）

如果第三种显著优于前两种，就证明逐层 build mismatch 是关键因素。

### 第三阶段：规模化（3–5 天）

确认 sequential build 有效后，采用：

```
build/install layer l → short recovery → build/install layer l+1
```

必要时配合局部 feature distillation，而不是从 17 层灾难性起点做 20 epoch 全局训练。

---

## 6. 当前判断（更新版）

- **小尺度联合是可行的**：L21–23 down + L17 o 联合微调后 PPL 20.38，证明两轴不冲突。
- **大尺度联合失败的具体根因尚未被隔离**：distribution cascade 是最值得优先验证的假设，但 tree build 质量下降、替换规模扩大、优化起点恶化都是混杂因素。
- **当前最有技术价值的方向是 deployment-aware sequential build**：每个 LUT 都在前序替换已部署的 student 分布上构建，而不是在原模型上一次性 build 所有 LUT。
- **hidden-state distillation 既是改进手段，也是诊断工具**：能直接观察误差是否逐层累积。
- **在隔离实验完成之前，不应继续盲目扩大替换规模**。

---

## 7. 需要新增的代码/脚本

| 需求 | 说明 |
|---|---|
| `build_lut_sequential.py` | 支持在已有部分替换的 student 模型上，逐层 build down_proj / o_proj |
| `finetune_joint.py --freeze_o / --freeze_down` | 支持分阶段冻结/解冻两轴 |
| hidden-state drift 记录 | 在 eval/fine-tune 中输出每层 `E_l = ||h_l^S - h_l^T|| / (||h_l^T|| + ε)` |
| normalized module-output MSE loss | 在 fine-tune 中加入被替换 projection 输出的归一化 MSE |

---

*更新时间：2026-07-10*
