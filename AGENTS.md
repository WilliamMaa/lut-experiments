# 项目核心目标与红线（Agent 必读）

## 1. 核心目标（不可偏离）

本项目的目的 **不是** 刷准确率。

> **目标：在存算一体（CIM）设备上，用 O(1) 计算量的 LUT（查表）方式，替代/辅助神经网络的部分计算，从而降低推理功耗与延迟。**

准确率只是验证"这种替代不会崩得太厉害"的指标，是手段，不是目的。

## 2. 我（Agent）犯的错误

### 错误 1：把 LUT 换成了 MLP
- 同事的原始代码和思路都是 **LUT 查表**（`LUT1_20Head_2D`、`LUT3_MultiScale_1D` 等）。
- 我在"端到端"阶段把 LUT 偷偷换成了 `nn.Linear(16, 128) → nn.Linear(128, 340)` 的 **MLP 生成**。
- MLP 虽然是固定维度，但它是矩阵乘法，计算量和功耗与静态层同级，**完全丧失了 LUT 查表的意义**。

### 错误 2：为了冲准确率不断扩张方法
- MNIST 上 LUT 两阶段做到 95.84%，端到端 MLP 做到 97.31%。
- 为了"突破"，我不断扩大方法（残差注入 backbone_main、更大的 query_proj、端到端联合训练）。
- 最终在 Fashion-MNIST 上跑 MLP 端到端，测试集 87.82%，还沾沾自喜，**完全忘了这是在验证一个和 LUT 无关的 HyperNetwork**。

### 错误 3：比较基准错误
- 正确的比较应该是：**"LUT 查表动态网络 vs 同等计算预算/功耗的静态网络"**。
- 我做的比较是：**"59K 参数的 MLP 动态注入 vs 13K 参数的静态基线"**，这在工程上毫无意义。
- 如果追求准确率，直接上 CNN 就能 99%，本项目的存在价值就是**在严格计算约束下找到最优解**。

## 3. 红线原则（以后任何实验必须遵守）

| # | 红线 | 违反示例 |
|---|------|----------|
| 1 | **动态参数必须通过 LUT 查表生成**，不能是 MLP/HyperNetwork/其他前馈网络 | `nn.Linear → ReLU → nn.Linear` 生成参数 |
| 2 | **比较基准必须是同等计算量/参数量**，不能拿大动态模型和小静态模型比 | 59K 动态 vs 13K 静态 |
| 3 | **准确率只是验证指标**，实验设计必须围绕"O(1) 查表加速"展开 | 为了涨 1% 准确率引入 O(N) 计算 |
| 4 | **新增方法必须先问：这对 LUT 查表有什么帮助？** | 不加思考地引入端到端 MLP |
| 5 | **禁止自动多卡分配**。模型必须显式加载到指定单卡；严禁 `device_map="auto"`、`accelerate` 等自动多卡切片/分配机制 | `from_pretrained(..., device_map="auto")` 导致模型切片到 0/1/2/3 所有 GPU，触发不可杀的 CUDA 死锁 |

## 4. 正确的实验方向

### 4.1 LUT 查表的定义
- 输入：一个低维 query 向量（如 16-dim 或 40-dim）。
- 输出：通过 **可微查表**（如多分辨率哈希表、多头 2D 表、多尺度 1D 表）直接索引得到参数残差。
- 计算量：与训练集大小无关，只与表的大小和插值次数有关，是 O(1)。

### 4.2 应该回答的问题
1. **LUT 查表在目标数据集上能做到多少准确率？**（基线 vs LUT 动态）
2. **LUT 表的存储开销有多大？**（KB 级还是 MB 级？）
3. **给定固定计算预算，LUT 动态网络是否优于纯静态网络？**（Pareto 曲线）
4. **LUT 查表在复杂数据（如 Fashion-MNIST）上是否仍有效？**

### 4.3 下一步的正确做法
- 把 **真正的 LUT 架构**（如 `LUT3_MultiScale_1D`）端到端联合训练到 Fashion-MNIST 上。
- 或者恢复两阶段流程：先训基线 → 冻结 → 用 LUT 查表替代 MLP 生成。
- **绝不**再引入 MLP/HyperNetwork 作为参数生成器。

## 5. 教训

> **当你发现自己在"调参冲准确率"的时候，停下来问自己：我是在验证核心假设，还是在刷数字？**

同事的方向是对的：LUT 查表、O(1) 计算、存算一体适配。
我的方向是错的：MLP 生成、端到端冲分、忘记硬件约束。

以后任何建议，先过一遍上面的 4 条红线。

---

# 当前阶段策略（v4 LLM_LUT，Agent 必读）

## 阶段目标

在 7B LLM（Qwen2.5-7B-Instruct）上验证 **多层 partial down_proj LUT 替换** 的可行性。

优先级：**MAC 削减 > LUT 存储 > 表现力别崩**。

## 核心原则

1. **优先多替换、多 layer**。目标是尽可能扩大全模型 MAC 削减比例，而不是保住 PPL/Acc。
2. **表现力崩了很正常**。当前方法把 dense 矩阵乘换成 LUT 均值查表，质量下降是预期结果。
3. **崩了之后用联合微调往上拉**，而不是回退到更小配置。
4. **LUT table 要小**。必须通过 FP16/INT8 量化控制存储，否则多层铺不开。
5. **key 必须来自已有输出**。当前方案用 residual 中挑选的 2 个 channel 作为 address，不引入额外计算。

## 判断标准（不是回到 baseline）

| 状态 | PPL | Acc | 行动 |
|---|---|---|---|
| 优秀 | < 30 | > 0.48 | 继续扩层数/group |
| 可用 | < 35 | > 0.45 | 继续扩，同时尝试量化 |
| 勉强可用 | < 45 | > 0.40 | 优化 lr/epoch/optimizer，不轻易放弃 |
| 不可用 | ≥ 45 或 Acc < 0.40 | — | 尝试更小替换量或改进 LUT 设计 |

> **不要以 baseline PPL 19.55 作为成功标准。** 只要 MAC 削减够大、表现力还在可用区间，就是有价值的结果。

## 当前正在执行的实验

- **9 层联合微调**：L17:8, L18:8, L19:8, L20:8, L21:12, L22:16, L23:16, L24:8, L25:8
- **目标**：验证 9 层、~1.56% MAC 削减在 10 epoch 联合微调后能否进入"可用"或"优秀"区间。
- **后续**：若成功，继续扩到 11/13 层或拉大 group；若失败，优先调 lr/epoch/optimizer，而非回退。

## 问题响应优先级

当指标下降/配置出现质量损失时，按以下顺序响应：

1. **承认事实**："PPL/Acc 确实下降了，这是替换的预期结果。"
2. **想办法改进**：
   - 增加微调 epoch
   - 调整 lr / optimizer / schedule
   - 改进 LUT 设计（更多 bins、更好的 address channel、残差补偿）
   - 上量化/压缩减少存储干扰
3. **尝试替代配置**：在同一规模内换非均匀分配、换层组合。
4. **最后才考虑回退**：只有当同一规模所有改进都试过且确实不可用时，才建议减小替换量。

**禁止**：把"回退到更小配置"作为第一个建议。项目目标是前进到更大 MAC 削减，不是保住 baseline 数字。


---

# v8 调试记录（2026-10-15，fact_acc 0.0156 排查）

## 已确认的事实（不要再重查）

1. **打包链路无辜**。`verify_lut_bundle.py pack`：L37/38/39 bundle 与 checkpoint 逐字段一致。
2. **权重 = 训练原样**。qual（公平 on-policy 链）cos 与训练 summary.json 自评严丝合缝：
   L37 0.75（训练 0.77）、L38 0.85（0.90）、L39 0.74（multilayer 0.80）。左尾烂是训练内在的
   （p10≈0.5），serve/转换没引入额外损伤。
3. **短上下文栈健康**：9 token prompt 下 LUT 输出与 dense 逐字级一致（三层全装）。
4. **64k 长上下文出现退化**：raw 补全下 LUT 有重复循环，dense 没有。
5. serve 环境约束：共享机唯一可用档位 `--gpu-memory-utilization 0.70`；
   `CUDA_VISIBLE_DEVICES=6,7` + `V8_LUT_LAYERS="37 38 39"` + `V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles`；
   模型路径必须写全 `/home/u/downloads/models/Qwen3.6-35B-A3B`（父目录没 tokenizer）；
   bench 参数是 `--concurrency` 不是 `--currency`；404 = 模型名和 serve 路径不一致。
6. triton 两个坑（已修）：kernel 体里的名字只从 `fn.__globals__` 解析（模块级 import，
   函数内 import 旧 triton 会炸 `NameError: tl`）；launch 必须在 `x.device` 的上下文里
   （跨卡时否则报骗人的 "cpu tensor?"）。

## 待验证的假设（用户的）

~~LUT 校准/训练用的 prompt 分布 vs bench 实际 prompt 分布错位。~~
**2026-10-15 被 gen 实验否掉**：32k 长文档×3 轮 QA，LUT 三轮全对（19千万元/王丽芳/30天，
均准确引用原文），短问答也全对。权重和分布在 32k 内都无辜（64k 双文档未测，但 8/512 是
全崩不像渐变）。嫌疑转移到 serve 运行时。

## 当前主嫌疑：vLLM serve 运行时（2026-10-15 已定罪）

同一 bundle 三层全装：HF 路径 32k QA 3/3，vLLM serve（64k 多轮）**0/16**。
关键推理：**0/16 是主动投毒不是 dense 回退**——插件失败会静默保持 dense（WARNING），
dense 输出是 baseline 质量（~0.734），不可能 0/16。所以 LUT 确实装着、确实在跑、跑得很错。
kernel 基本排除：test_lut_ffn 在 vllm_py310 下验证过 kernel ≡ reference（但那是小 n）。

投毒两类假设：
- **A. 与长度无关**：TP rank0 方案不成立（shared_expert 输出没被 all-reduce → rank 间状态分叉；
  或 is_tp_rank0() 两 rank 都 True → all-reduce 后翻倍）。
- **B. 与长度有关**：kernel 在 serve 大 n（64k prefill 一次前向几万 token）下有 bug，HF 只测到 32k。

定位靠实测不靠猜（probe 猜类别已废弃）：
1. **对照组**（控制实验，必须先做）：`bash tools/serve_debug.sh baseline 18004 "4,5"` 起无 LUT serve，
   eval --max-docs 1。到不了 ~0.73 → LUT 从头到尾无辜，之前全白测。
2. **插桩组**：`bash tools/serve_debug.sh lut 18002 "6,7"`（V8_LUT_DEBUG=1，插件前 4 次调用对真实
   输入同时算 dense(x) 和 LUT(x)）。`grep v8_debug logs/serve_lut_debug.log` 拿逐层 rank cos：
   cos ≈ HF 的 0.75~0.85 → LUT 给定输入是对的，毒在 rank 方案/下游；cos ≈ 0 → 输入/kernel 现形。

## 教训（时间红线）

- **不为零价值目标做战术勤奋**。为"严格复刻 bench 的 enable_thinking=False"绕了三轮
  （kwarg 不生效 → 手动插 /no_think → 撤回），而真正的阻塞只是 max_new 太短。
  动手前先问：这一步验证的是哪个假设？
- 长命令写进文件/脚本，chat 只贴短命令和结论。
- doc 43 是 Phase 1 唯一权威 runbook；v6 的 18 是迁移专用，不再加 serve/bench 内容。
- 排查顺序铁律：先 pack（秒级）→ qual 对训练自评（分钟级）→ gen 人眼看输出（分钟级），
  全 innocent 才轮到审计 serve kernel。
- **py_compile ≠ 验证**。本地跑不了的脚本（给服务器用的）必须在桌面上推演每个变量来源
  和边界后一次写完整；能用现有工具（curl/serve/bench）判读的，不造新 harness。
- **agent 不能在服务器执行，每轮试错都是用户买单**（同步+重跑+贴日志 10~30 分钟）。
  交付前自问：这一版能不能第一次就跑通？
