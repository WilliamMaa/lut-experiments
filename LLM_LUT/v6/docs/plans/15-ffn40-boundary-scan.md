# 15 - FFN 层数边界扫描计划（v6 → v8 eval 协议对齐；最大测点 24 层，40 层条件补测）

> 状态：待批准。本文档是 runbook：测什么、怎么测、每步命令、时间预算、判据。
> 前置结论（docs/41 修订版）：v6 不停。先把 FFN 替换从 3 层扩到 40 层，量出真正的
> 质量–覆盖–存储边界，再决定 routed experts 是否另开路线。

## 0. 实验问题

Qwen3.6-35B-A3B 全部 40 层的 shared_expert FFN 用 v6 方法（决策树地址 + 双级均值表）替换后：

1. 质量掉多少？（fact_acc + PPL，协议对齐 v8）
2. 替代计算占比多少？（算术：L 层 ≈ L × 3.15M MAC/token，40 层 ≈ 4%）
3. LUT 存储多少？（实测 MiB/层，逐层记录）

三个测点：**8 / 16 / 24 层**（嵌套子集，见 §2 D3）。40 层降级为条件补测：24 层过关后
再决定要不要补训剩余 16 层，不一次投入。

## 1. 关键成本事实（调研结论，决定本计划形态）

| 事实 | 数字 | 来源 |
|---|---|---|
| 建树+建表 | 6–15 h/层（单卡，calib 40万） | README_build_lut_ffn_output.md:395 |
| 采集 | 60万 token/层 ≈ 数小时（8卡） | 11-reflection.md、09-reflection.md |
| 单层表 | 320 MiB（FP16，14bit coarse + 16bit×32组 residual） | 三层 summary json |
| 三层质量 | PPL 6.075→9.123（+50%），L37 单层 delta≈0 | multilayer_l37_l39_5prompts_4096.json |
| eager LUT 速度 | ~54 ms/token/层（microbench 实测） | v8 docs/41 |
| v6 原 PPL 协议 | 3–5 条内置 prompt、512 截断 | run_multilayer_model_eval.py:33-40 |
| hook 机制 | 后置覆盖，**MAC 节省恒为 0** | v6_replacement_engine.py:226-245 |

**两个结构性结论：**

1. **训练是单卡的**（`--device cuda:0`），层与层之间完全独立 → 8 卡并行训练，
   37 层新训练 ≈ 5 批 × 6–15h ≈ **2–4 天墙钟**，而不是串行的 4–6 周。
2. **采集可以一次挂全部 40 层 hook**，一遍 rollout 采完所有层（现有脚本一次只支持
   一层，需小改，见 §5 改动 1）。采集从 37 遍降到 1 遍。

## 2. 方法论决策（已按推荐值定，有异议在此否决）

**D1. 采集：单遍全层、off-policy（不串 on-policy 链）。**
理由：on-policy 链是 39 级串行（训 L0 → 挂上采 L1 → …），与并行训练不兼容，
会把墙钟拉回到周级。代价：深层输入分布与真实部署有偏，40 层点的质量可能
被低估。扫描阶段接受此偏差；最终入选配置必须用 on-policy 重训复验。

**D2. 评测：v8 fact_acc 协议为主，PPL 为辅。**
- 主指标：v8 的 longctx 多轮 fact_acc（子串 AND 打分，逻辑照抄
  `v8/tools/eval_longctx_server.py`）。
- 数据：`v8/tools/gen_longctx_multiturn.py` 生成，**target 4k token、8 docs
  （64 题），固定 seed，全部测点共用同一份**（配对比较）。
- 辅指标：PPL（固定语料 32 seq × 512 token，HF `labels` 口径，全模型与
  各测点同语料对比）。
- 统计功效声明：64 题/点，±1 题 = ±1.6pt，相邻测点差 <5pt 不显著。
  PPL 作为更敏感的辅助判据。**这是 2.2 s/token 约束下买得起的最优组合，
  不是 v8 serving 规模的评测。**
- **必须保留逐题 correctness vector**（每题对/错，per-qtype 分组）。所有测点跑
  同一份数据 → 配对比较，报告 8→20→40 每一步**新坏哪些题、新好哪些题**。
  总 fact_acc 样本太少，逐题配对证据是主要分析材料，不允许只报聚合分。

**D3. 层集合（嵌套，折中版）：最大测点 24 层，不直冲 40。**
- 训练集合（24 层，Bresenham 均匀铺满 [0,39]）：
  {0,1,3,5,6,8,10,11,13,15,16,18,20,21,23,25,26,28,30,31,33,35,36,38}
- 评测点（同一批 checkpoint 的嵌套子集，**不额外训练**）：
  - 8 层：{0,5,11,16,21,26,31,36}
  - 16 层：{0,3,6,8,11,13,16,18,21,23,26,28,31,33,36,38}
  - 24 层：全部。
- 折中理由：40 层全训 = 5 批 × 6–15h ≈ 1.5–3.5 天；24 层 = 3 批 ≈ 1–2 天。
  24 层过关再补剩余 16 层（≈1 天）到 40；24 层崩则省掉全部 40 层投入（H3）。
- 采集仍一遍采全部 40 层（成本与单层相同），为将来补训留好数据。
- 旧产物 L37–39 checkpoint 不在本集合内（它们是相邻深层块，属于另一种配置形态），
  其价值是现成的对照锚点：已知"3 层相邻深层" PPL +50%，可与本扫描的"均匀散布"对比。

**D4. 层集合之外，表配置完全沿用 v6 三层验证过的参数**（group_size 64、
32 组、14+16 bit、tree 256 candidates），扫描期不改表设计——我们要量的是
**层数边界**，不是表容量边界。

## 3. 执行步骤

### 步骤 0：确认远端旧产物（0.5 h，手动）

```bash
ls /data/mamingyu/datasets/layer3{7,8,9}_shared_expert_v3_onpolicy/ | head
ls ~/lut-experiments/LLM_LUT/v6/outputs_ffn_lut_layer3*/checkpoints 2>/dev/null | head
```

- 数据集在（已确认 2026-10-08：`/data/mamingyu/datasets/`，L37–39 on-policy 齐全）
  → L0–36 新采集，L37–39 采集复用。
- `_as_v4/checkpoints` 在 → 训练只需 L0–36（37 层）；只有 v3 → 补 3 次转换；
  完全没有 → 训练 40 层。

### 步骤 1：抽 teacher 权重（全部 40 层，分钟级）

对每个新层跑 `scripts/conversion/extract_shared_expert.py --layer_idx L`。
（该脚本支持多层；输出 `qwen_35b_shared_expert_l{L}.pt`。）

### 步骤 2：单遍全层采集（8 卡，数小时）— 需改动 1

改动 1：`scripts/data_collection/collect_shared_expert_data.py` 支持
`--layer_idx` 传多层（如 `--layer_idx 0 1 2 ... 36`），hook 全部目标
shared_expert，输出按 `output_dir/layer{L}/input|output/sample_XXXXXX.pt`
分目录（training 脚本按层喂 `--dataset_dir`）。

```bash
python -u scripts/data_collection/collect_shared_expert_data.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --layer_idx 0 1 2 ... 36 \
  --calib_file candidate_prompts.jsonl \
  --output_dir /data/mamingyu/datasets/layers0_36_shared_expert_offpolicy \
  --max_prompts 1000 --max_new_tokens 512 --max_total_tokens 1200000 \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

### 步骤 3：并行建树建表（8 卡并行，1–2 天）— 需改动 2

改动 2：`scripts/training/run_all_layers.sh` 驱动脚本，
按 GPU 空闲把 24 个训练 run 派发上去（每 run `--device cuda:{g}`），失败重试、
日志按层归档。**不改训练脚本本身。** 层列表用 §2 D3 的 24 层集合：

```bash
LAYERS="0 1 3 5 6 8 10 11 13 15 16 18 20 21 23 25 26 28 30 31 33 35 36 38" \
DATA_ROOT=/data/mamingyu/datasets/layers0_39_shared_expert_offpolicy \
TEACHER_DIR=<teacher_dir> \
bash scripts/training/run_all_layers.sh
```

每层命令（与 14-plan §3 完全一致，仅换层号和路径）：

```bash
python -u scripts/training/build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path qwen_35b_shared_expert_l{L}.pt \
  --dataset_dir .../layer{L}/input --output_dataset_dir .../layer{L}/output \
  --output_root outputs_ffn_lut_layer{L}_shared_expert_offpolicy \
  --group_size 64 --group_ids "0-31" \
  --coarse_num_bits 14 --residual_num_bits 16 \
  --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
  --calib_size 600000 --eval_size 69000 \
  --finetune_epochs 50 --finetune_loss_mode multi --device cuda:{g}
```

再对每个新层跑 `convert_v3_to_v4_checkpoints.py`（分钟级）。

### 步骤 4：生成评测数据（一次性，分钟级）

```bash
python ../v8/tools/gen_longctx_multiturn.py --target-tokens 4096 --num-docs 8 \
  --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --out data/ffn40_eval_4k_8docs.jsonl
```

另备 PPL 语料（固定 32 段 × 512 token，写进 runbook 时确定来源）。

### 步骤 5：HF 离线 fact_acc + PPL 评测（需改动 3）

改动 3（新文件）：`scripts/evaluation/run_hf_factacc_eval.py`

- 加载模型（`balanced_low_0`, bf16），按 `--layer_idx`+`--checkpoint_dir`
  列表挂 v6_replacement_engine hook；
- 读 v8 格式 jsonl，应用 chat template，逐轮维护 `past_key_values`
  （**必须增量复用**，否则每轮全量重 forward，单 doc ~22h）；
- 打分逻辑逐行照抄 `v8/tools/eval_longctx_server.py`（gt 子串 AND）；
- 同一次 run 顺手算 PPL（固定语料，`labels` 口径）；
- 输出 JSON：`{fact_acc, n_correct, n_total, per_qtype, ppl, layers, table_mib}`。

测点矩阵（4 个配置 × 同一份数据）：

| 配置 | 层 | 预计单 doc 耗时 | 8 docs |
|---|---|---|---|
| full（baseline） | — | ~0.1 s/tok | ~1.5 h |
| 8 层 | D3 集合 | ~0.5 s/tok | ~6 h |
| 16 层 | D3 集合 | ~1.0 s/tok | ~12 h |
| 24 层 | 全部 | ~1.4 s/tok | ~17 h |

## 4. 结果表模板（每个测点一行）

| 配置 | 替换层数 | 替代 MAC/token | 表总 MiB | fact_acc (64q) | PPL | PPL vs full |
|---|---|---|---|---|---|---|
| full | 0 | 0 | 0 | | | 1.000 |
| 8 层 | 8 | 25.2M (~0.8%) | 2,560 | | | |
| 16 层 | 16 | 50.4M (~1.6%) | 5,120 | | | |
| 24 层 | 24 | 75.6M (~2.4%) | 7,680 | | | |
| 40 层（条件补测） | 40 | 126M (~4.0%) | 12,800 | | | |

（百分比以 ~3.1B active MAC/token 为分母；40 行只在 24 层过关后补训才填。）

## 5. 判据与出口

**硬规则（先于判据执行）：**

- **H1：off-policy 24 层崩 ≠ full-FFN LUT 判死。** off-policy 扫描中前层 LUT 替换会
  使后层输入分布 drift，误差随层数累积。若出现"8 层 OK / 16 层边缘 / 24 层崩"，
  下一步是挑 16 或 24 的一个候选做 **on-policy 复采复训**；on-policy 仍明显崩，
  才认定 coverage 边界真到了。
- **H2：存储不在扫描期前置否决。** 24 层 = 7.5 GiB，40 层 = 12.5 GiB；是否能
  由 v8 释放的 KV headroom 覆盖是下一道系统级算账，不是本扫描的否决项。
  若质量成立但存储最终被否，出口是表压缩（降 bit/减组/剪 leaf），不是砍层数。
- **H3：分阶段执行，不一次跑到底。** 采集后先训 8 层子集 → 评测 → 再补全 24 层
  → 评测。任一点明显崩可提前停（并行训练只在已决定扩展的集合内并行）。这是省
  GPU，不是追指标。

**出口判据：**

1. **24 层点 fact_acc 掉 ≤5pt 且 PPL 增幅 ≤25%** → 边界可接受。此时二选一：
   (a) 补训剩余 16 层到 40（+1–1.5 天），拿满 ~4% 覆盖数字；
   (b) 认为 ~2.4% 覆盖已够决策，直接进 fused kernel microbench（docs/41 的账用
   真实覆盖重算）。默认走 (a)，除非 24 层 PPL 增幅已接近 25% 的线。
2. **质量在 ≤16 层就崩** → 拿崩溃曲线和 FFN cosine/rel_l2 诊断，重启
   routed-expert 可行性议题（条件分布采集、热 expert 优先）。
3. **质量 24 层可接受** → 进入 H2 的系统级算账 + on-policy 复验，然后才到 kernel。
4. 所有质量结论附带 D1 的 off-policy 偏差声明；入选配置 on-policy 复验后才进
   kernel 阶段。

## 6. 不做的事（范围红线）

- 不改表设计、不改树算法、不改 hook 机制（本扫描只量边界，不救实现）。
- 不写 fused kernel（那是质量过关后的下一道门）。
- 不碰 routed experts（独立议题，等本扫描出口 2 才启动）。
- 不重开 serving/vLLM 线。

## 7. 时间预算总览（墙钟）

| 阶段 | 预估 |
|---|---|
| 步骤 0–1（确认产物 + 抽 teacher ×24 层） | <1 h |
| 步骤 2（单遍采集 40 层） | 数小时 |
| 步骤 3（24 层并行训练 + v4 转换，3 批） | 1–2 天 |
| 步骤 4（评测数据） | 分钟级 |
| 步骤 5（4 配置评测，串行） | ~1.5 天 |
| **合计** | **~2.5–4 天** |
| （可选）补训剩余 16 层到 40 | +1–1.5 天 |

## 8. 改动清单（已完成，待真机冒烟）

| # | 改动 | 文件 | 状态 |
|---|---|---|---|
| 1 | 多层采集（`--layer_idx` 收 list，多层输出按层分目录，单层布局不变） | scripts/data_collection/collect_shared_expert_data.py | ✅ 本机 py_compile 过；真机先 `--max_prompts 5` 小跑验证 |
| 2 | 8 卡并行训练驱动（派发/重试/日志/v4 补齐/汇总） | scripts/training/run_all_layers.sh | ✅ bash -n + 桩并发测试过；`LAYERS/DATA_ROOT/TEACHER_DIR` 必填 |
| 3 | HF 离线 fact_acc + PPL harness（chat template、past_key_values 增量、v8 子串 AND 打分、逐题 correctness vector） | scripts/evaluation/run_hf_factacc_eval.py | ✅ 本机 py_compile 过；**必须先 `--max_docs 1` 冒烟**（KV 增量前缀一致性、`generate` 返回 past 的行为只有真机能验） |

注意：改动 3 的 `--layer_idx/--checkpoint_dir` 是 `nargs="+"`（空格分隔），与旧脚本
`run_multilayer_model_eval.py` 的重复旗标风格不同，抄命令时注意。
