# v9 逐层推进实验记录

> 原载于 AGENTS.md，2026-10-16 迁入本文件。操作步骤见同目录 `01-phase2-runbook.md`。

## 方向 1 等价性小实验：已通过（2026-10-16，bit 级精确）

`compare_rerun_vs_propagation.py`（L37 装 as_v4 LUT，layers 37,38,39，8 条短 prompt）：
layer38/39 边界 cos_mean = 1.0000005 / 1.0000004，**abs_max = 0.0（bit 精确）**，
`equivalent=True`。推进语义与完整重跑等价，v9 前提成立。
结果：`/data/mamingyu/outputs/v9_compare/compare_result.json`。
checkpoint：`/data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints`（32 groups）。

## 第 2 步状态：三层训练已完成（2026-10-16），collect 管线不再重跑

挂死那轮实际已完成 L37/38/39 全部训练（train_layer{37,38,39}/checkpoints 各 33 文件），
卡死原因是训练子进程 CUDA 析构死锁（父子共卡；共享机无空 80G 卡，单卡/分卡方案都不可行，
不再尝试修复）。剩余收尾（L39 转换 → 打包 → serve bench）**不需要大卡**，
操作步骤见 `01-phase2-runbook.md`。

训练自评（train_layer*/summary.json）：L37 cos 0.7842 / L39 cos 0.8466（final eval）。
v9 表训练数据 ~20 万 token/层（v6 的 1/3），bench 解读记住这个 handicap。

## 已修的脚本 bug（勿回退）

推进/对比脚本（`compare_rerun_vs_propagation.py`）：
- `BoundaryCapture` 存的 `residual_in` 是 3 维 `[1,S,H]`，推进循环里再 `.unsqueeze(0)`
  会变 4 维在 linear_attn 炸掉；已去掉（collect 脚本 propagate() 同款提前修掉）。
- B 组推进链的 residual 分成 `residual`（原始 pass，只作首段起点）与 `prop_residual`
  （on-policy 推进产物，深层必须用这份），否则条数翻倍越界且语义错。

采集脚本（`collect_layerwise_propagation.py`）：
- `--max_prompts` 默认 256 只吃 prompt 文件前 1/4 → 改默认用满全文件；
  calib+eval 超采集量 → 按比例自动下调 + warn（不再白等一小时后报错）。
- 训练产物齐全自动跳过该层训练；engine 装前自动 v3→v4 转换（`ensure_v4_checkpoint`，
  转换产物缓存复用）；phase0 分片齐全自动跳过原始 pass（`_phase0_meta.json` 校验配置）。
- `load_model` 支持显式 JSON dict device_map（红线 5：禁 auto/auto-类字符串）。
- ~~`dual:N,M` 双卡映射~~ 已回退：homemade 层映射不对，模型加载恢复三层跑通过的
  `balanced_low_0` 方式（卡范围用 `CUDA_VISIBLE_DEVICES` 圈），不再折腾。

## serve bench：64 篇完整（2026-10-16）

三层全装（bundle `lut_bundles_v9/layer{37,38,39}.pt`），与 v8 Phase 1 完全同配置
（0.70 mem / max-num-seqs 4 / N=16 / 64 docs × 8 问 / max-model-len 131072）：

| 配置 | fact_acc | sess/h | wall |
|---|---|---|---|
| baseline dense | 0.9844（504/512） | 87.6 | 2629s |
| v8 LUT 三层（60 万 token/层 训练） | 0.9551（489/512） | 99.9 | 2306s |
| **v9 LUT 三层（20 万 token/层 训练）** | **0.9375（480/512）** | 101.4 | 2271s |

v9 用 1/3 训练数据交出与 v8 同档的 fact_acc（差 1.8pt，在 bench 噪声 ±2~3pt 内），
**Phase 3 门槛过**：v9 管线（便宜采集 + 逐层推进 + on-policy）方向成立，判据按
doc 42 记账（Δ吞吐 / Δ训练成本）。结果文件：`v8/results/phase1_v9_c16.json`。
注意：第一次 bench 只跑出 22 sessions（生成被截断，0.9148），已重生成完整数据后重跑，
以本次 0.9375 为准。扩层步骤见 `03-phase3-expansion-runbook.md`。
