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
