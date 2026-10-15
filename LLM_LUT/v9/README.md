# v9：逐层推进多层 LUT 构建

承接 `docs/00-setup.md` 方向 1（最高优先级）：**校准文本固定、上游 LUT 不改，前缀只算一次**。
不再每加一层就重新 generate 整条文本，而是 teacher-forced 地沿层推进边界 hidden states。

复用 v6 代码（`sys.path` 直引，不拷贝）：replacement engine、v3 shared-coarse 训练脚本、
评估脚本、checkpoint/pickle 类定义全部来自 `LLM_LUT/v6/`，v9 只新增推进逻辑。

## 原理

batch=1、无 padding、完整序列上手动依次执行 decoder layers `[a, b)` 与完整 forward **精确等价**
（每层在位置 t 的输出只依赖 ≤ t 的 hidden states，causal attention 自洽，无需 KV cache）。

```
Phase 0   原始 pass：完整 forward 一次，缓存各替换层 ffn_in / residual_in
layer j   1. 原始 shared_expert 给 ffn_in 打标签 y（不重跑 LLM）
          2. v6 训练脚本训 LUT_j（或 --existing_ckpt 复用）
          3. 注入 LUT_j，在 residual_in[j] 上跑 slice [j, k]，缓存 ffn_in[k]
            —— 训练第 k 层时输入已经过真实 LUT（on-policy），但不再重跑前缀
```

v6 旧法成本 ≈ N × 完整 generate；本方案 ≈ 1 × 原始 pass + 各层少量 block 推进 + 局部训练。
红线遵守：显式 device_map（禁 auto）；LUT 查表替代，无 MLP/HyperNetwork 生成器。

## 文件

- `scripts/propagation/lw_common.py` — v6 路径接入、模型加载、边界捕获、decoder slice 执行
- `scripts/propagation/collect_layerwise_propagation.py` — 逐层推进构建主脚本
- `scripts/propagation/compare_rerun_vs_propagation.py` — 等价性小实验（00-setup.md 要求）
- `configs/layerwise_l37_39.json` — 三层（L37→38→39）示例配置

## 运行（服务器）

0. 准备各层 teacher 权重（一次性）：

```bash
python ../../v6/scripts/conversion/extract_shared_expert.py \
  --model_path /path/Qwen3.6-35B-A3B \
  --layer_idx 37 --output_path /data/outputs/v9_teacher/teacher_l37.pt \
  --layer_idx 38 --output_path /data/outputs/v9_teacher/teacher_l38.pt \
  --layer_idx 39 --output_path /data/outputs/v9_teacher/teacher_l39.pt
```

1. 等价性小实验（先跑这个；8 条 × 1024 token，几分钟）：

```bash
python scripts/propagation/compare_rerun_vs_propagation.py \
  --model_path /path/Qwen3.6-35B-A3B \
  --layers 37,38,39 \
  --lut_checkpoint 37=/path/outputs_l37/checkpoints \
  --prompt_file ../../v6/example_custom_prompts.jsonl \
  --n_seqs 8 --max_len 1024 \
  --output_json /data/outputs/v9_compare/compare_result.json
```

判定：`equivalent=true`（各深边界 cos_mean > 0.999，bf16 噪声量级）即推进与重跑语义一致。

2. 三层构建：

```bash
python scripts/propagation/collect_layerwise_propagation.py \
  --model_path /path/Qwen3.6-35B-A3B \
  --prompt_file ../../v6/candidate_prompts.jsonl \
  --layers 37,38,39 \
  --teacher_weights 37=/data/outputs/v9_teacher/teacher_l37.pt \
                    38=/data/outputs/v9_teacher/teacher_l38.pt \
                    39=/data/outputs/v9_teacher/teacher_l39.pt \
  --output_root /data/outputs/v9_layerwise_l37_39 \
  --device_map balanced_low_0 --max_prompts 256 --max_len 2048
  # 256 x 2048 ≈ 52万 token/层，覆盖默认 calib 40万 + eval 10万
  # （v6 训练脚本要求 calib_size + eval_size <= 采集总量，调大请同步加 --max_prompts）
```

产物：`boundary/`（边界缓存）、`dataset/layer{idx}/{input,output}/`（训练数据）、
`train_layer{idx}/checkpoints/`（LUT）、`build_log.json`（各阶段耗时）。

3. 评估（v6 脚本，三层 PPL 判据见 v6 docs/plans/14：<20 成功 / 20-40 可用 / >40 分析回退）：

```bash
python ../../v6/scripts/evaluation/run_multilayer_model_eval.py \
  --model_path /path/Qwen3.6-35B-A3B \
  --layer_idx 37 --checkpoint_dir /data/outputs/v9_layerwise_l37_39/train_layer37/checkpoints \
  --layer_idx 38 --checkpoint_dir /data/outputs/v9_layerwise_l37_39/train_layer38/checkpoints \
  --layer_idx 39 --checkpoint_dir /data/outputs/v9_layerwise_l37_39/train_layer39/checkpoints \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output_json /data/outputs/v9_layerwise_l37_39/eval_result.json
```

## 复用已训好的浅层 checkpoint（混合流程）

若 L37 已有 v6 训好的 checkpoint，第一层可跳过训练：

```bash
python scripts/propagation/collect_layerwise_propagation.py ... \
  --no_train --existing_ckpt 37=/path/outputs_l37/checkpoints
```

（`--no_train` 模式下所有层都必须给 `--existing_ckpt`；要逐层混合就分两次跑，第二次用
`--layers 38,39` 且从第一次的 `boundary/prop/layer38/` 继续——当前版本自动识别 prop 缓存。）

## 已知限制

- 校准为固定文本 teacher forcing，不覆盖 v6 on-policy 生成的 rollout 分布差异；
  该差距用第 3 步 PPL 与 v6 旧法对比量化，作为方向 2（A/B/C 消融）的决策依据。
- 校准 batch=1（无 padding），换取 mask/rope 的严格正确性；吞吐靠序列级并行不行，
  但单次 forward 很快，256 条 × 2048 token 的原始 pass 在 35B 上约数分钟级。
