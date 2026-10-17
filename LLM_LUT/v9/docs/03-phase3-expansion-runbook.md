# v9 Phase 3 Runbook：扩层（37→39 扩到 33→39，7 层）

权威 runbook。前置结论：v9 三层 bench 0.9375 已过门槛（见 `02-experiment-record.md`），
判据按 doc 42 记账（Δ吞吐 / Δ训练成本），不是回 baseline。

模型是 40 层架构，最后一层就是 39，**不存在 40/41**。扩层只能从 37 往浅扩。

## 资源前提（先确认再动手）

- 共享机永远没有空 80G 单卡。模型 ~70GB，必须**显式**摊两张卡：
  `dual:N,M`（lw_common / extract_shared_expert 已支持，逐层对半分，红线 5 合法）。
- 训练子进程用第三张卡（`--train_device`），**父子不得共卡**（共卡会在训练子进程
  CUDA 析构时死锁——挂死那轮就是这个，产物其实已经齐了）。
- 开跑前 `nvidia-smi` 确认三张目标卡都空，卡号按实况填，不要照抄本文的数字。

## 第 1 步：提取新层 teacher 权重（占两张卡，一次性）

新层（33/34/35/36）各需要一个 teacher .pt；已训层（37/38/39）不需要重提：

```bash
cd ~/lut-experiments/LLM_LUT/v6
CUDA_VISIBLE_DEVICES=6,7 python -u scripts/conversion/extract_shared_expert.py \
  --model_path /data/mamingyu/models/Qwen3.6-35B-A3B \
  --layer_idx 33 --layer_idx 34 --layer_idx 35 --layer_idx 36 \
  --output_path /data/mamingyu/outputs/v9_teacher/teacher_l33.pt \
  --output_path /data/mamingyu/outputs/v9_teacher/teacher_l34.pt \
  --output_path /data/mamingyu/outputs/v9_teacher/teacher_l35.pt \
  --output_path /data/mamingyu/outputs/v9_teacher/teacher_l36.pt \
  --device_map dual:0,1 --torch_dtype bfloat16
```

（`dual:` 后面的卡号对应 `CUDA_VISIBLE_DEVICES` 暴露后的 0/1，按实况填。）

## 第 2 步：collect 扩层（三张卡：模型 dual 两张 + 训练一张）

**沿用同一个 output_root**（`/data/mamingyu/outputs/v9_layerwise_l37_39`），
37-39 的 phase0 / 训练 / 转换全部自动跳过，只做 33/34/35/36 的推进标注和训练：

```bash
cd ~/lut-experiments/LLM_LUT/v9
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/propagation/collect_layerwise_propagation.py \
  --model_path /data/mamingyu/models/Qwen3.6-35B-A3B \
  --prompt_file ../v6/candidate_prompts.jsonl \
  --layers 33,34,35,36,37,38,39 \
  --teacher_weights 33=/data/mamingyu/outputs/v9_teacher/teacher_l33.pt \
                    34=/data/mamingyu/outputs/v9_teacher/teacher_l34.pt \
                    35=/data/mamingyu/outputs/v9_teacher/teacher_l35.pt \
                    36=/data/mamingyu/outputs/v9_teacher/teacher_l36.pt \
  --output_root /data/mamingyu/outputs/v9_layerwise_l37_39 \
  --device_map dual:0,1 --train_device cuda:2 --max_len 2048
```

- 层列表**必须从浅到深排序**，推进语义要求按层序处理。
- 想扩更多层就把列表往浅续（32,31,...），只要小于 37 都是新层，需要 teacher .pt；
  记住标注成本随深度涨：浅层推进要经过更多层，但每层的标注都要从 phase0 边界
  重新推一遍，盯 `build_log.json` 里的 per-layer label_time_s 记账。
- 层数由脚本从 config / safetensors index 自己解析（`Qwen3_5MoeConfig` 没有
  `num_hidden_layers` 字段，已兼容），不用手填。

### 挂死预案（预期内，不是故障）

- 训练子进程打印 `[Final Evaluation]` / summary 后 **>10 分钟无输出** =
  CUDA 析构死锁，直接 `pkill -f build_lut_ffn` 杀掉该子进程；容错逻辑检测到
  v3 产物齐全会 warn 后继续下一层。
- collect 主进程全部层走完后自身退出时也可能挂死，**产物已齐**，直接杀即可，
  后续步骤不依赖它干净退出。

## 第 3 步：收尾（CPU，不占卡，命令与 01-phase2-runbook 第 1/2 步相同）

对每个**新层** L ∈ {33,34,35,36}：

```bash
python /home/mamingyu/lut-experiments/LLM_LUT/v6/scripts/conversion/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /data/mamingyu/outputs/v9_layerwise_l37_39/train_layer$L/checkpoints \
  --output_root /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer$L \
  --device cpu

python /home/mamingyu/lut-experiments/LLM_LUT/v6/scripts/conversion/prep_lut_bundle.py \
  --checkpoint_dir /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer$L/checkpoints \
  --output /data/mamingyu/lut_bundles_v9/layer$L.pt
```

七层目录下应有 7 × 32 = 224 个 `replacement_g*.pt`：

```bash
ls /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer*/checkpoints/replacement_g*.pt | wc -l
```

## 第 4 步：serve + bench（同 01-phase2-runbook 第 3/4 步，只改两处）

- `export V8_LUT_LAYERS="33 34 35 36 37 38 39"`（层数变了，其余环境变量不变）。
- bench `--out results/phase1_v9_l33_39_c16.json`（新文件名，别覆盖三层结果）。
- bench 数据若被删：按 01 第 3 步重新生成，**seed 必须默认 20260917**。

## 判读

- 核心判据只有 fact_acc 准不准；对照三层 0.9375 / v8 三层 0.9551 / baseline 0.9844。
- 七层 fact_acc 掉 2~4pt 以内 = 扩层没崩，继续；掉 >5pt 先看各层 train summary
  的 per-layer cos，区分深层标注质量问题与 LUT 容量问题，不要先回退层数。
- 吞吐记录 sess/h 记账即可，单次 ±8~10% 噪声，不做层间精细比较。
