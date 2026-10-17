# v9 Phase 2 Runbook：三层 LUT 收尾 → bundle → serve bench

权威 runbook。AGENTS.md 只记结论，本文记操作。前置状态见 AGENTS.md「v9 逐层推进实验记录」。

## 状态

- L37/38/39 训练全部完成：`/data/mamingyu/outputs/v9_layerwise_l37_39/train_layer{37,38,39}/checkpoints`（各 33 文件）。
- L37/38 v3→v4 转换已存在：`converted_layer{37,38}/checkpoints/`。
- collect 管线**不再重跑**（共享机无空 80G 卡，挂死不修）。

## 第 1 步：补 L39 转换（CPU，不占卡）

```bash
python /home/mamingyu/lut-experiments/LLM_LUT/v6/scripts/conversion/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /data/mamingyu/outputs/v9_layerwise_l37_39/train_layer39/checkpoints \
  --output_root /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer39 \
  --device cpu

# 验证：应输出 96（3 层 × 32 组）
ls /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer{37,38,39}/checkpoints/replacement_g*.pt | wc -l
```

## 第 2 步：打包 bundle（CPU，几分钟）

```bash
mkdir -p /data/mamingyu/lut_bundles_v9
for L in 37 38 39; do
python /home/mamingyu/lut-experiments/LLM_LUT/v6/scripts/conversion/prep_lut_bundle.py \
  --checkpoint_dir /data/mamingyu/outputs/v9_layerwise_l37_39/converted_layer$L/checkpoints \
  --output /data/mamingyu/lut_bundles_v9/layer$L.pt
done
```

注意：输出到独立目录 `lut_bundles_v9/`，**不得覆盖** v8 的 `/data/mamingyu/lut_bundles/`。

## 第 3 步：serve + 64k bench（与 v8 同一套）

serve（共享机 0.70 档位，参数与 v8 Phase 1 完全一致）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
CUDA_VISIBLE_DEVICES=6,7 V8_LUT_LAYERS="37 38 39" \
V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles_v9 \
bash tools/serve_debug.sh lut 18002 "6,7"
```

bench（另开终端，N=16，结果文件名带 v9 区分）：

```bash
python tools/bench_concurrency.py --port 18002 --concurrency 16 \
  --data data/longctx_multi_turn_65536_64docs.jsonl \
  --output results/phase1_v9_c16.json
```

（bench 参数以 `tools/bench_concurrency.py --help` 实际为准；对照文件 `results/phase1_l37l38l39_c16.json` 是 v8-LUT 的。）

## 判读

对照基准（同配置实测）：

| 配置 | fact_acc |
|---|---|
| baseline | 0.9844 |
| v8-LUT（60 万 token/层 训） | 0.9551 |

v9 表训练数据 ~20 万 token/层（v6 的 1/3），handicap 记在对解读里：

- v9 ≥ ~0.95：v9 成立，推进管线的表质量够用，可讨论扩层。
- 0.90 ~ 0.95：先做工况消融（对比 train summary 的 per-layer cos：L37 0.7842 / L39 0.8466），区分数据量问题与方法问题。
- < 0.90：方法或训练侧有问题，回 v9 数据链排查，不扩层。
