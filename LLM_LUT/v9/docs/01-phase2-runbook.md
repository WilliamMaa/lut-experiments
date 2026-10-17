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

## 第 3 步：serve + 64k bench（裸命令，不用 serve_debug.sh，配置全部可见）

serve（在 `~/lut-experiments/LLM_LUT/v8` 下跑；`python -m vllm_plugin.serve` 需要 v8 目录在
PYTHONPATH，cwd 在 v8/ 即可）：

```bash
cd ~/lut-experiments/LLM_LUT/v8

# 插件开关（这就是全部配置项）：
export V8_LUT_LAYERS="37 38 39"                 # 替换层
export V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles_v9  # bundle 目录
export V8_COMPRESS_SLOTS=512                    # KV 档位（0.70 util 对应）
unset V8_LUT_DEBUG                              # 正式 bench 必须关插桩

# 模型路径按服务器实况填；权重/tokenizer 不在同一目录时加 --tokenizer <目录>
MODEL=/data/mamingyu/models/Qwen3.6-35B-A3B

CUDA_VISIBLE_DEVICES=6,7 python -m vllm_plugin.serve "$MODEL" \
  --served-model-name "$MODEL" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-model-len 131072 \
  --tensor-parallel-size 2 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --port 18002
```

bench（另开终端，N=16；`--model` 必须和 `--served-model-name` 一致）：

```bash
cd ~/lut-experiments/LLM_LUT/v8
python tools/bench_concurrency.py \
  --base-url http://127.0.0.1:18002/v1 \
  --model /data/mamingyu/models/Qwen3.6-35B-A3B \
  --concurrency 16 \
  --data data/longctx_multi_turn_65536_64docs.jsonl \
  --out results/phase1_v9_c16.json
```

（对照文件 `results/phase1_l37l38l39_c16.json` 是 v8-LUT 的；baseline 0.9844 同配置实测。）

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
