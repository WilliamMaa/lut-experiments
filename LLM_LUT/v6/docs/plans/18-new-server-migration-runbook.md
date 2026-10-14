# 18 - 新服务器 runbook：onpolicy 多层链重跑（L37→38→39）

> 目标：在新机器上单机重跑 onpolicy 多层实验，所有产物写到 `/data/mamingyu/`。
> 依据：`docs/plans/14-multilayer-plan.md`（原执行记录，路径根为 `/home/u/mmy/`，
> 本文已全部改为 `/data/mamingyu/`）+ `v8/docs/43-phase1-combination-runbook.md`（打包/serve）。
>
> 全文固定路径约定（不随终端变化，无需 export 任何东西）：
> - repo：`~/lut-experiments/LLM_LUT`
> - 模型：`/home/u/downloads/models`

---

## 1. 新机器现状检查

```bash
nvidia-smi                      # 确认 8×A800、驱动 550.90.07 同代、有无他人占用
df -h /data                     # 预留 150GB+（模型 ~70GB + 产物 ~20GB）
ls ~/lut-experiments/LLM_LUT/v6/scripts/data_collection/collect_shared_expert_data.py
conda env list                  # 应已有 lut_py310 / vllm_py310；没有则 §3 装
```

GPU 命令遵循 AGENTS.md 红线：显式指定单卡或固定卡集合，**严禁**
`device_map="auto"`；采集用 `balanced_low_0` 前先用 `CUDA_VISIBLE_DEVICES`
圈定要用的卡。

## 2. 模型权重：已在机器上，确认即可

`/home/u/downloads/models/`（无名目录）就是本项目的模型，2026-10-10 已核对
config.json：多模态 MoE、40 层 hybrid（每 4 层 1 个 full attention）、256 routed
expert top-8、1 个 shared expert、hidden 2048，与 Qwen3.6-35B-A3B 架构一致。

```bash
ls /home/u/downloads/models/config.json && du -sh /home/u/downloads/models   # 应 ~70GB
```

## 3. 环境：两个 conda env

```bash
# lut_py310：训练/采集/prep 用
conda create -n lut_py310 python=3.10 -y
conda activate lut_py310
pip install -r ~/lut-experiments/LLM_LUT/v6/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# vllm_py310：数值测试 + vLLM serve 用
conda create -n vllm_py310 python=3.10 -y
conda activate vllm_py310
pip install vllm==0.19.1 torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

验证（vllm_py310 里跑）：`python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"`

> torch 版本说明（2026-10-10 实机踩过）：Qwen3.6 的 `qwen3_5_moe` 架构只有
> transformers 4.5x 认识，而 4.5x 依赖 `torch.accelerator`（torch 2.6+ 才有）。
> cu124 索引最高只有 torch 2.6.0（2.7 起不再发 cu124 wheel），所以
> `v6/requirements.txt` 钉的是 `torch==2.6.0+cu124`，新装不会踩坑；
> 若已有 env 报 `AttributeError: module 'torch' has no attribute 'accelerator'`，
> 补一句 `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` 即可。

## 4. teacher 权重（lut_py310）

候选 prompt 池 `v6/candidate_prompts.jsonl`（1000 条）已在 repo 里，不用重建。
teacher 权重直接从模型抽（一次抽三层）：

```bash
conda activate lut_py310
mkdir -p /data/mamingyu/datasets
cd ~/lut-experiments/LLM_LUT
python -u v6/scripts/conversion/extract_shared_expert.py \
  --model_path /home/u/downloads/models \
  --layer_idx 37 --layer_idx 38 --layer_idx 39 \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l37.pt \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l38.pt \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l39.pt \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

## 5. 重跑链（顺序执行，一律 lut_py310；采集占满圈定卡，训练只占 cuda:0，采集与训练不要并行）

### 5.1 采 L37 数据（无替换）

```bash
cd ~/lut-experiments/LLM_LUT
python -u v6/scripts/data_collection/collect_shared_expert_data.py \
  --model_path /home/u/downloads/models --layer_idx 37 \
  --calib_file v6/candidate_prompts.jsonl \
  --output_dir /data/mamingyu/datasets/layer37_shared_expert_v3_onpolicy \
  --max_prompts 1000 --max_tokens_per_prompt 512 --max_new_tokens 512 \
  --max_total_tokens 1200000 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  > collect_l37.log 2>&1 &
```

### 5.2 训 L37 v3

```bash
python -u v6/scripts/training/build_lut_ffn_output_v3_shared_coarse.py \
  --teacher_weight_path /data/mamingyu/qwen_35b_shared_expert_l37.pt \
  --dataset_dir /data/mamingyu/datasets/layer37_shared_expert_v3_onpolicy/input \
  --output_dataset_dir /data/mamingyu/datasets/layer37_shared_expert_v3_onpolicy/output \
  --output_root /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy \
  --group_size 64 --group_ids "0-31" --coarse_num_bits 14 --residual_num_bits 16 \
  --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
  --calib_size 600000 --eval_size 69000 \
  --coarse_finetune_epochs 10 --residual_finetune_epochs 10 --finetune_epochs 50 \
  --finetune_loss_mode multi --device cuda:0 \
  > train_l37.log 2>&1 &
```

token 数不足时脚本会报错，按实际把 `--calib_size`/`--eval_size` 调小（参考 collect 日志里的总 token 数）。

### 5.3 转 L37 v4

```bash
python -u v6/scripts/conversion/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy/checkpoints \
  --output_root /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4 \
  --device cuda:0
```

### 5.4 采 L38 数据（L37 已替换）

同 5.1，改 `--layer_idx 38`、输出目录 `layer38_shared_expert_v3_onpolicy`、日志
`collect_l38.log`，并追加：

```bash
  --replace_layer_idx 37 \
  --replace_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints
```

### 5.5 训 L38 v3 → 5.6 转 L38 v4

同 5.2/5.3，全部路径 `layer37`→`layer38`，`--eval_size 50000`。

### 5.7 采 L39 数据（L37+L38 已替换）

同 5.1，`--layer_idx 39`、输出目录 `layer39_shared_expert_v3_onpolicy_multilayer`、
日志 `collect_l39_multilayer.log`，替换参数给两组（argparse 是 `action="append"`，重复传参合法）：

```bash
  --replace_layer_idx 37 \
  --replace_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints \
  --replace_layer_idx 38 \
  --replace_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer38_shared_expert_v3_onpolicy_as_v4/checkpoints
```

### 5.8 训 L39 v3 → 5.9 转 L39 v4

同 5.2/5.3，路径用 `layer39_shared_expert_v3_onpolicy_multilayer` 全套，
`--calib_size 1000000 --eval_size 100000`。**不要**复用旧单层
`outputs_ffn_lut_layer39_shared_expert_v3_onpolicy`（plan 14 §1.4）。

## 6. 打包 + serve（vllm_py310）

```bash
conda activate vllm_py310
mkdir -p /data/mamingyu/lut_bundles
cd ~/lut-experiments/LLM_LUT
for spec in \
  "37 outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4" \
  "38 outputs_ffn_lut_layer38_shared_expert_v3_onpolicy_as_v4" \
  "39 outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer_as_v4"; do
  set -- $spec
  python v6/scripts/conversion/prep_lut_bundle.py \
    --checkpoint_dir /data/mamingyu/$2/checkpoints \
    --output /data/mamingyu/lut_bundles/layer$1.pt
done

# 数值测试过了再 serve（kernel vs torch reference，atol=0.05）
python vllm_plugin/tests/test_lut_ffn.py

# 起服务：2 张卡（TP=2），用 CUDA_VISIBLE_DEVICES 圈定空闲卡（老机器约定 6,7）；
# 完整日志落盘，别用 grep 过滤启动输出（报错会被吞）
cd ~/lut-experiments/LLM_LUT/v8
CUDA_VISIBLE_DEVICES=6,7 \
V8_LUT_LAYERS=39 V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles \
python -m vllm_plugin.serve /home/u/downloads/models \
  --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
  --tensor-parallel-size 2 --max-num-seqs 4 --gpu-memory-utilization 0.88 \
  --port 18002 > logs/lut_l39.log 2>&1 &
grep -E "LUT shared_expert|KV cache size|Error|Traceback" logs/lut_l39.log
```

## 7. 验收

| 步骤 | 通过标准 |
|---|---|
| 5.1/5.4/5.7 | 日志无 raise；`/data/mamingyu/datasets/layer{L}_*/` 下文件数 ~1000 |
| 5.2/5.5/5.8 | `outputs_*/summary.json` 生成，eval cos 有值 |
| 5.3/5.6/5.9 | `*_as_v4/checkpoints/` 下 32 个 `replacement_g*.pt` |
| 6 | 三个 bundle 各 ~626 MiB；`test_lut_ffn.py` 全 OK |
| 6 serve | 日志出现 `LUT shared_expert layer 39 installed`；端口 18002 可访问 |
