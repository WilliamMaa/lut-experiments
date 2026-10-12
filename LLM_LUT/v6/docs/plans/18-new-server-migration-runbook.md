# 18 - 新服务器迁移 runbook：onpolicy 多层链重跑（L37→38→39）

> 目的：把 onpolicy 多层实验从老机器（mamingyu@u，8×A800）迁到一台同配置新机器，
> 全部产物写到 `/data/mamingyu/`。
> 依据：`docs/plans/14-multilayer-plan.md`（原执行记录，路径根为 `/home/u/mmy/`，
> 本文已全部改为 `/data/mamingyu/`）+ `v8/docs/43-phase1-combination-runbook.md`（打包/serve）。

---

## 0. 变量约定

下文 `$MODEL` = 新机器上 Qwen3.6-35B-A3B 权重目录的绝对路径（见 §2 确定），
`$REPO` = 本仓库根（含 `v6/` 的目录）。

所有 GPU 命令遵循 AGENTS.md 红线：显式指定单卡或固定卡集合，
**严禁** `device_map="auto"`；采集用 `balanced_low_0` 前先用
`CUDA_VISIBLE_DEVICES` 圈定要用的卡。

## 1. 检查新机器现状

```bash
# 显卡
nvidia-smi                      # 确认 8×A800、驱动版本、有无他人占用
# 模型权重是否已存在（常见位置都查一遍；注意老机器权重可能没有模型名子目录，
# 直接躺在 downloads/models/ 根下，靠里面的 config.json 辨认）
ls -d /home/*/downloads/models/Qwen3.6-35B-A3B \
      /home/*/downloads/models \
      /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B 2>/dev/null
# 磁盘空间（模型 ~70GB，实验产物 ~20GB，预留 150GB+）
df -h /data /home
# 代码
ls $REPO/v6/scripts/data_collection/collect_shared_expert_data.py
# conda
conda env list                  # 应看到 lut_py310 / vllm_py310，没有则 §3 装
```

## 2. 模型权重：优先直接下载，不行再 rsync

模型是公开的 `Qwen/Qwen3.6-35B-A3B`（bf16，~70GB，多模态 MoE：40 层 hybrid、
256 routed expert top-8、1 个 shared expert、hidden 2048），新机器能联外网就直接下载，
比 rsync 省事（`hf` 自带断点续传）：

```bash
# lut_py310 或 vllm_py310 里装一次下载工具
pip install -U "huggingface_hub[hf_transfer]"

# 国内网络走镜像（可选但推荐）：
export HF_ENDPOINT=https://hf-mirror.com

hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir /data/mamingyu/models/Qwen3.6-35B-A3B \
  --local-dir-use-symlinks False
```

> ModelScope 备选（阿里云内网通常更快）：
> `pip install modelscope && modelscope download --model Qwen/Qwen3.6-35B-A3B --local_dir /data/mamingyu/models/Qwen3.6-35B-A3B`

下完校验再设路径：

```bash
du -sh /data/mamingyu/models/Qwen3.6-35B-A3B   # 应 ~70GB
export MODEL=/data/mamingyu/models/Qwen3.6-35B-A3B
```

**下载不可用（内网机/限速）时的回退**：从老机器 rsync（断点续传）。
注意（2026-10-10 已确认）：老机器 `mamingyu@u` 上的权重**直接放在
`/home/u/downloads/models/` 根下，没有模型名子目录**，但该目录的
config.json 已逐字段核对，与 Qwen3.6-35B-A3B 架构完全一致，就是本项目
一直在用的模型。rsync 源是这个目录本身：

```bash
rsync -avP mamingyu@u:/home/u/downloads/models/ \
  /data/mamingyu/models/Qwen3.6-35B-A3B/
```

> 老机器历史上还出现过 `/data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B` 的
> 写法（更早的机器），迁移时以实际 `ls` 到的为准。

## 3. 环境：两个 conda env

```bash
# lut_py310：训练/采集/prep 用（spec 见 v6/requirements.txt 头部注释）
conda create -n lut_py310 python=3.10 -y
conda activate lut_py310
pip install -r $REPO/v6/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# vllm_py310：数值测试 + vLLM serve 用（无安装脚本，按 v8/docs/28 规格手装）
conda create -n vllm_py310 python=3.10 -y
conda activate vllm_py310
pip install vllm==0.19.1 torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

验证：`python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"`
（vllm 只在 vllm_py310 里验证；driver 需 ≥ 550.90.07 同代，共享机不能动驱动。）

**已装好的 lut_py310 遇到 `AttributeError: module 'torch' has no attribute
'accelerator'`**（transformers 4.5x 需要 torch≥2.6，2026-10-10 实机踩过）：
只升级 torch 即可，不用重建 env——

```bash
conda activate lut_py310
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
```

## 4. 数据与 teacher 权重

- 候选 prompt 池 `v6/candidate_prompts.jsonl`（1000 条）**已在 repo 里，不用重建**。
- teacher 权重不拷贝，直接从模型抽（`extract_shared_expert.py` 支持一次多层）：

```bash
conda activate lut_py310
mkdir -p /data/mamingyu/datasets
python -u $REPO/v6/scripts/conversion/extract_shared_expert.py \
  --model_path $MODEL \
  --layer_idx 37 --layer_idx 38 --layer_idx 39 \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l37.pt \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l38.pt \
  --output_path /data/mamingyu/qwen_35b_shared_expert_l39.pt \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

## 5. 重跑链（顺序执行，一律 lut_py310；采集占满圈定卡，训练只占 cuda:0，采集与训练不要并行）

### 5.1 采 L37 数据（无替换）

```bash
cd $REPO && python -u v6/scripts/data_collection/collect_shared_expert_data.py \
  --model_path $MODEL --layer_idx 37 \
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

token 数不足时脚本会报错，按实际把 `--calib_size`/`--eval_size` 调小（参考 5.1 日志里的总 token 数）。

### 5.3 转 L37 v4

```bash
python -u v6/scripts/conversion/convert_v3_to_v4_checkpoints.py \
  --v3_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy/checkpoints \
  --output_root /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4 \
  --device cuda:0
```

### 5.4 采 L38 数据（L37 已替换）

同 5.1，改：`--layer_idx 38`、输出 `/data/mamingyu/datasets/layer38_shared_expert_v3_onpolicy`、日志 `collect_l38.log`，并追加：

```bash
  --replace_layer_idx 37 \
  --replace_checkpoint_dir /data/mamingyu/outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints
```

### 5.5 训 L38 v3 → 5.6 转 L38 v4

同 5.2/5.3，全部路径 `layer37`→`layer38`，`--eval_size 50000`。

### 5.7 采 L39 数据（L37+L38 已替换）

同 5.1，`--layer_idx 39`、输出 `/data/mamingyu/datasets/layer39_shared_expert_v3_onpolicy_multilayer`、日志 `collect_l39_multilayer.log`，替换参数给两组（argparse 为 `action="append"`，重复传参合法）：

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

## 6. 打包 + serve（切 vllm_py310）

```bash
conda activate vllm_py310
mkdir -p /data/mamingyu/lut_bundles
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

# 起服务：完整日志落盘，别用 grep 过滤启动输出（报错会被吞）
cd $REPO/v8
V8_LUT_LAYERS=39 V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles \
python -m vllm_plugin.serve $MODEL \
  --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
  --tensor-parallel-size 2 --max-num-seqs 4 --gpu-memory-utilization 0.88 \
  --port 18002 > logs/lut_l39.log 2>&1 &
grep -E "LUT shared_expert|KV cache size|Error|Traceback" logs/lut_l39.log
```

## 7. 回老机器取对照产物（可选）

老机器的中间产物（`onpolicy_layer39_v4_tail*`、`outputs_ffn_lut_*`、
`~/mmy/datasets/`）重跑链不依赖，但建议 rsync 到 `/data/mamingyu/` 留作对照：

```bash
rsync -avP mamingyu@u:'~/lut-experiments/outputs_ffn_lut_* ~/mmy/datasets ~/mmy/qwen_35b_shared_expert_l*.pt' \
  /data/mamingyu/from_old/
```

## 8. 验收

| 步骤 | 通过标准 |
|---|---|
| 5.1/5.4/5.7 | 日志无 raise；`datasets/layer{L}_*/` 下文件数 ~1000 |
| 5.2/5.5/5.8 | `outputs_*/summary.json` 生成，eval cos 有值 |
| 5.3/5.6/5.9 | `*_as_v4/checkpoints/` 下 32 个 `replacement_g*.pt` |
| 6 | 三个 bundle 各 ~626 MiB；`test_lut_ffn.py` 全 OK |
| 6 serve | 日志出现 `LUT shared_expert layer 39 installed`；端口 18002 可访问 |
