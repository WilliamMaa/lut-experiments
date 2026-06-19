# LLM_LUT v4

v4 是 LLM_LUT 的独立版本，不依赖 v3 的代码，只复用 v3 已经生成的 checkpoint/summary 数据。

核心目标：

1. **多层联合微调**：让多个层的 `down_proj.weight` 同时适应 LUT 的存在。
2. **非均匀层分配搜索**：基于单层敏感度，给不同层分配不同 group 数，寻找 Pareto 最优配置。
3. **LUT 量化**：把 FP32 table 量化到 FP16/INT8，降低多层部署的存储压力。

## 目录结构

```
LLM_LUT/v4/
├── README.md                    # 本文件
├── data.py                      # 数据准备（来自 v0，v4 独立副本）
├── metrics.py                   # 评估指标（来自 v0，v4 独立副本）
├── partial_linear.py            # V3PartialEngine（来自 v3，v4 独立副本）
├── triton_kernels.py            # LUT fill 内核（来自 v3，v4 独立副本）
├── trainable_engine.py          # TrainableV3PartialEngine + 数据加载
├── partial_linear_quantized.py  # 支持 INT8 的 V4PartialEngine
├── finetune_multi_layer.py      # 多层联合 KL 微调
├── search_layer_configs.py      # 非均匀 group 分配搜索
├── quantize_lut.py              # INT8/FP16 LUT 量化工具
└── results/                     # 默认输出目录
```

> **独立性说明**：v4 的所有 `.py` 文件都在本目录内互相导入（`from data import ...`、`from metrics import ...` 等），不再通过 `sys.path` 引用 `v3/` 或 `v0/` 的模块。未来修改 v4 不会影响 v3/v0 的可复现性。
>
> v4 唯一依赖 v3 的地方是**输入数据**：per-group checkpoint（`v3/outputs/checkpoints/...`）和 per-layer summary（`v3/results/summaries/...`）。这些数据由 v3 的 `expand_ratio.py` 生成。

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0（CUDA 版本，用于训练/扫描）
- transformers
- tqdm
- triton（可选，用于加速 LUT fill；没有则自动回退到 PyTorch）
- 显存：建议 ≥ 24 GB（Qwen2.5-7B-Instruct FP16 约需 14 GB，加上 activations/grads 需要更多）

当前开发机器只有 CPU torch + 8 GB GPU，因此脚本以**代码实现**为主，实际训练需在更大显存环境执行。

## 多 GPU 隔离

v4 脚本会在 `import torch` **之前**根据 `--device` 自动设置 `CUDA_VISIBLE_DEVICES`，让进程只能看到目标 GPU，避免多卡切片/死锁 bug。例如：

```bash
python finetune_multi_layer.py --device cuda:1 ...
# 等价于 CUDA_VISIBLE_DEVICES=1 python finetune_multi_layer.py --device cuda:0 ...
```

脚本内部会把 device 规范化为 `cuda:0`。

## 前置数据

v4 假设 v3 已经生成了 per-layer checkpoint：

```
LLM_LUT/v3/outputs/checkpoints/l{layer}/g{count}/replacement_l{layer}g{gid}.pt
```

以及 per-layer summary：

```
LLM_LUT/v3/results/summaries/expand_ratio_l{layer}.json
```

如果没有，请先运行：

```bash
cd LLM_LUT/v3
python expand_ratio.py --model Qwen/Qwen2.5-7B-Instruct --layer 21 --output_root outputs
# 对 L19-L23 每层都跑一遍，或用 run_expand_and_scan.sh
```

## 用法

### 1. 多层联合微调（all_layers_half）

```bash
cd LLM_LUT/v4
python finetune_multi_layer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layers "19,20,21,22,23" --groups_per_layer 8 \
    --checkpoint_root ../v3/outputs \
    --epochs 3 --lr 1e-5 \
    --output_dir results/finetune_all_layers_half
```

或显式指定每层 group 数：

```bash
python finetune_multi_layer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --configs "19:8,20:8,21:8,22:8,23:8" \
    --checkpoint_root ../v3/outputs \
    --epochs 3 --lr 1e-5 \
    --output_dir results/finetune_all_layers_half
```

### 2. 非均匀分配搜索

```bash
python search_layer_configs.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_root ../v3/outputs \
    --summary_root ../v3/results/summaries \
    --output_path results/layer_search_pareto.json
```

这会基于 `v3/results/summaries/expand_ratio_l*.json` 中的敏感度生成候选配置并评估，输出 Pareto 表。

### 3. LUT 量化

#### 3.1 离线量化所有 checkpoint

```bash
python quantize_lut.py \
    --checkpoint_root ../v3/outputs \
    --output_root ../v3/outputs_quantized \
    --dtype int8
```

#### 3.2 用 INT8 LUT 做微调

```bash
python finetune_multi_layer.py \
    --layers "19,20,21,22,23" \
    --groups_per_layer 8 \
    --checkpoint_root ../v3/outputs_quantized \
    --lut_dtype int8 \
    --epochs 3 --lr 1e-5 \
    --output_dir results/finetune_all_layers_half_int8
```

## 设计要点

- **冻结除目标层 down_proj.weight 外的所有参数**：保持 backbone 不变，只让被替换的层学会适应 LUT。
- **KL 目标来自无 LUT 的原始模型**：微调后的模型输出分布尽量接近原模型。
- **多层误差会耦合**：联合微调比逐层微调更能捕捉层间耦合。
- **非均匀分配优于均匀分配**：根据单层敏感度给不同层不同 group 预算，通常能在同样 MAC 削减下获得更好质量。
- **INT8 LUT 先 dequantize 再查表**：当前 Triton kernel 只接受 float table，因此 INT8 路径在推理时先 dequantize 到 FP16，再走 LUT fill；存储和传输时是 INT8。

## 与 v3 的关系

| 能力 | v3 | v4 |
|---|---|---|
| 单层 LUT 替换 | ✅ | 独立副本 |
| 多层同时评估 | ✅ | 独立实现 |
| 多层联合微调 | ❌ | ✅ |
| 非均匀分配搜索 | 手动 | ✅ 自动 |
| LUT 量化 | 无 | ✅ FP16/INT8 |

v4 的代码可以独立演进；v3 保持原样不动。
