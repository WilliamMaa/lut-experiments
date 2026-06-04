# LLM-LUT v0 — 服务器部署与运行指南

> 目标：在 Qwen2.5-0.5B-Instruct 上运行多层级敏感度扫描，识别适合 LUT 查表替代的组件。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0（CUDA 版本推荐，CPU 亦可跑通但慢） |
| GPU 显存 | ≥ 4GB（0.5B 模型 bf16 约需 2GB，留余量给激活） |
| 磁盘 | ≥ 5GB（模型缓存 + 结果） |
| 网络 | 首次运行需从 HuggingFace 下载模型（约 1GB） |

---

## 2. 安装依赖

在项目根目录（`LLM_LUT/v0/` 的父目录）执行：

```bash
# 建议先创建虚拟环境（可选，但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch transformers datasets tqdm scipy
```

如果你用 conda：

```bash
conda create -n llm_lut python=3.11
conda activate llm_lut
pip install torch transformers datasets tqdm scipy
```

---

## 3. 快速验证（Smoke Test）

在 **单张 GPU** 上跑一个最简扫描，验证 pipeline 无报错：

```bash
cd LLM_LUT/v0
python run_v0.py \
    --calib_size 64 \
    --eval_size 32 \
    --max_seq_len 128 \
    --batch_size 4 \
    --layer_ids 6
```

预期输出：
- 终端逐行打印 `ZERO -> KL=...`、`MEAN -> KL=...`、`BUCKET -> KL=...`
- 最后输出 `LLM-LUT v0 complete.`
- `results/` 目录下生成 `scan_results.json`、`rank_report.md`、`addr_stats.pt`

**耗时参考**：RTX 4090 上约 3-5 分钟；CPU 上约 20-30 分钟。

---

## 4. 正式扫描

### 4.1 最小完整版（IDEAS.md 推荐的 First-Round Scope）

```bash
cd LLM_LUT/v0
python run_v0.py \
    --calib_size 512 \
    --eval_size 256 \
    --max_seq_len 512 \
    --batch_size 8 \
    --layer_ids 6 12 18
```

| 参数 | 说明 |
|------|------|
| `--calib_size 512` | 校准集样本数，用于收集激活统计和建 bucket 表 |
| `--eval_size 256` | 评估集样本数，用于计算 KL/PPL/ACC |
| `--max_seq_len 512` | 截断长度，v0 固定 512，长文本延后 |
| `--batch_size 8` | 推理 batch size，根据显存调整 |
| `--layer_ids 6 12 18` | 扫描的层：early / middle / late |

**预期耗时**：RTX 4090 上约 40-60 分钟；A100 上约 20-30 分钟。

### 4.2 扩展版（Second-Round Scope）

第一轮有阳性信号后，扩大扫描范围：

```bash
python run_v0.py \
    --calib_size 1024 \
    --eval_size 512 \
    --max_seq_len 512 \
    --batch_size 8 \
    --layer_ids 3 6 10 12 14 18 21
```

---

## 5. 输出文件说明

运行结束后，`results/` 目录下会生成：

| 文件 | 内容 |
|------|------|
| `addr_stats.pt` | 校准统计量：每 (layer, type, group, head) 的地址通道索引、mean、std |
| `scan_results.json` | 原始扫描结果：每个 candidate 的 zero/mean/bucket KL、PPL、ACC、coverage |
| `rank_report.md` | 人工可读排名表：Top 50 候选 + 完整结果表格 |
| `rank_report.json` | 机器可读排名：含 sensitivity_penalty / addressability / final_score |

### 5.1 关键指标解读

在 `rank_report.md` 中，重点关注这几列：

- **KL Zero**：该 group 被抹零后的 logits KL。越小说明该 group 越不敏感。
- **KL Mean**：用 calibration mean 替换后的 KL。若显著低于 Zero，说明该 group 有稳定的偏置分量。
- **KL Bucket**：用 bucket lookup 替换后的 KL。若显著低于 Mean，说明该 group **可被 LUT 地址化**。
- **Coverage**：非空 bucket 比例。低于 50% 说明 calibration 数据不够，需要加量。
- **Addressability**：`(KL_mean - KL_bucket) / (KL_mean - KL_zero)`。越高说明 bucket 相对 mean 的优势越大。
- **Final Score**：`compute_saving + addressability - sensitivity_penalty`。越高越值得进入 v1 LUT 预训练。

### 5.2 阳性信号标准

一个候选值得进入 v1，需同时满足：

1. `KL_bucket < KL_mean × 0.8`（bucket 显著优于 mean）
2. `KL_bucket < 0.5`（模型级 KL 可控）
3. `bucket_coverage > 0.5`（表覆盖足够）
4. 替换该结构能移除真实稠密计算（down_proj / MLP 优先）

---

## 6. 使用自己的数据

默认使用代码内置的中英文 prompt 池生成 `data/calib.jsonl` 和 `data/eval.jsonl`。如需替换：

```bash
# 准备自己的数据，格式：每行一个 JSON {"text": "..."}
head -n 512 my_calib.jsonl > v0/data/calib.jsonl
head -n 256 my_eval.jsonl > v0/data/eval.jsonl
```

然后带 `--skip_calib` 运行（跳过内置数据生成，直接读取现有文件）：

```bash
python run_v0.py --calib_size 512 --eval_size 256 --max_seq_len 512 --batch_size 8 --layer_ids 6 12 18
```

> 注意：`data.py` 中的 `prepare_data()` 只在文件不存在时生成内置数据。如果 `data/calib.jsonl` 已存在，会直接复用。

---

## 7. 参数完整列表

```bash
python run_v0.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace 模型名 |
| `--calib_size` | 128 | 校准集样本数 |
| `--eval_size` | 64 | 评估集样本数 |
| `--max_seq_len` | 128 | 最大序列长度 |
| `--batch_size` | 2 | 推理 batch size |
| `--layer_ids` | `6 12 18` | 要扫描的层号列表 |
| `--skip_calib` | False | 跳过校准，复用 `results/addr_stats.pt` |
| `--skip_scan` | False | 跳过扫描，直接对已有 `scan_results.json` 重新排序 |
| `--result_dir` | `results` | 结果输出目录 |

---

## 8. 代码架构速览

| 文件 | 职责 | 对应 IDEAS.md 章节 |
|------|------|-------------------|
| `config.py` | 超参数、模型结构常量、hook 目标定位 | §5, §6 |
| `data.py` | 内置 prompt 池、jsonl 读写、TextDataset | §9 |
| `hooks.py` | `PerturbationHook`（zero/mean/noise/bucket 注入） | §7 |
| `calibrate.py` | 地址通道选择（variance + correlation proxy） | §6, §10.3 |
| `bucket.py` | Bucket table 构建（address → bin → avg） | §7.4, §10.6 |
| `metrics.py` | Local MSE / Cosine + 模型级 KL / PPL / ACC | §8 |
| `scan.py` |  orchestrates 逐层/逐类型/逐 group 扫描 | §10.5 |
| `rank.py` | 候选打分、Markdown/JSON 报告 | §10.7 |
| `run_v0.py` | 一键入口：加载 → 校准 → 扫描 → 排序 | — |

---

## 9. 从 v10 迁移的关键设计

| v10 (COCO/YOLO) | v0 (LLM/Qwen) |
|---|---|
| 1x1 Conv 输出通道 | MLP `down_proj` 输出 hidden groups |
| Spatial group (H×W) | Sequence-wise group (token 维度上的通道分组) |
| Multi-head address (per-spatial) | Multi-head address (per-token scalar) |
| `raw = x + alpha * delta` | `output_group += alpha * LUT(address)`（残差保持） |
| Phase0 prefit + Phase1 distill | v0 不训练，只用扰动 + bucket 评估敏感度 |

---

## 10. 常见问题

**Q: 第一次运行卡在 "Loading model..." 很久？**
A: 模型约 1GB，首次需从 HuggingFace 下载。设置 `HF_ENDPOINT=https://hf-mirror.com` 可换国内镜像。

**Q: CUDA out of memory？**
A: 降低 `--batch_size`（如 2 或 1），或减小 `--max_seq_len`（如 256）。

**Q: Bucket coverage 很低（< 20%）？**
A: 增加 `--calib_size`（512→2048）或减小 `--num_bins`（代码默认 64，已较保守）。如果仍低，说明该 candidate 的地址分布太分散，可能不适合 LUT。

**Q: 能否扫描 1.5B 模型？**
A: 代码兼容，只需改 `--model_name Qwen/Qwen2.5-1.5B-Instruct`。显存需 ≥ 6GB。

**Q: 如何只跑一种 candidate type？**
A: 目前需改 `config.py` 中的 `candidate_types`。后续版本可加命令行参数。

---

## 11. v0 → v1 的过渡条件

v0 成功后，进入 v1（实际 LUT 预训练）的条件：

1. 至少一个 candidate 满足 §5.2 的阳性信号标准。
2. `addr_stats.pt` 中的地址通道稳定（不同 calibration seed 下 addr_idx 基本一致）。
3. Bucket replacement 在 **多个 layer** 上均表现出 addressability > 0.3。

v1 将引入可训练的 `LLM_GroupLUTModule`（类似 v10 的 `SpatialGroupMultiHeadLUTDelta`），执行真正的 Phase0 prefit + Phase1 distill。
