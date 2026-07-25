# run_model_eval.py 使用说明

把 V6 LUT 替换接入完整 LLM 前向，对比 **Baseline（不替换）** 和 **LUT 替换** 的 PPL 与生成文本。

---

## 需要的输入

### 1. 完整模型（`--model_path`）

- Qwen3.6-35B-A3B 的本地路径或 HuggingFace model id。
- 例：`/data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B`
- 必须是 transformers 能直接加载的 causal LM 格式。

### 2. LUT checkpoint 目录（`--checkpoint_dir`）

- `build_lut_ffn_output.py` / `build_tail_aware_hard_correction.py` 生成的 `checkpoints/` 目录。
- 里面必须包含 `replacement_g{gid}.pt` 文件。
- 例：`./worstcase_32g_full_ffn/checkpoints`、`./outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints`
- 兼容三种 checkpoint 来源：
  - 原始 v6 单 LUT（`build_lut_ffn_output.py`）
  - v3 shared coarse + per-group residual（由 `build_tail_aware_hard_correction.py` 导出）
  - v4 tail-aware hard correction（`build_tail_aware_hard_correction.py`）

### 3. （可选）PPL 评估文本（`--eval_file`）

- JSONL 或纯文本文件，每行一条文本。
- 如果不提供，脚本用内置的 6 条 prompts 跑 PPL 和生成（只是示意，不是真实 PPL）。
- 建议从真实语料中切 64~128 条短文本。

---

## 环境依赖

```bash
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
```

依赖与 `build_lut_ffn_output.py` 一致：Python 3.10 + PyTorch 2.5.1 (cu124) + transformers + accelerate + numpy + tqdm。

---

## 运行示例

### 单卡（小模型调试）

```bash
cd LLM_LUT/v6
python run_model_eval.py \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --checkpoint_dir ./worstcase_32g_full_ffn/checkpoints \
  --layer_idx 1 \
  --device cuda:0 \
  --torch_dtype bfloat16 \
  --max_new_tokens 128 \
  --output_json ./worstcase_32g_full_ffn_model_eval.json
```

### 8 卡 35B 模型（推荐）

```bash
cd LLM_LUT/v6
python run_model_eval.py \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --checkpoint_dir ./worstcase_32g_full_ffn/checkpoints \
  --layer_idx 1 \
  --hook_path "model.model.layers[1].mlp" \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --max_new_tokens 128 \
  --output_json ./worstcase_32g_full_ffn_model_eval.json
```

> 注意：`--device_map balanced_low_0` 是 transformers 的显式映射策略，**不是** `auto`。脚本里也对 `auto` 做了禁止。

如果 `model.model.layers[1].mlp` 定位不到，请用 `print` 或 `model.named_modules()` 查看实际路径，然后传 `--hook_path`。

---

## 关键参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--model_path` | 模型路径或 HF id | 必填 |
| `--checkpoint_dir` | LUT checkpoint 目录 | 必填 |
| `--layer_idx` | 要替换 FFN 的层号 | 必填 |
| `--hook_path` | hook 模块定位表达式，例如 `model.model.layers[1].mlp` | `model.model.layers[layer_idx].mlp` |
| `--eval_file` | 评估文本（JSONL 或 txt） | 无，用内置 prompts |
| `--max_eval_samples` | 最多用于 PPL 的样本数 | 128 |
| `--max_new_tokens` | 生成最大长度 | 128 |
| `--max_length` | 输入截断长度 | 512 |
| `--device` | 单卡推理时使用 | `cuda:0` |
| `--device_map` | 多卡映射策略，如 `balanced_low_0` | 无 |
| `--torch_dtype` | 模型 dtype | `bfloat16` |
| `--output_json` | 输出结果 JSON 路径 | 无 |
| `--prompt` | 自定义生成 prompt，可多次传入 | 无 |
| `--verify_replacement` | 安装 hook 后验证 LUT 输出与原始 MLP 输出不同 | True |
| `--no_verify_replacement` | 跳过替换验证 | False |

---

## 输出说明

脚本会先后打印：

1. **Baseline（不 hook）** PPL 和 6 条生成。
2. **LUT（hook 替换）** PPL 和 6 条生成。
3. **Summary**：PPL 差值。

如果 `--output_json` 提供，会保存一个 JSON，包含：

- 模型路径、checkpoint 目录、层号、device_map
- `baseline_ppl`、`lut_ppl`、`ppl_delta`
- `baseline_generations` 和 `lut_generations` 的 prompt/output 对

---

## 注意事项

1. 35B 模型在 8 卡上加载时，`--device_map` 必须显式给，如 `balanced_low_0`。脚本内部会根据当前 hook 模块所在 GPU 把 LUT 表放到对应卡上，不会把所有表压到 `cuda:0`。
2. 32-group 完整替换会用 LUT 输出**覆盖**对应层 `mlp` block 的完整输出，因此模型下游接收到的不是原始 MLP 输出。脚本默认会做一次验证，打印 `Replacement verified`，确认 LUT 输出和原始 MLP 输出不同。
3. 但当前实现仍**先执行 MLP forward，再用 hook 覆盖输出**，所以并不能证明 FFN 矩阵乘法被真正跳过。要验证 MAC 削减，需要额外 monkey-patch 或直接替换模块 forward。
4. 如果生成直接崩溃或乱码，属于预期结果——本次实验就是要在“最坏情况”下测下限。
