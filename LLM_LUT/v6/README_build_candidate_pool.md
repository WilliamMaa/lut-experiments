# build_candidate_pool.py 使用说明

为方案 1 的 `collect_onpolicy_data.py` 构造 1000 条混合候选 prompt 池。

核心思想：
- **公开数据集提供真实任务原料**
- **固定模板把短问题改写成 500–2048 token 长输出任务**
- **按配比混合，输出统一 JSONL**

---

## 默认配比

| 来源 | 数量 | 目的 |
|-----|------|------|
| LIFEBench | 300 | 长文生成与长度约束 |
| LongGenBench | 150 | 多约束长生成 |
| Infinity-Instruct | 200 | 综合真实 instruction |
| 数学/推理 | 100 | 公式、分步推导、回溯 |
| 代码/技术 | 100 | 代码块、自然语言、符号交替 |
| Aya 多语言 | 100 | 中/英/日/混合 |
| 用户自定义 | 50 | 真实复杂任务 |
| **合计** | **1000** | |

---

## 安装依赖

```bash
pip install datasets
```

---

## 运行示例

### 默认：从所有公开数据集生成 1000 条

```bash
cd LLM_LUT/v6
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --n_total 1000 \
  --seed 42
```

### 加上用户自定义 prompt

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --custom_file ./my_complex_prompts.jsonl \
  --n_total 1000
```

### 跳过某些来源

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --no-use-code \
  --no-use-math \
  --n_total 800
```

### 调整各来源数量

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --n_lifebench 400 \
  --n_longgenbench 100 \
  --n_infinity 150 \
  --n_aya 150
```

---

## 自定义 prompt 文件格式

JSONL，每行一个：

```json
{"prompt": "分析罗马帝国衰落的多重原因", "language": "zh", "task": "history_analysis", "format": "essay", "target_length": 2048}
{"prompt": "Write a design doc for a distributed KV store", "language": "en", "task": "design", "format": "tech_report", "target_length": 2048}
```

如果 `prompt` 本身已经是长输出 prompt，可直接使用；如果太短，脚本会用模板自动改写。

---

## 输出 JSONL 格式

```json
{
  "prompt": "改写后的长输出 prompt",
  "source": "lifebench",
  "language": "zh",
  "task": "long_analysis",
  "format": "sectioned_essay",
  "target_length": 2048,
  "original": "原始 prompt"
}
```

直接作为 `collect_onpolicy_data.py --prompt_file` 的输入。

---

## 内置模板

- `multi_dim_analysis`：多维分析 + 因果关系 + 分章节
- `tech_report`：背景/需求/方案/实现/风险/测试
- `reasoning_selfcheck`：拆解条件 + 多思路 + 证明 + 边界检查
- `long_format`：8 个有实质内容章节 + 总结自检
- `cross_language_ja`：中文任务、日文回答
- `cross_language_en`：中文任务、英文回答
- `code_full`：需求/算法/代码/解释/测试/边界
- `math_full`：条件/目标/多思路/证明/边界/常见错误

---

## 数据源适配

脚本中内置的数据集名称和字段是**基于常见公开数据集的猜测**。远端跑的时候很可能会遇到数据集名称不对、字段名不同、网络/SSL 失败、需要 HF token 等情况。

### 远端 HF 访问不稳定时的推荐做法

默认行为是：**缺多少用 Aya 补齐**，保证最终仍能输出 `--n_total` 条 prompt。

**做法 A：只用 Aya（最稳）**

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --n_total 1000 \
  --n_aya 1000 \
  --no-use-lifebench \
  --no-use-longgenbench \
  --no-use-infinity \
  --no-use-math \
  --no-use-code
```

**做法 B：尝试所有来源，缺多少 Aya 补多少（默认行为）**

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --n_total 1000
```

**做法 C：提供本地 prompt 文件补齐**

```bash
python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --fallback_file ./my_local_prompts.jsonl
```

**做法 D：手动改 loader**

如果某个数据集名称/字段不对：

1. 修改 `build_candidate_pool.py` 中对应 loader 的 `load_hf_dataset(...)` 名称和 split/config；
2. 修改 `_collect_texts_from_dataset(...)` 里的候选字段名；
3. 脚本启动时会打印第一个 example 的 keys，方便你定位字段名。

优先保证能跑通，不要卡在找不到某个数据集上。

---

## 建议工作流

```bash
# 1. 建候选池
python build_candidate_pool.py --output_file ./candidate_prompts.jsonl --n_total 1000

# 2. 用 collect_onpolicy_data.py 跑方案 1
python collect_onpolicy_data.py \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints \
  --prompt_file ./candidate_prompts.jsonl \
  --output_root ./onpolicy_data_layer39_v4 \
  --layer_idx 39 \
  --hook_path "model.model.layers[39].mlp" \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --enable_medium_stage \
  > collect_onpolicy_v2.log 2>&1 &
```
