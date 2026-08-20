# LLM_LUT v8

v8 在 v6 FFN LUT 成果基础上并行开启两条新线：

1. **VQK-based Transformer 权重量化** (`vqk/`)
2. **KV Cache Compression** (`kv_cache/`)

两条线共用 `common/` 下的统一模型级 evaluation framework。

## 目录结构

```text
v8/
├── common/
│   ├── evaluator.py      # 统一 Evaluator（baseline + patch）
│   ├── metrics.py        # PPL、logit KL、top-k agreement、generation metrics
│   ├── prompts.py        # 默认 prompts 和 eval text 加载
│   └── utils.py          # 模型加载（device_map 安全检查）
├── vqk/
│   └── README.md         # VQK 实验入口说明
├── kv_cache/
│   └── README.md         # KV Cache Compression 实验入口说明
├── run_baseline_eval.py  # 基线模型评估（无 patch）
└── docs/
    └── 02-understanding-and-next-steps.md  # v8 工作理解与下一步
```

## 统一评估入口

所有实验最终都通过 `common.evaluator.Evaluator` 跑模型级评估：

```python
from v8.common.evaluator import Evaluator
from my_method import MyPatch

ev = Evaluator(
    model_path="/path/to/Qwen3.6-35B-A3B",
    device_map="balanced_low_0",
    torch_dtype="bfloat16",
    logit_metrics=True,  # 需要加载两份模型，计算 KL / top-k agreement
)

result = ev.evaluate(
    patch=MyPatch(...),
    texts=texts,
    prompts=prompts,
    max_length=512,
    max_new_tokens=256,
    output_json="results/my_method.json",
)
```

## 输出指标

每个实验至少输出：

- **模型质量**：PPL、Logit KL、Top-1 / Top-5 agreement、Generation quality、EOS success rate、Repetition rate
- **系统指标**：peak GPU memory、PPL compute time

后续 VQK 实验还会输出 weight memory、bytes read/token 等；KV Cache 实验会输出 KV bytes/token、retained attention mass 等。

## 红线

- **禁止 `device_map="auto"`**，必须使用显式 map（如 `balanced_low_0`）。
- 评估只看 **PPL / generation / logit KL**，不看 local cosine。

## 快速开始

### 1. 跑基线

所有 v8 脚本默认从 `LLM_LUT/v8/` 目录运行：

```bash
cd LLM_LUT/v8
python -u run_baseline_eval.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file eval.jsonl \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/v8_baseline.json
```

### 2. 实现 VQK / KV Cache patch

参考各子目录 README 实现一个 `EvalPatch` 子类，然后调用 `Evaluator.evaluate()`。
