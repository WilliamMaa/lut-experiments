# LLM_LUT v8

v8 不再继续扩展 v6 的 FFN LUT，而是**在 v6 成果旁边新开两条独立实验线**：

1. **VQK-based Transformer 权重量化** (`vqk/`)
2. **KV Cache Compression** (`kv_cache/`)

三条线（v6 FFN LUT、v8 VQK、v8 KV Cache）共享同一个底层思想：

> **用小型离散表示替代/压缩神经网络中的密集张量，在降低推理开销的同时只保留功能真正需要的信息。**

但它们的**替换对象和优化目标完全不同**。

---

## v8 与 v6 的核心区别

| | v6 FFN LUT | v8 VQK | v8 KV Cache |
|---|---|---|---|
| **替换对象** | `mlp.shared_expert` 的输出激活 | `self_attn.o_proj` 等 Linear 的**权重矩阵** | attention 的 K/V cache |
| **替换层级** | 替换整个子层的**计算结果** | 替换 Linear 的**权重存储/精度** | 替换 attention 历史状态的存储 |
| **核心操作** | 用 LUT 查表生成 FFN 输出 | 把 FP16/BF16 权重 `W` 变成低 bit `W_q` + block-wise scale `S` | 量化 / 驱逐 / codebook 压缩 K/V |
| **优化目标** | 减少 FFN **计算量 (MAC)** | 减少 Linear **权重表示精度 / 存储** | 减少 KV **显存占用和读取带宽** |
| **形式化** | `y_LUT = table[addr]` | `W ≈ S ⊙ W_q`（运行时反量化为 BF16 再 GEMM） | `KV_i = f(KV_i)` |

### 具体区别：FFN LUT vs VQK

**v6 FFN LUT（已完成）：**

- 把 `layer 39 mlp.shared_expert` 这个**前馈网络子层**替换成 LUT。
- 原 `shared_expert` 做矩阵乘法 + SiLU + gate，计算量大。
- LUT 用 tree 结构直接查表得到 2048 维输出，**绕过矩阵乘法**。
- 目标是**减少 FFN 推理时的 MAC 数量**。

**v8 VQK（当前实验）：**

- 把 `layer 39 self_attn.o_proj` 这个**线性层的权重**替换成低 bit 表示。
- `o_proj` 本身的计算仍然是 `y = x @ W.T`，但权重被表示为 `W ≈ S ⊙ W_q`。
- `W_q` 是 4-bit 整数，`S` 是每个 block 一个 FP16 scale；**运行时反量化回 BF16 再做普通 GEMM**。
- 当前阶段验证的是：**这种低 bit + block-wise scale 的权重表示，能否比简单 RTN INT4 更好地保持模型质量**。
- 目标不是立刻做 INT4 GEMM 加速 kernel，而是先验证**权重表示本身**是否有优势。

**一句话总结：**

> v6 是**换激活 / 换计算**；v8 VQK 是**换权重表示精度**，计算图本身没变，当前也不承诺计算收益。

---

## 目录结构

```text
v8/
├── common/
│   ├── evaluator.py           # 统一 Evaluator（baseline + patch）
│   ├── metrics.py             # PPL、logit KL、top-k agreement、generation metrics
│   ├── prompts.py             # 默认 prompts 和 eval text 加载
│   ├── utils.py               # 模型加载（device_map 安全检查）
│   ├── example_patch.py       # EvalPatch 接口示例
│   └── generate_eval_texts.py # 从 prompt 生成 PPL eval 文本
├── vqk/
│   ├── vqk_linear.py          # VQK Linear 实现
│   ├── standard_quant.py      # RTN INT 量化 baseline
│   ├── vqk_patch.py           # VQKPatch（替换/恢复指定 Linear）
│   ├── eval_vqk.py            # VQK 评估入口
│   ├── summarize_vqk_results.py
│   └── README.md              # VQK 实验说明
├── kv_cache/
│   └── README.md              # KV Cache Compression 实验说明
├── run_baseline_eval.py       # 基线模型评估（无 patch）
└── docs/
    ├── 02-understanding-and-next-steps.md
    └── 03-vqk-plan.md
```

---

## 统一评估入口

所有实验最终都通过 `common.evaluator.Evaluator` 跑模型级评估：

```python
from common.evaluator import Evaluator
from vqk.vqk_patch import VQKPatch

ev = Evaluator(
    model_path="/path/to/Qwen3.6-35B-A3B",
    device_map="balanced_low_0",
    torch_dtype="bfloat16",
    logit_metrics=True,  # 需要加载两份模型，计算 KL / top-k agreement
)

patch = VQKPatch(
    layer_idx=39,
    module_path="self_attn.o_proj",
    bits=4,
    block_size=64,
    quant_method="vqk",
)

result = ev.evaluate(
    patch=patch,
    texts=texts,        # 长文本，算 PPL
    prompts=prompts,    # 短 prompt，跑生成
    max_length=512,
    max_new_tokens=256,
    output_json="results/my_method.json",
)
```

---

## 输出指标

每个实验至少输出：

- **模型质量**：PPL、Logit KL、Top-1 / Top-5 agreement、Generation quality、EOS success rate、Repetition rate
- **系统指标**：peak GPU memory、PPL compute time

后续 VQK 实验还会输出 weight memory、bytes read/token 等；KV Cache 实验会输出 KV bytes/token、retained attention mass 等。

---

## 红线

- **禁止 `device_map="auto"`**，必须使用显式 map（如 `balanced_low_0`）。
- 评估只看 **PPL / generation / logit KL**，不看 local cosine。

---

## VQK 快速开始

### 0. 理解你在做什么

这是 **Transformer weight representation 实验**，不是新的计算图，不是 LUT，也不是 attention approximation。

你要把 Qwen3.6-35B-A3B 第 39 层 `self_attn.o_proj` 的权重，从 BF16 压成 VQK-4（4-bit 整数 + block-wise scale），然后看模型级指标退化多少。最后和 **RTN INT4** 对比，判断 block-wise scale 是否比简单 per-channel 量化更好。

`self_attn.o_proj` 是 attention 最后一个投影层：把多头 attention 的输出重新投影到 hidden size。选 layer39 不是因为它“安全”，而是因为它靠近输出端、对最终 logits 影响直接，恰好能测出这种权重表示的真实敏感度。

### 1. 准备 PPL eval 文本

如果只有 prompts（如 `candidate_prompts.jsonl`），先用 baseline 模型续写成长文本：

```bash
cd LLM_LUT/v8
python -u common/generate_eval_texts.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --prompt_file /data/1000_prompts.jsonl \
  --max_samples 128 \
  --max_new_tokens 512 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --output_jsonl /data/v8_eval_texts.jsonl
```

### 2. 跑 BF16 baseline

```bash
python -u run_baseline_eval.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/v8_baseline.json
```

### 3. 跑 VQK 单点实验

```bash
python -u vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --quant_method vqk \
  --bits 4 \
  --block_size 64 \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/vqk_l39_o_proj_b4_blk64.json
```

### 4. 跑 RTN INT4 baseline（必须对比）

`vqk/standard_quant.py` 实现的是 per-channel symmetric RTN（Round-To-Nearest）INT4，作为第一轮 baseline。

```bash
python -u vqk/eval_vqk.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --layer_idx 39 \
  --module_path self_attn.o_proj \
  --quant_method int \
  --bits 4 \
  --max_eval_samples 128 \
  --max_new_tokens 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --logit_metrics \
  --output_json results/rtn_int4_l39_o_proj_b4.json
```

### 5. 汇总对比

```bash
python -u vqk/summarize_vqk_results.py \
  --result_dir results \
  --pattern '*.json' \
  --output_json results/vqk_summary_l39_o_proj.json
```

---

## 决策标准

继续 VQK 路线的前提：在相同 bit-width 下，VQK 的 PPL / KL / agreement **明显优于 RTN INT4**。

如果只是 local cosine 更高但 PPL 没优势，直接停止该配置。

## 后续路线

如果 VQK-4 在 RTN INT4 上有稳定优势，下一阶段再引入更强的 LLM quantization baseline：

```text
RTN INT4
→ VQK INT4
→ AWQ / GPTQ reference
→ activation-aware VQK (combining block scale + activation-aware channel scaling)
→ multi-layer
```

当前第一轮只回答三个问题：

1. VQK-4 vs RTN INT4：block-wise scale 有没有价值？
2. block 32/64/128：distribution shift 要多细才够？
3. PPL / logit KL / generation：这种优势是否真的传到模型级？
