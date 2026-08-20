# KV Cache Compression

寻找比直接 LUT 化 attention 更稳、更容易部署的 attention-side 优化方法。

## 目标问题

> **在显著降低 KV memory / bandwidth 的前提下，尽可能保留 attention 行为、PPL 和长上下文能力。**

## 阶段路线

| Phase | 内容 |
|-------|------|
| Phase 0 | 统一 evaluation framework（已完成在 `common/`） |
| Phase 1 | KV 量化敏感度：INT8/4/3/2 + KIVI-style K per-channel / V per-token |
| Phase 2 | Token eviction：recent window、recent + sink、heavy-hitter |
| Phase 3 | Layer sensitivity map |
| Phase 4 | Layer-adaptive KV budget |
| Phase 5 | Basic learned codebook KV |
| Phase 6 | Attention/logit-aware codebook training |
| Phase 7 | On-policy KV 数据收集 |
| Phase 8 | Hot / Warm / Cold 混合 KV |

## 首轮实验矩阵

```text
B0  BF16 full KV
B1  INT8 KV
B2  INT4 KV
B3  KIVI-style 2-bit
B4  Recent-window 50%
B5  Heavy-hitter 50%
B6  Heavy-hitter 25%
B7  Layer-adaptive 25% average
M1  PQ / codebook KV
M2  Attention-aware codebook KV
```

先跑 B0–B4，后续再扩展。

## 待实现文件

- `kv_cache_patch.py`：`KVCachePatch(EvalPatch)` 实现
- `eval_kv_cache.py`：baseline sweep 入口
- `kv_quantizers.py`：INT / KIVI / codebook 量化器
- `kv_evictors.py`：recent-window / heavy-hitter eviction 策略

## 首个实验

```bash
python -u LLM_LUT/v8/kv_cache/eval_kv_cache.py \
  --model_path /data/models/Qwen3.6-35B-A3B \
  --kv_bits 4 \
  --quant_mode uniform \
  --eval_file eval.jsonl \
  --device_map balanced_low_0 \
  --output_json results/kv_uniform_int4.json
```

## 关键指标

- PPL、Logit KL、Top-1/Top-5 agreement
- Generation quality、EOS success rate、Repetition rate
- KV bytes/token、Total KV cache size、Compression ratio
- Attention output cosine、Attention score correlation、Retained attention mass

## 决策标准

1. 如果 2–4 bit KV 几乎无损，则 learned codebook 必须**在同 memory 下质量更好**才继续。
2. 如果 25% retained tokens 仍保持高 long-context performance，则 token selection 比 vector compression 更值得做。
3. 如果 learned codebook 在相同 KV bytes/token 下不如 scalar quant，直接停止 codebook 线。
