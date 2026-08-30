# v8 GDN Recurrent State Low-Rank Plan

## 0. 我们到底在做什么

v6 已经验证：FFN 的 dense 输出激活可以用 LUT/离散表示替代，且 MAC 削减是核心收益。  
v8 的新目标是：**验证 attention 侧是否也存在类似的“可压缩核心状态”**。

具体对象：Qwen3.6-35B-A3B 里 30 层 Gated DeltaNet (GDN) 的 recurrent state `S_t`。

`S_t` 在模型内部已经是 `(B, 32, 128, 128)` 的 FP32 张量，每层 2 MB、30 层共 60 MB。它不是 KV cache，而是 GDN 的固定尺寸 recurrent memory。问题变成：

> **这个 recurrent state 能不能用更小的表示替代，而不让模型行为明显退化？**

这不是 training，也不是刷准确率；这是一个**压缩/表示实验**。

---

## 1. 已确认的事实

1. **GDN 占 30 层，full attention 占 10 层**，每 4 层一组：3 GDN + 1 full attention。
2. **Recurrent state 形状**：`(B, 32, 128, 128)`，FP32，每层 2 MB。
3. **每个 head 的 128×128 state 矩阵强烈低秩**：
   - 90% 能量：rank 12–18
   - rank 32 重建误差：~1%
   - rank 64 重建误差：~0.2%
4. **State 随时间变化很大**：pos 256→512 的相对 Frobenius 变化约 0.6–0.7，不是 sparse-delta。
5. **Head 能量分布不均**：少数 head 占 25%，top-8 占 50–60%。

**结论**：`S_t` 是低秩矩阵，但**不是静态的**。真正的研究问题不是“能不能把 snapshot 压缩”，而是“**压缩后的表示能不能在 delta-rule 更新下稳定传播**”。

---

## 2. 当前代码资产

- `attention_compact/probe_gdn.py`：确认 state/cov_state 形状、dtype、bytes。
- `attention_compact/collect_gdn_states.py`：采集真实 state。
- `attention_compact/analyze_gdn_states.py`：低秩/能量/delta 分析。
- `attention_compact/low_rank_state_patch.py`：把 DynamicCache 里的 recurrent state 透明地替换为低秩 SVD 因子（读时重建、写时压缩）。
- `attention_compact/eval_low_rank_state.py`：复用 `common.evaluator` 跑 baseline + patched 对比。

**问题**：`low_rank_state_patch.py` 的实现是“事后压缩”——每一层、每一步都重新 SVD，开销巨大。它只是**可行性探针**，不是最终方案。如果 feasibility 通过，下一步必须做“真正的低秩 GDN 更新”（见 §5）。

---

## 3. 阶段性问题与停止条件

| 阶段 | 问题 | 通过标准 | 失败标准 | 时间预估 |
|------|------|----------|----------|----------|
| **Phase 1** | 把最终 prefill state 压缩到 rank 32/64，PPL 会涨多少？ | ΔPPL ≤ 5% | ΔPPL > 20% 或输出乱码 | 1 小时内 |
| **Phase 2** | 压缩后的 state 随 32-token 生成传播，质量是否稳定？ | EOS 成功率与 baseline 相当，无明显重复/乱码 | 生成立刻崩溃、重复率飙升 | 2–4 小时 |
| **Phase 3** | 设计真正的低秩/量化 GDN update，而不是事后 SVD | 待定 | Phase 1/2 不过则不进 Phase 3 | 待定 |

**核心红线**：如果 Phase 1 的 snapshot 压缩就让 PPL 崩掉，说明低秩这条路在 attention state 上不如 FFN activation 上稳，直接停止，不再浪费时间优化 SVD 速度。

---

## 4. Phase 1 实验（只做这个）

### 4.1 目的

验证“state 本身可压缩”这件事在模型级是否成立。**不涉及生成传播**，只测 snapshot。

### 4.2 方法

用 `low_rank_state_patch.py` 跑 PPL-only 评估。由于 PPL 只用到 prefill 的最终 state，每 text 每 GDN 层只压缩一次，不会随 token 反复 SVD，因此很快。

### 4.3 命令

```bash
cd /data/mamingyu/v8
python - <<'PY'
import sys, torch
sys.path.insert(0, '.')
from common.utils import load_model_and_tokenizer
from common.prompts import load_eval_texts
from common.metrics import compute_ppl
from attention_compact.low_rank_state_patch import LowRankStatePatch

texts = load_eval_texts('/data/v8_eval_texts.jsonl', 32)
model, tokenizer, device = load_model_and_tokenizer(
    '/home/u/downloads/models/Qwen3.6-35B-A3B',
    torch_dtype='bfloat16', device_map='balanced_low_0'
)

# Baseline: rank=128 == no compression
for rank in [128, 64, 32, 16]:
    patch = LowRankStatePatch(rank=rank)
    patch.install(model)
    ppl = compute_ppl(model, tokenizer, texts, device, max_length=512)
    patch.uninstall(model)
    print(f'rank={rank:3d}  PPL={ppl:.4f}')
PY
```

### 4.4 预期与判读

- `rank=128` 应该和正常 baseline 一致，作为 sanity check。
- **ΔPPL ≤ 5%（rank=32）**：snapshot 低秩可行，进入 Phase 2。
- **ΔPPL > 20%（rank=32）**：state 低秩性虽然数学上成立，但模型对精度敏感，停止这条线。
- 如果 rank=64 好、rank=32 差，则最优 trade-off 在 32–64 之间。

### 4.5 为什么先不做生成

生成阶段 state 每 token 都要传播，会触发大量 SVD，这是**工程实现问题**，不是科学问题。在解决工程问题之前，必须先回答科学问题：state snapshot 压缩本身是否可接受。

---

## 5. Phase 2（只有 Phase 1 通过才做）

### 5.1 目的

验证压缩后的 state 能在 autoregressive generation 下稳定传播。

### 5.2 方法

用 `eval_low_rank_state.py`，但参数压到最小：
- 4 prompts
- 32 tokens
- logit_metrics=False
- 只比较生成输出和 EOS 成功率

```bash
python -u attention_compact/eval_low_rank_state.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file /data/v8_eval_texts.jsonl \
  --prompt_file /data/1000_prompts.jsonl \
  --rank 32 \
  --max_eval_samples 4 \
  --max_new_tokens 32 \
  --max_length 256 \
  --device_map balanced_low_0 \
  --torch_dtype bfloat16 \
  --output_json results/gdn_low_rank_r32_gen_smoke.json
```

### 5.3 判读

- 生成文本没有立刻重复/乱码/EOS 失败率不暴跌 → 继续。
- 生成质量明显劣化 → 说明 delta-rule 对低秩误差敏感，需要真正的低秩 update，而不是事后压缩。

---

## 6. Phase 3（只有 Phase 2 通过才做）

如果 Phase 1+2 都通过，说明：
1. State snapshot 可压；
2. 压缩后的 state 能传播。

但“每 token 做 SVD”仍然不可接受。Phase 3 要设计**真正的低秩 GDN 更新**，例如：

- **Factorized delta rule**：不维护 dense `S_t`，而是维护 `S_t = U_t V_t^T`，并推导出 `U_t, V_t` 的递推更新。
- **Incremental SVD**：用 power iteration 或 randomized SVD 增量更新低秩因子。
- **Per-head rank allocation**：根据 head 能量给不同 rank。
- **Factor quantization**：把 U/V 压到 BF16/INT8。

这才是最终能落地到 memory savings 的方案。

---

## 7. 现在立刻停止做的事

1. **不要再跑 64 samples + 256 tokens 的长 eval**：那是 Phase 2 之后才该考虑的规模。
2. **不要优化 SVD 速度**：Phase 1 没过之前，优化没有意义。
3. **不要加 KV cache 或 VQK**：v8 当前只聚焦 GDN recurrent state。
4. **不要上 multi-layer / multi-rank 扫描**：先过单层 rank 32/64 的可行性。

---

## 8. 当前下一步

**只跑 Phase 1 的命令**（见 §4.3），预计 1 小时内出结果。  
跑完把 `rank=128/64/32/16` 的 PPL 贴回来，根据 §4.4 的判读标准决定停还是进 Phase 2。
