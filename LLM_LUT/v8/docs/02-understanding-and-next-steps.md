# v8 工作理解：下一步围绕 VQK 与 KV Cache Compression 启动

## 1. 背景定位

v6 的核心任务（FFN 输出 LUT 化，尤其是 `layer39.shared_expert`）已经完成，multi-layer 也已验证。v8 不再继续扩展 FFN LUT，而是**在 v6 成果旁边并行开启两条新线**：

| 路线 | 目标对象 | 核心问题 | 来源文档 |
|------|----------|----------|----------|
| **VQK-based Quantization** | Transformer Linear 权重 | 低 bit 下 VQK + block-wise KDS 是否优于 RTN INT quantization？ | `01-ideas-about-proj.md` |
| **KV Cache Compression** | Attention KV memory / bandwidth | 能否用量化、驱逐、learned codebook 显著降低 KV 开销，同时保留 attention 行为和长上下文能力？ | `00-ideas.md` |

两条线共享同一个底层信念：

> 不要追求内部张量逐元素精确，而是用小型离散表示保留模型功能真正需要的信息。

最终若都走通，可与 FFN LUT 统一为：

> **Compositional discrete representations for reducing LLM inference computation and memory movement.**

## 2. 对两条线的具体理解

### 2.1 VQK 线

- **基本形式**：`W ≈ S ⊙ W_q`
  - `W_q`：2/3/4/6/8-bit 整数权重。
  - `S`：每个 block 一个 FP16/BF16 scale，block 沿 input dimension 划分。
- **首轮目标模块**：先 `layer39.o_proj`，再 `v_proj` / `down_proj`，最后才碰 `q_proj` / `k_proj` / `gate_proj` / `up_proj`。
- **第一轮实验矩阵**：B0（BF16）、B1（RTN INT8）、B2（RTN INT4）、V1–V7（VQK 8/6/4/3 bit + block 32/64/128/256）。
- **关键对比**：不是 VQK vs BF16，而是 **VQK-4 vs RTN INT4、VQK-3 vs RTN INT3**。同 bit 下没优势就直接停止。
- **后续升级**：activation-aware scale calibration → logit-aware calibration → 多层扩展 → VQK + activation quantization → bit-sliced / LUT arithmetic。

### 2.2 KV Cache Compression 线

- **目标**：显著降低 KV memory / bandwidth，同时保留 attention 行为、PPL、长上下文能力。
- **阶段路线**（必须按顺序）：
  1. Phase 0：统一 evaluation framework。
  2. Phase 1：2/3/4/8-bit KV quantization（含 KIVI-style K per-channel / V per-token）。
  3. Phase 2：token eviction（recent window、recent + sink、heavy-hitter）。
  4. Phase 3：layer sensitivity map。
  5. Phase 4：layer-adaptive KV budget。
  6. Phase 5：learned codebook KV。
  7. Phase 6：attention/logit-aware codebook 训练。
  8. Phase 7：on-policy KV 数据收集。
  9. Phase 8：Hot/Warm/Cold 混合 KV。
- **第一阶段实验矩阵**：B0–B7 都是 baseline，M1/M2 是 codebook 方法；**先不做 Hot/Warm/Cold**。
- **关键指标**：PPL、logit KL、top-1/top-5 agreement、generation quality、attention mass recall、KV bytes/token、decode bandwidth。

## 3. 共享基础设施

两条线都需要一套统一的模型级 evaluation framework，不能再像早期 LUT 实验那样只看 local cosine。必须至少输出：

- **模型质量**：PPL、Logit KL、Top-1/Top-5 agreement、Generation quality、EOS success rate、Repetition rate。
- **Attention 行为**：Attention output cosine、Attention score correlation、Top-k attention recall、Retained attention mass。
- **系统指标**：KV bytes/token、weight memory、peak GPU memory、decode latency、throughput。

**建议复用 v6 已有的 `run_model_eval.py` 风格的评估入口**，把两类压缩方法都挂到同一个模型级 eval 上。

## 4. 推荐启动顺序

两条线可以**并行启动**，但都需要先解决一个共同问题：**统一的 Phase 0 evaluation framework**。在此基础上：

### 4.1 建议 A：先启动 VQK（更快验证价值）

原因：
- 思路直接，实现量相对可控（一层 Linear 的替换 + scale 搜索）。
- 只要跑 `layer39.o_proj` 的 bit/block sweep，就能在一两天内得到"VQK 是否值得做"的决策依据。
- 失败成本低；如果 VQK-4 不如 RTN INT4，可以立刻停掉。

第一步具体动作：
1. 在 `LLM_LUT/v8/` 下创建 `vqk/` 子目录。
2. 实现一个最小 VQK Linear wrapper：
   - 输入：原始 FP16/BF16 `nn.Linear` 权重。
   - 输出：量化后的 VQK 权重 + block-wise scale。
   - 支持 bit={8,6,4,3,2}，block={32,64,128,256}，默认 L2 initialization 求 scale。
3. 写一个 `apply_vqk_to_model.py` 工具：把指定 module（如 `model.model.layers[39].self_attn.o_proj`）替换成 VQK 版本。
4. 复用 v6 的模型级 eval 入口跑 B0/B1/B2 + V1–V7。
5. 输出对比表：同 bit 下 VQK vs RTN INT 的 PPL、logit KL、generation score。

### 4.2 建议 B：同时启动 KV Cache Compression Phase 0

原因：
- 这条线更长、更复杂，但和 VQK 线共用模型级 eval。
- Phase 0 本身的实现可以很快落地，后续阶段可以按需按结果决定。

第一步具体动作：
1. 在 `LLM_LUT/v8/` 下创建 `kv_cache/` 子目录。
2. 实现基线压缩方法：
   - uniform INT quantization（K/V 分别支持 8/4/3/2 bit）。
   - KIVI-style asymmetric quantization（K per-channel、V per-token）。
   - recent-window eviction（keep 100%/75%/50%/25%/12.5%）。
3. 写一个 `patch_kv_cache.py` 工具：把指定 Transformer 层的 KV 缓存替换为压缩版本。
4. 复用同一套模型级 eval，跑 B0–B3（量化基线）和 B4（recent window）。
5. 输出第一条 **Memory ↔ Quality** 曲线。

## 5. 两条线的依赖关系

```text
Phase 0: 统一模型级 eval framework
        ├─→ VQK Phase 1: o_proj bit/block sweep
        └─→ KV Cache Phase 1: INT/KIVI quantization sweep
```

先做 Phase 0 可以同时养活两条线。Phase 0 完成后，两条线可以独立并行推进。

## 6. 风险与决策点

| 风险 | 影响 | 应对 |
|------|------|------|
| VQK-4 不如 RTN INT4 | 直接停止 VQK 线 | 第一阶段就设好决策标准 |
| KV 量化到 2-bit 时 PPL 崩掉 | 说明当前模型对 KV 精度敏感，后续重点转向 eviction / codebook | 记录 baseline 曲线即可 |
| 模型级 eval 太慢 | 影响迭代速度 | 先用小 eval 集（128 samples / 2048 tokens），后续再补全 benchmark |
| 自动多卡分配死锁 | 违反项目红线 | 所有加载必须显式指定 `device` 或 `device_map`，严禁 `device_map="auto"` |

## 7. 下一步行动清单

1. **建立 v8 目录结构**：`LLM_LUT/v8/vqk/`、`LLM_LUT/v8/kv_cache/`、`LLM_LUT/v8/common/`（共享 eval）。
2. **复用 / 整理 v6 的模型级 eval 脚本**，提取成 v8 公共入口。
3. **并行实现**：
   - VQK Linear wrapper + `apply_vqk_to_model.py`。
   - KV cache 压缩 wrapper + `patch_kv_cache.py`。
4. **跑第一轮 baseline**：VQK 跑 `layer39.o_proj`；KV 跑 INT8/INT4/KIVI/recent-window。
5. **输出对比表和 Pareto 图**，决定下一阶段投入方向。
