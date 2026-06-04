# LLM-LUT v0 设计方案：从 YOLO 到 LLM 的敏感度扫描与 LUT 适配

## 0. 设计目标

本文档的目标是将 `LLM_LUT/IDEAS.md` 中的**多层级敏感度扫描方法论**与 `lut_coco_v10.py` 中验证过的**Group-Multi-Head LUT 工程实践**相结合，产出一份可直接落地的 LLM-LUT v0 实施设计。

**核心约束（承自 AGENTS.md 红线）：**
- 动态参数必须通过 LUT 查表生成，不能引入额外 MLP / Linear / 矩阵乘法。
- 比较基准必须是同等计算量/参数量。
- 准确率只是验证指标，实验设计必须围绕"O(1) 查表加速"展开。

---

## 1. 从 v10 到 LLM 的技术迁移映射

| v10 (COCO/YOLO) | LLM-LUT v0 (Qwen2.5-0.5B) | 迁移说明 |
|---|---|---|
| 1x1 Conv 输出通道 | MLP `down_proj` 输出 hidden groups | 同为线性投影后的输出，可直接对应 |
| Spatial group (H×W 上的通道分组) | Sequence-wise group (token 维度上的通道分组) | 图像空间维度 → 序列长度维度，group 操作沿 hidden_dim 进行 |
| Multi-head address (每 group 多地址通道) | Multi-head address (每 group 多 scalar address) | 保留多地址平均降低方差的思路 |
| Per-spatial scalar quantization | Per-token scalar quantization | 每个 token 位置独立量化地址 |
| `raw = x + alpha * delta` 残差注入 | `output_group += alpha * LUT(address)` 残差注入 | 保持残差结构，不直接替换全部输出 |
| Phase0 prefit → Phase1 distill → Phase2 QAT | Calibration → Sensitivity Scan → Bucket Prefit → (Future) LUT Prefit | 先做零训练扰动扫描，再决定下一步 |
| `calibrate_v10_addr` 地址通道校准 | `calibrate_llm_addr` 基于校准集统计地址均值/方差 | 流程一致，统计对象变为 token-level activations |

---

## 2. v0 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM-LUT v0 Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: Calibration                                               │
│    ├─ Load Qwen2.5-0.5B-Instruct (bf16, eval)                       │
│    ├─ Run calib.jsonl (512~2k sequences, max_len=512)               │
│    └─ Collect per-layer activation statistics                       │
│         ├─ down_proj output means/stds per group                    │
│         ├─ MLP residual delta means/stds per group                  │
│         ├─ attention output means/stds per group                    │
│         └─ Address channel selection (variance-based + corr-based)  │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2: Sensitivity Scan (Zero / Mean / Noise)                    │
│    ├─ For each layer in [6, 12, 18]                                 │
│    ├─ For each candidate type (down_proj, MLP_delta, attn_out)      │
│    ├─ For each group                                                │
│    │    ├─ Zero ablation: replace group output with 0               │
│    │    ├─ Mean replacement: replace with calibration mean          │
│    │    └─ Noise perturbation: add sigma * std * noise              │
│    └─ Record: Local MSE, Logits KL, Next-token Acc                  │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3: Bucket Replacement Scan (最重要的预 LUT 测试)              │
│    ├─ For promising candidates from Stage 2                         │
│    ├─ Build bucket table: address scalar → quantized bin            │
│    ├─ bin_avg = mean(target_group) for all tokens in bin            │
│    └─ Evaluate: bucket MSE, bucket coverage, logits KL              │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 4: Ranking & Report                                          │
│    ├─ Compute sensitivity_score, compute_saving_score,              │
│    │    addressability_score for each candidate                     │
│    ├─ Rank by: compute_saving + addressability - sensitivity        │
│    └─ Output candidate map markdown table                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 Hook-Based Perturbation Engine (`hooks.py`)

这是整个 v0 的底层基础设施。相比 v10 中直接 `nn.Module` 替换，`LLM` 扫描阶段不需要训练，因此全部用 **forward hook** 实现临时注入，避免修改模型结构。

```python
class PerturbationHook:
    """
    通用扰动 hook，支持 zero / mean / noise / bucket 四种模式。
    """
    def __init__(self, 
                 layer_id: int,
                 candidate_type: str,      # "down_proj", "mlp_delta", "attn_out"
                 group_size: int,
                 group_id: int,
                 mode: str,                # "zero", "mean", "noise", "bucket"
                 calib_stats: dict,        # from Stage 1
                 bucket_table: dict = None # for mode="bucket"
                ):
        ...
    
    def __call__(self, module, input, output):
        """
        output shape: [batch, seq_len, hidden_dim] 或 [batch, seq_len, intermediate_dim]
        对指定 group 进行扰动，其余 group 保持不变。
        """
        # 1. 将 output 沿 hidden_dim 分 group
        # 2. 按 mode 修改目标 group
        # 3. 重组输出
        return modified_output
```

**关键适配点（LLM vs YOLO）：**
- v10 中 tensor 是 `[B, C, H, W]`，group 沿 `C` 分，地址从 `H×W` 上取。
- LLM 中 tensor 是 `[B, seq_len, hidden_dim]`，group 沿 `hidden_dim` 分，地址从 `seq_len` 上取（每个 token 有独立地址）。

### 3.2 Address Calibration (`calibrate.py`)

承自 v10 的 `calibrate_v10_addr`，但适配到 LLM 的 token-level activation。

```python
def calibrate_llm_address(
    model, 
    tokenizer,
    calib_loader,
    layer_ids: List[int],
    candidate_types: List[str],
    group_size: int = 64,
    heads: int = 2,
    max_seq_len: int = 512,
):
    """
    为每个 layer / candidate_type / group / head 选择地址通道并统计 mean/std。
    
    地址选择策略（同 v10）：
      - head 0: group 内 activation variance 最大的通道
      - head 1: group 内与 residual magnitude 相关性最高的通道
      - head h: 交替从两个排序列表中选择不重复的通道
    """
    ...
```

**输出格式（与 v10 兼容）：**
```python
{
    (layer_id, "down_proj"): {
        "addr_idx":  Tensor[G, heads],      # 全局通道索引
        "addr_mean": Tensor[G, heads],
        "addr_std":  Tensor[G, heads],
    },
    ...
}
```

### 3.3 Bucket Table Builder (`bucket.py`)

这是连接"敏感度扫描"与"真实 LUT"的桥梁。

```python
def build_bucket_table(
    model,
    calib_loader,
    layer_id: int,
    candidate_type: str,
    group_id: int,
    addr_idx: Tensor,
    addr_mean: Tensor,
    addr_std: Tensor,
    num_bins: int = 64,
    addr_clip: float = 3.0,
):
    """
    1. 遍历校准集，收集 (address_scalar, target_group_output) 对
    2. address = (activation - mean) / std，clip 到 [-clip, clip]
    3. quantize to [0, num_bins-1]
    4. table[bin] = mean(target_group_output for tokens in bin)
    
    Returns:
        table: Tensor[num_bins, group_size]
        coverage: float (非空 bin 比例)
        per_bin_var: Tensor[num_bins]
    """
    ...
```

**关键设计决策：**
- 地址来源必须是**已有激活值**（如 group 内某通道的 hidden state），不能引入新投影。
- 每个 token 独立做 scalar → bin → lookup，不存在序列依赖的计算。
- `num_bins=64` 是 v0 的起点，因为 64×group_size 的表非常小（64×64×2B = 8KB）。

### 3.4 Metrics Collector (`metrics.py`)

```python
def compute_local_metrics(original, perturbed):
    """
    计算局部指标：MSE, Cosine Similarity, Relative Error Reduction
    """
    ...

def compute_model_metrics(model_original, model_perturbed, eval_loader):
    """
    计算模型级指标：
      - Logits KL divergence (最重要，快速)
      - Perplexity (pipeline 稳定后添加)
      - Next-token accuracy
      - Generation sanity (小 prompt 集，人工抽查)
    """
    ...
```

---

## 4. 扫描执行流程（伪代码级）

```python
# ========== Stage 1: Calibration ==========
calib_data = load_jsonl("calib.jsonl")
calib_loader = build_loader(calib_data, max_len=512, batch_size=8)

model = load_qwen_0_5b()
addr_stats = calibrate_llm_address(
    model, tokenizer, calib_loader,
    layer_ids=[6, 12, 18],
    candidate_types=["down_proj", "mlp_delta", "attn_out"],
    group_size=64,
    heads=2,
)

# ========== Stage 2 & 3: Scan ==========
results = []
eval_loader = build_loader("eval.jsonl", max_len=512, batch_size=8)

for layer_id in [6, 12, 18]:
    for cand_type in ["down_proj", "mlp_delta", "attn_out"]:
        # 确定分组数和维度
        num_groups, dim = get_shape(model, layer_id, cand_type, group_size=64)
        
        for group_id in range(num_groups):
            # --- Zero Ablation ---
            hook = PerturbationHook(layer_id, cand_type, group_id, mode="zero")
            kl_zero = run_eval(model, hook, eval_loader)
            
            # --- Mean Replacement ---
            mean_vec = addr_stats[(layer_id, cand_type)]["group_means"][group_id]
            hook = PerturbationHook(layer_id, cand_type, group_id, mode="mean", mean_vec=mean_vec)
            kl_mean = run_eval(model, hook, eval_loader)
            
            # --- Bucket Replacement (核心) ---
            bucket = build_bucket_table(
                model, calib_loader, layer_id, cand_type, group_id,
                addr_idx=addr_stats[(layer_id, cand_type)]["addr_idx"][group_id],
                addr_mean=addr_stats[(layer_id, cand_type)]["addr_mean"][group_id],
                addr_std=addr_stats[(layer_id, cand_type)]["addr_std"][group_id],
                num_bins=64,
            )
            hook = PerturbationHook(layer_id, cand_type, group_id, mode="bucket", bucket=bucket)
            kl_bucket = run_eval(model, hook, eval_loader)
            
            results.append({
                "layer": layer_id,
                "type": cand_type,
                "group": group_id,
                "kl_zero": kl_zero,
                "kl_mean": kl_mean,
                "kl_bucket": kl_bucket,
                "bucket_coverage": bucket.coverage,
                "addressability": (kl_mean - kl_bucket) / max(kl_mean - kl_zero, 1e-8),
            })

# ========== Stage 4: Ranking ==========
for r in results:
    r["sensitivity_penalty"] = r["kl_bucket"]  # 越小越好
    r["compute_saving"] = estimate_mac_saving(r["type"], r["group"])
    r["final_score"] = r["compute_saving"] + r["addressability"] - r["sensitivity_penalty"]

top_candidates = sorted(results, key=lambda x: x["final_score"], reverse=True)[:20]
generate_report(top_candidates)
```

---

## 5. 关键工程细节

### 5.1 Group Size 与维度映射

| Candidate Type | Tensor Shape | Group 维度 | Group Size 建议 |
|---|---|---|---|
| `down_proj` output | `[B, seq, hidden_dim]` | `hidden_dim` | 64 |
| `mlp_delta` residual | `[B, seq, hidden_dim]` | `hidden_dim` | 64 |
| `attn_out` | `[B, seq, hidden_dim]` | `hidden_dim` | 64 |
| `intermediate` (v0 暂不做) | `[B, seq, intermediate_dim]` | `intermediate_dim` | 128 |
| `attention_head` (v0 暂不做) | `[B, seq, hidden_dim]` | 按 `hidden_dim // num_heads` | natural |

Qwen2.5-0.5B 的 hidden_dim = 896，因此 `group_size=64` 时，每层有 `896/64 = 14` 个 groups。

### 5.2 Hook 点选择（精确到模块路径）

```python
# 承自 IDEAS.md Section 10
HOOK_TARGETS = {
    "down_proj": lambda model, i: model.model.layers[i].mlp.down_proj,
    "mlp":       lambda model, i: model.model.layers[i].mlp,          # 用于 mlp_delta
    "attn":      lambda model, i: model.model.layers[i].self_attn,     # 用于 attn_out
}
```

**`mlp_delta` 的特殊处理：**
- 需要对 `mlp` 模块注册 hook，在其输出后减去输入 `x`，得到 delta。
- 然后对 delta 的指定 group 进行扰动，最后加回 `x`。

### 5.3 Bucket Replacement 的地址来源

承自 v10 的地址选择逻辑，但对每个 group 只选 `heads=2` 个通道：

```
address_g,h = (activation[channel_g,h] - mean_g,h) / std_g,h
bin = quantize(address_g,h, clip=3.0, bins=64)
replacement_g = table_g[bin]   # [group_size]
```

如果 multi-head 选了多个地址，可以取平均或分别查表再平均（同 v10 的 `d.mean(dim=2)`）。

### 5.4 计算节省估算

承自 IDEAS.md 的理念，给出量化的 saving score：

| Candidate Type | 替换后可移除的计算 | Saving Score |
|---|---|---|
| `down_proj` group | 该 group 对应的 `down_proj` 权重行 | `group_size / hidden_dim` |
| `mlp_delta` group | 该 group 对应的 MLP 完整路径（gate+up+down 的一部分） | 更高，但实现更复杂 |
| `attn_out` group | 该 group 对应的 `o_proj` 权重行 | `group_size / hidden_dim` |

---

## 6. 文件结构建议

```
LLM_LUT/
├── IDEAS.md                          # 原始理论文档（已存在）
├── LLM_LUT_v0_DESIGN.md              # 本设计文档（已存在）
├── v0/
│   ├── README.md                     # v0 快速开始
│   ├── config.py                     # 模型、层、group_size 等配置
│   ├── data.py                       # calib/eval 数据加载
│   ├── hooks.py                      # PerturbationHook 实现
│   ├── calibrate.py                  # 地址校准
│   ├── bucket.py                     # Bucket table 构建
│   ├── metrics.py                    # 局部 + 模型级指标
│   ├── scan.py                       # 主扫描流程（Stage 2-3）
│   ├── rank.py                       # 候选排序与报告生成
│   └── run_v0.py                     # 一键运行入口
```

---

## 7. 与 v10 的代码级对应关系

| v10 函数/类 | v0 对应 | 说明 |
|---|---|---|
| `SpatialGroupMultiHeadLUTDelta` | `BucketReplacer` (概念级) | v0 不训练 LUT，只存 bucket 均值表 |
| `calibrate_v10_addr` | `calibrate_llm_address` | 核心逻辑完全一致，维度处理改适配 LLM |
| `_quant_float_all` | `quantize_address` |  scalar 量化公式直接复用 |
| `_interp_lookup_all` | `bucket_lookup` | v0 用最近邻（无插值），因为 bin 均值不需要插值 |
| `raw_distill_loss` | `compute_local_metrics` | v0 没有训练，只有统计距离 |
| `replace_target_convs_with_lut` | `register_forward_hook` | v0 用临时 hook 而非结构替换 |
| `Phase0/1/2` | `Stage1/2/3/4` | 流程重新映射：Calibrate → Scan → Bucket → Rank |

---

## 8. 风险与规避

| 风险 | 影响 | 规避方案 |
|---|---|---|
| Hook 实现不当导致梯度泄漏或内存爆炸 | 扫描失败 | 全程 `torch.no_grad()`，不存储中间激活，只存统计量 |
| Bucket table 覆盖率低（大量空 bin）| 替换效果差 | 记录 coverage，低于阈值时标红，不作为候选 |
| Attention 输出 context-dependent，bucket 难以捕捉 | 地址性差 | 预期中，v0 重点放在 MLP 组件，attention 只做基线扫描 |
| 校准集分布偏移导致 mean/bucket 不准 | 指标不可靠 | 校准集覆盖中英文+instruction+推理，eval 集独立 |
| 序列长度变化导致地址统计不准 | bucket 失效 | v0 固定 `max_len=512`，长文本延后处理 |

---

## 9. 成功标准与出口条件

同 IDEAS.md Section 14，但增加工程可验证的量化标准：

1. **Pipeline 可用**：`run_v0.py` 能在单张 GPU 上完成 `[6, 12, 18]` 三层的完整扫描（预计 < 2 小时）。
2. **敏感度表格产出**：至少覆盖 `zero/mean/bucket` 三种扰动 × 三种候选类型 × 三层 × 14 groups = ~378 行的结果表。
3. **存在阳性信号**：至少找到一个候选满足：
   - `kl_bucket < kl_mean * 0.8`（bucket 显著优于均值替换）
   - `kl_bucket < 0.5`（模型级 KL 可控）
   - `bucket_coverage > 0.7`（表覆盖足够）
4. **明确 v1 目标**：从 top 5 候选中选出 1~2 个，给出具体的 LUT 模块接口设计。

---

## 10. 下一步（v1 预览）

v0 验证某个结构（如 `layer_12 down_proj group_3`）是低敏感度且高地址性后，v1 将：

1. **实现可训练的 `LLM_GroupLUTModule`**：
   - 输入：token hidden state
   - 输出：group delta（同 v10 的 `forward_raw`）
   - 参数：`tables[G, heads, L, group_size]`, `alpha_raw[G]`
   - 推理：scalar quant → LUT lookup → weighted add（O(1)）

2. **Phase0 prefit**：冻结原模型，用校准数据预训练 LUT 表拟合 `down_proj` 输出。
3. **Phase1 端到端微调**：类似 v10 的 distill trainer，但适配到 causal LM 的 next-token prediction loss + KL distill。

**v1 与 v10 的最大差异**：
- v10 替换的是 Conv1x1（空间局部），v1 替换的是 Linear projection（序列全局但每个 token 独立）。
- v10 的地址来自输入 feature map 的 channel 值，v1 的地址来自 hidden state 的 channel 值。
- **核心不变**：都是 `scalar address → LUT → vector add`，没有矩阵乘法。
