# LLM-LUT v0.5 扩展实验设计文档

## 0. 事故教训与本次设计原则

| 上次问题 | 本次措施 |
|---------|---------|
| `device_map="auto"` 多卡死锁 | **显式单卡 `cuda:0`**，加载后 assert 其他 GPU 显存为 0 |
| CPU 内存爆炸（存储 logits） | **Online KL 计算**，不存储完整 logits |
| Hook 异常时不清理 | **所有 hook 注册必须用 try-finally** |
| 没有前置 GPU 健康检查 | **保留 `gpu_sanity_check.py`**，必须先跑通 |
| 盲目全量扫描 | **只测已验证的 candidate**（Layer 6 mlp_delta），不浪费算力 |
| Final Score 不可靠 | **废弃原有 scoring**，用 Recovery + Bucket Advantage |
| 没有分析 coverage | **新增 occupancy entropy + empty-bin ratio** |
| 单 group 结果不可推广 | **必须做多 group 组合测试** |

---

## 1. 实验目标

验证 Layer 6 `mlp_delta` 的 LUT 信号是否：
1. **可复现**（扩大数据量后仍然稳定）
2. **可扩展**（多 group 组合不会 nonlinear collapse）
3. **可优化**（quantile binning 优于 uniform binning）

---

## 2. 实验配置

```python
# 模型与硬件
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda:0"  # 显式单卡，禁止 device_map

# 数据（比 v0 扩大）
calib_size = 1024
eval_size = 512
max_seq_len = 256
batch_size = 8  # 根据显存调整

# 扫描范围（只测已验证最优的 candidate）
layer_ids = [6]
candidate_type = "mlp_delta"  # 只测这一个
groups_to_test = [4, 3, 8, 1, 13, 9, 0]  # v0 top 7

# Binning 配置
binning_modes = ["uniform", "quantile"]
num_bins_list = [32, 64, 128, 256]
```

---

## 3. 核心指标（废弃 Final Score）

| 指标 | 公式 | 用途 |
|------|------|------|
| **Recovery** | `(KL_Zero - KL_Bucket) / KL_Zero` | bucket 相对 zero 的恢复比例，越大越好 |
| **Bucket Advantage** | `KL_Mean - KL_Bucket` | bucket 相对 mean 的绝对改善，>0 说明有 addressable 结构 |
| **Coverage** | `非空 bin 数 / 总 bin 数` | 表利用率，>50% 才算可用 |
| **Occupancy Entropy** | `-sum(p_i * log(p_i))` | bin 分布的均匀度，越高说明 address 越分散 |
| **Empty-bin Ratio** | `空 bin 数 / 总 bin 数` | 与 coverage 互补 |

Ranking 输出列：
```
Layer | Group | Binning | NumBins | KL_Zero | KL_Mean | KL_Bucket | Recovery | BucketAdv | Coverage | Entropy
```

---

## 4. 三组实验

### 实验 A：单 Group × Binning 全面扫描

对 7 个 groups × 2 binning modes × 4 num_bins = **56 个配置**跑 zero/mean/bucket。

目标：找到最优的 (group, binning, num_bins) 组合。

### 实验 B：多 Group 组合测试（最关键）

固定最优的 (binning, num_bins)，测试以下组合：

```
Config 0: baseline（无替换）
Config 1: group 4 only
Config 2: group 4 + 3
Config 3: group 4 + 3 + 8
Config 4: group 4 + 3 + 8 + 1
Config 5: group 4 + 3 + 8 + 1 + 13
Config 6: group 4 + 3 + 8 + 1 + 13 + 9
Config 7: group 4 + 3 + 8 + 1 + 13 + 9 + 0
```

每个配置只跑 **bucket replacement**（不复测 zero/mean）。

目标：看 KL 是否随 group 数量近似线性增长，还是出现 collapse。

### 实验 C：Two-head Address Ablation

对 top 3 groups 测试：

```
Heads = 1（v0 默认取 variance 最大的通道）
Heads = 2（v0 默认：variance + correlation）
```

目标：确认 multi-head address 是否比 single-head 更好。

---

## 5. 模块改动点

### 5.1 `bucket.py` — 新增 Quantile Binning

```python
def build_bucket_table(..., binning_mode: str = "uniform"):
    if binning_mode == "uniform":
        # 现有逻辑：clip + uniform quantize
    elif binning_mode == "quantile":
        # 新逻辑：用 calibration data 的地址分布计算 quantile boundaries
        boundaries = torch.quantile(all_addresses.float(), q=torch.linspace(0, 1, num_bins+1))
        # bin_id = searchsorted(boundaries, address)
```

### 5.2 `hooks.py` — 新增 MultiGroupPerturbationHook

```python
class MultiGroupPerturbationHook:
    """同时替换多个 groups，用于实验 B 的多 group 组合测试。"""
    def __init__(self, group_ids: List[int], mode: str, ...):
        ...
```

### 5.3 `metrics.py` — 新增 Occupancy Entropy

```python
def compute_occupancy_entropy(per_bin_count: Tensor) -> float:
    probs = per_bin_count / per_bin_count.sum()
    probs = probs[probs > 0]
    return (-probs * torch.log(probs)).sum().item()
```

### 5.4 `rank.py` — 新 Ranking 输出

废弃 Final Score，输出 Recovery + Bucket Advantage + Coverage + Entropy。

### 5.5 `run_v0_5.py` — 新入口

- 先调用 `gpu_sanity_check.py`
- 运行实验 A → B → C
- 每一步都有明确的进度输出和中间结果保存
- 任何异常都会清理 hooks 并退出

---

## 6. 输出文件

```
results/
├── v0_5_experiment_A.json    # 单 group × binning 全面扫描
├── v0_5_experiment_B.json    # 多 group 组合测试
├── v0_5_experiment_C.json    # Two-head ablation
├── v0_5_report.md            # 汇总报告 + ranking 表
└── v0_5_addr_stats.pt        # 校准统计量（复用 v0 的或重新生成）
```

---

## 7. 资源估算

| 实验 | 配置数 | 每配置 eval 次数 | 总 forward 次数 | A100 预估时间 |
|------|--------|-----------------|----------------|--------------|
| A | 56 | 1 (只测 bucket) | 56 × 64 batches | ~10-15 分钟 |
| B | 8 | 1 | 8 × 64 batches | ~2-3 分钟 |
| C | 6 | 1 | 6 × 64 batches | ~2 分钟 |
| **总计** | | | | **~20 分钟** |

---

## 8. 成功标准

实验 A：
- 至少一个 (group, binning, num_bins) 组合的 **Recovery > 50%**
- Quantile binning 的 Coverage 显著高于 Uniform

实验 B：
- 3-4 个 group 组合时 KL 不会崩（不会比单 group 的累加和暴增 2 倍以上）

实验 C：
- Two-head 的 Recovery 优于或等于 Single-head

如果全部通过，进入 v1（trainable LUT prefit）。
