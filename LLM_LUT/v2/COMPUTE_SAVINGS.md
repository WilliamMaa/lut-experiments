# LLM-LUT 计算节省分析

> 回答核心问题：当前 replacement 到底省了哪部分计算？

---

## 1. 当前 Stage：Functional Replacement（计算节省 = 0）

当前实现（R1/R2）的 replacement 流程：

```
1. 完整计算 MLP（gate_proj + up_proj + down_proj）
2. mlp_delta = MLP_output - x
3. Hook: mlp_delta[:, group_4] = LUT(address)  ← 只替换 slice
4. x_next = x + mlp_delta
```

**问题**：第 1 步仍然计算了完整的 MLP，包括 group 4 对应的原始 dense path。Hook 只是在输出后做覆盖，没有跳过任何 FLOPs。

**结论**：当前阶段的计算节省 = **0**。这是 functional proof-of-concept，不是 compute optimization。

---

## 2. Stage 2：Compute Removal 的路径

要做到真正的计算节省，必须**跳过被替换 group 对应的原始 dense computation**。

### 2.1 Qwen2.5 MLP 结构

```python
def forward(self, x):
    # x: [B, seq, hidden_size]
    gate = self.gate_proj(x)      # [B, seq, intermediate_size]
    up   = self.up_proj(x)        # [B, seq, intermediate_size]
    hidden = self.act_fn(gate) * up   # [B, seq, intermediate_size]
    down = self.down_proj(hidden)     # [B, seq, hidden_size]
    return down
```

`mlp_delta = down - x`。group 4 是 `down` 的 slice：`down[..., 4*gs:(4+1)*gs]`。

### 2.2 哪里可以省？

**不能省的部分**：
- `gate_proj` 和 `up_proj`：它们产生 `intermediate_size` 的激活，被**所有** output groups 共享。即使只替换 group 4，仍然需要完整的 intermediate activation。
- `act_fn(gate) * up`：同上，element-wise 激活作用于所有通道。

**可以省的部分**：
- `down_proj` 中对应 group 4 的输出通道：`down[:, :, 4*64:(4+1)*64]`
- 具体：`hidden @ W_down[4*64:(4+1)*64, :].T`

### 2.3 理论 FLOPs 节省

| 模型 | hidden | intermediate | group_size | 省掉 FLOPs | 占 down_proj 比例 |
|------|--------|-------------|------------|-----------|-------------------|
| 0.5B | 896 | 4864 | 64 | 4864×64 = 311K | **7.1%** |
| 1.5B | 1536 | 8960 | 64 | 8960×64 = 573K | **4.2%** |
| 3B | 2048 | 11008 | 64 | 11008×64 = 705K | **3.1%** |
| 7B | 3584 | 18944 | 64 | 18944×64 = 1.21M | **1.8%** |

**单 group 的节省比例随模型增大而下降**。因为 hidden_size 增长，但 group_size 固定为 64。

### 2.4 Multi-group 的节省

如果同时替换 N 个 groups，节省比例线性增长：

| 模型 | 1 group | 2 groups | 4 groups | 8 groups |
|------|---------|----------|----------|----------|
| 0.5B | 7.1% | 14.3% | 28.6% | 57.1% |
| 1.5B | 4.2% | 8.3% | 16.7% | 33.3% |
| 3B | 3.1% | 6.3% | 12.5% | 25.0% |
| 7B | 1.8% | 3.6% | 7.1% | 14.3% |

**关键结论**：
- 小模型（0.5B）替换 4-8 groups 才有显著节省
- 大模型（7B）需要替换更多 groups 才能达到同等比例
- 但大模型的绝对 FLOPs 基数更大，**绝对节省量**（MFLOPs/token）可能更大

---

## 3. 工程实现路径

### 3.1 最小实现（Python loop）

重写 `down_proj.forward`：

```python
def partial_down_proj(self, hidden, skip_groups, lut_fn):
    B, seq, _ = hidden.shape
    num_groups = self.out_features // self.group_size
    output = torch.empty(B, seq, self.out_features, dtype=hidden.dtype, device=hidden.device)
    
    for g in range(num_groups):
        if g in skip_groups:
            output[..., g*gs:(g+1)*gs] = lut_fn(g, hidden)  # O(1) lookup
        else:
            w = self.weight[g*gs:(g+1)*gs, :]  # [gs, intermediate]
            b = self.bias[g*gs:(g+1)*gs] if self.bias is not None else None
            output[..., g*gs:(g+1)*gs] = F.linear(hidden, w, b)
    
    return output
```

**问题**：Python 循环 + 多次小矩阵乘法，可能比单次大矩阵乘法**更慢**（PyTorch 的优化矩阵乘法对大块数据更高效）。

### 3.2 更高效实现

**方案 A：分组权重矩阵**
- 预先把 `down_proj.weight` 拆成 `num_groups` 个 `[group_size, intermediate]` 子矩阵
- 对非替换 groups 做并行 batch matmul
- 替换 groups 直接填入 LUT 输出

**方案 B：CUDA kernel**
- 写一个 custom CUDA kernel，在单次 kernel launch 中：
  - 对非替换 groups 做 matmul
  - 对替换 groups 做 table lookup
- 这是最优路径，但需要 CUDA 开发

**方案 C：混合（当前可做的）**
- 用 `torch.nn.ParameterList` 把 down_proj 拆成 groups
- forward 时用 `torch.cat` 组合
- 评估 latency 和 FLOPs 节省

---

## 4. 当前建议

### 先回答"有没有节省"

> **当前没有。**
>
> R1/R2 是 functional replacement，只证明"可以替换"，不证明"可以加速"。

### 下一步做什么

1. **先确认 multi-group 在 PPL/Acc 上可行**
   - 如果同时替换 2-4 个 groups 后 PPL 仍可控，才有 compute removal 的价值
   - 如果 multi-group 崩了，单 group 的 1-7% 节省没有意义

2. **再做 compute removal 的 latency demo**
   - 用方案 C（PyTorch 分组实现）测量实际 latency
   - 对比：original vs partial_down_proj vs full down_proj
   - 目标：证明理论 FLOPs 降低能转化为实际 latency 降低

3. **如果 latency demo 有效，再上 CUDA kernel**
   - 只有 PyTorch 层面的延迟改进不够，才需要 custom kernel

### 为什么不现在就做 compute removal？

- **单 group 节省太小**（1.5B 只有 4%），不值得工程投入
- **multi-group 的可行性还没验证**（不知道同时替换几个 groups 后 PPL 会不会崩）
- **PyTorch 层面的 partial linear 可能比 full linear 慢**（矩阵乘法效率问题）
- **当前 priority 是 scaling validation**（1.5B→3B→7B），不是 micro-optimization

---

## 5. 研究叙事如何说

**现在可以说**：

> "我们证明了 selected MLP residual contribution group 可以被 2-head LUT functionally replaced with controlled degradation."

**后续可以说**（如果 multi-group 可行 + latency demo 有效）：

> "通过同时替换多个 groups 并用 partial down-projection 跳过对应的 dense computation，我们实现了 X% 的理论 FLOPs 降低和 Y% 的实际 latency 降低。"

**现在的红线**：
- **不 claim 已经加速**
- **不 claim 已经省计算**
- **明确区分 functional replacement 和 compute removal**
