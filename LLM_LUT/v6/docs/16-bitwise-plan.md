# Bitwise / Compositional Address LUT 实验计划

> 基于 `@LLM_LUT/v6/docs/15-bitwise-reflection.md` 的思路，把当前 16-bit residual leaf 表拆成两级可加地址表，验证在大幅压缩存储的同时能否保持生成质量。

---

## 1. 核心思路

当前每层 residual 表大小：
- 32 groups × 65,536 leaves × 64 dims × 2 bytes ≈ 256 MiB

目标：把每个 group 的 16-bit residual 地址拆成两级：

```
address = (coarse_code, fine_code)
y_LUT = T_c[coarse_code] + T_f[fine_code]
```

例如 8 + 8 拆分：
- `T_c`: 256 entries
- `T_f`: 256 entries
- 总 entries：512（对比完整表 65,536，压缩 128×）

关键约束：**coarse/fine 位必须来自树路径，而不是把 leaf ID 机械拆成二进制**。这样每个 bit 段才是“有语义的层次地址”。

---

## 2. 实现方案

### 2.1 独立训练两级树

对每个 group 的 residual 目标：

1. **第一棵树（coarse）**：深度 `K` bit，直接拟合 residual。
   - 输出 `T_c[c]`，每个 coarse code 对应一个 64-dim 向量。
2. **计算残差**：`r_i' = r_i - T_c[c_i]`
3. **第二棵树（fine）**：深度 `D-K` bit，在 `r_i'` 上建树。
   - 输出 `T_f[f]`，每个 fine code 对应一个 64-dim 向量。
4. **预测**：`y_hat = T_c[c] + T_f[f]`

这和当前 `shared_coarse + residual` 结构精神上完全一致，只是把原来单棵 16-bit residual 树换成“coarse tree + fine tree”的组合。

### 2.2 支持的拆分配置

| 配置 | coarse entries | fine entries | 总 entries | 压缩比（相对 65K） |
|---|---|---|---|---|
| 8+8 | 256 | 256 | 512 | 128× |
| 8+4 | 256 | 16 | 272 | 241× |
| 6+5 | 64 | 32 | 96 | 683× |
| 10+6 | 1024 | 64 | 1088 | 60× |

建议先试 **8+8** 和 **8+4**，风险/收益比较均衡。

### 2.3 对 shared global coarse 也做拆分

当前 shared coarse 是 14-bit（16,384 entries），占 64 MiB。也可拆成 7+7：
- `T_c`: 128 entries
- `T_f`: 128 entries
- 总 256 entries（对比 16,384，压缩 64×）

但 shared coarse 影响所有 group，质量更敏感。建议先不动，或只做小实验对比。

---

## 3. 实验顺序

### Stage 1：单层 l39 快速验证

用已有的 `datasets/layer39_shared_expert_v3_onpolicy_multilayer/input,output` 数据：

```bash
python -u build_lut_ffn_output_v4_bitwise.py \
  --teacher_weight_path qwen_35b_shared_expert_l39.pt \
  --dataset_dir datasets/layer39_shared_expert_v3_onpolicy_multilayer/input \
  --output_dataset_dir datasets/layer39_shared_expert_v3_onpolicy_multilayer/output \
  --output_root outputs_ffn_lut_layer39_v4_bitwise_8_8 \
  --group_size 64 --group_ids "0-31" \
  --coarse_bits 8 --fine_bits 8 \
  --calib_size 600000 --eval_size 60000 \
  --device cuda:0
```

判断标准：
- cosine_similarity 与当前 v3 的 0.8375 相比，掉多少？
- PPL delta 与 v3 的 +3.05 相比，涨多少？
- 生成是否还连贯？
- 表大小从 320 MiB 降到多少？

### Stage 2：如果 8+8 可接受，试 8+4 和 6+5

找压缩比和质量的 Pareto 前沿。

### Stage 3：集成到多层

如果单层某配置明显优于当前 v3，再训练 l37/l38/l39 多层版本。

---

## 4. 预期结果

| 情况 | 行动 |
|---|---|
| 8+8 cosine > 0.80，PPL delta < 5 | 非常值得继续，直接替代当前 v3 |
| 8+8 cosine 0.75-0.80，PPL delta 5-8 | 有潜力，加 finetune 或试 10+6 |
| 8+8 cosine < 0.75，PPL delta > 8 | 当前地址可加性不够，放弃或改 conditional fine |

---

## 5. 下一步

先写 `build_lut_ffn_output_v4_bitwise.py`，跑 l39 单层 8+8 实验。
