# build_lut_ffn_output.py 使用说明

本脚本是 LLM_LUT v6 纠偏后的第一个最小实验：单层单专家 FFN output group LUT。

它完全对应 `docs/00-ideas.md` 里的 **Phase A 最小实验**：

- **不**做最近邻搜索（ANN）
- **不**用 JVP / Jacobian
- 固定 O(1) tree address 查表
- 先验证 LUT 对真实 FFN 输出的近似能力

---

## 需要的输入数据

和 `docs/exp11.py` 保持一致，至少需要两类文件。如果已经有预计算的 FFN 输出，还可以加上第三类，避免重复 forward Teacher：

### 1. Teacher 专家权重（`--teacher_weight_path`）

- 一个 `.pt` 文件，包含单个 Qwen MoE 专家的 `state_dict`。
- 必须包含以下键：
  - `gate_proj.weight`
  - `up_proj.weight`
  - `down_proj.weight`
- 键名如果带有 `expert.` 前缀（例如 `expert.gate_proj.weight`），脚本会自动去掉。
- 即使使用预计算输出，目前仍需要加载 Teacher 以获取 `hidden_size` / `intermediate_size`。

### 2. FFN 输入数据集（`--dataset_dir`）

- 一个目录，里面包含若干 `.pt` 文件。
- 每个 `.pt` 文件是一个 `float` tensor，形状为 `[N, hidden_size]`，表示该专家的 N 个 FFN 输入 token。

### 3. （可选）预计算 FFN 输出数据集（`--output_dataset_dir`）

- 如果目录里已经有 `input` 对应的 `output`，可以直接复用，不用每次 forward Teacher。
- 每个 `.pt` 文件必须是 `[N, hidden_size]`，且 **文件名**与 `--dataset_dir` 中的输入文件一一对应。
- 例如 `input/0000.pt` 对应 `output/0000.pt`，形状相同。
- 如果指定了 `--output_dataset_dir`，脚本会直接读取预计算输出；否则把输入喂给 Teacher 得到 target。

数据切分规则和 `exp11.py` 一样：
- 除最后 100 个 `.pt` 文件外的所有文件 → **calibration（构建 LUT）**
- 最后 100 个 `.pt` 文件 → **evaluation（验证精度）**

---

## 环境依赖

- Python >= 3.9
- PyTorch
- tqdm

---

## 运行示例

### 方式一：只提供输入，脚本自己 forward Teacher 得到 target（较慢）

```bash
cd LLM_LUT/v6
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_3y_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups \
  --group_size 64 \
  --group_ids "0,1,2,3" \
  --num_bits 12 \
  --device cuda:0
```

### 方式二：有预计算输出，直接复用（更快）

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_3y_0711 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_3y_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups \
  --group_size 64 \
  --group_ids "0,1,2,3" \
  --num_bits 12 \
  --device cuda:0
```

### 关键参数说明

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--teacher_weight_path` | 专家权重 `.pt` 路径 | 必填 |
| `--dataset_dir` | 输入 `.pt` 目录 | 必填 |
| `--output_dataset_dir` | 预计算输出 `.pt` 目录（可选） | `None` |
| `--output_root` | 输出目录 | 必填 |
| `--group_size` | 每个输出 group 的通道数 | 64 |
| `--group_ids` | 要替换的输出 group 编号，逗号分隔 | `0,1,2,3` |
| `--num_bits` | tree 深度，表项数为 `2^num_bits` | 12 |
| `--channels_per_bit` | 每个 tree 分裂节点随机选用的输入通道数 | 4 |
| `--tree_candidates` | 每个分裂节点尝试的随机投影数 | 64 |
| `--tree_min_samples` | 节点不再分裂的最小样本数 | 16 |
| `--tree_max_samples` | tree 构建时最多使用的 calibration 样本数 | 65536 |
| `--calib_size` | 用于构建 LUT 的样本数 | 65536 |
| `--eval_size` | 用于评估的样本数 | 8192 |
| `--batch_size` | 读取数据时每次处理的 batch size | 256 |
| `--device` | 使用的单卡 | `cuda:0` |

---

## 输出说明

### 1. 每个 group 的 checkpoint

`outputs/checkpoints/replacement_g{gid}.pt` 包含：

- `tree_state`: 可序列化的 tree 地址生成器
- `lut_table`: FP16 的 LUT 表，形状 `[1, 2^num_bits, group_size]`
- `group_id`, `group_size`, `num_bits`, `channels_per_bit`

### 2. 汇总结果

`outputs/summary.json` 包含：

- 每个 group 的 MSE、relative MSE、relative L2、cosine similarity
- 把所有被替换 group 拼回完整 FFN 输出后的完整输出指标
- LUT 总存储量（按 FP16 估算）
- 被替换通道数 / 占比
- 理论 MAC 削减比例（按跳过被替换输出通道对应的 `down_proj` slice 估算）

---

## 注意事项

1. **本脚本只验证 LUT 输出对真实 FFN 输出的近似精度**，还没有真正在完整模型前向里跳过 `down_proj` 计算。真正的 MAC 削减需要下一步把本脚本生成的 LUT 接入 `HybridFFNOutputEngine`。
2. 存储量按 LUT 表值 **FP16** 计算；tree 地址元数据很小，未计入。
3. MAC 削减比例是理论估算：假设被替换输出通道对应的 `down_proj` slice 不再执行，节省 `replaced_channels * intermediate_size` 次 MAC。
4. 本脚本只使用单卡，不启动多进程，也不使用 `device_map="auto"` 等多卡自动分配机制。

---

## 后续方向

如果 4/8 个 group 的结果显示容量-精度曲线成立，可以继续按 `00-ideas.md` 扩展：

- 扩到单专家 10% MAC reduction（约 18-20 个 group）
- 尝试 `Coarse + Residual LUT`
- 接入真正的 `HybridFFNOutputEngine` 做模型级前向
- 做 LUT-only 联合微调
