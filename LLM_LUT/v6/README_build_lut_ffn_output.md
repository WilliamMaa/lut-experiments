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
- 每个 `.pt` 文件是一个 `float` tensor，形状可以是：
  - `[hidden_size]`：单个 FFN 输入 token
  - `[N, hidden_size]`：N 个 FFN 输入 token 的 batch
- 对应的预计算输出文件形状需与输入文件一致。

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

- Python >= 3.9（推荐 3.10）
- PyTorch 2.5.1 + cu124
- transformers / accelerate / numpy / tqdm

安装命令：

```bash
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
```

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
| `--group_ids` | 要替换的输出 group 编号，支持逗号或连字符范围，如 `0,1,2,3`、`0-7`、`0-3,8,10-15` | `0-3` |
| `--num_bits` | tree 深度 / 每表 bit 数，表项数为 `2^num_bits` | 12 |
| `--coarse_num_bits` | **Coarse LUT** 的 bit 数。需要和 `--residual_num_bits` 同时指定才会启用 coarse+residual 模式 | `None` |
| `--residual_num_bits` | **Residual LUT** 的 bit 数。需要和 `--coarse_num_bits` 同时指定才会启用 coarse+residual 模式 | `None` |
| `--channels_per_bit` | 每个地址位随机选用的输入通道数 | 4 |
| `--address_mode` | 地址生成器：`tree`（决策树）、`high_order`（随机高阶投影）、`2d`（两通道分箱） | `tree` |
| `--num_tables` | `high_order` 模式下的表数（总表项 = `num_tables * 2^num_bits`） | 1 |
| `--num_bins` | `2d` 模式下每轴 bins 数（表项 = `num_bins^2`） | 64 |
| `--tree_candidates` | 每个 tree 分裂节点尝试的随机投影数 | 64 |
| `--tree_min_samples` | 节点不再分裂的最小样本数 | 16 |
| `--tree_max_samples` | tree 构建时最多使用的 calibration 样本数 | 65536 |
| `--target_mode` | `direct`：LUT 存完整 FFN 输出；`residual_mean`：LUT 存 `输出 - group 均值`；`residual_input`：LUT 存 `输出 - 输入残差` | `direct` |
| `--calib_size` | 用于构建 LUT 的样本数 | 65536 |
| `--eval_size` | 用于评估的样本数 | 8192 |
| `--batch_size` | 读取数据时每次处理的 batch size | 256 |
| `--finetune_epochs` | 均值初始化后，是否离线 finetune LUT 表值。`0` = 不微调 | 0 |
| `--finetune_lr` | LUT 表值 finetune 的学习率 | 1e-3 |
| `--finetune_batch_size` | LUT 表值 finetune 的 batch size | 1024 |
| `--finetune_loss_mode` | 离线 finetune 损失函数：`mse`、`mse+cosine`、`cosine`。`mse+cosine` 同时优化数值和方向 | `mse` |
| `--finetune_cosine_alpha` | `mse+cosine` 中 cosine 项的权重 | 1.0 |

---

## 强化 finetune 用于最后一层

最后一层 FFN 替换难度远高于第一层，建议从以下角度强化 finetune：

1. **使用 `mse+cosine` 损失**：最后一层对输出方向敏感，cosine 项能让 LUT 更好地保持 FFN 输出的方向。
2. **增加 finetune epoch**：从默认 10 提升到 50，让表值充分收敛。
3. **使用 `residual_input` target**：LUT 只学习 `FFN(x) - x`，减少需要拟合的量。
4. **增加 LUT 容量**：例如 `coarse 14 + residual 16`，用存储换精度。

示例：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer_39_ffn_3000w_0721 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer_39_ffn_3000w_0721 \
  --output_root ./outputs_ffn_lut_layer39_32groups_mse_cosine \
  --group_size 64 \
  --group_ids "0-31" \
  --coarse_num_bits 14 \
  --residual_num_bits 16 \
  --target_mode residual_input \
  --finetune_epochs 50 \
  --finetune_lr 1e-3 \
  --finetune_batch_size 1024 \
  --finetune_loss_mode mse+cosine \
  --finetune_cosine_alpha 1.0 \
  --calib_size 200000 \
  --eval_size 20000 \
  --device cuda:0
```

---

## 放大实验

当前 4 group、num_bits=12 的结果只替换了约 **4.2%** 的 FFN MAC，表也只有 **2 MiB**。从 `00-ideas.md` 的目标看，还有很大放大空间：

1. **先换地址生成器**：`tree` 在这个模型上效果不佳（单 group cos_sim ~0.5），可以先试 `high_order` 随机高阶投影地址。
2. **如果直接预测完整输出困难，尝试 `residual_input`（v5 的做法）**：LUT 只学 `FFN 输出 - 输入残差`，也就是 MLP 那一部分残差。对 Transformer 来说，输入残差通常是一个很好的 baseline，比 group 均值更稳。
3. **再扩替换比例**：在这个模型上（hidden=2048, intermediate=512, 32 个 group）：
   - 8 个 group → 约 8.3% MAC reduction
   - 10 个 group → 约 10.4% MAC reduction（`00-ideas.md` 的主目标）
   - 16 个 group → 约 16.7% MAC reduction
   - 19 个 group → 约 20.0% MAC reduction

表存储按 FP16 估算（high_order 模式下乘以 `num_tables`）：
- `num_bits=12`，`num_tables=4`：每个 group 2 MiB
- `num_bits=14`，`num_tables=4`：每个 group 8 MiB
- `num_bits=16`，`num_tables=1`：每个 group 8 MiB

### 放大示例

**4 groups，high_order 地址，num_bits=12，num_tables=4**：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_1000w_0711 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_1000w_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups_1000w_high_order_nb12_t4 \
  --group_size 64 \
  --group_ids "0-3" \
  --address_mode high_order \
  --num_bits 12 \
  --num_tables 4 \
  --channels_per_bit 4 \
  --calib_size 200000 \
  --eval_size 20000 \
  --device cuda:0
```

**4 groups，num_bits=16，用 residual_input 目标（v5 风格）**：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_1000w_0711 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_1000w_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups_1000w_nb16_residual_input \
  --group_size 64 \
  --group_ids "0-3" \
  --num_bits 16 \
  --tree_max_samples 200000 \
  --tree_min_samples 16 \
  --calib_size 200000 \
  --eval_size 20000 \
  --target_mode residual_input \
  --device cuda:0
```

**8 groups，num_bits=14，目标 ~8.3% MAC reduction**：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_3y_0711 \
  --output_root ./outputs_ffn_lut_layer1_8groups_nb14 \
  --group_size 64 \
  --group_ids "0-7" \
  --num_bits 14 \
  --device cuda:0
```

**10 groups，num_bits=16，目标 ~10.4% MAC reduction**：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_3y_0711 \
  --output_root ./outputs_ffn_lut_layer1_10groups_nb16 \
  --group_size 64 \
  --group_ids "0-9" \
  --num_bits 16 \
  --device cuda:0
```

**4 groups，Coarse + Residual LUT（high_order，coarse 12-bit + residual 16-bit）**。
这个配置每张 LUT 的容量都会显著变大，4 groups 合计约 40 MiB 左右，适合验证“大容量 LUT”是否能压住残差：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_1000w_0711 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_1000w_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups_1000w_coarse12_residual16 \
  --group_size 64 \
  --group_ids "0-3" \
  --address_mode high_order \
  --num_tables 1 \
  --coarse_num_bits 12 \
  --residual_num_bits 16 \
  --calib_size 200000 \
  --eval_size 20000 \
  --device cuda:0
```

说明：
- coarse LUT 用 `coarse_num_bits=12` 捕捉 FFN 输出的主要分量。
- residual LUT 用 `residual_num_bits=16` 对 coarse 预测后的残差做第二次查表。
- 两张表在线相加得到最终预测，summary 里的总存储量会自动按两张表之和估算。
- 在 `tree` 或 `2d` 模式下，`coarse_num_bits` / `residual_num_bits` 同样有效；`2d` 模式下两张表仍共用 `--num_bins`。

**4 groups，Coarse + Residual + 离线 finetune（10 epochs）**：

```bash
python build_lut_ffn_output.py \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_1000w_0711 \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/output_qwen3_layer1_ffn_1000w_0711 \
  --output_root ./outputs_ffn_lut_layer1_4groups_1000w_coarse12_residual16_finetune10 \
  --group_size 64 \
  --group_ids "0-3" \
  --address_mode high_order \
  --coarse_num_bits 12 \
  --residual_num_bits 16 \
  --finetune_epochs 10 \
  --finetune_lr 1e-3 \
  --finetune_batch_size 1024 \
  --calib_size 200000 \
  --eval_size 20000 \
  --device cuda:0
```

说明：
- 先按 calibration 均值初始化 coarse 和 residual 两张表。
- 然后固定 address，只优化 LUT 表值，最小化被替换 group 的 MSE。
- finetune 不增加存储，只改变表值，仍然属于 O(1) LUT 查表。

---

**建议推进顺序**：
1. 先跑 `high_order` + `num_bits=12` + `num_tables=4` + `direct` 目标，看单 group cos_sim 是否比 `tree` 好。
2. 如果好，再试 `high_order` + `residual_input`。
3. 如果单 group 到 0.9 以上，再扩到 10 groups。
4. 如果直接 LUT 精度不够，优先试 `Coarse + Residual`（`--coarse_num_bits` + `--residual_num_bits`），用更大容量压残差。

---

## 输出说明

### 1. 每个 group 的 checkpoint

`outputs/checkpoints/replacement_g{gid}.pt` 包含：

- 单 LUT 模式下：
  - `address`: 地址生成器对象（tree / high_order / 2d 之一）
  - `lut_table`: FP16 的 LUT 表，形状 `[num_tables, num_entries, group_size]`
- Coarse + Residual 模式下：
  - `addresses`: 地址生成器对象的列表，长度 2（coarse 在前，residual 在后）
  - `lut_tables`: 对应 LUT 表的列表
- 公共字段：
  - `group_id`, `group_size`, `num_bits`, `coarse_num_bits`, `residual_num_bits`, `channels_per_bit`, `target_mode`, `address_mode`
  - `finetune_epochs`, `finetune_lr`：如果做了离线 LUT 表值微调，会记录这些参数
  - `group_mean`: 用于 `residual_mean` 目标

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
