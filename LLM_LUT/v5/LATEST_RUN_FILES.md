# 最近一次有效跑通的结果：Phase 2 Large（Sequential deployment-aware build）

## 结果摘要
- 脚本：`run_phase2_large.sh`
- 指标：PPL 28.08 / Acc 0.482 / MAC 削减 2.89% / LUT 24.5 MiB
- 输出目录：`outputs_sequential_large/`（刚被清理脚本删除）
- 微调结果：`results/finetune_joint_sequential_large/`

## 代码文件清单

| 文件 | 作用 | 是否手动修改过 |
|------|------|----------------|
| `run_phase2_large.sh` | 入口脚本：先 sequential build，再 joint fine-tune | 是，定义了 L15-L27 配置 |
| `build_lut_sequential.py` | **核心**：逐层部署 aware 地建 down_proj / o_proj LUT 并安装 | 是，修过 checkpoint save_dir bug |
| `finetune_joint.py` | **核心**：联合微调 down_proj + o_proj 的 LUT 表 | 是，支持 freeze_down / freeze_o |
| `eval_joint.py` | 只评估、不训练，用于快速看替换后指标 | 否 |
| `engine.py` | `HybridPartialEngine` / `HybridOProjEngine`，把 LUT 挂到模型上 | 否 |
| `lut.py` | `LUTGroup` 查表容器 | 否 |
| `address.py` | 地址生成器：`Address2D` / `AddressHighOrderRandom` / `AddressGreedyTree` | 否 |
| `build_lut.py` | 原始 down_proj LUT 构建器，被 `build_lut_sequential.py` 复用捕获/评估函数 | 否 |
| `build_lut_o_proj.py` | 原始 o_proj LUT 构建器，被 `build_lut_sequential.py` 复用捕获/评估函数 | 否 |
| `data.py` | 加载 `calib.jsonl` / `eval.jsonl` | 否 |
| `metrics.py` | PPL / Acc / KL / MAC 统计 | 否 |
| `utils.py` | 模型加载、基线 logits 收集 | 否 |

## 输入数据
- `../v0/data/calib.jsonl`：校准集（`CALIB_SIZE=512`）
- `../v0/data/eval.jsonl`：评估集（`EVAL_SIZE=128`）

## 关键配置（Phase 2 Large）
```bash
--down_configs "15:12,16:12,17:12,18:12,19:12,20:12,21:12,22:16,23:16,24:12,25:12,26:12,27:12"
--o_configs    "15:8,16:8,17:8,27:8"
--o_modes      "15:direct,16:direct,17:direct,27:delta"
--address_mode tree
--num_bits 10
--channels_per_bit 4
--tree_candidates 32
--tree_min_samples 32
--tree_max_samples 16384
```

## 生成的产物（跑完才有）
- `outputs_sequential_large/checkpoints/l{layer_id}/g{group_count}/address.pt`
- `outputs_sequential_large/checkpoints/l{layer_id}/g{group_count}/lut.pt`
- `outputs_sequential_large/checkpoints/l{layer_id}/g{group_count}/meta.json`
- `results/finetune_joint_sequential_large/summary.json`
- `results/finetune_joint_sequential_large/best_model.pt`（最佳 LUT 表权重）
