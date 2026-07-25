# LLM_LUT v7: Jacobian 引导的大规模直接寻址

基于 `docs/00-ideas.md` 和 `docs/01-jacobian_direct_addressing_development_plan.md` 的实现。

## 项目结构

```
v7/
├── configs/                  # 配置文件
│   └── phase0_config.yaml   # Phase 0 配置示例
├── docs/                     # 设计文档
│   ├── 00-ideas.md          # 核心思路
│   └── 01-jacobian_direct_addressing_development_plan.md  # 开发计划
├── src/                      # 源代码
│   ├── data/                # 数据收集与处理
│   │   ├── activation_collector.py  # 激活值收集（参考 v6 参数风格）
│   │   └── anchor_builder.py        # Anchor 选择
│   ├── teacher/             # Teacher Path（仅离线使用）
│   │   └── exact_search.py  # 暴力 NN 搜索 + Jacobian 计算
│   └── evaluation/          # 评估指标
│       └── metrics.py       # 各类评估指标
├── scripts/                  # 运行脚本
│   └── run_phase0.py        # Phase 0 主脚本（参考 v6 参数风格）
└── README.md
```

## 核心约束（重要）

### 设备限制
- **必须使用单卡或 CPU**: `--device cuda:0` 或 `--device cpu`
- **禁止 auto mapping**: 严禁 `device_map="auto"` 等多卡分配机制
- **远端运行**: 所有代码设计为在远端服务器运行，本地仅审查

### 红线原则（来自 AGENTS.md）
1. **动态参数必须通过 LUT 查表生成**，不能是 MLP/HyperNetwork
2. **比较基准必须是同等计算量/参数量**
3. **准确率只是验证指标**，实验设计围绕"O(1) 查表加速"
4. **禁止自动多卡分配**

## Phase 0: 数据固化与 30k 基线复现

### 目标

1. 固化 activation 数据集
2. 复现 30k 暴力 NN + true Jacobian/JVP
3. 复现裸 anchor 基线
4. 固化评估脚本
5. **分离 teacher path 和 deployment path**

### 运行方式

参考 v6 的参数风格：

```bash
cd LLM_LUT/v7

python scripts/run_phase0.py \
    --teacher_weight_path /path/to/expert_0_L17.pt \
    --dataset_dir /path/to/activation_inputs \
    --output_root ./outputs/phase0_baseline \
    --device cuda:0 \
    --n_anchors 30000 \
    --calib_size 65536 \
    --eval_size 8192
```

可选参数（完整列表见脚本）：

```bash
python scripts/run_phase0.py --help
```

### 关键参数说明

| 参数 | 说明 | 参考 v6 |
|------|------|---------|
| `--teacher_weight_path` | Expert 权重路径 | ✅ 同 v6 |
| `--dataset_dir` | 输入数据目录 | ✅ 同 v6 |
| `--output_dataset_dir` | 预计算输出目录（可选） | ✅ 同 v6 |
| `--output_root` | 输出目录 | ✅ 同 v6 |
| `--device` | 设备（cuda:0 或 cpu） | ✅ 同 v6 |
| `--n_anchors` | Anchor 数量 | 🆕 v7 新增 |
| `--anchor_method` | 采样方法 | 🆕 v7 新增 |
| `--skip_jacobian` | 跳过 Jacobian 计算 | 🆕 v7 新增 |

### 输出文件

```
outputs/phase0_baseline/
├── checkpoints/
│   └── anchors.pt          # Anchor 数据
└── summary.json            # 完整评估结果
```

### Summary 内容

```json
{
  "bare_anchor_metrics": {
    "cosine_similarity": 0.85,
    "relative_l2": 0.15,
    ...
  },
  "jacobian_metrics": {
    "cosine_similarity": 0.95,
    "relative_l2": 0.05,
    ...
  },
  "storage_metrics": {
    "total_mib": 45.8,
    "full_jacobian_mib": 3662.1,
    "compression_ratio": 80.0
  }
}
```

## 核心公式

### 原方案（Teacher Path）

```
a*(x) = argmin_{a_i} ||x - a_i||
F_hat(x) = F(a*) + J(a*) @ (x - a*)
```

### 目标方案（Deployment Path）

```
i = A(x)                    # 直接地址映射（无搜索）
delta = x - a_i
F_hat(x) = F(a_i) + U_b [ c_i ⊙ V_b^T @ delta ]  # 共享低秩修正
```

## 评估指标

| 类别 | 指标 |
|------|------|
| **输出质量** | MSE, RMSE, Relative L2, Cosine Similarity, Norm Ratio |
| **路由质量** | Exact-address rate, Misrouting regret, Catastrophic routing rate |
| **存储** | Total MiB, Compression ratio vs full Jacobian |
| **成本** | MACs, Bytes/query, Lookup count |

## 开发阶段

| 阶段 | 内容 | 状态 | 代码入口 |
|------|------|------|----------|
| Phase 0 | 数据收集 + 30k 基线 | 🚧 进行中 | `scripts/run_phase0.py` |
| Phase 1 | 300k anchor scaling | ⏳ 待开始 | - |
| Phase 2 | Jacobian action 蒸馏 | ⏳ 待开始 | - |
| Phase 3 | Direct functional addressing | ⏳ 待开始 | - |
| Phase 4 | Robust addressing | ⏳ 待开始 | - |
| Phase 5 | 完整模拟闭环 | ⏳ 待开始 | - |
| Phase 6 | Triton kernel | ⏳ 暂不实现 | - |
| Phase 7 | 物理地址布局 | ⏳ 暂不实现 | - |

## 关键约束（再强调）

1. **禁止在线搜索**: Deployment path 不允许 FAISS/exact NN/bucket reranking
2. **禁止完整 Jacobian**: 不保存、读取或计算完整 Jacobian
3. **固定地址映射**: 在线阶段必须是直接映射，无二次检索
4. **纯 PyTorch**: 当前阶段只用 PyTorch，不做 CUDA kernel
5. **单卡限制**: 明确 `--device cuda:0`，禁止 auto mapping
