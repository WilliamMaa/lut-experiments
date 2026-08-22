# V6 文件夹整理计划

> 目标：把当前 V6 根目录下 40+ 个文件按功能归档，保留可运行的主线流程，把实验性的方向（bitwise、pairwise、tail-aware 等）放进 `future_work/`，避免混淆。

---

## 1. 当前问题

- 根目录文件太多：训练、数据收集、评估、诊断、辅助脚本全混在一起。
- `__pycache__/` 没有被忽略，仓库里有 Python 编译缓存。
- `scripts/02_lut_training/archive/` 嵌套太深，且根目录仍残留旧版本（`build_lut_ffn_output.py` 等）。
- bitwise 诊断结果一般（k-means 0.77），但文件和 plan 散落在根目录和 docs，容易让人误以为还在主线上。
- 没有一份清晰的 README 告诉新人哪个脚本是当前流程。

---

## 2. 整理原则

1. **最小破坏**：所有根目录脚本先保留；新结构通过复制/移动实现，旧路径用软链或 README 标注 deprecation。
2. **服务器同步友好**：V6 脚本通常在 `/data/mamingyu/` 平铺运行，彼此用同目录 import。新结构在仓库内按功能分组，但提供一键生成“服务器平铺视图”的工具。
3. **主线清晰**：当前主线只有一条：
   - 数据：`collect_shared_expert_data.py` → 可选 `collect_onpolicy_data.py`
   - 训练：`build_lut_ffn_output_v3_shared_coarse.py`
   - 转换：`convert_v3_to_v4_checkpoints.py`
   - 评估：`run_model_eval.py` / `run_multilayer_dialogue_eval.py`
   - 引擎：`v6_replacement_engine.py`
4. **实验性方向归档**：bitwise、pairwise、tail-aware、hard correction、lowrank 等未进入主线的实现，统一放到 `future_work/` 或 `archive/`。

---

## 3. 目标目录结构

```
LLM_LUT/v6/
├── README.md                         # V6 总览 + 快速开始 + 文件索引
├── requirements.txt
├── scenarios_eval.json
├── example_custom_prompts.jsonl
│
├── docs/
│   ├── README.md                     # docs 索引
│   ├── reflections/                  # 历次反思
│   │   ├── 00-ideas.md
│   │   ├── 01-reflection.md
│   │   └── ...
│   ├── plans/                        # 已执行或待执行的计划
│   │   ├── 06-plan-prompt-selection.md
│   │   ├── 14-multilayer-plan.md
│   │   └── 17-cleanup-plan.md
│   ├── conclusions/                  # 阶段性结论
│   │   ├── 13-best-onpolicy-result.md
│   │   └── v6-final-conclusion.md    # 新增：V6 最终结论
│   └── future_work/                  # 暂停/未验证的方向
│       └── bitwise/
│           ├── 15-bitwise-reflection.md
│           ├── 16-bitwise-plan.md
│           └── diagnose_bitwise_compressibility.py
│
├── scripts/                          # 可执行脚本（仓库内分组，服务器可平铺）
│   ├── data_collection/
│   │   ├── collect_shared_expert_data.py      # 主线
│   │   ├── collect_onpolicy_data.py           # 主线（可选）
│   │   ├── collect_moe_block_data.py
│   │   └── collect_moe_block_data_from_pt.py
│   ├── training/
│   │   ├── build_lut_ffn_output_v3_shared_coarse.py   # 主线
│   │   └── README.md                          # 说明这是当前唯一推荐训练器
│   ├── conversion/
│   │   ├── convert_v3_to_v4_checkpoints.py      # 主线
│   │   ├── extract_shared_expert.py
│   │   └── reconstruct_metadata.py
│   ├── evaluation/
│   │   ├── run_model_eval.py                  # 主线
│   │   ├── run_multilayer_model_eval.py
│   │   ├── run_multilayer_dialogue_eval.py    # 主线
│   │   ├── sanity_check_lut.py
│   │   └── check_onpolicy_sanity.py
│   ├── diagnostics/
│   │   ├── diagnose_leaf_coverage.py
│   │   ├── diagnose_leaf_residual_pca.py
│   │   ├── diagnose_pairwise_interaction.py
│   │   ├── diagnose_lut.py
│   │   └── analyze_data_distribution.py
│   └── utils/
│       ├── v6_replacement_engine.py             # 主线
│       ├── build_candidate_pool.py
│       ├── select_100_prompts.py
│       ├── cleanup_v6_outputs.py
│       ├── inspect_onpolicy_data.py
│       └── download_model.py
│
├── archive/                          # 旧版本/不再维护的训练器
│   ├── build_lut_ffn_output_v2.py
│   ├── build_lut_ffn_output_v3_independent.py
│   ├── build_lut_ffn_output_v3_shared_coarse.py   # 早期版本
│   ├── build_lut_ffn_output_v4_tail_aware.py
│   ├── build_lut_ffn_output.py
│   ├── build_lut_ffn_output_v3_lowrank.py
│   ├── build_pairwise_correction_v3.py
│   └── build_tail_aware_hard_correction.py
│
├── results/                          # 保留，但新增 README 索引
│   ├── README.md
│   └── ... (现有文件不动)
│
├── llm_lut/                          # 可选：未来把 v6_replacement_engine 拆成包
│   └── __init__.py
│
└── tools/                            # 仓库维护工具
    ├── flatten_for_server.py         # 把 scripts/ 下脚本平铺导出到指定目录
    └── .gitignore                    # 覆盖/更新根目录 .gitignore
```

---

## 4. 关键处理事项

### 4.1 Import 关系

当前大量脚本同目录 import：

```python
import build_lut_ffn_output_v3_shared_coarse as v3
from v6_replacement_engine import V6ReplacementEngine
```

移动到子目录后，有两种处理方案：

**方案 A：平铺部署时保留同目录 import（推荐）**
- 仓库内脚本分好组。
- `tools/flatten_for_server.py` 在部署时把所有脚本 + 引擎 + 训练器复制到同一个输出目录。
- 服务器上继续平铺运行，import 不变。

**方案 B：改成相对/包 import**
- 把 `llm_lut/` 发展为真正的包。
- 脚本变成 `from llm_lut.engine import V6ReplacementEngine`。
- 需要服务器上 `pip install -e .` 或设置 `PYTHONPATH`。
- 更干净，但改动量大，容易出路径问题。

**建议：先按方案 A 执行**，只改文件位置、不改 import；后续如果要把 V6 做成可安装包，再做方案 B。

### 4.2 Bitwise 归档

- `docs/15-bitwise-reflection.md` → `docs/future_work/bitwise/15-bitwise-reflection.md`
- `docs/16-bitwise-plan.md` → `docs/future_work/bitwise/16-bitwise-plan.md`
- `diagnose_bitwise_compressibility.py` → `docs/future_work/bitwise/diagnose_bitwise_compressibility.py`
- 在 `v6-final-conclusion.md` 中写明：bitwise k-means 拆分只能到 0.77，未达 0.85 可用线，暂时放 future work。

### 4.3 旧训练器归档

以下脚本不再作为主线推荐，移到 `archive/`：
- `build_lut_ffn_output.py`
- `build_lut_ffn_output_v3_lowrank.py`
- `build_lut_ffn_output_v4_tail_aware.py`
- `build_pairwise_correction_v3.py`
- `build_tail_aware_hard_correction.py`

注意：`scripts/training/build_lut_ffn_output_v3_shared_coarse.py` 是当前唯一保留在 training 下的训练器。

### 4.4 `__pycache__` 清理

- 删除现有 `LLM_LUT/v6/__pycache__/`。
- 在 `LLM_LUT/v6/.gitignore` 或根目录 `.gitignore` 中加入：

```gitignore
__pycache__/
*.py[cod]
*.so
*.pt
*.log
```

---

## 5. 执行步骤

### Step 1：文档与结论（先不碰脚本）
1. 创建 `docs/reflections/`、`docs/plans/`、`docs/conclusions/`、`docs/future_work/bitwise/`。
2. 把现有 reflection/plan md 按编号/主题移入对应目录。
3. 新建 `docs/conclusions/v6-final-conclusion.md`，总结：
   - 主线方法：shared coarse + per-group residual，on-policy 蒸馏，L37/38/39 多层替换。
   - 最好结果：三层联合对话评估基本保持 baseline 质量。
   - 放弃/暂停方向：bitwise（0.77 < 0.85）、pairwise、lowrank、tail-aware。

### Step 2：脚本分组（复制式移动，不破坏旧路径）
1. 创建 `scripts/` 子目录。
2. 把脚本复制到对应子目录。
3. 在根目录保留 **软链接** 或 **deprecated 说明**，避免用户旧命令立刻失效。
4. 把旧训练器移到 `archive/`。

### Step 3：部署工具
1. 实现 `tools/flatten_for_server.py`：
   - 输入：仓库 V6 路径、输出目录。
   - 输出：把 `scripts/*/*.py`、`v6_replacement_engine.py`、`build_lut_ffn_output_v3_shared_coarse.py` 等平铺复制到输出目录。
   - 这样 `/data/mamingyu/` 可以继续平铺运行。

### Step 4：验证
1. 对平铺后的目录跑 `python -m py_compile *.py`。
2. 确认关键脚本能 import 彼此。
3. 删除根目录的 `.pyc` 和 `__pycache__`。

---

## 6. 风险与回退

- **风险**：移动脚本后，服务器上的旧路径命令失效。
  - ** mitigation**：根目录保留 1-2 周软链接；README 明确新路径；提供 flatten 工具。
- **风险**：import 路径在子目录里失效。
  - **mitigation**：方案 A 不改 import，平铺部署即可。
- **风险**：归档的 bitwise/pairwise 脚本以后找不到。
  - **mitigation**：`archive/` 和 `future_work/` 在 README 中索引清楚。

---

## 7. 建议的下一步

1. 如果同意方案 A（平铺部署，不改 import），我可以先执行 **Step 1 文档整理** + **Step 2 创建新目录并复制脚本** + **Step 4 清理 pycache**。
2. `flatten_for_server.py` 可以第二步再做，因为目前手动把脚本拖到 `/data/mamingyu/` 也成立。
3. 整理完后，V6 的“当前可跑流程”应该只剩：
   - `scripts/data_collection/collect_shared_expert_data.py`
   - `scripts/training/build_lut_ffn_output_v3_shared_coarse.py`
   - `scripts/conversion/convert_v3_to_v4_checkpoints.py`
   - `scripts/evaluation/run_model_eval.py`
   - `scripts/evaluation/run_multilayer_dialogue_eval.py`
   - `scripts/utils/v6_replacement_engine.py`
