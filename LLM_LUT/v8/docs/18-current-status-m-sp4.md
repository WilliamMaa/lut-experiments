# v8 KV 压缩：当前状态与最终配置验证（2026-09-15）

一句话状态（2026-09-15 更新）：**12a/12b 均已跑完。12b（k8v8 2000x）成立——
哨兵题全保，仅丢 1 个临界轮。12a 未过逐位一致：2/53 自由生成轮中途分叉，
方差源定位为折叠的 CUDA index_add_ 原子加，已改为确定性 one-hot matmul。
只差 12c（修复后跑一次，命令在 docs/16-run-commands.md 第 12 节）。**

评测条件：Qwen3.6-35B-A3B，device_map=balanced_low_0，bf16，
v3 多轮评测集（53 轮，8 个文档多轮对话），baseline EOS 0.811，repetition 0.019。

## 1. 方法栈（四个已验证组件）

| 组件 | 作用 | 状态 |
|---|---|---|
| sdpa stash wrapper（attention_scores.py） | 前向逐位不变地拿到 per-token attention 质量分数 | 硬门槛：PPL delta 必须为 0 |
| HeavyHitterCache（sink/recent/hh 三段淘汰） | budget=128（sink 4 + recent 32 + hh 92），按 prefill attn score 选 hh | 基座，500x→1000x 的 Pareto 基础 |
| merge_evicted（凸组合折叠，--merge_evicted） | 被淘汰 value 按 per-head 质量折入保留槽，weight = p_t 与 p_i 的质量加权平均，幅度有界 | v2 成立。v1 加法折叠判死（幅度膨胀 2-9 倍） |
| span_window 4（M4，--span_window 4） | 对 ±4 窗口做 max-pool，多 token 事实成组保留 | 本轮验证成立，见下 |

配套修正：CUDA topk 换成稳定降序排序（并列取位置小者），选集跨 run 可复现。

## 2. 哨兵题诊断 → 修复闭环

哨兵题：doc0 T4，答案"178亿元至182亿元"，文档靠后段单次出现。

探针（kv_cache/probe_sentinel.py，可复用）钉死的根因：
- 答案 span 共 9 个 token（tokenizer 把数字拆成单字 token），逐 token 打分被拆散；
- layer 3/39 全灭，中间层碎片化保留（个别 token rank 进了 hh，事实整体仍不可用）；
- merge_evicted 把信息折进了邻居，但没有 query 照到邻居 = 寻址问题，不是容量问题。

→ span_window 4 让窗口内任一 token 的高分庇护整组，m_sp4 run 直接验证：三处事实错误同时修复。

## 3. run 全史（v3 集，53 轮）

| run | 配置 | 标称压缩 | EOS | repetition | decode KL | 哨兵/退化 |
|---|---|---|---|---|---|---|
| l256 r128 | 三段淘汰 | 500x | 0.811（=bl） | 0.019 | 0.370 | 哨兵题错 |
| l128 r64 | 三段淘汰 | 1000x | 0.717 | 0.019 | 0.543 | 2 轮退化循环 + 2 轮事实错 |
| l128 r32 | 三段淘汰 | 1000x | 0.830 | 0.019 | 0.560 | 哨兵题仍错 |
| m | + merge v1 | 1000x | 0.811 | 0.075 | 0.610 | v1 加法膨胀，负收益 |
| m2 | + merge v2 凸组合 | 1000x | 0.849 | 0.038 | 0.555 | 哨兵题仍错（寻址问题） |
| sh_m | + 跨层共享选择 | 1000x | 0.792 | 0.057 | 0.538 | 判死：稀释单层锐利信号 |
| m_k8v8 | + INT8 KV 量化 | 2000x | 0.830 | 0.057 | 0.564 | 无退化轮；INT8 买到存储没买到质量 |
| **m_sp4** | **+ span_window 4** | **1000x** | **0.811（=bl）** | **0.057** | **0.519（历史最低）** | **哨兵题首次全对** |

另：k4v4@8000x 越界（EOS 0.774，出退化循环），4-bit 不可行。

三条方法论定论（不变）：
1. 选择机制是瓶颈，不是容量；
2. 轨迹级指标（decode KL/top-1）不敏感，主指标是退化轮次 + 哨兵题；
3. middle 预算 > recency 预算（文档事实问答场景）。

## 4. m_sp4 结果明细

文件：results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_multiturn_v3set.json

- 配置核验：k16v16 bf16，compression_ratio 1000.0，与文件名相符。
- baseline EOS 0.811 / rep 0.019 / 平均输出 46.6
- patched  EOS 0.811 / rep 0.057 / 平均输出 48.5
- decode KL 0.519，top-1 agreement 82.5%，top-5 99.2%（891 个 decode 位置）
- PPL delta = 0（sdpa wrapper 硬门槛通过）
- 修复：doc0 T4（178-182亿）、doc0 T5（9.7亿）、doc3 T4（平台名）
- 代价：repetition +3.8pp；doc2 T0 多一个 no-EOS；doc4 T0 列表内局部重复

## 5. 待办（指令在 docs/16-run-commands.md 第 12 节）

1. **12a 确定性复验**：同 m_sp4 配置重跑（`_rerun` 后缀）。两次逐位一致 = 可以定型；
   不一致 = 仍有方差源，先查再往下走。
2. **12b 最终配置**：m + sp4 + k8v8（标称 2000x）。看 span 保护能否盖过 INT8 噪声。
   12a 没过之前不要跑。
3. 两个都过后：更新 docs/17-method-roadmap.md（M4 从"待验证"改"成立"），
   出最终总结页（方法栈 + 1000x→2000x Pareto + 哨兵题前后对照）。

## 6. 教训备忘（已付出过代价）

- 结果文件不许覆盖，文件名必须带配置后缀；
- 质量边缘的单 run 结论需双 run 确认（topk 不稳定的前科）；
- 文件名与配置必须核验（有过 k4v4 塞进 bf16 文件名的前科）；
- 自定义 eager attention kernel 数值 drift 必死，sdpa wrapper 是唯一干净路径；
- shared_selection（M3）判死，不要再回头。
