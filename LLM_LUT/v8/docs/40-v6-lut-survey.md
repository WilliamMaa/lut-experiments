# 40 · v6 LUT 家底调研（docs/39 前置：只调研，未改代码）

日期：2026-10-08 · 方式：只读盘点 v6 全部 docs/results/code（v0–v5、v7、v8 对照）

## 0. 一个重要纠正

用户记忆中的 **"down L21–23 + o L17" 不是 v6，是 v3/v4 线**（Qwen2.5-7B-Instruct dense）：13 层 down INT8（含 L21:12/L22:16/L23:16）MAC ↓2.21%、LUT 40.5MiB、PPL 27.67/Acc 0.449（`v4/docs/01-june-27-progress.md:50,69,79-83`）；o_proj 计划 `17:4,15:4,16:4,20:4`（`v4/docs/o_proj_experiment_plan.md:55`），最终结果本地无（只在远端跑过）。

**v6 本身是另一条线：Qwen3.6-35B-A3B（与 v8 同模型）**，替换 MoE 第 37–39 层的 `shared_expert` FFN。docs/39 的 "v6" 是混称，但论证不受影响。

## 1. v6 是什么

HF transformers（8 卡 balanced_low_0，bf16；**不是 vLLM**）。贪心决策树地址 + coarse/residual 双级均值表，整体替换 `layers[37..39].mlp.shared_expert`（2048 维 = 32 group × 64 通道）：14-bit shared coarse tree（16K leaf）+ 每组 16-bit residual tree（65K leaf），leaf 存 FP16 均值，输出 = coarse + Σ residual。每 token 33 次树遍历 + 33 次 gather，O(1)。训练：生成式采集 60 万样本 → 建树 + multi-loss 表值微调 → on-policy 重打标（17,988 条）resume finetune，多层从浅到深逐层 on-policy。

**致命现状：推理是 forward hook 后置覆盖，原 FFN 照算——MAC 节省恒为 0，且从未测过任何延迟/吞吐**（`v6/scripts/utils/v6_replacement_engine.py:226-245`、`v6/results/worstcase_32g_full_ffn_analysis.md:81`）。

## 2. 结果盘点（v6/results，全是 3–5 prompt 小样本 PPL，无 fact_acc / 无 benchmark / 无 MAC% / 无延迟）

| 配置 | PPL（baseline→LUT） | 表大小 | 出处 |
|---|---|---|---|
| **L37 单层（质量最好）** | 10.248→10.255（delta≈0） | ~320MiB | `outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_summary.json` |
| L37+L38 | 10.248→10.731 | ~640MiB | `generation_l37_l38_multilayer.json` |
| **L37–39 三层（最终形态）** | 6.075→9.123（+50%） | ~1GiB | `multilayer_l37_l39_5prompts_4096.json` |
| L39 单层（最佳单层但不是无损） | 8.00→11.68 | ~320MiB | `docs/conclusions/13-best-onpolicy-result.md:128-148` |

## 3. 慢的归因（有实测 + 结构证据）

- **唯一直实测**（v3 线，Qwen2.5-7B L21 down_proj，`v3/V3_IMPROVEMENT.md:23-56`）：真跳过 GEMM 的机制本身成立（partial matmul 省 24.4%），但 +LUT 生成 +输出重建后总计 **比 dense 慢 54%**。归因：dense GEMM 极强，LUT 的 index 生成 / kernel launch / 中间张量 / 重建全是额外开销。
- **v6 结构更重**：地址生成是 bit-serial 树遍历（15–17 轮顺序小 kernel × 33 棵树，每轮只处理 [N] 标量索引），gather 是 65K leaf × 64 维 × FP16 = 8KiB/行的随机访存（表 320MiB 超 L2，无合并）。decode 下 N=batch×1，纯靠 launch 开销和访存延迟——正是 docs/39 说的 memory-latency-bound。
- v6 全目录无 triton/CUDA（纯 PyTorch）；v4 有 triton fill kernel 但仍慢。

## 4. 代码资产

- 可复用：`build_lut_ffn_output_v3_shared_coarse.py`（建树+表值微调，模型无关输入 .pt 对，但类定义在 `__main__`，有序列化耦合）、`v6_replacement_engine.py`（查表循环本身模型无关；hook 路径焊死 HF）、convert 脚本。
- 焊死 HF、vLLM 不可用：采集脚本（hook_path eval HF forward）、eval 脚本（评估协议太弱，弃用）。
- **vLLM 侧无可抄嫁接**：v8/vllm_plugin 是 KV 压缩 attention backend，与 LUT 无关；v8/qwen_toolkit 的 transition LUT patch 是 HF 侧 crude 实验。v8 可复用的是评估脚手架（bench_concurrency / run_capacity_sweep / eval_longctx_server）和插件部署环境。

## 5. docs/39 最小验证的素材与判断

**LUT-small 候选 = v6 现成的 L37 单层 shared_expert checkpoint**（PPL delta≈0，320MiB，存在于 `outputs_ffn_lut_layer37_shared_expert_v3_onpolicy_as_v4/checkpoints`）。存储对 v8 释放的 headroom 是零头，瓶颈全在算得快不快。

落地 vLLM 的四步（工作量从大到小）：①真跳 GEMM（patch shared_expert module forward，必须）；②查表 kernel 化（拍平 33 棵树为 batched 遍历 + 单次 gather，理想 fused triton——**成败关键**）；③vLLM 集成（中等，复用 v8 的 serve/runbook 环境）；④评估复用 v8 脚手架（小）。

**风险判断**：v3 实测慢 54% 且 v6 表结构更重；docs/39 的门（v8+LUT ≥118 sess/h）能否过几乎全取决于 kernel 化质量。**建议先动 serving 之前跑一个 microbenchmark：L37 shared_expert 的 dense GEMM vs 拍平后的查表——单这一个数字就能预判整条路线，成本一小时以内。**
