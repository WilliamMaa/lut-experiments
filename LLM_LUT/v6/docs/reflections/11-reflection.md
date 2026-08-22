# v3 On-Policy 改进实验记录

## 背景

从 0 重建 LLM_LUT/v6 pipeline（新服务器，8×A300，存储敏感）。
目标模型：Qwen3.6-35B-A3B，替换目标为 layer 39 的 `shared_expert`。

## 重建步骤

1. **提取 teacher**：从完整 35B MoE checkpoint 提取 `model.model.layers[39].mlp.shared_expert`，保存为 `/home/u/mmy/qwen_35b_shared_expert_l39.pt`。

2. **生成式 calibration 数据采集**：
   - 使用 1000 条 candidate prompts；
   - 每条 prompt 生成 512 个新 token；
   - 捕获 `shared_expert` 的输入/输出对；
   - 共采集约 60 万 token-level 样本，保存在 `datasets/layer39_shared_expert_v3_rollout/`。

   关键教训：必须让模型真正生成，而不是只过 prompt。prompt-only 数据只能覆盖 encoding 状态，看不到长生成中 layer 39 的真实访问分布。

3. **训练 v3 shared_coarse base**：
   - 14-bit shared global coarse + 16-bit per-group residual；
   - 32 个 group，group_size=64；
   - calib 500k / eval 100k；
   - 最终 eval cosine = 0.775，rel_l2 = 59.96%，norm_ratio = 0.958。

4. **格式转换**：将 v3 checkpoint 转为 v6 engine 可用的 `replacement_g*.pt` 格式。

5. **Sanity check**：
   - `original vs teacher` ≈ 1.0；
   - `offline_lut vs runtime_lut` ≈ 1.0；
   - `teacher vs offline_lut` ≈ 0.837（单 prompt）。

   确认实现本身无 bug，差距来自表达能力。

6. **开放长度生成测试**：
   - 用 1500 token、2048 token 分别测试；
   - 改用 prompt 内指令控制长度（"约2000字"/"around 1500 words"），而非硬截断；
   - 覆盖中文历史、英文经济、中文技术三个领域。

   初步结论：单 layer 替换后，生成长文本仍基本连贯，结尾较自然。

## On-Policy 数据与重训

7. **采集 on-policy 数据**：
   - 使用当前 v3 LUT 替换 layer 39 shared_expert；
   - 让模型在 100 条 prompt 上各生成 1024 token；
   - 记录 LUT 真实访问到的 FFN input；
   - 用独立 teacher 给这些状态重新打标签；
   - 按位置 + 困难度 + leaf 新颖性采样，最终保留 17988 条高价值样本。

8. **混合数据 resume finetune**：
   - 将 60 万 teacher rollout 样本与 17988 on-policy 样本合并；
   - 从已有 v3 checkpoint resume，不重建树，只更新 table 值；
   - coarse_finetune_epochs=0，residual_finetune_epochs=10，joint finetune 50 epochs。

## 结果

- **PPL**：baseline 8.00 → LUT 11.68（+3.68）。
- **主观生成质量**：相比纯 teacher rollout 训练的 v3，on-policy 后的 LUT 在长文本生成中更连贯，重复和突兀截断明显减少。
- **关键确认**：on-policy 数据确实能帮助 LUT 适应自己的 rollout 分布。

## 当前瓶颈

- PPL 仍比 baseline 高约 46%，单 layer 已能感知影响。
- 尚未做定量诊断（leaf coverage、residual PCA、pairwise interaction）。
- 尚未尝试 multi-layer 替换，这是最终目标。

## 下一步候选方向

1. **Multi-layer 扩展**：同时替换 layer 37/38/39 的 shared_expert，评估误差累积。
2. **更多 on-policy 数据**：扩大 prompt 数量或单次生成长度。
3. **结构改进**：若诊断显示 leaf residual 低秩，则加 per-leaf 低秩修正；若 group 交互强，则加 pairwise correction。
4. **诊断先行**：在扩层前先跑 leaf coverage / residual PCA / pairwise interaction，确认误差来源。

## 文件位置

- Teacher: `/home/u/mmy/qwen_35b_shared_expert_l39.pt`
- Rollout data: `/home/u/mmy/datasets/layer39_shared_expert_v3_rollout/`
- On-policy data: `/home/u/mmy/datasets/layer39_onpolicy_v3_100x1024/`
- Mixed data: `/home/u/mmy/datasets/layer39_mixed_v3/`
- v3 base: `/home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_rollout/`
- On-policy finetuned: `/home/u/mmy/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy/`
- Generation reports: `/home/u/mmy/generation_v3_openlength_3prompts.json`, `/home/u/mmy/generation_v3_onpolicy_2048_3prompts.json`
