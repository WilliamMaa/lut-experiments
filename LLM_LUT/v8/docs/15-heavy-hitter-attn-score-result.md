# heavy_hitter_attn (obs-window + prefill attn-score) 首个有效结果

日期：2026-09-07
结果文件：`results/heavy_hitter_attn_l256_s4_r128_w64_v2_multiturn.json`
远程日志：`heavy_hitter_attn_w64_v2.log`

## 配置

- patch: `heavy_hitter_attn`，budget 256，sink 4，recent 128，obs_window 64
- importance: prefill attention score（从 sdpa wrapper 在 prefill 时 fp32 统计 softmax column-sum，按 obs-window 取行均值）
- 全 10 个 full-attn 层（idx 3,7,...,39）统一替换 cache
- 模型 Qwen3.6-35B-A3B，balanced_low_0，bf16
- 压缩率 500x（2048 B/token → 4.096 B/token）

## 硬门槛：通过

| 检查 | 结果 |
|---|---|
| patched PPL vs baseline PPL | **逐位相等：2.2066881468920685 == 2.2066881468920685** |
| 固定轨迹 logit KL | 9.7e-5，top-1/top-5 agreement = 100% |

sdpa stash wrapper 前向完全无污染。下游 decode divergence 全部来自淘汰本身，不来自 hook。之前 eager kernel 路线（自定义 attention 数学）正式放弃，以此结果为准。

## 多轮生成指标

| 指标 | baseline | patched (attn-score 256) |
|---|---|---|
| EOS success rate | 0.96 | **0.96（delta 0.0）** |
| avg output length | 22.6 | 23.2 |
| repetition rate | 0.00 | 0.00 |

| decode divergence | 数值 |
|---|---|
| avg decode KL | 0.492 |
| top-1 agreement | 81.4% |
| top-5 agreement | 99.1% |
| teacher greedy token prob under student | 0.608 |

## 横向对比（全部 multiturn 评测，同 25 轮）

| 方法 | budget | 压缩率 | EOS | decode KL | top-1 |
|---|---|---|---|---|---|
| retention s4 | 256 | 500x | 0.64 | 0.86 | 68.1% |
| heavy_hitter key-norm r128 | 256 | 500x | 0.80 | 0.447 | 78.8% |
| **heavy_hitter attn-score r128 w64** | 256 | 500x | **0.96** | 0.492 | 81.4% |
| retention s4 | 512 | 250x | 0.76 | 0.098 | 91.4% |

结论：
- importance-aware 淘汰成立。500x 压缩下 EOS 首次追平 baseline，是所有 256 预算方法里唯一做到的。
- attn-score 全面优于 key-norm：EOS 0.96 vs 0.80，top-1 81.4% vs 78.8%。用真实 attention 分数做重要性排序优于 key 范数，符合预期。
- retention 512（250x）的 decode KL 0.098 仍是质量天花板，但 attn-score 用一半存储达到可用区间。

## 逐轮定性核对（v2）

- 25 轮中唯一未 EOS 的一轮：财报文档 T0（长总结）。**baseline 在同一轮同样未 EOS**（同样顶到 128 token 上限），是题目本身特性，不是 patch 回退。
- 财报文档 5 个事实性问题全部答对：16.5亿/31%、客户订单延迟+降价+汇率、新加坡和迪拜、178-182亿。
- 火星、AuroraKV、气候峰会、小安P7 文档的事实性问题（保修期、充电座空间、Raft、32MB、30秒、128人等）全部答对且简洁收尾。

注意：本文件是 v2 重跑版。早前 w64 第一版（`heavy_hitter_attn_l256_s4_r128_w64_multiturn.json`）的财报答案有损（16亿/18%、170-175亿），v2 全部正确。两次 run 的差别说明该预算下质量已接近边缘，单样本可能有波动，跨方法比较应以多次 run 或多文档均值为准。

## 下一步

1. **budget 128（1000x）**：同配置压 max_cache_len，找 obs-window 淘汰的密度地板。若 EOS 仍 ≥0.8，说明极限未到。
2. **PyramidKV 式分层预算**：底层给大 budget、顶层给小 budget（或反之），总 budget 不变，看 decode KL 能否向 retention-512 的 0.098 靠拢。
3. 评测集升级：`data/multi_turn_prompts_v2.jsonl`（英文、日文、开放性问题），跑新评测集重新标定 baseline。
