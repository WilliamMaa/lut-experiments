# v8 并发测试 Runbook（操作手册）

日期：2026-09-17
适用机器：远程 `/data/mamingyu/v8`（模型 `/home/u/downloads/models/Qwen3.6-35B-A3B`，
≥8×80GB GPU，所有命令在该目录下执行）。
设计背景与改动说明见 `docs/21-concurrent-serving.md`，本文只讲**怎么跑**。

**原则：严格按 step 0→6 顺序执行。任何一步的判定不过，停止并排查，不要往下走。**

---

## 1. 测什么

在**固定 HBM 预算**（默认 512GB）下扫描并发数 N，比较 5 个配置的 serving 行为：

| 配置 | 含义 | 标称压缩 |
|---|---|---|
| `full` | 无压缩（DynamicCache），baseline | 1x |
| `hh` | attn-score 选择 l128/s4/r32/w64 | 1000x |
| `hh_merge` | + 凸组合折叠 | 1000x |
| `hh_merge_m4` | + span_window 4（定型配置 m_sp4） | 1000x |
| `m4_k8v8` | + INT8 KV | 2000x |

每个 cell（配置 × N）产出 6 项指标：

1. **TTFT**（prefill 均值）——各配置应基本持平（淘汰 deferred 到 decode）；
2. **TPOT**（逐 decode 步均值）——压缩配置在高档 N / 长上下文应显著更快；
3. **吞吐**（output tok/s，wall-clock 实算）；
4. **峰值 HBM**（全卡 `max_memory_allocated` 求和）——full 随 N 线性涨，压缩配置平线；
5. **EOS 率 / repetition 率**（质量不崩的底线指标）；
6. **fact accuracy**（对合成负载的 ground truth，含 multi-instruction 的 AND 判定）。

**判定规则**（写死在 harness 里，结果带 `sustainable` 标记）：cell 可持续 ⇔
HBM ≤ 预算 **且** EOS ≥ 同 N 下 full 配置的 EOS − 2pp。

**测试矩阵**：

| 长度档 | 配置 | 并发档 |
|---|---|---|
| 32k（主档） | 全部 5 个 | 1, 8, 16, 32, 64 |
| 64k | full, m4_k8v8（两端） | 8, 32 |
| 128k | full, m4_k8v8（两端） | 8, 32 |

负载：`data/longctx_multi_turn_{32k,64k,128k}.jsonl`，每文档 6 记录（区域唯一、
多位数事实）+ 无数字 filler + 8 问（≥1/4 digit_span，≥2 multi_instruction），
8 文档轮换分配给并发会话。

---

## 2. 前置条件

- [ ] 远程代码已同步本批改动（6 个文件）：
  `kv_cache/attention_scores.py`、`kv_cache/heavy_hitter_cache.py`、
  `kv_cache/probe_sentinel.py`、`kv_cache/concurrent_serve.py`（新）、
  `tools/gen_longctx_multiturn.py`（新）、`tools/analyze_concurrency.py`（新）
- [ ] GPU 空闲：`nvidia-smi` 确认无残留进程（历史 CUDA 死锁教训，见红线 5）
- [ ] 磁盘空间：`results/concurrency/` 每 cell JSON 数 MB 级，无压力

---

## 3. 总览与耗时估计

```
step 0  CPU cache 等价性 selftest     1 分钟        任意机器
step 1  生成 32k/64k/128k 负载        10–30 分钟    远程（带 tokenizer 校准）
step 2  GPU selftest（padding 不变性） ~1 小时       远程，占卡
step 3  B=1 回归门（复现定型档案）     ~2 小时       远程，占卡（可与 step 2 同日排队）
step 4  pilot cell（m4_k8v8 × N=8）   ~20 分钟      远程，占卡
step 5  全量矩阵                       数小时–1 天   远程，nohup，可断点续跑
step 6  出表 + 回填                    1 分钟
```

step 5 耗时以 pilot 实测外推：一个 32k / N=64 / 8 turns 的 cell，prefill 总量约
64 路 × 32–36k token × 8 轮 ≈ 18M token，加 decode。单个 cell 预计 15–40 分钟，
全矩阵（25 cell + 8 cell）预计 **4–16 小时**。

---

## 4. Step 0：CPU cache 等价性 selftest（1 分钟，先跑）

**测什么**：B=1 跑 N 次 与 B=N 批处理，在选择 / 凸折叠 / 量化（bf16 与 k8v8 两档）
上是否逐 session 一致。这是批化改动的第一道门，不依赖 GPU。

```bash
python kv_cache/concurrent_serve.py --cache-selftest
```

**判定**：输出两行 `PASS k16v16 ...` / `PASS k8v8 ...` + `ALL PASS`。

**失败怎么办**：断言会打印 `K/V/orig_idx mismatch b=...`。说明批化有形状/广播
bug——把完整输出发回来，不要继续。

---

## 5. Step 1：生成长上下文负载（10–30 分钟）

**测什么**：不测模型，生成三档长度负载并自检（token 长度、答案 grounded、
digit_span 占比）。

```bash
for T in 32768 65536 131072; do
  python tools/gen_longctx_multiturn.py \
    --target-tokens $T --num-docs 8 \
    --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --out data/longctx_multi_turn_${T}.jsonl
done
```

**判定**（生成器自带断言）：
- 每行 `tokens=` 在 target ±2% 内；
- 每个 ground truth 都 assert 在文档内（不在会直接抛错）；
- `digit_span` 占比 ≥ 1/4（日志行可见）。

**失败怎么办**：长度超差 → 调 `--tol 0.03` 或 `--chars-per-token`（默认 1.6）；
ground truth 断言失败 → 生成器 bug，带 doc 编号回报。

---

## 6. Step 2：GPU selftest——padding 不变性（占卡，~1 小时）

**测什么**：同一批会话，B=1 逐条跑 vs 拼成 B=4 批处理跑，m_sp4 配置下答案必须
**逐字一致**。这是 left-pad mask 与 per-batch 分数正确性的实证（批化第二道门）。

```bash
CUDA_LAUNCH_BLOCKING=1 python kv_cache/concurrent_serve.py --selftest \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --device_map balanced_low_0 --torch_dtype bfloat16
```

**判定**：`[selftest] PASS: 64 answers token-identical (B=1 sequential vs B=4 batched, m_sp4)`。

**失败怎么办**：会逐条打印 `MISMATCH #i: seq=... bat=...`。
- 所有答案都不同 → attention mask / 填充方向类大 bug；
- 个别开放式长输出分叉（事实题全对）→ 参照 docs/16 12a 的先例，轨迹混沌导致，
  可接受，但要在结果文档里记录；
- digit_span / factoid 题分叉 → 真 bug，停止，带样例回报。

---

## 7. Step 3：B=1 回归门——复现定型档案（占卡，~2 小时）

**测什么**：批化改动后，单流（batch=1）路径必须与定型结果逐位复现，证明算法未被
碰过。命令原样复制 docs/16 第 12c 节（m_sp4 det 配置，v3 评测集）：

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py \
  --patch heavy_hitter_attn \
  --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 \
  --merge_evicted --span_window 4 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl \
  --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl \
  --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 \
  --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics \
  --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_regress.json \
  > logs/heavy_hitter_attn_l128_m_sp4_det_regress.log 2>&1 &
```

**判定**（与 docs/19 定型档案对比）：

```bash
python tools/analyze_result.py results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_regress.json \
  --compare results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_multiturn_v3set.json
```

- EOS 0.8113207547、repetition 0.038、decode KL ≈ 0.519；
- 哨兵题（doc0 T4 178-182亿 / doc0 T5 9.7亿 / doc3 T4 平台名）逐字一致。

**失败怎么办**：聚合指标不符 → 批化在 B=1 路径引入了数学变化，停止排查
（重点查 `_position_scores` 的 mean 回退与 obs_window 排除的维度）。
聚合一致但自由生成分叉 → 参照 12a/12c 先例可接受，记录即可。

---

## 8. Step 4：pilot cell——单点打通（占卡，~20 分钟）

**测什么**：正式矩阵前，用一个 cell 验证 harness 端到端产出全部 6 项指标且数值合理。

```bash
CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --configs m4_k8v8 --concurrency-list 8 --turns 2 --max-new-tokens 64 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency \
  > serve_pilot.log 2>&1 &
```

**判定**：`results/concurrency/m4_k8v8_n8.json` 存在且：
- `status: ok`，6 项指标齐全无 null；
- fact accuracy 显著 > 0（合成题答案在文档里，正常情况下应 > 0.5；若 < 0.3
  先怀疑负载模板/聊天模板没对齐，别怀疑模型）；
- `peak_hbm_mb` 合理（8 路 × 32k × 20KB/token ≈ 5GB KV + 模型权重，应为几十 GB 级）。

**失败怎么办**：OOM → 降 `--concurrency-list 4` 重试并记录；CUDA assert → 带
`CUDA_LAUNCH_BLOCKING=1` 的完整栈回报。

---

## 9. Step 5：全量矩阵（nohup，可断点续跑）

**测什么**：step 1 表格里的全部 cell。每个 cell 独立 JSON，**重跑同配置同 N 即
覆盖续跑**，中断后从缺的 cell 继续即可。

32k 主档（25 cell，全 ladder × N∈{1,8,16,32,64}）——**以这条为准**（auto-pick
自动剔除被邻居霸占的卡；`PYTHONUNBUFFERED` 让 log 实时；`-u`/buffered 二选一）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --configs full,hh,hh_merge,hh_merge_m4,m4_k8v8 \
  --concurrency-list 1,8,16,32,64 \
  --turns 8 --max-new-tokens 128 \
  --kv-budget-gb 512 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency \
  > logs/concurrency_32k.log 2>&1 &
```

64k / 128k 两端对比（full vs m4_k8v8；64k 全 ladder，128k 预计 full 在 mid-N
即 OOM 可砍到 1,8,16）：

```bash
for T in 65536 131072; do
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --data_file data/longctx_multi_turn_${T}.jsonl \
    --configs full,m4_k8v8 --concurrency-list 1,8,16,32,64 \
    --turns 8 --max-new-tokens 128 \
    --kv-budget-gb 512 \
    --device_map balanced_low_0 --torch_dtype bfloat16 \
    --output-dir results/concurrency_${T} \
    > logs/concurrency_${T}.log 2>&1 &
done
```

**断点续跑语义**（2026-09-30 版 harness）：已有 cell JSON 且 `status: ok/oom` →
skip；`status: cuda_error` → 重试。**OOM/cuda_error 的 cell 若记录在被共享
邻居污染期间，删掉对应 JSON 再重启**（否则会被 skip 掉，拿不到干净判决）。

**运行中监控**（nohup 下 log 已实时，无需看缓冲 tricks）：

```bash
tail -f logs/concurrency_32k.log    # 每 cell 完成打一行 TTFT/TPOT/tok/s/HBM/EOS/fact
ls results/concurrency/             # *_n*.json 逐个出现
grep auto-selected logs/concurrency_32k.log   # 确认选卡剔除了邻居卡
```

**判定**（32k 实测修正版，详见 docs/21 结果节）：full 的 TPOT 随 N 恶化
~10×（32k KV 注意力），压缩配置仅 ~1.45×（128 slot）；**峰值 HBM 压缩配置
反超 full**（prefill 期间 KV 无约束增长，128-slot 红利只在 decode 后兑现）；
fact/EOS 无 N 趋势（并发不降解质量）。任何 cell 的 `status: oom` 本身是有效
数据点（记录，继续）。

**时间不够时的砍单顺序**（保留结论价值）：先砍 128k → 再砍 64k → 32k 矩阵至少保
N∈{1,16,64} × 5 配置。**不要**砍 step 0-3 验证门。

---

## 10. Step 6：出表与回填（1 分钟）

```bash
python tools/analyze_concurrency.py results/concurrency
python tools/analyze_concurrency.py results/concurrency_65536
python tools/analyze_concurrency.py results/concurrency_131072
python tools/analyze_concurrency.py results/concurrency --markdown > results/concurrency/table.md
```

把终端表格 + `table.md` 路径回填到 `docs/21-concurrent-serving.md` 的结果节，
并在 `results/concurrency/README.md` 记一句运行日期 / 机器 / 数据文件版本（seed）。

---

## 11. 故障速查

| 现象 | 第一动作 |
|---|---|
| `--cache-selftest` mismatch | 停止；带 `b=` 编号与 k/v bits 回报，批化形状 bug |
| `--selftest` MISMATCH | 看打印的 seq/bat 对照；事实题分叉=停止，开放题分叉=记录后继续 |
| step 3 聚合指标不符 | 停止；`_position_scores`/`_fold_evicted_values` 批化回归 |
| cell `status: oom` | 数据点，勿重试同档；降 N 补一个 cell。**但若 OOM 时 log 里出现某邻居 PID 占了 ~20GB，先删该 cell JSON 再重启**（auto-pick 会换卡重跑，否则被 skip） |
| `CUDA error: unspecified launch failure` | 多为内存墙异步爆发或邻居 XID；cell 记 `cuda_error` 后进程干净退出，**直接重跑同命令**（resume 自动重试该 cell）；第三次在同一 cell 复现才用 `compute-sanitizer` |
| CUDA device-side assert | `CUDA_LAUNCH_BLOCKING=1` 重跑同一命令拿真实栈 |
| fact accuracy 全 ~0 | 查负载（`data_file` 对不对、questions/answers 是否错位），不是模型问题 |
| 想改判定阈值 | `--kv-budget-gb` / `--eos-tolerance-pp`，改完重跑受影响 cell 并注明 |

**禁止事项**（项目红线）：`device_map="auto"` / accelerate 自动多卡分配；
在验证门（step 0-3）未全过时跑 step 5 全量矩阵。
