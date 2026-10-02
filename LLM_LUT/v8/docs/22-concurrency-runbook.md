# v8 并发测试 Runbook（操作手册）

适用机器：远程（模型 `/home/u/downloads/models/Qwen3.6-35B-A3B`，≥8×80GB GPU，
所有命令在该目录下执行）。

**本文只讲怎么跑**。结论与机制见 `docs/24-concurrency-postmortem.md`，
完整结果表与统计检验见 `docs/21-concurrent-serving.md`。
**下一阶段（真实 serving 栈 concurrency 测量，docs/25 定案）见
`docs/26-next-phase-serving.md`；本 runbook 覆盖的是其 diagnostic 前身的
lockstep harness。**

**命令一律用单行版**。多行反斜杠命令复制时换行处空格会被吃掉导致
argparse 报错（`unrecognized arguments`）。

**原则：严格按 step 0→6 顺序执行。任何一步的判定不过，停止并排查，不要往下走。**

---

## 1. 测什么

在**固定 HBM 预算**（默认 512GB，参数 `--hbm-budget-gb`）下扫描并发数 N，
比较 5 个配置的 serving 行为：

| 配置 | 含义 | 标称压缩 |
|---|---|---|
| `full` | 无压缩（DynamicCache），baseline | 1x |
| `hh` | attn-score 选择 l128/s4/r32/w64 | 1000x |
| `hh_merge` | + 凸组合折叠 | 1000x |
| `hh_merge_m4` | + span_window 4（定型配置 m_sp4） | 1000x |
| `m4_k8v8` | + INT8 KV | 2000x |

每个 cell（配置 × N）产出指标：TTFT、TPOT、吞吐（output tok/s）、
峰值 HBM（全卡 max_memory_allocated 求和）、EOS 率、fact accuracy
（对合成负载 ground truth 的子串判定），以及三个新字段
`fact_details` / `steady_decode_hbm_gb` / `kv_resident_bytes_end`。

harness 内建 sustainability 规则（quality-blind，只作记录不作结论）：
HBM ≤ 预算 且 EOS ≥ 同 N full 配置 EOS − 2pp。该规则的解读限制见
docs/21 判定标准节。

**测试矩阵**：

| 长度档 | 配置 | 并发档 |
|---|---|---|
| 32k（主档） | 全部 5 个 | 1, 8, 16, 32, 64 |
| 64k | full, m4_k8v8 | 1, 8, 16, 32, 64 |
| 128k | full, m4_k8v8 | 1, 8, 16, 18, 32, 64 |

负载：`data/longctx_multi_turn_{32768,65536,131072}.jsonl`，每文档 6 记录
（区域唯一、多位数事实）+ 无数字 filler + 8 问（≥1/4 digit_span，
≥2 multi_instruction），8 文档轮换分配给并发会话。

---

## 2. 前置条件

- [ ] 代码已同步并**验证**（传完必须跑验证命令，没见过输出等于没传）：
  ```bash
  grep -n "SERVE_MEMLOG" kv_cache/concurrent_serve.py     # 期望: memlog_on / def memlog 等多行
  grep -n "hbm_allocated_gb" kv_cache/concurrent_serve.py # 期望: def hbm_allocated_gb(...) 行
  grep -n "fact_details" kv_cache/concurrent_serve.py     # 期望: 多处
  grep -n "max-cache-len" kv_cache/concurrent_serve.py    # 期望: add_argument 行
  grep -n "steady" tools/analyze_concurrency.py           # 期望: METRICS 定义行
  ls tools/paired_analysis.py                             # 期望: 文件存在
  ```
- [ ] GPU 空闲：`nvidia-smi` 确认无残留进程（历史 CUDA 死锁教训，见红线 5）
- [ ] 磁盘空间：`results/` 每 cell JSON 数 MB 级，无压力
- [ ] 目录存在：`mkdir -p results logs data`

---

## 3. 总览与耗时

```
step 0  CPU cache 等价性 selftest     1 分钟        任意机器
step 1  生成 32k/64k/128k 负载        10–30 分钟    远程（带 tokenizer 校准）
step 2  GPU selftest（padding 不变性） ~1 小时       远程，占卡
step 3  B=1 回归门（复现定型档案）     ~2 小时       远程，占卡（可与 step 2 同日排队）
step 4  pilot cell（m4_k8v8 × N=8）   ~20 分钟      远程，占卡
step 5  全量矩阵                       数小时–1 天   远程，nohup，可断点续跑
step 6  出表                           1 分钟
```

---

## 4. Step 0：CPU cache 等价性 selftest（1 分钟，先跑）

**测什么**：B=1 跑 N 次 与 B=N 批处理，在选择 / 凸折叠 / 量化（bf16 与 k8v8 两档）
上是否逐 session 一致。批化改动的第一道门，不依赖 GPU。

```bash
python kv_cache/concurrent_serve.py --cache-selftest
```

**判定**：输出两行 `PASS k16v16 ...` / `PASS k8v8 ...` + `ALL PASS`。

**失败怎么办**：断言会打印 `K/V/orig_idx mismatch b=...`。说明批化有形状/广播
bug——把完整输出发回来，不要继续。

---

## 5. Step 1：生成长上下文负载（10–30 分钟）

```bash
for T in 32768 65536 131072; do
  python tools/gen_longctx_multiturn.py \
    --target-tokens $T --num-docs 8 \
    --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --out data/longctx_multi_turn_${T}.jsonl
done
```

**判定**（生成器自带断言）：每行 `tokens=` 在 target ±2% 内；每个 ground
truth 都 assert 在文档内；`digit_span` 占比 ≥ 1/4。

**失败怎么办**：长度超差 → 调 `--tol 0.03` 或 `--chars-per-token`（默认 1.6）；
ground truth 断言失败 → 生成器 bug，带 doc 编号回报。

---

## 6. Step 2：GPU selftest——padding 不变性（占卡，~1 小时）

**测什么**：同一批会话，B=1 逐条跑 vs 拼成 B=4 批处理跑，m_sp4 配置下答案必须
**逐字一致**。left-pad mask 与 per-batch 分数正确性的实证（批化第二道门）。

```bash
CUDA_LAUNCH_BLOCKING=1 python kv_cache/concurrent_serve.py --selftest --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --device_map balanced_low_0 --torch_dtype bfloat16
```

**判定**：`[selftest] PASS (harness-clean): turn 0 token-identical for all 4 sessions`。
turn≥1 的分叉打印记录但不判死（批处理浮点噪声在有损重压缩上的固有性质）。

**失败怎么办**：逐条打印 `MISMATCH #i: seq=... bat=...`。
- turn 0 任何分叉 = harness/pad/mask 硬 bug，停止；
- 事实题分叉 → 真 bug，停止，带样例回报。

---

## 7. Step 3：B=1 回归门——复现定型档案（占卡，~2 小时）

**测什么**：批化改动后，单流（batch=1）路径与定型结果复现，证明算法未被碰过。
命令原样复制 docs/16 第 12c 节（m_sp4 det 配置，v3 评测集）：

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/eval_kv_cache.py --patch heavy_hitter_attn --max_cache_len 128 --sink_tokens 4 --recent_tokens 32 --obs_window 64 --merge_evicted --span_window 4 --model /home/u/downloads/models/Qwen3.6-35B-A3B --eval_file v8_eval_texts.jsonl --prompt_file candidate_prompts.jsonl --multi_turn --multi_turn_file data/multi_turn_prompts_v3.jsonl --max_eval_samples 8 --max_new_tokens 128 --max_length 4096 --device_map balanced_low_0 --torch_dtype bfloat16 --logit_metrics --output_json results/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_regress.json > logs/heavy_hitter_attn_l128_m_sp4_det_regress.log 2>&1 &
```

**判定**（与 docs/19 定型档案对比）：

```bash
python tools/analyze_result.py results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_regress.json --compare results/heavy_hitter/heavy_hitter_attn_l128_s4_r32_w64_m_sp4_det_multiturn_v3set.json
```

- EOS 0.8113207547、repetition 0.038、decode KL ≈ 0.519；
- 哨兵题（doc0 T4 178-182亿 / doc0 T5 9.7亿 / doc3 T4 平台名）逐字一致。

**失败怎么办**：聚合指标不符 → 批化在 B=1 路径引入了数学变化，停止排查
（重点查 `_position_scores` 的 mean 回退与 obs_window 排除的维度）。

---

## 8. Step 4：pilot cell——单点打通（占卡，~20 分钟）

```bash
CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs m4_k8v8 --concurrency-list 8 --turns 2 --max-new-tokens 64 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_pilot > logs/serve_pilot.log 2>&1 &
```

**判定**：`results/concurrency_pilot/m4_k8v8_n8.json` 存在且：
- `status: ok`，指标齐全无 null；
- fact accuracy 显著 > 0（正常情况下应 > 0.5；若 < 0.3 先怀疑负载/聊天模板
  没对齐，别怀疑模型）；
- `peak_hbm_mb` 为几十 GB 级（8 路 × 32k × 20KB/token ≈ 5GB KV + 权重分片）。

**失败怎么办**：OOM → 降 `--concurrency-list 4` 重试并记录；CUDA assert →
`CUDA_LAUNCH_BLOCKING=1` 重跑拿真实栈。

---

## 9. Step 5：全量矩阵（nohup，可断点续跑）

**resume 语义**：已有 cell JSON 且 `status: ok/oom` → skip 不重跑；
`status: cuda_error` → 重试。**要重跑某格必须换 `--output-dir` 或先删该格
JSON**。OOM 的 cell 若是有效数据点（探到显存墙），**不要删**。

**32k 主档**（25 cell，全 ladder × N∈{1,8,16,32,64}）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs full,hh,hh_merge,hh_merge_m4,m4_k8v8 --concurrency-list 1,8,16,32,64 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency > logs/concurrency_32k.log 2>&1 &
```

**64k**（上一条跑完后手动串行）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_65536.jsonl --configs full,m4_k8v8 --concurrency-list 1,8,16,32,64 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_65536 > logs/concurrency_65536.log 2>&1 &
```

**128k**（64k 跑完后手动串行；N=18 为探墙加档，可删）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_131072.jsonl --configs full,m4_k8v8 --concurrency-list 1,8,16,18,32,64 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_131072 > logs/concurrency_131072.log 2>&1 &
```

**运行中监控**：

```bash
tail -f logs/concurrency_32k.log    # 每 cell 完成打一行 TTFT/TPOT/tok/s/HBM/EOS/fact
ls results/concurrency/             # *_n*.json 逐个出现
grep auto-selected logs/concurrency_32k.log   # 确认选卡剔除了邻居卡
```

**判定**：各配置指标随 N 的形态预期与解读，见 docs/21 结果节与
docs/24 §1（显存墙机制）。任何 cell 的 `status: oom` 本身是有效数据点
（记录，继续）。

**时间不够时的砍单顺序**：先砍 128k → 再砍 64k → 32k 至少保
N∈{1,16,64} × 5 配置。**不要**砍 step 0-3 验证门。

---

## 10. Step 6：出表

```bash
python tools/analyze_concurrency.py results/concurrency --markdown
python tools/analyze_concurrency.py results/concurrency --markdown > results/concurrency/table.md
```

budget 扫描档（§12b）与复现档（§12c）同理对各自目录跑。

---

## 11. 故障速查

| 现象 | 第一动作 |
|---|---|
| `--cache-selftest` mismatch | 停止；带 `b=` 编号与 k/v bits 回报，批化形状 bug |
| `--selftest` turn 0 MISMATCH | 停止；attention mask / 填充方向 bug |
| step 3 聚合指标不符 | 停止；`_position_scores`/`_fold_evicted_values` 批化回归 |
| cell `status: oom` | 数据点，勿重试同档。**但若 OOM 时 log 里出现邻居 PID 占 ~20GB，先删该 cell JSON 再重启**（auto-pick 会换卡重跑，否则被 skip） |
| `CUDA error: unspecified launch failure` | 多为内存墙异步爆发或邻居 XID；cell 记 `cuda_error` 后进程干净退出，直接重跑同命令（resume 自动重试）；同一 cell 第三次复现才用 `compute-sanitizer` |
| CUDA device-side assert | `CUDA_LAUNCH_BLOCKING=1` 重跑同一命令拿真实栈 |
| fact accuracy 全 ~0 | 查负载（`data_file` 对不对、questions/answers 是否错位），不是模型问题 |
| 想改判定阈值 | `--hbm-budget-gb` / `--eos-tolerance-pp`，改完重跑受影响 cell 并注明 |

**禁止事项**（项目红线）：`device_map="auto"` / accelerate 自动多卡分配；
在验证门（step 0-3）未全过时跑 step 5 全量矩阵。

---

## 12. 追加实验指令

以下指令对应 docs/23 反馈后的扩展实验。**已完成的批次清单见
docs/24 §4，跑之前先查那张表，不要重复跑。**

### 12a. 带新字段的整档重跑（v2 档）

需要 `fact_details` / 稳态列的数据时（配对分析、稳态 HBM 分析的前提），
整档换 output-dir 重跑（旧档一个字节不动）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs full,hh,hh_merge,hh_merge_m4,m4_k8v8 --concurrency-list 1,8,16,32,64 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_v2 > logs/concurrency_v2_32k.log 2>&1 &
```

### 12b. budget × fact Pareto 扫描

`--max-cache-len` 覆盖压缩配置的 slot 预算（配置名不变，**每个 budget
必须用独立 output-dir**）。模板（换 budget 值与目录/log 名即可，手动串行）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs m4_k8v8 --concurrency-list 8 --max-cache-len 256 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/budget_256_32k > logs/budget_256_32k.log 2>&1 &
```

扩展：32k 已扫完 128-2048；把 `data_file` 换 65536/131072、`output-dir`
换 `budget_*_64k/128k` 即迁档。

出表（逐目录抽 m4_k8v8 行）：

```bash
for d in results/budget_256_32k results/budget_512_32k; do echo "== $d =="; python tools/analyze_concurrency.py $d --markdown | grep -E "m4_k8v8" | grep -E "ms|0\."; done
```

### 12c. 单格复现（reps）

判定某格指标是否 run 波动：同一格换 output-dir 重跑 2-3 次，跑完汇总对比：

```bash
for d in results/repro_X_run1 results/repro_X_run2 results/repro_X_run3; do echo "== $d =="; python tools/analyze_concurrency.py $d --markdown | grep -E "EOS success|Fact accuracy" -A2; done
```

### 12d. 组件配对分析

前提：目录里的 cell 带 `fact_details`（12a 的 v2 档已满足）：

```bash
python tools/paired_analysis.py results/concurrency_v2 --matrix
python tools/paired_analysis.py results/concurrency_v2 -n 8 -a hh_merge_m4 -b m4_k8v8
```

判读：McNemar p<0.05 才算配对差异显著；聚合 fact 差 <10pp 不具分辨率
（噪声底，docs/21 v2 节）。

### 12e. 显存诊断（memlog / 探针）

memlog 单 cell 诊断跑（每 chunk 打 8 卡 allocated/peak，区分累积型 vs
transient 型增长）：

```bash
SERVE_MEMLOG=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_65536.jsonl --configs m4_k8v8 --concurrency-list 16 --turns 1 --max-new-tokens 8 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/memdiag > logs/memdiag.log 2>&1 &
```

```bash
grep "memlog" logs/memdiag.log
```

backend/GQA 形状探针：

```bash
python tools/probe_sdpa.py
python tools/probe_gqa.py
```

已知结论（不要重跑探针去"验证"）：mask 不打 math fallback、chunk cap
无效、`enable_gqa` 与 5-D stride-0 均被 torch 2.6 拒绝/回退——明细见
docs/24 §2。

### 12f. HBM 五分解 profiling（待办：解释 74GB 差额）

步骤：
1. `SERVE_MEMLOG=1` 跑 compressed 与 full 两个对照 cell（命令同 12e，
   两次跑换 `--configs` 与 `--output-dir`），逐 chunk 对比 allocated
   曲线差：差值从第几个 chunk 出现、斜率多少 → 圈定 b·B·K 是累积型
   还是 transient 型；
2. v2 cell 的 `kv_resident_bytes_end` 给稳态 KV 精确值，
   `steady_decode_hbm_gb` − KV bytes ≈ weights + stash 常驻 + allocator；
3. repeat_kv transient 用公式直接算（2·B·16·K·256·2B，chunk 无关），
   从峰值减掉，剩下就是 stash + other。

stash 目前无禁用开关（`attention_scores.py` 无 env 开关）；若第 1 步
确认 stash 是主嫌，再加 `--no-stash` 诊断开关（一次性改动，不进正式配置）。

### 12g. 参数备忘

`--kv-budget-gb` 是 `--hbm-budget-gb` 的兼容别名；新命令一律写
`--hbm-budget-gb`（聚合进程 HBM 峰值预算，不是 KV budget）。
