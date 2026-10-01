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
- [ ] （2026-09-30 排查批）`kv_cache/concurrent_serve.py` 已同步最新
  （含 `SERVE_MEMLOG` 开关），`tools/probe_sdpa.py` 已同步（含 `S_REAL`）。
  **传完必须验证，没见过输出等于没传**：
  ```bash
  grep -n "SERVE_MEMLOG" kv_cache/concurrent_serve.py   # 期望: memlog_on / def memlog 等多行
  grep -n "S_REAL" tools/probe_sdpa.py                  # 期望: S_REAL = 60380 等 2 行
  ```
- [ ] （2026-09-30 反馈后批，docs/23）三个文件已同步：
  `kv_cache/concurrent_serve.py`（新增 `hbm_allocated_gb` /
  `fact_details` / 稳态 HBM 探针 / `--hbm-budget-gb` 改名 /
  `--max-cache-len` 覆盖）、`tools/analyze_concurrency.py`（新增
  steady / kv 两列）、`tools/paired_analysis.py`（新文件）。
  **验证（缺任何一条 = 没同步，不要开跑）**：
  ```bash
  grep -n "hbm_allocated_gb" kv_cache/concurrent_serve.py     # 期望: def hbm_allocated_gb(...) 行
  grep -n "fact_details" kv_cache/concurrent_serve.py         # 期望: 多处
  grep -n "max-cache-len" kv_cache/concurrent_serve.py        # 期望: add_argument 行
  grep -n "steady" tools/analyze_concurrency.py               # 期望: METRICS 定义行
  ls tools/paired_analysis.py                                 # 期望: 文件存在
  ```
- [ ] GPU 空闲：`nvidia-smi` 确认无残留进程（历史 CUDA 死锁教训，见红线 5）
- [ ] 磁盘空间：`results/concurrency/` 每 cell JSON 数 MB 级，无压力
- [ ] 目录存在：`mkdir -p results logs data`

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

32k 主档（25 cell，全 ladder × N∈{1,8,16,32,64}）——**实际执行请用
§12 Step D 的单行版命令**（多行反斜杠版仅供阅读，复制时换行处空格
会被吃掉导致 argparse 报错）：

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
即 OOM）。**先完成 §12 的显存诊断并拿到修法，再跑这两组**（2026-09-30：
64k N=32/64 的小缺口 OOM 根因尚未定位，直接重跑会原样复现）。**两条独立
命令，手动串行**：先启动 64k，等进程退出（`ps -ef | grep concurrent_serve`
无输出）再启动 128k。不要用 for 循环 + `&`（循环体内的 `&`
会让两次跑同时抢卡）。单行版同样见 §12 Step D。

```bash
# 第一条：64k
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_65536.jsonl \
  --configs full,m4_k8v8 --concurrency-list 1,8,16,32,64 \
  --turns 8 --max-new-tokens 128 \
  --kv-budget-gb 512 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency_65536 \
  > logs/concurrency_65536.log 2>&1 &

# 64k 跑完后再跑这条：128k
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_131072.jsonl \
  --configs full,m4_k8v8 --concurrency-list 1,8,16,32,64 \
  --turns 8 --max-new-tokens 128 \
  --kv-budget-gb 512 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency_131072 \
  > logs/concurrency_131072.log 2>&1 &
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

---

## 12. 追加：64k/128k 高并发 OOM 排查流程（2026-09-30）

**背景**：32k 主档矩阵已完成（结果在 docs/21）。64k 两端对比中，m4_k8v8
N=16 通过（TTFT 276.8s / TPOT 185ms / HBM 279.4GB / EOS 0.953 /
fact 0.4375），但 N=32、N=64 均以极小缺口 OOM（只差 258MiB / 512MiB，
卡上 PyTorch 已分配 77.97GiB）。本节定位根因并给出修法；**§9 的 64k/128k
命令在本节判定完成前不要跑**。

**已排除的假说**（`tools/probe_sdpa.py` 实测，torch 2.6.0+cu124）：

- ❌ "4D additive mask 把 sdpa 打到 math fallback 物化 64GB scores"——
  探针强制逐 backend 测试 64k 下 N=8/16/32/64 的精确形状，**efficient
  全部接受**（B=64/64k peak 63.7GiB 也能跑），math 才会 OOM 但不会被选中。
- ❌ "chunk 太大"——efficient 的 peak 构成是 HF `repeat_kv` 的 GQA 拷贝
  （K/V 2→16 头 expand+reshape，2×B×16×k×256×2B，**与 chunk 无关**：
  N=32/64k ≈ 31.7GB，N=64/64k ≈ 63GB）。曾据此加过 12e9 的 chunk cap，
  方向错误，**已回退**。
- ⚠️ 账算不平：模型分片 ~8.75GB + prefill 期全量 KV（update() 只追加
  不淘汰，N=32/64k ≈ 39.6GB 摊开）+ repeat_kv transient ~34.5GB +
  m4 mask ~3.75GB，最忙的卡预测峰值 ~52GB，**与实际 78GB 差 ~25GB 待查**。

### Step B：memlog 诊断跑（单 cell，~10 分钟，占卡）

**测什么**：`[memlog]` 每 chunk 打印 8 卡 allocated/high-water (GiB)，
区分"随 chunk 稳步爬升（累积型）"vs"某 chunk 突跳（transient 型）"，
以及最后一行停在 prefill 中间还是 decode begin 之后（=第一次淘汰/折叠）。

```bash
SERVE_MEMLOG=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_65536.jsonl \
  --configs m4_k8v8 --concurrency-list 32 --turns 1 --max-new-tokens 8 \
  --kv-budget-gb 512 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/memdiag_64k \
  > logs/memdiag_64k.log 2>&1 &
```

**取数**（跑完或 OOM 后均可）：

```bash
grep "memlog" logs/memdiag_64k.log
```

把完整输出发回来定修法。主要候选：prefill 期全量 KV 常驻
（`heavy_hitter_cache.py` update() 的 deferred-eviction 分支）或
repeat_kv 拷贝与 masks 在单卡叠加。

### Step B 结果（2026-09-30 实测，N=32/64k，chunk=1032）

memlog 显示两类增长，根因双双定位：

1. **resident 累积（真泄漏型）**：allocated 只有两张卡涨——gpu1
   +0.10GiB/chunk、gpu5 +0.55GiB/chunk（52 chunk 后 gpu5 allocated
   42.6GiB，即单卡攒了 ~30GB）。这就是 `heavy_hitter_cache.py`
   update() 的 deferred-eviction：prefill 期间只追加不淘汰，N=32×60k
   token × 20KB/token 的满血 KV 常驻到 decode 第一步。**机制结论：
   chunked prefill 期间（每轮最吃显存的阶段）压缩配置和 full 一样
   持有全量 KV，128-slot 红利只在 decode 第一步后兑现——这就是
   32k 矩阵峰值 HBM 倒挂的根源。**
2. **peak 棘轮（transient 型）**：每卡 peak 随 k_total 线性爬
   ~1.1GiB/chunk（gpu6 恰好减半 = 1 个 full-attn 层的量）。构成是
   HF `repeat_kv` 的 GQA 拷贝（K/V 2→16 头，0.52GiB/chunk/层）叠加
   cache cat 与 stash 的 fp32 副本。efficient backend 本身不物化
   scores（探针已证），但绕不开 repeat_kv 的实体拷贝。
3. **OOM 点**：chunk52/59（k≈54.7k，prefill 88%），gpu1 peak 77.5
   顶墙。N=16 能通过只是因为同样的棘轮减半。

### Step C：修法（绕过 repeat_kv 的 GQA 拷贝）

峰值里最大最无谓的一块是 repeat_kv 拷贝（N=32/64k ≈ 28GB，纯
内存搬运，数学零贡献）。**已否决：`enable_gqa=True`**——torch 2.6 的
efficient kernel 要求 dense 输入 Q/K/V 同头数，带 mask 时直接拒绝。
**现行方案：5-D stride-0 视图**（kernel 报错信息自己建议的
unsqueeze+expand 路线）：K/V 扩成 `[B, H_kv, n_rep, S, D]` stride-0
视图（零拷贝），Q 用 view 分组——head (a,b)=a·n_rep+b 读 kv-head a，
与 repeat_kv 的 interleaved 配对逐位一致。harness 的 stash wrapper
以"逐位委托 HF 原函数"为正确性铁律，所以**先探针验证位级等价且
无 backend 静默回退，任何非零差异/回退就否决此路**。

**C1. 同步 `tools/probe_gqa.py`（本地已写好）并跑位级等价探针**：

```bash
python tools/probe_gqa.py
```

判定：三行全 `bitwise_equal=True` 且 `fallback_warn=0` 且
`peak_5d ≪ peak_repeat` → 走 C2；否则放弃此路，直接跑 Step D
（按 N≤16 收尾 64k/128k）。

**C1 结果（2026-09-30 实测，三变体 v1/v2/v3 全灭，此路关闭）**：

- `enable_gqa=True`：torch 2.6 efficient kernel 要求 dense Q/K/V 同头数，
  带 mask 直接拒（flash 拒 mask）。
- 5-D stride-0 视图（kernel 报错建议的 expand 路线）：B=8 形状全部 OOM
  （29.75GiB 单次分配 = fp32 scores 物化，静默落到 math 系 backend）；
  小形状能跑但 `bitwise_equal=False`（max diff 2.4e-4，kernel/累加序不同），
  违反 stash wrapper"逐位委托 HF 原函数"的正确性铁律。
- 结论：**serving 侧接受 repeat_kv 的 GQA 拷贝为已知开销**，不再尝试
  零拷贝绕过。高并发上限由它 + prefill 满血 KV 共同决定（见 Step B
  结果节的两条机制）。

### Step D：重跑 64k/128k（Plan B 定案版）

**不要删 OOM cell**——`status: oom` 的 JSON 是有效数据点（探到显存墙的
位置本身就是结论），resume 会 skip 它们。先确认现有 cell：

```bash
ls results/concurrency_65536/
```

然后按 §9 两条命令手动串行重跑（64k 完再 128k；resume 自动跳过已有
ok/oom cell，只补缺的）。预期：64k/128k 下 full 因稳态 KV 线性膨胀
在 mid-N 即 OOM，m4_k8v8 通过 N=16、N≥32 撞墙（两条机制见 Step B
结果节）。最后 §10 出表，把机制结论 + 表格回填 docs/21。

**128k 补跑（单行版，整行复制——多行反斜杠命令在复制时会被吃掉
空格导致 argparse 报错，一律用下面的单行版）**：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_131072.jsonl --configs full,m4_k8v8 --concurrency-list 1,8,16,18,32,64 --turns 8 --max-new-tokens 128 --kv-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_131072 > logs/concurrency_131072.log 2>&1 &
```

（64k 若需补跑同理：把 `131072` 换成 `65536`、`concurrency_131072`
换成 `concurrency_65536`、log 文件名换掉，其余不动。）


---

## 13. 追加：docs/23 反馈后实验（2026-09-30）

背景：docs/23（同事/导师反馈）10 条意见全部成立，docs/21 已按此修订。harness
已扩展出新字段与新工具，但**旧 cell JSON（results/concurrency{,_65536,_131072}/）
没有这些字段**——它们是老版本 harness 跑的。本节给出获取新数据的命令。

### 13a. 重跑 cell 以获得新字段（fact_details / 稳态 HBM 列）

> **状态（2026-09-30 晚）：32k 全矩阵已在 `results/concurrency_v2/` 跑完**
> （命令即本节单行版），结果（稳态 HBM / KV 常驻表 + 配对分析 McNemar
> 矩阵 + v1-vs-v2 噪声底标定）已回填 docs/21 "v2 重跑"小节。**不要重跑。**
> 仍缺的：64k/128k 的 v2 cell（带新字段）与 64k m4_k8v8 N=16 的 EOS
> 复现 reps（§13a 砍单建议第 2 条）。

新 cell JSON 相比旧的多三个关键数据：

- `fact_details`：per-question `{session, turn, qtype, correct}` →
  `tools/paired_analysis.py` 的原料（docs/23 第 10 条）；
- `steady_decode_hbm_gb`：decode 循环后的稳态 HBM（首次淘汰后）；
- `kv_resident_bytes_end`：turn 结束时 cache 精确字节 →
  analyzer 的 steady / kv 两列（docs/23 第 6 条："压缩省 HBM"迄今无直接证据，
  这两列就是补证据的）。

**resume 语义注意**：已有且 `status: ok/oom` 的 cell 会被 **skip，不会重跑**。
要新数据只有两个办法：换 `--output-dir`（推荐，旧档保留可对照），或删掉想重跑
的 cell JSON。建议新档 `results/concurrency_v2/` 整档重跑 32k 主矩阵（旧档
results/concurrency/ 一个字节都不动）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs full,hh,hh_merge,hh_merge_m4,m4_k8v8 --concurrency-list 1,8,16,32,64 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/concurrency_v2 > logs/concurrency_v2_32k.log 2>&1 &
```

**砍单建议**（GPU 时间有限时，按价值排序保留）：
1. `full` 只保 N=8（EOS 基线 + 稳态对照），其余 N 用旧档数字；
2. 64k EOS 复现格（docs/23 第 3 条）：`m4_k8v8` × N=16 跑 **2–3 次**（每次
   换 output-dir 或先删 JSON），判定 0.953 是并发所致还是 run 级波动；
3. 压缩四配置 × N∈{8,32} 是 paired analysis 性价比最高的格子（n=64/格，
   McNemar 检出力最大）。

出表（analyzer 自动多 steady / kv 两列，旧档 JSON 无字段会显示空白，属正常）：

```bash
python tools/analyze_concurrency.py results/concurrency_v2 --markdown
```

### 13b. 组件配对分析（docs/23 第 10 条：ladder 叙事不成立）

需要 13a 的新 cell（有 `fact_details`）。对同一道题跨配置算 2×2 转移计数 +
精确二项 McNemar p，回答"M4 修了哪些题、merge 破坏了哪些题"：

```bash
python tools/paired_analysis.py results/concurrency_v2 --matrix
python tools/paired_analysis.py results/concurrency_v2 -n 8 -a hh_merge_m4 -b m4_k8v8
```

**判定**：`k8v8` vs `hh_merge_m4` 的 discordant 对（b+c）若 McNemar p>0.05，
则"INT8 不比 bf16-M4 差"成立，ladder 最后一级的质量损失叙事要改写；若 p<0.05
且 c>b，则量化确实在特定题上更差，需要看 qtype 分布找规律（输出里没有 qtype
分列，必要时把 fact_details 导出来按 qtype 手动分桶）。

### 13c. budget × context × fact Pareto 扫描（docs/23 第 9 条，真正的科学问题）

问题：128-slot 预算对长上下文 factual recall 过于激进（fact 0.44-0.53 vs
full ~1.0），Pareto 前沿在哪？扫
`max_cache_len ∈ {128,256,512,1024,2048} × {32k,64k,128k}`，测 fact / TPOT /
稳态 KV bytes。

harness 已支持 `--max-cache-len`（覆盖所有压缩配置的 128 默认值；配置名不变，
**每个 budget 必须用独立 output-dir**，否则 resume 会 skip）。

模板（**每个 budget 一条命令，手动串行**；先小档 pilot 再铺矩阵）：

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python kv_cache/concurrent_serve.py --model_path /home/u/downloads/models/Qwen3.6-35B-A3B --data_file data/longctx_multi_turn_32768.jsonl --configs m4_k8v8 --concurrency-list 8 --max-cache-len 256 --turns 8 --max-new-tokens 128 --hbm-budget-gb 512 --device_map balanced_low_0 --torch_dtype bfloat16 --output-dir results/budget_256_32k > logs/budget_256_32k.log 2>&1 &
```

- 只扫 `m4_k8v8`（末端配置，前面 hh/hh_merge/hh_merge_m4 的递进已被第 10 条
  证伪，不必重复）；
- N 先固定 8（质量问题的检出力来自 n=64 题数，不来自并发档数）；拿到
  初步 Pareto 后再决定是否补 N=16；
- 128 档不用跑（= results/concurrency_v2 已有）；2048 档预计 TPOT 优势收窄，
  但它回答"前沿在哪一端饱和"；
- 出表：各 budget 目录各跑一次 `analyze_concurrency.py`（看 fact + steady +
  kv 三列），手动拼 Pareto 表回填 docs/21。

### 13d. HBM 五分解 profiling（docs/23 第 5 条：74GB 未解释差额）

目标：把峰值拆成 weights / resident KV / stash / repeat_kv transient / other，
定位 compressed path 的 b·B·K 项。现有工具可分三步做，**不需要改代码**：

1. `SERVE_MEMLOG=1` 重跑 64k N=16 的 m4_k8v8 与 full 两个对照 cell
   （命令同 §12 Step B，把 `--concurrency-list` 改成 16、`--data_file` 换
   65536、`--output-dir` 换 `results/memdiag_64k_n16`），逐 chunk 对比两配置
   的 allocated 曲线差——差值从第几个 chunk 开始出现、斜率多少，直接圈定
   b·B·K 是累积型还是 transient 型；
2. 新 cell 的 `kv_resident_bytes_end`（13a）给稳态 KV 精确值，
   `steady_decode_hbm_gb` − KV bytes ≈ weights + stash 常驻 + allocator；
3. repeat_kv transient 用探针公式直接算（2·B·16·K·256·2B，chunk 无关），
   从峰值里减掉，剩下就是 stash + other。

**stash 目前无禁用开关**（`attention_scores.py` 无 env 开关），如果第 1 步的
差值曲线确认 stash 是主嫌，再考虑加 `--no-stash` 诊断开关（一次性改动，不进
正式配置）。

### 13e. 参数改名备忘

`--kv-budget-gb` 仍是 `--hbm-budget-gb` 的兼容别名，旧命令照跑不误；但新命令
一律写 `--hbm-budget-gb`（它是聚合进程 HBM 峰值预算，**不是 KV budget**——
docs/23 第 7 条）。固定 KV budget 的问题由 13c 的 budget 扫描回答。
