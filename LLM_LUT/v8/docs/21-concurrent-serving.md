# v8 并发 Serving 测试（docs/20 第二步落地）

**操作手册（怎么跑、每步测什么、失败怎么办）见 `docs/22-concurrency-runbook.md`，本文是设计/变更/结果记录。**

日期：2026-09-17
远程目录：`/data/mamingyu/v8`（所有命令在该目录下执行）
前置：方法栈已定型（docs/19：m_sp4 1000x EOS=baseline，m_sp4+k8v8 2000x 哨兵题全保）。

## 目标

把 v8 KV 压缩放进真实并发场景。在**固定 HBM 预算**下扫描并发数 N，按 ladder 比较：

```text
full          无压缩（DynamicCache，baseline）
hh            attn-score 选择（l128/s4/r32/w64）
hh_merge      + 凸组合折叠
hh_merge_m4   + span_window 4（= m_sp4，1000x）
m4_k8v8       + INT8 KV（= 2000x）
```

指标：sustainable concurrency、TTFT、TPOT、吞吐（padded/effective）、峰值 HBM、
EOS / repetition / fact accuracy（对 ground truth）。

## 本批改动（为并发做的扩展，算法不变）

| 文件 | 改动 |
|---|---|
| `kv_cache/attention_scores.py` | score bank 批化：`scores [K]→[B,K]`、`scores_per_head [H,K]→[B,H,K]`。B=1 数值不变 |
| `kv_cache/heavy_hitter_cache.py` | 逐 batch：`orig_idx [B,len]`、逐 batch 选择/折叠/量化。`_shared_position_scores`（M3，已判死）batch>1 显式 raise |
| `kv_cache/probe_sentinel.py` | 探针适配 `[1,K]` bank 形状（取 `[0]`） |
| `kv_cache/concurrent_serve.py` | **新**：并发 serving harness + 两级 selftest |
| `tools/gen_longctx_multiturn.py` | **新**：长上下文多轮负载生成器 |
| `tools/analyze_concurrency.py` | **新**：ladder × N 汇总出表 |

多轮语义与 `common/metrics.py` 一致：每轮 apply_chat_template 重灌全量累积对话 +
全新 cache。每轮 = 一次 batched prefill + 手写 greedy decode 循环（finished 槽位喂
pad、mask 隔离、首个 EOS 截断）。

## 诚实记录（指标解释必读）

- 手写逐 token decode 循环，吞吐绝对值远低于 vLLM 级生产引擎；**只做跨配置相对比较**。
- lockstep 逐轮批处理，非连续批处理（sessions 不同步进出）。
- 每轮重灌全量上下文：prefill 成本各配置相同（淘汰 deferred 到首个 decode 步），
  压缩的收益体现在 decode 注意力（128 vs 131072 个 key 的 KV 读取）和 HBM residency。
- fact accuracy 判定：ground truth 归一化后子串匹配；含数字/% 的答案带非数字边界
  （`9%` 不匹配 `19%`）；multi_instruction 为 AND（所有 gt 必须出现）。
- 负载是**合成文档**：记录块（多位数事实，考验 M4）+ 无数字 filler（可压缩噪声），
  区域每文档唯一保证问题无歧义，gt 生成时校验必在文档内。
- decode 步的 attention mask 必须建在**压缩后 slot 空间**：transformers 会把传入
  mask 与 cache 自己报告的 `get_seq_length()` 对齐（实测：传 128-slot mask 被扩展回
  全量 prefill 长度），长度不符直接炸 sdpa。harness **不读 cache 内部状态**（
  `out.past_key_values` 可能被重包装，自定义属性不可靠），纯 harness 侧推导：淘汰后
  布局恒为 sink|hh|recent，pad 只可能存活于 sink 区，故压缩配置的 mask 是常量
  `arange(128) ≥ pad_len`（sink 后强制 1）；未淘汰的小 prefill 用 prefill mask +
  补 1。hh 选入 pad 的极端角落由 B=1-vs-batched selftest 实证兜底。单流 batch=1
  评测从未暴露此问题：无 pad 时 mask=None 走 is_causal，根本不经过 4D mask。

## 运行步骤（按顺序，每步过再进下一步）

### 0. CPU cache 等价性 selftest（无需模型，任何机器可跑）

验证 B=1×N 与 B=N batched 在选择/折叠/量化（bf16 与 k8v8）上逐 session 一致：

```bash
python kv_cache/concurrent_serve.py --cache-selftest
# 期望输出两行 PASS + ALL PASS
```

### 1. 生成长上下文负载（在远程机跑，带 tokenizer 精确校准）

```bash
for T in 32768 65536 131072; do
  python tools/gen_longctx_multiturn.py \
    --target-tokens $T --num-docs 8 \
    --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --out data/longctx_multi_turn_${T}.jsonl
done
# 期望：每行 actual_tokens 在 target ±2% 内；digit_span 占比 >= 1/4
```

### 2. GPU selftest：padding 不变性（B=1 逐条 vs B=K 批处理，答案逐字一致）

m_sp4 配置，抓 left-pad mask / per-batch 分数两类 bug：

```bash
python kv_cache/concurrent_serve.py --selftest \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --device_map balanced_low_0 --torch_dtype bfloat16
# 期望：[selftest] PASS: 64 answers token-identical
```

### 3. B=1 回归门：单流结果必须复现定型档案

重跑 docs/16 第 12c 节命令（m_sp4 det 配置），核对：
EOS 0.8113207547 / rep 0.038 / decode KL ≈0.519 / 哨兵题逐字一致
（docs/19 表格）。bank/cache 批化后 B=1 路径数学不变，此步证实。

### 4. Pilot：单个 cell 打通三类指标

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --configs m4_k8v8 --concurrency-list 8 --turns 2 --max-new-tokens 64 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency \
  > serve_pilot.log 2>&1 &
```

检查 `results/concurrency/m4_k8v8_n8.json`：TTFT / TPOT / tok/s / HBM / EOS /
fact accuracy 字段齐全且合理（fact accuracy 显著 > 0）。

### 5. 全量矩阵

32k 主档全 ladder × N∈{1,8,16,32,64}；64k/128k 跑两端（full 与 m4_k8v8）控时间：

```bash
CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/concurrent_serve.py \
  --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data_file data/longctx_multi_turn_32768.jsonl \
  --configs full,hh,hh_merge,hh_merge_m4,m4_k8v8 \
  --concurrency-list 1,8,16,32,64 \
  --turns 8 --max-new-tokens 128 \
  --kv-budget-gb 512 \
  --device_map balanced_low_0 --torch_dtype bfloat16 \
  --output-dir results/concurrency \
  > serve_32k_matrix.log 2>&1 &

# 64k / 128k 两端对比
for T in 65536 131072; do
  CUDA_LAUNCH_BLOCKING=1 nohup python -u kv_cache/concurrent_serve.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --data_file data/longctx_multi_turn_${T}.jsonl \
    --configs full,m4_k8v8 --concurrency-list 8,32 \
    --turns 4 --max-new-tokens 128 \
    --device_map balanced_low_0 --torch_dtype bfloat16 \
    --output-dir results/concurrency_${T} \
    > serve_${T}.log 2>&1 &
done
```

OOM 的 cell 记 `status: oom` 不崩溃，继续扫下一档。

### 6. 出表

```bash
python tools/analyze_concurrency.py results/concurrency
python tools/analyze_concurrency.py results/concurrency --markdown > results/concurrency/table.md
```

## 判定标准（不是回到 baseline）

某 config 在并发 N 下**可持续**，当且仅当：

1. `peak_hbm_mb/1024 ≤ --kv-budget-gb`（默认 512GB）；
2. `eos_success_rate ≥ full 配置同 N 的 EOS − 2pp`（Fact accuracy 同时记录、
   随表报告，但不作为硬门槛——合成 gt 子串判定是下界）。

预期看到的结论形态：full 的 HBM 随 N 线性涨（128k 时每路 ~2.6GB）、decode 注意力
随上下文线性变慢；m_sp4/k8v8 的 HBM 平线（每路 ~2.6MB/13MB 级）、TTFT 各配置
持平、TPOT 与 HBM 在高档 N 拉开数量级差距、sustainable N 差出 1-2 个数量级。

## 结果

（待 step 0-6 跑完回填）

| config | max sustainable N (32k) | 备注 |
|---|---|---|
| full | | |
| hh | | |
| hh_merge | | |
| hh_merge_m4 | | |
| m4_k8v8 | | |
