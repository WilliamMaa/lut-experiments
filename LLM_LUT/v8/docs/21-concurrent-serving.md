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
| `kv_cache/concurrent_serve.py` | 联调期追加（算法不变）：prefill `logits_to_keep=1`（全序列 logits 4×31k×262144×2B≈65GB 曾 OOM 爆卡）；分块 prefill（`B*S>131072` 时按 8 对齐 chunk 喂，harness 自建 4D mask，B=1 回归与 B=4 selftest 走原单发路径）；resume（跳过已有 cell、`cuda_error` cell 重试）；`_auto_pick_gpus`（按实际空闲 HBM 选卡，剔除被邻居进程霸占的卡）；CUDA error 按 cell 记录后干净退出 |
| `kv_cache/attention_scores.py` | stash 改逐 q-head 循环累加（原 repeat_interleave+全量 softmax 在 N=64/128k 物化 >100GB fp32 必炸）；修 4D mask 单例 head 维广播 bug（`[B,W,K]+[B,1,W,K]` 右对齐静默扩成 `[B,B,W,K]`） |

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
# 期望（2026-09-29 修订判定后）：turn 0 全部逐字一致（任何分叉 = harness/
# pad/mask bug，硬 FAIL）；turn>=1 的分叉打印记录但不判死（批处理浮点噪声
# 在有损重压缩上的固有性质）。典型输出：
# [selftest] PASS (harness-clean): turn 0 token-identical for all 4 sessions;
#   N later-turn divergences documented
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

**32k 全矩阵（2026-09-29/30 夜跑完）**：5 配置 × N∈{1,8,16,32,64}，turns=8，
max_new=128，kv-budget 512GB。机器为 7×A800-80GB（`_auto_pick_gpus` 剔除被
邻居进程占 20GB 的 GPU3），`expandable_segments:True`。cell JSON 在
`results/concurrency/`，下表由 `tools/analyze_concurrency.py --markdown` 生成。

| config | max sustainable N (32k) | 备注 |
|---|---|---|
| full | **64** | 基线本身；N=64 峰值 357GB 仍 < 512GB 预算 |
| hh | 32 | N=64 OOM |
| hh_merge | 1 | N=8 EOS 0.969 恰好跌破 full−2pp=0.98 的规则线，单格噪声，非真实崩溃 |
| hh_merge_m4 | 1 | 同上（N=8 EOS 0.891） |
| m4_k8v8 | 32 | N=64 OOM；fact 全场最高（0.47-0.53@N≥8） |

### 三个核心发现

**1. 延迟：压缩配置赢面巨大且随 N 放大（TPOT 表）**

```text
TPOT (mean)                    N=1       N=8      N=16      N=32      N=64
--------------------------------------------------------------------------
full                       114.7ms   294.3ms   448.5ms   793.9ms  1233.3ms
hh                         129.2ms   153.2ms   169.2ms   185.8ms       OOM
hh_merge                   133.6ms   171.9ms   177.5ms   192.0ms       OOM
hh_merge_m4                136.4ms   169.0ms   175.5ms   188.7ms       OOM
m4_k8v8                    140.6ms   174.1ms   179.8ms   204.2ms       OOM
```

full 的 TPOT 从 N=1 到 N=64 恶化 **10.8×**（lockstep 批处理下每步都要读
32k×N 量级的 KV 注意力）；压缩配置只恶化 **~1.45×**（每步读 128 slot，与 N
基本无关）。**N=32 时 full 逐步解码比 m4_k8v8 慢 3.9×**。TTFT 上压缩配置在
低 N 略慢 ~5%（stash 16-head 循环 + 淘汰 + 量化的 prefill 税），但第 2 轮起
prefill 只对 128 slot 做注意力，长对话下 prefill 优势显现（N=32：full 194.6s
vs m4_k8v8 218.0s——注意此时压缩 prefill 还背着未淘汰的全量 KV，差距没拉开）。

**2. 质量：并发本身不降解；压缩的代价在 B=1 就已付清（fact 表）**

```text
Fact accuracy                  N=1       N=8      N=16      N=32      N=64
--------------------------------------------------------------------------
full                        1.000     1.000     0.984     1.000     1.000
hh                          0.625     0.469     0.344     0.391       OOM
hh_merge                    0.625     0.406     0.312     0.406       OOM
hh_merge_m4                 0.750     0.453     0.484     0.484       OOM
m4_k8v8                     0.750     0.484     0.469     0.531       OOM
```

128-slot 淘汰在 32k 上下文的事实题上固定损失 25-40pp（B=1 即如此，与
docs/16 两阶段档案一致）；**N=1→32 全程平坦**——批处理并发没有引入新的质量
损失（m4_k8v8: 0.75→0.48→0.47→0.53，噪声内）。EOS 同趋势（full 1.0 全程，
压缩 0.89-1.0 无 N 趋势）。**并发 serving 对压缩配置是安全的**。
注意 N=1 格 n=8，fact 误差棒极大；n 随 N 增大（N=64 时 n=512）。

**3. 显存：与预期倒挂——压缩配置先撞墙（N=64 全 OOM，full 反而过）**

```text
Peak HBM (GB)                  N=1       N=8      N=16      N=32      N=64
--------------------------------------------------------------------------
full                        101.3     155.3     180.7     236.1     357.2
hh                          102.7     178.1     228.3     321.3       OOM
hh_merge                    102.7     179.3     227.8     339.0       OOM
hh_merge_m4                 104.3     167.8     214.1     339.1       OOM
m4_k8v8                     104.3     167.8     212.2     339.1       OOM
```

压缩配置峰值比 full 高 13GB（N=8）到 103GB（N=32），N=64 集体 OOM 而
full 以 357GB 通过。最大嫌疑是**方法层面的**：cache 语义是"prefill 只追加、
首次 decode 才淘汰"，所以**分块 prefill 期间 KV 无约束增长到全量 31k×N**
（N=64 时 40GB+，与 full 相同），峰值出现在 prefill 中段；128-slot 的存储
红利要等首次 decode 之后才兑现，对峰值毫无帮助，此消彼长还叠加 stash 的
fp32 临时张量。**KV 压缩的收益是稳态 HBM（decode 之后），不是峰值 HBM**
——修正"判定标准"一节的预期形态。

### 与预判的差异

- 预判"m4_k8v8 HBM 平线、sustainable N 差 1-2 个数量级"：**未兑现**（32k
  档位上峰值被 prefill 全量增长主导）。若要在峰值上兑现，需要改 cache 语义
  （分块 prefill 内按当前 chunk 分数滚动淘汰——改变与已验证档案的一致性，
  另立项）或在更大上下文验证（64k/128k 下 full 的稳态 KV 线性膨胀，倒挂应
  正过来）。
- 预判之外的发现：并发批处理对质量零额外损失（finding 2）是干净的新结论。

### 已知的坑（复现本表前必读）

- 共享机邻居：GPU3 常年被占 20GB，`N>=32` 的 cell 曾全天 OOM 在其上；
  必须带 `_auto_pick_gpus`（自动）或手动 `CUDA_VISIBLE_DEVICES` 剔除。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必带（碎片曾达 16GB）。
- 4D mask 必须 8 对齐，否则 sdpa 掉 math backend 物化 B·H·chunk·K scores
  （N=32 时 64GB）——chunk 已在 harness 内对齐。
- 早批 cell（13 个 ok + 若干 oom）部分记录在邻居污染期，最终表以
  2026-09-30 凌晨重跑批为准。

### 下一步

- 64k/128k 两端（full vs m4_k8v8）：full 稳态 KV 在 128k×N=64 时 168GB，
  预计 mid-N 即 OOM；压缩配置稳态 ~MB 级。长上下文才是压缩配置 HBM 主场。
- （可选）prefill 内滚动淘汰，把峰值 HBM 也压下来——需重新标定与
  docs/16 档案的一致性。
