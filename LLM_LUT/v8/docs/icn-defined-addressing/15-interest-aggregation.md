# 15 — Interest Aggregation（PIT）机制 spec（矩阵行 7）

矩阵位置、研究问题、负结果政策见 14 号总纲。本文只做一件事：
把行 7 从"❌ 未做"变成可实现的 spec + 可判文的实验。

## 1. 机制映射

| ICN | 本原型 |
|---|---|
| Interest 包 | 一个 turn 的 resume 需求（一组内容命名块 / 一条前缀链） |
| PIT（pending interest table） | scheduler 侧的 in-flight 需求表：同名/同前缀需求挂为等待者，只算/搬一次 |
| Data 满足多个 pending interest | 一份 fetch payload 送达多个等待 target；一次 prefill 发布后被多个 waiter 复用 |

解决的问题（14 号行 7）：高并发同前缀风暴下的重复劳动。testbed
里它**已经在发生**——closed-loop share=2 下 8 条 session 前缀完全
相同且同时起步；open-loop zipf 共享文档天然产生并发同前缀需求。

## 2. 原型中的两个聚合点

**A. 计算聚合（compute merge）**：两个 turn 前缀相同（fp 相同），
被派到不同 worker 各自 prefill 同一条链。PIT 行为：后到 turn
挂为 waiter，serving turn 发布块后 waiter 唤醒，resume 现成块
——同一条链只算一次。

**B. 取块聚合（fetch merge）**：两个 worker 同时向同一 holder
demand-fetch 同名块。PIT 行为：合并为一次 fetch，payload 到
手后 multicast deliver 给全部 waiter target——同一批字节只搬
一次。

v1 两个都做，共用一个 PIT 表。只对会跨 worker 取块/复用的策略
有意义：b3、ours（b0/b1/b2 的 waiter 唤醒后无 fetch 能力，留
v2）。

## 3. Step 0：机会量预估（先量化，再动工）

实现前先用现有 16 格 JSON 回答"值不值得做"。records 里已有
`fp`（前缀指纹）、`E`、`xfer_blocks`、`t_arrive`、`queue_s`、
`latency_s`，零新实验：

```bash
cd ~/lut-experiments/LLM_LUT/v8
# E2 格（open-loop）：2026-09-28 实测 A_pairs 2-4/格、B_pairs≈0 ——
# poisson 到达 + think time 把同前缀 session 错开，风暴前提不满足
python -m icn_proto.pit_opportunity results/icn_proto/matrix_e2.json
# 闭链强前提（16 session 同时起步、share=2、8 对完全相同前缀链）：
# 一格新 cell 同时是 ①活性修法 smoke（rc 必须 0）②行 7 闭链机会量
# ③将来 PIT A/B 的 pit=off 基线。不要用任何老 manifest。
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 16 --turns-per-session 3 --q-tokens 40 \
  --shares 2 --policies b3 --reps 1 --cell-timeout 1800 \
  --manifest results/icn_proto/matrix_pit_smoke.json
python -m icn_proto.pit_opportunity results/icn_proto/matrix_pit_smoke.json
```

脚本（本目录仓库 `icn_proto/pit_opportunity.py`）输出每格：
同 fp 计算窗口重叠数（A 类机会）、同块集合并发 fetch 数（B 类
机会）。判读：机会 ≈ 0 的 workload 配置不进实验矩阵；机会显著
的配置进 §6。

## 4. 设计（scheduler 侧，worker 零改动）

### 4.1 PIT 表

```python
self._pit = {}   # fp -> {"rid": serving rid, "worker": ident,
                 #        "waiters": [turn, ...],
                 #        "t": time.time()}
```

- **挂为 waiter 的条件**：ready turn 的 `prefix_fingerprint()`
  与表中 serving turn 完全相同（v1 不做前缀覆盖匹配——链式
  前缀的子集关系留给 v2），且 serving turn 仍在飞（busy）。
- **park 语义**：turn 移出 ready，trace 发 `pit_wait`；
  不产生 record（和未调度状态一致）。

### 4.2 挂接点

1. `dispatch()` 取 ready turn 时先查 PIT（A 类）：命中且
   serving worker 未灰 → park 为 waiter，continue。
2. fetch 路径（`dispatch()` 发 fetch 前，B 类）：`self._xfer`
   里查 `x["names"] == need` 且 stage 在 fetch/deliver 的在途
   项 → 不新发 fetch，把 (target, turn, E_loc, decision) 挂到
   该 xfer 的 `waiters` 列表，target worker 标 busy，turn 移出
   ready。
3. `result`（serving turn 完成）：发布块进目录后，唤醒该 fp 的
   全部 waiter：`turn.t_ready = now` 重新进 ready（走正常
   choose()——块已在 serving worker 上，b3/ours 会单发一次
   fetch 拿到，或与其他 waiter 的 B 类合并）。
4. `delivered`（B 类 payload 到手）：主 target 按原路径
   send_assign；随后对 xfer.waiters 逐个发 `deliver`（同一
   payload 复用）并各自 send_assign；trace 发 `pit_wake`。

### 4.3 与活性修法的接口（必须，事故四的延伸）

- `_grey()` 释放路径新增两类：serving turn 被灰 → 其 PIT
  waiter 全部唤醒重排队；带 waiter 的 xfer 被灰 → 主 target 走
  原 degrade，waiter 各自 degrade/重排队。
- waiter 挂起上限：waiter 等待 > 60s（serving 异常但未触发灰）
  → 强制唤醒重排队，防无限 park。

### 4.4 记账与观测量（机制计数器优先，比分只是附带）

summary JSON 新增：

- `pit_compute_merged`：A 类合并次数（挂为 waiter 的次数）
- `pit_fetch_merged`：B 类合并次数
- `pit_waiters_served`：被合并服务而最终 ok 的 waiter 数
- `pit_wait_s`：waiter 平均挂起时长
- `pit_recompute_tokens_saved`：waiter 因聚合少算的 prefill
  tokens（Σ waiter 的 would-be fresh tokens；A 类独有）

trace 事件：`pit_wait / pit_wake / pit_fetch_merge`，带 fp 和
rid。transfers / transfer_bytes 只计一次 wire fetch（B 类
multicast 不产生新 wire 传输；deliver 是本地消息），这点必须在
代码注释里写明，避免"合并了但字节数没省"的记账歧义。

## 5. 不变式（实现时不得破坏）

1. worker 协议零改动——聚合纯控制面。
2. 单个 serving turn 崩溃/变灰时 waiter 不丢（释放路径 4.3）。
3. PIT 不改变放置决策本身：waiter 唤醒后走的就是现有
   choose()，不做"为了聚合而改派"。
4. 关掉 `--pit` 时代码路径与现状逐字节等价（回归保证）。

## 6. 实验与判文

- 开关：`--pit`（默认 off，保住旧格可比性）。
- 矩阵：E1 闭链 share=2 + E2 open-loop zipf（机会量大的档位，
  由 Step 0 定）× {b3, ours} × {pit off, on}。
- 判文标准（对应 14 号负结果政策）：
  - 机会量显著且计数器 > 0 → 机制成立，regime 边界用机会量
    分布说清（哪个并发度/共享度区间有聚合价值）；
  - 机会量显著但计数器 ≈ 0 → 实现有洞，回炉；
  - 机会量 ≈ 0 → 该 workload 下判否，说明边界。
- 不看的：rps/SLO 比分（除非计数器先成立，否则比分无意义）。

## 7. 实现顺序

1. `pit_opportunity.py`（Step 0，半天内）。
2. `--pit` + PIT 表 + A 类（计算聚合）+ 计数器 + trace。
3. B 类（fetch multicast）。
4. `_grey` 释放接口 + waiter 上限。
5. 回归：runbook §2 五条单测 + `--pit off` 与旧格逐指标一致
   （同 seed 重跑一格 diff JSON，除时间戳外应完全相同）。
6. 实验矩阵 + 判文写入 13/14 号。

**验证（2026-09-28 实现后，按顺序）**：

```bash
# 1. runbook §2 五条单测，全 ALL PASS
# 2. pit=on 对照格：与 matrix_pit_smoke 完全同配置仅加 --pit，
#    新 manifest（闭链一格 ~35s）
cd ~/lut-experiments/LLM_LUT/v8
python -m icn_proto.matrix_step5 --model-path /home/u/downloads/models/Qwen3.6-35B-A3B \
  --gpu-pool 0,1,2,3,4,5,6,7 --sessions 16 --turns-per-session 3 --q-tokens 40 \
  --shares 2 --policies b3 --reps 1 --cell-timeout 1800 --pit \
  --manifest results/icn_proto/matrix_pit_on.json
```

判读（判文锚点 = smoke 格机会量 A_pairs=4）：pit=on 格
`pit_compute_merged > 0` 且 `pit_recompute_tokens_saved ≈ 6082`
→ A 类机制成立；`failed=0` 且 hit/new_tok 不比基线崩 → 没破坏
正常路径；二者全满足 → 按 §6 铺 {b3, ours} × {off, on} 小矩阵
补齐 regime 边界。计数器全 0 → 实现有洞，回炉。
