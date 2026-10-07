# v8 并发扫描结果（docs/33）

2026-10-07，vLLM 插件 v2026-10-06i，Qwen3.6-35B-A3B @ 2×GPU (TP2, GPU 6,7)，
vLLM 0.19.1，`--enforce-eager`，prefix caching off，chunked prefill 8192。
数据：`data/longctx_multi_turn_65536.jsonl`（8 docs × ~64k tokens × 8 轮 QA）。
每配置 8 会话 × 8 题 = 64 题，单遍。命令：`bash tools/run_concurrency_sweep.sh`。

本表回答的问题：**v8 自身在并发下的质量/吞吐行为**（Gate 4c 的扩展）。
它**不能**回答"v8 比 full-KV 多扛多少并发"——那需要 full-KV 同条件
容量基线（docs/34），见 `tools/run_capacity_sweep.sh`。

## 结果总表

| slots | N | fact_acc | 错误 | wall (8 会话) | 单会话均时 | 吞吐 |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.984 | 0 | 623s | 78s | 46.3 sess/h |
| 1024 | 8 | 0.953 | 0 | 454s | 429s | 63.4 sess/h |
| 1024 | 16* | 0.984 | 0 | 456s | 444s | 63.1 sess/h |
| 4096 | 1 | 0.969 | 0 | 638s | 80s | 45.2 sess/h |
| 4096 | 8 | 0.969 | 0 | 435s | 398s | 66.2 sess/h |
| 4096 | 16* | 0.969 | 0 | 432s | 397s | 66.7 sess/h |

\* **N=16 无效作为并发数据点**：bench 只有 8 个 doc（8 会话），N=16
实际仍是 8 并发 + 空转。N=8 ≈ N=16 不说明任何 saturation。
全部配置：0 崩溃、0 contract raises、0 HTTP 错误。

## 可以下的结论

1. **质量对并发不敏感**：N=1→8（有效范围内），fact_acc 在 0.95–0.98
   波动（每格 64 题，±1 题 = ±0.016，属噪声）。per-request 压缩布局由
   request_id 隔离，并发未引入交叉污染。
2. **当前 64k workload 上，没有观察到 1024→4096 带来明确 factual-quality
   增益**（0.953–0.984 vs 0.969，噪声内）。注意这不等于"1024 足够"——
   是否足够取决于业务质量门槛（Acc≥0.95 两档都过；Acc≥0.99 都没过），
   门槛由应用目标定，不由实验自定。
3. **v8 自身从串行到并发的吞吐收益 1.43×**（46→66 sess/h，N=1→8）。
   这是 vLLM batch scheduling 的收益，**不能归因给 KV 压缩**；对 full
   的归因必须等容量基线。
4. N=8 时 4096 比 1024 快 ~7%（398s vs 429s）。**无机制解释**（按方法
   描述 decode 不存在"信息丢失重建"，且 1024 attention 读得更少、应更
   便宜）：单遍 run，只能记为 run-to-run/scheduling variance，需同配置
   重复 2–3 遍才能判断是否真实。

## 明确不说的

- ~~"full-KV 在 2 卡上只能扛 1–2 个 64k 并发"~~——**已删除**。按官方
  结构粗算（10 full-attn 层 × 2 KV heads × 256 dim × 2B × 2(K,V)
  ≈ 20KB/token，64k ≈ 1.25GB/请求，8 并发 ≈ 5GB/GPU），该旧假设很
  可能严重低估 full baseline。full 的真实容量由
  `tools/run_capacity_sweep.sh` 实测钉死。
- ~~"v8 并发能力大幅超过 full"~~——未证明，等容量基线。

## 下一步（docs/34 定向，唯一优先级）

`bash tools/run_capacity_sweep.sh`：64 会话 backlog × 64k × 8 轮，
full / v8-1024 / v8-4096 三后端 × N {1,2,4,8,16,32}，同卡同 flags，
记录 OOM/admission、peak KV、peak HBM、sessions/h、P50/P95、fact_acc。
产出："On 2×A800 at 64k and ≥X factual accuracy, v8 increases supported
concurrency from A to B and throughput from C to D."
