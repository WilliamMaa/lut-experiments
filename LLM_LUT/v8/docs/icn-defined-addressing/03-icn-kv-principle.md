# ICN × KV Memory Allocation：思想、映射、切入点与评估

> 本文档是 `docs/icn-defined-addressing/` 系列重新定义后的基准文档。
> 定位：回答五个问题——**思想到底是什么、ICN 哪些东西能用、我们该怎么做、切入点是什么、怎么评估**。
> 与已有文档的关系：`00-ideas.md` 是方向原始陈述（保留，作为出处）；`01` 是已取代的模拟方案（数字锚点仍引用）；`02` 是实现计划（其中的实验设计以本文档 §5 为准，凡与本文冲突处本文优先）。

## 1. 思想到底是什么

**一句话**：把 KV cache 从"某请求在某 GPU 上的私有运行时状态"，变成"**cluster 级的、按名字寻址的、可再生成、可多编码的内容对象**"，让 memory allocation 围绕这个名字解析系统统一决策。

当前 serving 的固定顺序（`00-ideas.md` §1）：

```text
request → 选 worker → 在 worker 上找/分配 KV
```

我们把它倒过来一部分：

```text
request → 解析需要哪些 KV name → 查 name→locations 索引
        → 按 locality + load + transfer cost 决策（含"以哪种编码存"）
        → 执行：路由过去 / 搬过来 / 重新生成
```

这个思想成立依赖 KV-as-content 的三个**物理性质**，是普通数据不具备的，也是本方向区别于"给 PagedAttention 加 content hash"（`00-ideas.md` §10 明确要避免的假东西）的全部根据：

1. **可从名字派生**：KV 是 token 序列的纯函数——同一 (model, encoding, token span) 必然产出同一份 KV。所以名字可以是**内容的哈希**，不是会话的编号；相同内容自动共享。
2. **可再生成**：丢了可以重新 prefill。allocation 因此多出一个 ICN 没有的选项——recompute（§3）。
3. **可多编码**：同一份逻辑 KV 有 bf16 / m_sp4 / k8v8 等多种 representation（v8 的压缩结果），质量-体积可换——对应 ICN 的"同一内容多编码"（类比视频码率分层），这是本方向**最有新意的维度**（`00-ideas.md` §6）。

ICN 在这里提供的是**组织原则**，不是协议：identity/location 解耦、name→locator-set 解析、按 locality 的 caching/routing 决策。我们不碰 NDN packet、PIT/FIB 全套（那些是手段，第一版不需要，`00-ideas.md` §3/§9）。

## 2. ICN 哪些东西可以用到这个问题上

### 2.1 映射表（承重概念 → 本系统）

| ICN 概念 | KV serving 对应物 | 说明 |
|---|---|---|
| Content Data | 某 token span 在某 encoding 下的 KV 状态 | per-layer tensor 组 |
| Content Name（location-independent） | `H(model, encoding, token span)` | **内容寻址，不是会话编号**（见 §4 当前修正项） |
| Host / address | worker id、GPU HBM、CPU RAM | 只是 locator，不构成身份 |
| NRS（RFC 9236：name → locators） | scheduler 的 KVNameIndex：`name → {locations, reuse_count, encoding, size}` | 第一版的"网络"全部在这里 |
| In-network cache / tier | GPU HBM ↔ CPU RAM ↔ NVMe | 第二版才做 tier，第一版只有 HBM |
| Interest | 请求的 prefix 需求 | scheduler 从请求推出需要的 name 集合 |
| Interest 聚合（PIT） | 并发同 name fetch 去重 | 第二阶段（§9） |
| Content replication | 同一 KVName 驻留多个 worker | 第二阶段（§8），由 reuse_count 驱动 |
| 多编码内容（码率分层） | bf16 / m_sp4 / k8v8 | **v8 独有维度**：allocation 同时决定 where + which encoding |
| Managed caching（RFC 8793） | promote / demote / replicate / move / drop | 以 encoding 转换（transcode）为扩展点 |

### 2.2 不能直接用、必须改造的

- **ICN 内容不可再生成，KV 可以**。所以 allocation 的决策空间比 ICN 多一维：`route（compute follows KV）/ move（KV follows compute）/ recompute（重新生成）`。recompute 的可用性由 prefill 成本决定，这是 KV 系统的独有问题。
- **ICN 内容生产者发布后即不可变，KV 的"生产"（prefill）昂贵**。所以 name→data 的绑定关系（谁生产、花多少算力）必须进入 cost model，不能只算传输。
- **请求本身携带生成新内容的能力**：一个请求未命中任何已有 name 时，它既是消费者也是生产者——它产出的新 KV object 要以 name 发布进索引，供后续请求共享。

### 2.3 明确不用的

NDN 报文格式、内容签名安全、PIT/FIB 数据面、off-path caching 路由。这些在本问题里没有对应的痛点。

## 3. 我们该怎么做：请求生命周期与 allocator 决策

每个请求到达（含多轮对话的每个 turn）：

```text
1. 从 prompt 解析所需 KV name：H(model, encoding, prefix token ids)
2. NRS lookup：name → LocationSet（可能在多个 worker / 多个 encoding）
3. 对每个候选 worker j 计算：
     C_j = C_queue,j + min( C_transfer,j, C_recompute,j ) + C_quality
   其中：
     C_transfer  = object_bytes(encoding) / 实测带宽      —— 编码决定体积
     C_recompute = span_tokens / 实测 prefill 速率        —— 与编码无关
     C_quality   = EOS_delta(encoding)                    —— 压缩的质量代价
4. argmin 选 j*，执行三选一：
   A. compute follows KV（j* 已持有 object）→ 直接 resume
   B. KV follows compute（object 在别处）→ fetch+deliver 后 resume
   C. recompute（无 object 或 min 选中了它）→ 全量 prefill
5. 本 turn 产出的新 KV object 以 name 发布进 NRS
   （记录 location、encoding、bytes、reuse_count）
```

其中 **C_transfer 由 encoding 决定而 C_recompute 不由**——这就是 §1 性质 3 进入 cost model 的入口：压缩不改变重算成本，只改变搬运成本。由此产生本文档最核心的可检验预言（§5 E3/E5）。

## 4. 切入点与当前状态

### 4.1 切入点（已拍板，不变）

**只做 memory allocation，不碰网络协议**（`00-ideas.md` §3/§7）。第一版系统 = NRS + 三选一 allocator + 真实 representation（复用 `kv_cache/` 已标定的 m_sp4/k8v8），单机多 GPU 真机（`02` 文档的实现层细节仍然有效）。

### 4.2 当前 prototype 与思想的差距（修正清单）

实现过程中发现两处**结构性偏差**，在跑任何指标前必须修正，否则测的不是 ICN：

| # | 偏差 | 后果 | 修正 |
|---|---|---|---|
| 0 | **workload 里文档从未被 prefill**：question turn 的 input 只有问题本身（15-80 token），8000-char 文档只存在于 cum_tokens 标签里 | **比命名偏差更严重**：之前所有"长上下文"实验的计算是假的——Step 2.5 标定的 "prefill 30-62K tok/s" 全部作废，"L=35K 骑在 crossover 上"是幻影；诚实 prefill 速率实测仅 ≈2.4-2.7K tok/s（见 E3） | 每个 session 增加合成 doc-turn（turn=-1，prefill 完整文档、decode_steps=0、以 `H(doc_ids)` 发布 object）；question turn 只 prefill q_ids |
| 1 | name 是 session-scoped（`/session/doc3/turn/2/...`），不是 content-addressed | 内容相同的 prefix 跨 session 无法共享——**ICN 最核心的自动共享行为结构上不可能发生**，系统退化为 per-session cache（LMCache 式） | name 改为 `H(model, encoding, prefix token ids)`（prefix_hash/span_end/repr 三字段，sha256）；scheduler 持有全部 token ids，可自行计算 |
| 2 | turn 0 不做 NRS lookup（prev=None 直接重算） | 即使名字改对，入口也不查表，共享链条断在起点 | 每个 turn（含首 turn、含 doc-turn）都走 §3 的 lookup→三选一（reuse/resume/recompute） |
| 3 | NRS 不记录 reuse_count | 第二阶段 replication（§8）无驱动信号 | 索引结构顺带记录（`nrs_reuse` 计数器已埋点） |

已确认可接受的第二版事项（明确不在第一版）：多 replica、memory tier（CPU/NVMe）、PIT 聚合、object 生成后的 transcode（换编码）。

### 4.3 与相邻工作的边界

- **LMCache** 提供"KV 是可迁移 object"（我们的 transport 层等价物）；不做 name 解析与 allocation 决策。
- **CacheRoute** 提供 affinity routing（我们的选项 A）；没有 move/recompute 权衡，更没有 encoding 决策。
- **vLLM/PagedAttention** 提供 block 虚拟化；是 address-oriented，不是 name-oriented。
- 我们的增量：**统一的 name-addressed allocation**（A/B/C 三选一 + encoding 选择 + 发布/复用计数）。少其中任何一条就退回上述某个已有系统。

## 5. 怎么评估

### 5.0 原则

评估对象是**ICN 假设本身的一组可证伪预言**，不是"哪个策略赢 wall time"。共享机计时噪声大，wall time 只作辅助证据；每个预言给出行为级判据。（此前把实验组织成 P0/P1/P2 赛马是错误的框架，已废止。）

### 5.1 五条预言

**E1 identity/location 解耦成立（机制）**
同名 object 可在任意 worker 间迁移、注入、被无感消费，resume 后行为与连续执行一致。
判据：迁移后 decode 与本地 resume 的 token 一致率 ≈ 1（同 encoding）；codec 往返无损。
状态：✅ 已验证（Step 1-2，hit_rate 恒理论上限，bf16/m_sp4/k8v8 往返差异全 0）。

**E2 自动共享发生（行为）**
内容相同的 prefix 跨 session 自动 collapse：第 k 个拥有相同内容的 session，其整链应为已有 object 的命中，零重算。
实验：content-addressed 改造后，16 session 由 8 篇文档循环生成（session i 与 i+8 内容全同）。
判据：后到的同内容 session 的 doc-turn 以 NRS reuse 命中，零文档重算；NRS 中同 name 被多 session 引用。
状态：✅ 已验证（2026-09-19，`results/icn_proto/cluster_p2_m_sp4_s16t4_20260919_184451.json`）：`nrs reuse names: 8 names, total 8 cross-session reuses`——16 session 中后 8 个的 doc-turn 全部命中先到者的 content-addressed object，文档零重算；hit_rate 0.9 = 72/80 恰为理论上限（8 个 fresh doc-turn 是必要首算）。

**E3 locality 成为 cost term（决策经济学）**
move vs recompute 的选择边界与标定 cost model 一致：迁移胜出 ⟺ `L > L* = S(encoding)·R_prefill/R_xfer`。
已知标定（2026-09-19 晚，修正 §4.2#0 后的诚实值）：R_prefill ≈ 2.4-2.7K tok/s（heavy-hitter 路径，有 eviction 开销）、R_xfer ≈ 100MB/s → **m_sp4（S≈66-141MB）L* ≈ 1.6-3K token；bf16 L* ≈ 17K token**。此前 "R_prefill 30-62K、L* 22-45K" 是假 workload 幻影，作废。
判据：allocator 决策日志（每 turn 记录三选项预测成本与选择）在 L 扫描下于 L* 附近翻转；L ≪ L* 时应以 recompute 为主，L ≫ L* 时应以 transfer 为主。
状态：系数已实测（E2 实验顺带）；E2 之后跑 L 扫描验证翻转点。（次要缺口：transfer 类决策尚未写入 per-turn records，待补。）

**E4 encoding-aware allocation 可行且质量可控（质量腿）**
显存预算紧张时 allocator 把长链分配到 m_sp4/k8v8、短链保 bf16，输出质量退化在 encoding 本身的质量锚点内。
质量锚点：v8 评测 EOS delta——m_sp4 ≈ 0、k8v8 ≈ 0.019（`docs/19-final-summary.md`，不重测）。
判据：cluster 内跨 policy 的 decode token 一致率——同 encoding 链 P0-vs-P2 ≈ 1；k8v8 链发散率与 EOS delta 0.019 相容。
状态：机制已就绪（worker 上报 decode tokens），待 E2 后跑。

**E5 压缩改变 allocation 自由度（经济学对比，本方向的核心卖点）**
同一逻辑 KV，m_sp4（~66MB 定长，doc-turn object 实测 140.6MB，差异原因待查）vs bf16（20KiB×L）：使"KV follows compute"从不可行变为可行的上下文区间移动约 10 倍（L* 之比）。
判据：E3 的 L* 实测值对比即结论；辅以 transfer_bytes 曲线（m_sp4 平坦 vs bf16 线性增长）。
状态：✅ 物理量已实测（诚实带宽下 140.6MB/1.405s ≈ 100MB/s）；按诚实系数重算 L*：m_sp4 1.6-3K vs bf16 17K，差一个 E3 的正式边界扫描出最终对比数字。

### 5.2 明确不评估

- 端到端 wall-time 策略赛马（共享机噪声主导，不是假设本身）；
- 生成质量榜单（质量只做 sanity，见 AGENTS.md 红线 #3：准确率是验证指标不是目的）；
- 第二版事项（replication/tier/PIT）未实现前的任何"预热""聚合"指标。

## 6. 与项目总目标的关系（红线自查）

- 本方向服务于总目标：m_sp4/k8v8 是 LUT/存算友好的编码形式，allocation 层是让压缩产生**系统级收益**（省显存、省重算、可迁移）的机制。
- 红线 #1（O(1) 查表）：本文所有机制只做存储/搬运/路由决策，不引入新的 O(N) 计算。
- 红线 #3（准确率为手段）：E4 是唯一质量测量，且以"退化不超过 encoding 锚点"为判据，不追分。
- 红线 #2（同预算比较）：E5 的 L* 对比天然同预算（同一 workload、同一 hardware）。
