# 17 — Testbed v2 设计：vLLM × LMCache × 35B hybrid（提案 v2）

> 状态：提案 v2（2026-09-28，按外部反馈修订），未动工。
> 动机与范围见 16 号 §7 遗留项。本文只写思路、架构、判据、
> Gate、风险，不写过程。

## 1. 定位与动机

**模型决策：沿用 Qwen3.6-35B-A3B（大模型），不换小模型。** 三个
理由：

1. 它是 GDN + full-attention hybrid，正是行 1 最硬结论
   （named state 对 recurrent 模型良定义）的载体，换 dense 小
   模型会丢这条；
2. LMCache 官方验证的 hybrid 配方（`mamba-cache-mode=align`、
   `--separate-object-groups`）覆盖 Qwen3.5/3.6/3.8 家族，
   35B-A3B 在官方支持面上，不用赌兼容性；
3. 数据已在盘上，零新增模型风险。

**修正一个事实错误**（v1 提案曾写"Qwen3 系列小尺寸本身就是
hybrid"）：Qwen3-4B 是标准 `Qwen3ForCausalLM`（36 层 GQA，无
GDN），**不是 hybrid**。小模型备选仅 Qwen3.5-0.8B（官方验证的
小 hybrid），本提案不采用。

动机排序（大模型前提下修正）：

- **主动机：在工业 substrate 上执行遗留项**。LMCache 已原生
  提供 token lookup、P2P、controller move/copy、CPU tier、
  observability——16 号缺的最干净对照（replication 隔离、
  PIT 正式化、tier regime 复测）在它的原语上反而比 v1 自写
  数据面更容易做干净。
- **次动机：外部效度**——机制在工业系统暴露的抽象粒度上是否
  仍成立。
- ~~便宜 rep~~（小模型动机，大模型下不成立，删除；35B 一格
  仍是 ~分钟级，rep 成本与 v1 同量级）。

## 2. RQ1 在 v2 的两层判据（关键修正）

LMCache 官方明确：hybrid 模型 cached vs fresh **不 bit-exact**
（GDN backend 不支持 vLLM batch-invariant mode），其 CI 用
GSM8K store-vs-retrieve 的 **score-level equivalence** 做门。
因此 v2 的 RQ1 拆两层：

- **RQ1a（硬门）state reuse 正确性**：store → retrieve →
  resume → 无 crash / 无 state corruption → **答案/score 级
  等价**。
- **RQ1b（观察项）token exactness**：decoded_ids 逐 token
  一致率，只记录不卡门。

不能用工业 backend 的数值非确定性去否定 ICN 抽象本身——
v1 的 token 级一致是 v1 自写数据面下的结论，v2 判据独立。

## 3. 架构：policy → adapter → LMCache 原语

**不机械复用 v1 的 fetch/deliver 消息协议**——那会在 LMCache
上重新造一层 LMCache。scheduler 保留的是**科学策略层**，
数据面全部交给 LMCache：

```
        Our scheduler / PIT 层（策略）
        ├─ 请求路由：哪个 request → 哪个 vLLM
        ├─ PIT-A：park / wake 相同 fp 的并发请求
        ├─ 复制策略：何时 controller copy
        └─ 判文计数器（hit/new_tok/lookup/retrieve/store）
                    │ adapter
        ┌───────────┴───────────┐
        ▼                       ▼
     vLLM W0                 vLLM W1        （TP=2/卡组，同 v1 布局）
        │                       │
    LMCache MP              LMCache MP
        │                       │
        └──── LMCache Controller ────┘
          lookup(tokens) → instance → (location, matched_len)
          P2P retrieve / move(copy=True)
          CPU RAM tier（pinned + LRU）
          官方 metrics（hit/lookup/retrieve/store tokens, latency）
```

| v1 操作 | v2 实现 |
|---|---|
| resolve（name → holder set） | Controller `lookup(tokens)` |
| fetch（跨 worker 取块） | LMCache P2P retrieve |
| replicate | Controller `move(..., copy=True)` |
| spill tier | CPU RAM backend |
| evict | LMCache eviction |
| 传输/序列化/位置一致性 | LMCache 内部，我们不重写 |

scheduler 不再管理 binary payload、fetch/deliver、CPU 序列化。

## 4. 三个遗留项在 v2 上的设计

### 4.1 Replication 隔离对照（最值得先做）

LMCache controller 的显式 `move(copy=True)` 使这个对照第一次
可以做得真正干净：

```text
B3 腿：lookup/retrieve only
Ours 腿：同样的 lookup/retrieve + 按我们的复制策略调 copy
其余一切相同（同模型同卡同 workload 同 seed）
```

这正是 16 号 §7 缺的 "same everything except proactive
replication"，且不需要自写搬运层。

### 4.2 PIT：A 类为主实验，B 类先 characterization

- **A 类（计算聚合）**：在 scheduler/frontend 层、请求进 vLLM
  之前 park 相同 fp 的 waiter——vLLM 看不到重复请求，与
  vLLM 内部实现无关。干净，作为主实验。
- **B 类（fetch 聚合）**：LMCache 自身已有 lookup lock、
  prefetch job 跟踪（request_id）、P2P LOOKUP_AND_LOCK、读锁、
  in-flight 上限——**它已经做了一部分并发协调**。若直接在上层
  加 PIT-B，"pit on/off 无差异"将不可解释（可能是 LMCache
  已合并）。因此 B 类先做 substrate characterization（Gate 3）：
  两 worker 并发同前缀，计量 LMCache 实际发出几次物理
  transfer、多少字节，再决定外加 PIT-B 有无空间。

### 4.3 分层驻留 regime 复测

LMCache CPU RAM backend 即 GPU → pinned host RAM → recall，
容量满 LRU eviction；官方 metrics 直接给 hit/lookup/retrieve/
store tokens 与延迟。`new_tok` 仍需自行从 vLLM 侧计算，但
v1 担心的"计数器断链"大部分不存在。sp-1/sp96 两档在 v2 复测
"落点决定一切"。## 5. 物理约束

8×A800 机器、**1 GPU = 1 worker**（一张卡开两个 vLLM 实例会
引入共享 compute/HBM 的干扰变量，不做）。35B 单卡放不下，
沿用 v1 布局：**TP=2 每 worker，4 worker**。PIT 风暴用
2/4/8 worker 梯度（8 worker 需一卡一 worker 的小模型变体，
留作可选项，不在第一版）。

## 6. 最小验证：Gate 0-3（先做这个再谈矩阵）

**Gate 0 — hybrid substrate**：Qwen3.6-35B-A3B + vLLM +
LMCache 按官方 hybrid 配方（`mamba-cache-mode=align`、
`--separate-object-groups`）起实例，单实例正常跑通。

**Gate 1 — 跨实例复用（RQ1a/b）**：W0 prompt P → store；
W1 同 P + suffix → lookup/retrieve → resume。硬门：答案/score
等价、无 corruption；记录：lookup hit、retrieve tokens、
fresh 计算削减、token exactness（RQ1b）。

**Gate 2 — controller 语义**：`lookup(P)` 能看到
`W0/CPU` 的 locator set；`move(copy=True)` 后再 lookup，
locator set 变化符合预期。**此门一过，replication 隔离对照
的地基就位。**

**Gate 3 — PIT substrate characterization**：两 worker 并发
请求同一 missing/cached prefix，计量 LMCache 的 lookup 次数、
物理 transfer 次数、字节数——弄清 substrate 自己合并到什么程度，
再决定外部 PIT-B 是否实现。

**任一 Gate 不过即停**，不投入矩阵；说明 LMCache 抽象与我们的
策略/记账不兼容，遗留项回 v1 栈做贵版本。

## 7. 风险

- **数值非确定性**：hybrid cached/fresh 不 bit-exact（LMCache
  官方 caveat），判据已按 §2 分层规避，但 token exactness
  记录若显著低于 v1，需在判文中解释是 backend 特性而非机制失效。
- **粒度差异**：LMCache 统一块 544 token（Qwen3.5 配方），与
  v1 的 16-token 块不可直接比 fetch/merge 机会量口径；跨
  testbed 比较只到"机制成立/regime"层，不下计数器级对比结论。
- **vLLM 对 Qwen3.6-35B-A3B 的支持度**：Gate 0 见分晓；hybrid
  MoE + GDN 的 vLLM 路径比 Qwen3.5-0.8B 重，这是大模型决策
  的主要代价。
- **计数器换源**：new_tok 与 pit 计数器需从 vLLM/scheduler
  侧自行推导，口径要对齐 v1 定义并写进判文。

## 8. 决策点

- Gate 0-3 全过 → 按 §4 顺序排矩阵（replication 隔离优先，
  PIT-A 次之，tier 复测顺带）。
- 任一不过 → v2 停，遗留项回 v1 执行。
- 本提案不改变 v1 任何已判结论；v1 仍是机制研究主体，v2 是
  工业 substrate 上的验证与对照补齐。
