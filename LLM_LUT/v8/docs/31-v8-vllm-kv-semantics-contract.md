# V8–vLLM KV Semantics Contract

> 状态：草案 v1。**在本文档的不变量全部通过自动化验证（Gate 1–3）之前，
> 禁止写任何 integration 代码（没有 v2026-10-04u），禁止让用户跑实机 repro。**
> 背景：docs/29（问题清单）、docs/30（反思）。本文档是补丁路线的正式截断点。

## 0. 事故定论

崩溃现场（v2026-10-04t）：

```
write-back OOB: L2=16384 jb2_max=511 n_blk=256 C=8192 blocks_len=256
scheduler: block_ids=([1],[2],[3],[4..11]), num_scheduled_tokens=8192
```

三个数字各自合法，但来自三种不同的 block 单位：

- `jb2_max=511` ← 插件用 **pool 物理页粒度**（kv_cache.shape[2]=32）换算；
- `n_blk=256` ← builder 用 **spec.block_size** 切片块表；
- `8 blocks / 8192 tokens` ← 调度器用 **group manager 的 block_size** 记账。

根因不是某个写错的常数，而是**把三种单位当成了一种**。以下全部条款围绕
"每个量必须声明自己的单位、每个换算必须显式、每个不变量必须可检查"展开。

## 1. 五个单位及其唯一真值来源（vLLM 0.19.1 源码定位）

| # | 单位 | 符号 | 谁定义 | 0.19.1 真值来源 |
|---|------|------|--------|------------------|
| 1 | Scheduler token | T | scheduler | `num_scheduled_tokens` / `num_computed_tokens`（EngineCore 纯 token 计数，无块概念） |
| 2 | Group manager block | B_g | 每个 KV cache group 的 spec | `kv_cache_config.kv_cache_groups[g].kv_cache_spec.block_size`；调度器分配按它记账（`get_num_blocks_to_allocate` = cdiv(tokens, B_g)） |
| 3 | Hash block | H | BlockPool | `block_pool.hash_block_size`；多 group 不等时 group block 是 H 的整数倍（block_pool.py:140-143, 244-250） |
| 4 | Kernel/pool page | P | 每层 pool 张量 | 运行时 `kv_cache.shape[2]`（impl 实际索引用的粒度） |
| 5 | v8 compact slot | S | v8 算法 | 逻辑 token 槽位（sink|HH|recent），与 1–4 无任何天然等值关系 |

源码已核实的关键事实（spike/vllm-src-019）：

- **F1** 多 group 的合法粒度本来就不同：每个 manager 有自己的
  `block_size`（`single_type_kv_cache_manager.py`），prefix-cache 命中对齐用
  `lcm_block_size = lcm(B_0..B_n)`（`kv_cache_coordinator.py:451`），
  hash 用 `hash_block_size`（可为 GCD 级）。**所以"assert 所有 block size
  相等"本身非法**（docs/30 第 1 节的修正是对的）。
- **F2** hybrid GDN 模型（Qwen3.5/3-Next 属此类）会**强制上调 attention
  block_size** 使 attention page ≥ mamba page
  （`HybridAttentionMambaModelConfig`，`model_executor/models/config.py:156-317`）。
  即 B_fullattn 不是用户设的 16/32，可能是 512/1024 这种大页。
- **F3** `CommonAttentionMetadata`（`v1/attention/backend.py:323`）**不含任何
  request 身份字段**；它按 group 分别在
  `worker/gpu/attn_utils.py:209 build_attn_metadata` 构造。
- **F4** worker 侧 batch 序 → request_id 的真值是
  `input_batch.req_ids`（`gpu_model_runner.py`，构建 metadata 的同一进程、
  同一步内有效）。这就是身份的正路，不需要 blocks[0] 启发式。
- **F5** 调度器侧 per-request 块所有权本来就是显式状态：
  `kv_cache_manager.get_block_ids(request_id)`（`kv_cache_manager.py:522`）。

## 1.1 实测钉值（Gate 1 probe，2026-10-06，Qwen3.6-35B-A3B @ vLLM 0.19.1）

`tools/probe_kv_units.py`（CPU、只读配置）输出：

```
architecture     : Qwen3_5MoeForConditionalGeneration
max_model_len (T): 131072
B_g (attention)  : 1056    ← HybridAttentionMambaModelConfig 强制上调
mamba_block_size : 1056    (mamba_cache_mode=align; mamba page 2162688 B)
```

两个推论，均已用源码分支数学核实：

1. **B_g=1056 与 prefix caching 开关无关**：开（align 模式）走
   `chunk_size*cdiv(...)` 分支、关（我们的 serve 配置）走
   `16*cdiv(mamba_page, 16*attn_1tok)` 分支，mamba page 恰好是
   2162688 B = 66×16×2048 B，两分支都得出 1056。
2. **事故数字完全闭合**：调度器给 8192-token chunk 分配
   `cdiv(8192,1056)=8` 块（崩溃 dump 的 `[4..11]`）；插件 builder 按
   spec 的 32 去切块表（抓满 padding 宽度 256）；impl write-back 按
   32 换算需要 `16384/32=512` 块 > 256 → OOB。三套单位同处一行代码。

**对 integration 的硬约束**：v8 的 spec/换算一律以运行时
`kv_cache_groups[g].kv_cache_spec.block_size` 为准，禁止任何 32/16
字面量；启动时 assert `spec.block_size == cache_config.block_size`（同
进程 config 真值），pool 页 P 在第一次 forward 用 `kv_cache.shape[2]`
对账（I2 的 P vs B_g 关系就此钉死）。

### 1.2 block_table.py 精读补充（2026-10-06，vLLM 0.19.1 worker 侧）

- **块表张量已经是 kernel 单位**：`BlockTable(block_size=B_g,
  kernel_block_size=P)`，当 B_g != P 时 `use_hybrid_blocks=True`，
  `append_row` 把每个 manager 块展开为 `blocks_per_kv_block = B_g//P`
  个 kernel 块（`kernel_id = mgr_id*q + r`，block_table.py:47-68,
  110-118）。所以 builder 看到的就是 P=32 单位的表，pool 索引用 P 作
  除数是**对的**；崩溃的真因是 builder 切片宽度与 frontier 脱节 +
  spec 侧常数错误，不是这里要再换算。
- **行内没有有效长度哨兵**：`clear()`/`clear_row()` 用 0 填充
  （block_table.py:124-171），越界读到的是陈旧块号或 0（0 是合法物理
  块）。因此"读表数有效长度"不可能——有效长度只能由本步 T 前沿
  `computed + scheduled` 推导，这就是 I3 的 fail-closed 检查存在的
  原因（vllm_plugin/blockplan.py）。
- **spec.py 实锤 bug**：`blocks_per_request = ceil(131072/32)+1 = 4097`
  按 B_g=32 算；真值必须按 B_g=1056 → **125**。allocator cap 与
  `max_memory_usage_bytes` 全部要改按 group 真值。

### 1.3 Gate 结果（2026-10-06，本机纯 CPU）

| Gate | 模块 | 结果 |
|---|---|---|
| Gate 1 | `vllm_plugin/units.py` + `tests/test_units.py`（5 组 × 20k 随机） | **PASS** |
| Gate 1 probe | `tools/probe_kv_units.py`（服务器，只读配置） | **B_g=1056 钉死** |
| Gate 2 | `vllm_plugin/identity.py` + `tests/test_identity.py`（7 种调度序列） | **PASS** |
| Gate 3 | `vllm_plugin/blockplan.py` + `tests/test_blockplan.py`（4 组 × 20k 随机） | **PASS** |
| Gate 4a | integration harness（docs/32）：`tests/test_integration.py` 8 case + 2194 request-steps 随机交错 + `tests/test_forbidden.py` 禁止项扫描 | **PASS**（2026-10-06h，远程 lut_py310） |

剩余唯一未钉值：serve 时 P 的实机确认（integration 启动 assert 自动完成）。

**Gate 4 拆分（docs/32）**：原 Gate 4 混了两个问题，拆为两关——
- **Gate 4b integration 正确性**：2 并发 32k、chunked prefill + decode +
  multi-turn，标准 = 不崩、无 OOB、无跨请求泄漏、contract raises = 0
  （`tools/repro_concurrency.sh` 的 verification 段）；
- **Gate 4c 算法质量**：fact_acc ≥ 0.7（slots 扫描，`run_concurrency_sweep.sh`）。
fact_acc 下降不再阻塞 integration 判定，反之亦然。

## 2. 必须成立的不变量（每个都写成可执行检查）

### I1 单位声明
代码中每个长度/偏移量出现处，必须能静态说出它是 T、B_g、H、P 还是 S。
`impl.py` 的 `bs = kv_cache.shape[2]` 这类裸变量名一律改为带单位后缀
（`pool_page` / `mgr_block`）。

### I2 换算显式且单向
slot→物理地址只允许一条换算链，且每一步的除数从真值来源读取：

```
S (compact slot)
  --(÷B_g, 向上取整)--> manager block 序号 j      (除数 = group spec.block_size)
  --(查块表)-->         physical block id b_j       (块表 = 该 group 的 block_table_tensor 行)
  --(×B_g + r)-->      pool 内 token 偏移          (r = S % B_g)
```

**禁止**用 `kv_cache.shape[2]`（P）去除 manager block 序号，**禁止**反过来。
若 vLLM 对 hybrid 有 P≠B_g 的 reshape 约定（F2 使得这必须核实），换算链在
Gate 1 用真机 metadata 一次性钉死并写进本文档，之后只许引用、不许重推。

### I3 需求 ≥ 供给时 fail-closed
builder 为每请求每步产出 `BlockPlan`：

```python
BlockPlan(
    request_id,          # 来自 F4，不是猜的
    group_id,
    token_start, token_end,      # T 单位
    mgr_block_size,              # B_g，来自 group spec（F1 真值）
    required_mgr_blocks,         # ceil(token_end / B_g)
    allocated_block_ids,         # 块表行的有效前缀（长度 = required 时才允许继续）
)
```

不变量：`required_mgr_blocks <= len(allocated_block_ids)`，不成立就在
builder 里 raise（带 I6 的全套现场），**禁止 min() clamp**。
builder 的切片宽度 = write-back 的真实需求，不再用 seq_lens/snap_len 等
滞后量近似。

### I4 身份唯一来源
`request_id → v8 state` 的映射只许来自 `input_batch.req_ids`（F4）。
具体注入点：monkeypatch `worker/gpu/attn_utils.py:209` 的
`build_attn_metadata`（或 `CommonAttentionMetadata` 构造点），把
`req_ids[:num_reqs]` 按 group 附到 metadata 上。
**blocks[0] / 前缀匹配 / 变长规则 / n_computed==0 全部禁用**，
包括作为 fallback。

### I5 状态生命周期
state 的创建只在"batch 中出现了一个从未见过的 request_id"时；
request 结束信号用 vLLM 的 finished 集合（worker 每步都有），
state 销毁与块释放解耦——块复用不再影响 state，因为身份不靠块。

### I6 失败现场最小完备集
任何 raise 必须打印：`request_id, group_id, T(start/end/computed/
scheduled), B_g, P, required, available, block_ids 前 16 个`。
字段不齐的 raise 视为 bug。

## 3. 验收 Gate（全绿前不动 GPU）

### Gate 1 — 单位换算 contract test（纯 CPU，不加载模型）
- 用假数据驱动 §2 I2 的换算函数：随机 B_g ∈ {16,32,64,512,1024}、
  随机 token 长度到 131072，断言 S→(j,b_j,offset) 全链一致；
- 用 vLLM 0.19.1 源码里的真实 helper（cdiv/lcm）交叉验证；
- 产出：换算函数 + 本档 §2 I2 的最终确认（P 与 B_g 是否相等，以
  `get_kv_cache_group_metadata` 和 pool shape 的打印为准——这条需要一个
  一次性、加载配置但不加载权重的 probe 进程，允许跑）。

### Gate 2 — request identity test（纯 CPU，模拟调度器）
模拟序列：新请求 → 下一步 → 等长重排 → 前缀命中 → 请求结束块复用 →
两并发交错。断言：每步 `req_ids[i] → state` 唯一且稳定；任何启发式字段
（blocks[0]、块表内容）不参与判定（代码 review + 测试双重禁绝）。

### Gate 3 — builder/impl 不变量 property test（纯 CPU）
随机生成 `(seq_lens, scheduled_tokens, computed_tokens, 各 group 块表,
B_g)`，跑 builder 产出 BlockPlan，断言 I3/I6；再对 impl 的索引数学做
同构检查（write span == BlockPlan 声明 span）。几千个随机 case。

### Gate 4 — 实机最小验证（仅当 1–3 全绿）
加载 35B，只跑 1 → 2 → 4 并发，数据 32k，通过标准：
版本号为新版（u 跳过，直接新版本命名）、无 raise、state reset 计数 0、
fact_acc ≥ 0.7。**禁止直接上 N=8/16。**

## 4. 与旧代码的关系

- `vllm_plugin/` 现有 builder 的身份逻辑（backend.py 整个 states 字典 +
  blocks[0] key）和 impl 的 `bs = kv_cache.shape[2]` 换算按 §2 I2/I4 判定为
  设计错误，重写时不作参考实现，只保留 eviction/attention 数学（这部分与
  vLLM 无关、已单测过）。
- 单请求 64k slots 扫描结果（0.844–0.938）与 ICN 线不受影响，结论继续
  有效；但"v8 能提高 vLLM 并发多少"在 Gate 4 通过前不回答。
