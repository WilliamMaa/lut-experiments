是，这个问题确实很大，而且我看完你这份清单、再对照 vLLM 当前的 hybrid KV cache 设计后，我觉得要比你这里写的还再往前推一步：

> **现在不能修 OOB。必须停掉 v8→vLLM 的并发集成，重新定义“v8 在 vLLM 的 KV 生命周期里到底操作哪一种 block，以及这些 block 之间怎么映射”。**

而且有一个特别重要的地方：你清单第 1 条里准备加的

```python
assert spec.block_size == pool_bs == scheduler_block_size
```

**很可能本身也是错的。**

因为 vLLM 在 hybrid KV cache 下，本来就可能存在多套合法但不同的 block granularity。

vLLM 当前源码明确区分：

- **manager/group block size**：每个 KV cache group 实际分配的 block 大小；
- **scheduler block size**：scheduler 做 token 对齐和 allocation accounting 用的粒度，多 group 时是各 group effective block size 的 **LCM**；
- **hash block size**：prefix hash 的粒度，多 group 时可以是各 group block size 的 **GCD**；
- attention kernel 甚至还可能有自己的 **kernel block size**。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py?utm_source=chatgpt.com)

而 hybrid manager 代码也明确维护每个 group 自己的 `manager.block_size`，不是所有 group 都共享一个物理 block size。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py?utm_source=chatgpt.com)

所以你现在看到：

```text
pool bs = 32
scheduler allocation 看起来像另一种单位
spec 又是一个值
```

**不一定说明 vLLM 自己单位乱了。**

更可能说明：

> 我们把不同层级的 block unit 当成了同一个东西。

这比“hardcode 写错一个数字”严重得多，但也意味着现在终于找到真正的问题层级了。

---

## 一、先彻底停掉 n→t 这种修法

这点你自己的判断完全正确。

现在任何类似：

```text
if n_computed == 0 ...
if block prefix matches ...
if snap_len increases ...
min(...)
```

都应该停止。

因为这些东西是在猜：

> “vLLM 现在大概处于 request lifecycle 的哪个阶段。”

而 vLLM 本身已经有真正的 request identity。

当前 V1 engine 的 `EngineCoreRequest` 明确带有：

```python
request_id: str
```

KV cache manager 也直接提供：

```python
get_blocks(request_id)
get_block_ids(request_id)
get_block_ids_for_computed_tokens(request_id, ...)
```

也就是说 request identity 和 request→block ownership 本来就是 scheduler/core 的显式状态，不需要用 `blocks[0]` 猜。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py?utm_source=chatgpt.com)

更直接的是 LMCache 官方集成本身就在这么干：

```python
LMCacheMPRequestTracker:
    request_id
    allocated_block_ids
    num_scheduled_tokens
    num_stored_blocks
```

它不是靠 block ID 猜 request 生命周期。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py?utm_source=chatgpt.com)

所以你第 5 条其实已经可以升级成：

> **当前 v8 backend 的 request-state tracking architecture 应废弃。**

不是修规则。

---

# 二、我们现在首先要定义 4 个“单位”

在写任何代码以前，我会先做一张这样的表。

| 单位 | 谁定义 | 用来干什么 |
|---|---|---|
| Scheduler token unit | vLLM scheduler | admission / `num_computed_tokens` / scheduling |
| KV group block | 每个 KV cache manager/spec | 真正 physical KV allocation |
| Hash block | prefix cache | content identity |
| Kernel block/page | attention backend | slot mapping / kernel access |

然后对于 Qwen3.6 这种 hybrid 模型，把**每个 KV group**列出来：

```text
Group 0 full-attention:
manager block = ?
physical pool shape = ?
kernel block = ?

Group 1 GDN:
manager block = ?
state shape = ?
scheduler alignment = ?

Group 2 ...
```

vLLM 自己甚至提供 `get_kv_cache_group_metadata()`，返回每个 group 的：

```text
group_idx
kind
block_size
sliding_window
```

这本来就是给 external consumers 看 KV group metadata 的。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py?utm_source=chatgpt.com)

所以第一件事情不是：

```python
assert all block sizes equal
```

而是：

```text
打印 vLLM 自己认为的每一层 block semantics
↓
建立显式 mapping
↓
只有“不满足合法映射关系”时才 fail
```

例如：

\[
B_{\text{scheduler}}
=
\mathrm{LCM}(B_0,B_1,\ldots)
\]

这是 vLLM 当前源码明确采用的规则。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py?utm_source=chatgpt.com)

这才是应该 assert 的不变量。

---

# 三、builder → impl 之间必须有一个明确的数学 contract

你现在的核心事故其实就是：

```text
builder 认为给了 256 blocks
impl 认为需要 512 blocks
```

这件事根本不应该运行到 CUDA write-back 才知道。

应该在 builder 完成时就形成类似：

```python
BlockPlan(
    request_id,
    group_id,
    logical_token_start,
    logical_token_end,
    manager_block_size,
    kernel_block_size,
    required_manager_blocks,
    required_kernel_blocks,
    allocated_block_ids,
)
```

然后必须成立：

\[
N_{\text{allocated}}
\ge
N_{\text{required}}
\]

如果不成立：

```text
raise before model forward
```

不能 `min()`。

你这一条批评完全对：

```python
n_need = min(required, available)
```

在这里不是 defensive programming。

它是在：

> **隐藏 impossible state。**

应该直接变成：

```python
if required > available:
    raise BlockPlanInvariantError(...)
```

并把：

```text
request_id
group_id
scheduler tokens
manager block size
kernel block size
required
available
block ids
```

一次全部打印出来。

---

# 四、还有一个更根本的问题：v8 compression 的“block”到底是哪一种 block？

这个现在必须明确。

你的算法原来操作的是：

```text
token positions
→ heavy hitter selection
→ compact 128/512/1024 slots
```

它本质上是**模型层 KV token layout**。

而 vLLM 的 block/page 是：

> allocator/storage granularity。

这两个概念不是一回事。

不能因为都叫 block 就直接一一对应。

真正需要定义的是：

\[
\text{vLLM physical pages}
\longrightarrow
\text{v8 logical compact positions}
\]

或者反过来：

\[
\text{v8 retained token slots}
\longrightarrow
\text{which physical KV pages/slots remain valid}
\]

如果这一层映射没定义清楚，你后面永远会遇到：

- scheduler 认为 capacity 是 X；
- backend 认为 compact length 是 Y；
- allocator 已经 reuse 了 block ID；
- v8 还认为是上一 request 的 state。

所以目前真正要重写的不是一个函数。

而是：

> **v8 cache representation ↔ vLLM KV manager representation adapter。**

---

# 五、特别是 hybrid Qwen，这件事更不能靠统一 block 假设

这一点你现在遇到事故反而很有价值。

vLLM 对 hybrid KV cache 就明确考虑了：

> 不同 KV cache group 可以有不同 block sizes，而 scheduler 必须通过协调规则统一管理。

它甚至明确写到：`num_gpu_blocks * block_size` 对 hybrid model 可能是错的，因此专门引入了 group-aware `kv_cache_size_tokens` / `kv_cache_max_concurrency`。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py?utm_source=chatgpt.com)

所以你之前：

> “scheduler 给了 8 blocks，所以每 block 应该是多少 token”

这种逆推，在 hybrid 情况下本身就危险。

**8 个 scheduler-visible allocation units 不一定等于 8 个 attention-pool physical pages。**

这正是当前最需要验证的。

---

# 六、我完全同意“不再让用户跑 repro”

下一步应该全部是静态/CPU 层面的。

而且我会比你的清单再严格一点：

### Gate 1 — Block-unit contract test

不加载模型。

构造：

```text
1 group
multi group
不同 group block size
8192 / 32768 / 65536 token
```

验证：

```text
scheduler unit
manager unit
kernel unit
v8 compact unit
```

之间的换算。

---

### Gate 2 — Request identity test

模拟：

```text
new request
same request next scheduler step
equal-length reschedule
prefix hit
block reuse after request finishes
two concurrent requests
```

每一步确保：

```text
request_id → v8 state
```

唯一稳定。

完全禁止：

```text
blocks[0]
prefix heuristic
length-growth heuristic
```

参与 identity。

---

### Gate 3 — Builder/impl invariant test

随机生成：

```text
seq_len
scheduled_tokens
computed_tokens
block tables
group block sizes
```

做 property test：

\[
required\_slots
\le
available\_slots
\]

以及 builder 计算出的 write span 与 impl 完全一致。

这个用 Python 就能测几千个随机 case。

---

### Gate 4 — 只有前三个全绿，才加载 35B

然后只跑：

```text
1 request
2 requests
4 requests
```

不直接上 32 并发。

---

# 七、我现在反而建议先不要承诺“用 request_id”具体从 attention backend 直接拿

方向一定是对的，但接口位置得先查清。

vLLM engine/core 层肯定有 `request_id`，KV manager 也以 request ID 为 key。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py?utm_source=chatgpt.com)

但：

> attention metadata builder / custom backend 的那个具体 callback 上是否直接暴露 request_id

需要根据你固定的 **vLLM 0.19.1** 源码确认。

如果 backend interface 没直接给，就应该：

```text
scheduler/model-runner 显式建立
batch slot ↔ request_id
```

传下来。

而不是因为 backend 没现成字段就退回 block heuristic。

这个设计原则应该钉死。

---

# 八、所以这次事故到底意味着什么？

我会非常明确：

**它使现在所有“vLLM concurrency integration”的正结果暂时都不能继续往上堆。**

不是说之前算法结果失效。

这些仍然成立：

- 原 v8 单流 compression；
- quality/budget 结果；
- 原 HF harness 的 decode TPOT 现象；
- ICN 那条线。

但是：

> **新的 vLLM integration 还没有证明其 cache lifecycle semantics 正确。**

所以在这个 adapter 重新验证以前，不能用它回答：

> “v8 可以把 vLLM concurrency 提高多少？”

这不是坏消息。

至少我们现在没有继续拿错误 integration 跑 30 个 cell 再回头发现全部无效。

---

## 我现在会把任务压缩成一句话

不要做 v2026-10-04u。

不要再修一个 if。

下一项工作应该叫：

> **V8–vLLM KV Semantics Contract**

先写清楚并通过自动化验证：

\[
\boxed{
\text{request identity}
+
\text{scheduler token unit}
+
\text{KV-group block unit}
+
\text{kernel slot unit}
+
\text{v8 compact slot unit}
}
\]

之间唯一且可检查的关系。

**只有 contract 成立以后，我们再写 integration。**

你这次列出来的问题本质上不是“代码质量差了一点”，而是我们一直没有先定义这个 contract。现在既然已经看到了，就应该直接在这里截断补丁路线，重建这一层，而不是再让实机替我们做类型检查。