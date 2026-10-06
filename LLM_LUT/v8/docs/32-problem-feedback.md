这个方向我赞成，而且这次我会直接说：**要写 harness，而且这是现在最合理的下一步。**

因为你现在遇到的不是模型问题、不是数值问题、也不是“某个 corner case”。真正的问题是：

> **我们的 v8 backend 对 vLLM scheduler 生命周期的理解没有被可执行地定义。**

而 fake scheduler harness 正好把这件事从“8 卡机器上试错”变成“本地几秒钟验证状态机”。

vLLM 自己的 scheduler 本来就不是简单的“prefill / decode 两阶段”。它每一步只根据 `num_computed_tokens` 和当前需要追赶的 token 数决定本轮 schedule 多少 token，因此天然会出现 chunked prefill、cached/resumed request、preemption 等多种 step 形态。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py?utm_source=chatgpt.com) 所以你现在说的这五种场景：

- chunk1；
- chunk2；
- decode；
- 两请求交错；
- preemption/resume；

确实就是 integration 层必须明确覆盖的基本状态，而不是边角 case。

而且 vLLM 自己的测试也是这么干的：它会直接创建 scheduler/request，在不跑真实模型的情况下连续调用 `schedule()` / `update_from_output()`，验证第一次 chunk 1024、第二次 1024、第三次剩余 452 之类的行为。[GitHub](https://github.com/vllm-project/vllm/blob/main/tests/v1/core/test_scheduler.py?utm_source=chatgpt.com) 所以 **fake scheduler / fake metadata harness 不是临时 debug hack，本身就是符合 vLLM 测试方式的正确工程路径。**

但我建议你不要只写一个“能复现现在 crash 的 harness”。要一次性把它写成 **integration contract test**。

最重要的是让它验证下面这些守恒关系。

### 1. Request identity 必须来自 scheduler 真值

不要再允许：

```text
blocks[0]
长度变化
prefix match
snap_len
```

决定 request identity。

vLLM 的 scheduler 输出本来就携带 request ID；KV connector 工具也直接从 `scheduled_new_reqs` / `scheduled_cached_reqs` 取 `req_id` 和 block IDs。[GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/utils.py?utm_source=chatgpt.com)

所以 harness 里应该显式制造：

```text
request A
request B
```

然后验证所有 backend state 永远按：

```text
request_id
```

归属。

特别要覆盖：

```text
A chunk1
B chunk1
A chunk2
B chunk2
A decode
```

确保 B 的任何行为都不能污染 A。

---

### 2. Builder 和 impl 必须共享同一个 BlockPlan

这是你现在最缺的东西。

不要：

```text
builder 自己算长度
impl 再根据 tensor shape 重新猜一次
```

应该 builder 明确生成：

```text
BlockPlan {
    request_id
    group_id
    scheduled_tokens
    compact_len_before
    compact_len_after
    required_blocks
    allocated_block_ids
    write_start
    write_end
}
```

然后 `impl.forward()` **只能消费这个 plan**。

不能再次独立推导：

```text
我觉得应该有 512 blocks
```

否则你永远可能再次出现：

```text
builder: 256
impl: 512
```

这是 architecture contract，不是 assert 修 bug。

---

### 3. block size 不能要求“三处相等”

这一点要特别小心。

vLLM 的 attention backend/spec 本来就允许 backend 根据 kernel layout 调整 KV cache spec，而且 chunked prefill / scheduler 的 token accounting 也不是简单等同于 physical cache tensor shape。[vLLM](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backend/?utm_source=chatgpt.com)

所以 harness 真正要 assert 的不是：

```python
scheduler_bs == spec_bs == pool_bs
```

而是：

> **它们之间的映射关系是否明确且满足。**

例如：

```text
scheduled tokens
    ↓ ceil / alignment
logical compact slots
    ↓ mapping
physical pool blocks
```

最终要求：

\[
\text{writeable capacity} \ge \text{planned write span}
\]

以及：

\[
\text{plan.tokens}=
\text{scheduler.num\_scheduled\_tokens}
\]

不是所有 block unit 强行数值相等。

---

## fake harness 我会要求至少覆盖 8 个 case

你现在列了五个，我再加三个非常关键的。

1. **单请求 chunk1**
2. **同请求 chunk2**
3. **prefill → decode**
4. **A/B 两请求交错**
5. **preempt → resume**
6. **等长 scheduler step**
   - 防止之前“严格变长才算同 request”的启发式死灰复燃。
7. **block ID recycle**
   - A 完成释放后，同一个 physical block ID 被 B 重用，必须不能继承 A 的 v8 state。
8. **不同 KV group / block unit**
   - 这是这次事故最核心的 regression case。

如果这 8 个 case 都能在 CPU/small tensor 上跑：

```text
1000 次随机交错
0 invariant failure
```

再去上 35B。

---

## 最好再加 property-based test，而不是只写固定 case

因为你这次的问题就是固定 happy path 没暴露。

随机生成：

```text
num_scheduled_tokens
previous computed tokens
request interleaving
preemption point
physical block IDs
group/block sizes
compact lengths
```

然后永远检查：

\[
0 \le write\_start \le write\_end \le pool\_capacity
\]

\[
required\_blocks \le allocated\_blocks
\]

\[
state.request\_id = scheduler.request\_id
\]

\[
compact\_len \le configured\_budget + current\_new\_chunk
\]

这样很多逻辑错误根本不需要 GPU。

---

## 我还会再加一个“禁止项”

harness 过之前，代码里搜索并禁止：

```text
min(required, available)
```

这种 capacity clamp。

以及任何类似：

```text
if blocks[0] == ...
if snap_len ...
if n_computed == 0 then new request
```

的 request identity heuristic。

如果必须存在，测试直接 fail。

这样可以防止下一版又偷偷绕回去。

---

## Gate 4 也应该改

现在 Gate 4 是：

> 2 并发、32k、fact_acc ≥ 0.7。

我觉得这其实混了两个问题。

integration correctness gate 应该先只验证：

```text
2 requests
32k
chunked prefill
decode
multi-turn/resume

→ no crash
→ no OOB
→ no cross-request state leak
→ full vs v8 expected cache layout correct
```

**fact_acc ≥ 0.7 是算法质量 gate，不是 integration contract gate。**

它可以紧跟在后面，但不要把：

> backend state machine 正确

和：

> 128/512/1024 slots 质量够不够

混成一关。

否则下一次 fact fail，你又不知道是在 debug integration 还是方法。

---

所以我会回复那个 Agent：

> **写。现在停止所有远程 repro，先实现 fake scheduler harness。**
>
> 目标不是复现当前 16384/8192 crash，而是把 vLLM→v8 integration contract 变成可执行测试：request identity、chunk progression、block-plan/writeback consistency、preemption/resume、block recycling、multi-request interleaving、multi-group block mapping 全部在本地验证。
>
> builder 必须产出单一 BlockPlan，impl 只消费它；禁止用 `blocks[0]` / 长度变化猜 identity，也禁止用 `min()` 把 required>available 静默裁掉。
>
> Harness 全绿前，不再上 8 卡机。

我觉得这次一定要把反馈周期从：

```text
15 分钟 × 一次猜测
```

变成：

```text
几秒 × 几千状态组合
```

否则继续集成没有意义。