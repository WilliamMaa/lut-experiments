我把这份结果从头到尾按“**数字是否互相支持、归因是否真的成立、有没有 harness/实验定义把结论带偏**”重新过了一遍。结论是：**确实有几处明显不合理，而且不是小措辞问题。最严重的是 quality gate、显存墙归因、以及所谓 sustainable concurrency 的定义。**

### 1. 最大的问题：你现在不能说“并发 serving 对压缩配置是安全的”

这是我认为目前最需要纠正的结论。

你自己的 32k 数据是：

| config | N=1 | N=8 | N=16 | N=32 |
|---|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 0.984 | 1.000 |
| m4_k8v8 | **0.750** | **0.484** | **0.469** | **0.531** |

:chatgpt-content-reference{index="1"}

你现在的解释是：

> N=1 样本只有 8，所以 0.75→0.48 属于噪声，并发没有额外质量损失。

从统计上说，**确实不能证明 N=1→N=8 有显著下降**——N=1 只有 8 个样本，这一点没错。

但反过来也不能证明：

> “并发不降解”或者“并发 serving 安全”。

正确结论只能是：

> **在 N≥8 的较大样本区间，没有观察到随 concurrency 单调恶化的趋势。**

这个结论是成立的：

```text
32k m4_k8v8:
N=8   0.484
N=16  0.469
N=32  0.531
```

64k：

```text
N=1   .500
N=8   .500
N=16  .438
```

128k：

```text
N=1   .500
N=8   .469
```

确实没有明显 concurrency trend。:chatgpt-content-reference{index="2"} :chatgpt-content-reference{index="3"}

但是另一个更大的事实是：

> **压缩本身的 factual quality 已经很差。**

32k 大约 0.47–0.53，64/128k 大约 0.44–0.50，而 full≈1.0。

所以现在最准确的结论不是：

> concurrent serving is safe

而是：

> **Batching does not show additional degradation beyond the substantial quality loss already introduced by the 128-slot compression regime.**

这是完全不同的意思。

---

# 2. 你的 sustainable concurrency 定义现在其实不成立

这是第二个严重问题。

你定义 sustainable：

1. peak HBM ≤ 512 GB；
2. EOS ≥ full − 2pp；
3. fact accuracy 只记录，不设门槛。:chatgpt-content-reference{index="4"}

然后得到：

```text
full       sustainable N=64
m4_k8v8   sustainable N=32
```

但 m4_k8v8 的 fact accuracy 大约只有：

```text
0.47–0.53
```

而 full 是：

```text
~1.0
```

那么：

> **一个丢掉一半事实的配置怎么能和 full 在同一个“可持续 serving”定义里比较？**

EOS 根本不是足够的 quality constraint。

模型能正常停下来：

```text
EOS = 1
```

不代表回答还是对的。

所以这里其实是你 evaluation definition 有问题，而不是结果有问题。

至少应该有一个 **quality-preserving concurrency**：

\[
N_{\max}^{quality}
\]

要求例如：

\[
Acc_{compressed}
\ge
Acc_{full}-\epsilon
\]

或者不用人为定 \(\epsilon\)，直接同时报告：

```text
memory-feasible N
EOS-feasible N
fact-quality curve
```

不要硬压成一个 max sustainable N。

按照你当前 fact 数据，如果要求接近 full，**m4_k8v8 在这个 benchmark 上可能 N=1 都不满足。**

这不是坏结果。

它告诉我们的恰恰是：

> v8 原来那个 53-turn benchmark 把 compression 的 information-loss 严重低估了。

这是很重要的新发现。

---

# 3. 64k 的 EOS=0.953 不能直接叫“规则线噪声”

64k N=16：

```text
full       EOS = 1.000
m4_k8v8    EOS = 0.953
```

:chatgpt-content-reference{index="5"}

N=16 × 8 turns，如果每个 turn 一次输出，大约是 128 个样本。

0.953 大约意味着 **6 个 EOS failure**。

这不像 N=8 下：

```text
0.969 ≈ 2/64 failures
```

那么容易直接归为小样本波动。

在没有重复 run 之前，我不会写：

> 属规则线噪声。

应该写：

> **N=16 exhibits a measurable EOS degradation; replication is required to determine whether it is concurrency-induced or run-level variability.**

这个格值得补 2–3 reps。

---

# 4. 最大的显存归因矛盾：你说“墙与是否压缩无关”，但数字恰恰说明有关

这里是我觉得文档内部最明显的逻辑冲突。

你总结：

> 墙的位置由 \(B\times K\) 决定，与是否压缩 KV 无关。

但是紧接着你的表：

```text
              32k   64k   128k   B×K @ wall

full           64    32     18    ~2.0–2.4M
m4_k8v8        32    16      8    ~1.0M
```

:chatgpt-content-reference{index="6"}

这实际上说明：

\[
(BK)_{\text{wall, compressed}}
\approx
\frac12(BK)_{\text{wall, full}}
\]

所以真正的结论应该是：

> **两种路径的 memory wall 都近似随 \(B\times K\) 缩放，但 compression path 的比例常数明显更大，因此更早 OOM。**

而不是：

> 与是否压缩无关。

这是一个本质区别。

---

# 5. `repeat_kv` 能解释共同的 B×K 墙，但解释不了为什么 compressed 比 full 早一倍撞墙

你现在定位出：

```text
repeat_kv:
2 × B × 16 × K × 256 × 2 bytes
```

并且 PyTorch 的 SDPA/GQA 路径确实存在不同 backend 对 GQA 支持不同的问题；官方文档也明确说明 `enable_gqa` 的支持有 backend 约束，而 vLLM 会使用 FlashAttention/FlashInfer/Triton 等专门 GQA backend 来避免普通 serving stack 的一些开销。:chatgpt-content-reference{index="7"}

所以：

> **HF/PyTorch serving harness 的 GQA transient 是真实问题。**

这一点我认可。

但是：

```text
full wall      ~2.0M BK
compressed     ~1.0M BK
```

说明 compressed path 还有一个很大的额外 \(O(BK)\) 成本。

而你自己的数据也特别明显：

64k N=16：

```text
full       205.4 GB
m4_k8v8    279.4 GB
```

差 **74 GB**。:chatgpt-content-reference{index="8"}

full 和 compressed 都：

- 做 full prefill；
- 都走 repeat_kv；
- 都还没进行首次 decode eviction。

那么这 74GB 从哪来？

不能用共同的 `repeat_kv` 解释。

最大的嫌疑还是：

- attention-score stash；
- stash wrapper 的 intermediate；
- fp32 score accumulation；
- compression-specific bookkeeping；
- 或者额外 attention calculation。

所以现在真实的 memory model 应该是：

\[
M_{\text{full}}
=
M_{\text{weights}}
+
aBK
+
M_{\text{KV}}
\]

而：

\[
M_{\text{compressed}}
=
M_{\text{weights}}
+
aBK
+
M_{\text{KV}}
+
\boxed{bBK}
+
M_{\text{compression}}
\]

这个 \(bBK\) 目前还没被解释掉。

**这应该成为下一步最优先的 memory profiling。**

否则现在说：

> 第一瓶颈就是 repeat_kv

还太早。

更准确是：

> **repeat_kv explains a major shared B×K transient, but an additional compression-specific B×K overhead remains unresolved.**

---

# 6. 你的“压缩节省 HBM”实际上到现在还没有直接测出来

目前表里全是：

```text
Peak HBM
```

而你自己已经证明 peak 发生在：

> eviction 之前的 prefill。

那么 Peak HBM 天生看不到真正的 128-slot storage benefit。

但是后面你写：

> decode 后 MB 级稳态 HBM。

:chatgpt-content-reference{index="9"}

这个理论上应该是对的，但**当前结果表没有直接给证据**。

应该增加：

```text
HBM after prefill
HBM immediately before eviction
HBM after first eviction
steady decode HBM
KV-resident bytes only
non-KV allocated bytes
```

这样可以直接得到：

\[
M_{\text{steady,full}}
\]

vs

\[
M_{\text{steady,m4}}
\]

否则现在出现一个很尴尬的情况：

> 论文想讲 KV memory compression，
> 结果唯一正式 HBM 表显示 compressed 比 full 更费内存。

虽然我们知道为什么，但 reviewer 不会替我们脑补 steady-state。

---

# 7. “512GB KV budget”其实不是 KV budget

你现在 gate 是：

```text
peak_hbm_mb / 1024 <= 512GB
```

但 `peak_hbm` 显然包含：

- model weights；
- activations；
- attention transient；
- stash；
- KV；
- allocator overhead。

所以这个 **512GB 不是 KV budget**。

它实际上是：

> aggregate process HBM peak budget。

而且机器是：

```text
7 × 80GB = 560GB
```

还要考虑 allocator、CUDA context 等。

所以建议彻底改名字：

```text
--hbm-budget-gb
```

而不是：

```text
--kv-budget-gb
```

然后如果真想测固定 KV budget：

\[
M_{\text{KV only}}
\]

必须单独记账。

这个不只是名字问题，因为你最开始的研究问题就是：

> fixed HBM/KV capacity 下 compression 能增加多少 concurrency？

目前这件事实际上**还没被 cleanly tested**。

---

# 8. TPOT 3.4–3.9× 是真实而且很有价值，但不要把它解释成整体 serving speedup

这一项我认为数据很漂亮：

32k N=32：

```text
full       793.9ms
m4_k8v8    204.2ms
≈3.9×
```

64k N=16：

```text
651.6ms vs 184.6ms
≈3.5×
```

128k N=8：

```text
666.8ms vs 196.7ms
≈3.4×
```

:chatgpt-content-reference{index="10"} :chatgpt-content-reference{index="11"} :chatgpt-content-reference{index="12"}

这三个独立长度都落在 **3.4–3.9×**，非常稳定。

而且机理明确：

\[
K_{\text{decode}}:
32K/64K/128K
\rightarrow
128
\]

full-attention 层 decode 的 KV walk 大幅减少。

这个我认为是目前 concurrency experiment **最可靠的正结果**。

但它只能叫：

> **decode-step latency reduction**

不能叫：

> throughput 3.9×

因为你的 harness 是 lockstep batch、手写 decode，不是 continuous batching，文档自己也承认绝对 throughput 不对应生产 serving。:chatgpt-content-reference{index="13"}

真正 throughput 要迁 vLLM 类 serving stack 后再讲。

---

# 9. 更大的科学问题：新 benchmark 已经推翻了“m_sp4 几乎无质量代价”的旧印象

这个我觉得反而特别重要。

旧 docs/19：

```text
m_sp4 1000x
EOS == baseline
sentinel all correct
```

现在真正 long-context factual retrieval：

```text
32k: ~0.48
64k: ~0.50
128k: ~0.47
full: ~1.00
```

这说明：

> **原来的 sentinel/53-turn benchmark 对 information preservation 太宽松。**

不是 concurrency 把算法搞坏了。

而是这个新的 benchmark 第一次真正暴露：

\[
128\text{-slot budget}
\]

对于长上下文 factual recall 本身就过于激进。

这意味着 v8 下一步最重要的问题已经不是：

> 还能不能搞 2000×？

而是：

> **compression ratio × context length × factual preservation 的 Pareto frontier 到底在哪？**

例如应该扫：

\[
budget
\in
\{128,256,512,1024,2048\}
\]

在：

\[
32K,64K,128K
\]

下测：

```text
Fact Accuracy
TPOT
steady KV bytes
```

这才会产生真正有价值的 Pareto 图。

---

# 10. 现在的 ladder 也没有显示“每个组件都在改善质量”

32k：

```text
N=8:
hh             .469
hh_merge       .406
hh_merge_m4    .453
m4_k8v8        .484

N=32:
hh             .391
hh_merge       .406
hh_merge_m4    .484
m4_k8v8        .531
```

:chatgpt-content-reference{index="14"}

所以不能再讲一个简单故事：

```text
HH
→ merge improves
→ M4 improves
→ INT8 slight loss
```

新数据显然不是这样。

特别有意思的是：

> **k8v8 居然没有比 bf16 M4 更差，甚至有些格更高。**

这很可能只是 sampling/statistical fluctuation，也可能 quantization 改变了 trajectory。

但总之需要：

> **paired per-question analysis**

而不是只看 aggregate accuracy。

同一个 question：

```text
HH       correct?
Merge    correct?
M4       correct?
INT8     correct?
```

做 McNemar / transition matrix。

这样才能真正回答：

> M4 修了哪些题？
> merge 又破坏了哪些题？

这对于你们解释方法远比再加一个平均值重要。

---

## 我现在会把整批结果重新归纳成四句话

**A. 可信的正结果**

> 极端 KV reduction 在 decode 阶段确实稳定降低 long-context attention cost；32–128k 下，受测并发点 TPOT consistently 改善约 **3.4–3.9×**。

**B. 可信的负结果**

> 当前 deferred-eviction 实现**没有扩大并发容量**；反而因为 compression-specific prefill overhead，使压缩路径更早撞 HBM wall。

**C. 新暴露的问题**

> 128-slot 极端 compression 在新的 long-context factual benchmark 上有严重质量损失（约 25–55pp），旧 benchmark 明显不足以刻画 information preservation。

**D. 尚未回答**

> 真正 KV-storage compression 在固定 KV/HBM budget 下到底能提升多少 production-serving concurrency，目前**还没被这个 harness cleanly 测出来**，因为 peak 被 full-prefill residency + attention transient 主导。

---

所以我现在最不建议做的是继续跑更多 `N`。

**下一步应该先解决两件事：**

1. 把 HBM 分解成：
   `weights / resident KV / stash / repeat_kv transient / other`，解释 compressed path 为什么约在 full 一半 \(B\times K\) 就 OOM；
2. 扫 `KV budget = 128/256/512/1024/...`，画真正的：

\[
\boxed{
\text{Fact accuracy}
\;\leftrightarrow\;
\text{TPOT}
\;\leftrightarrow\;
\text{steady KV bytes}
}
\]

这会比现在那个“max sustainable N”更接近 v8 真正要回答的科学问题。

至于 `repeat_kv`：你的诊断方向是可信的，PyTorch 官方 SDPA 对 GQA 的 fused backend 支持确实有限，而生产 serving 系统如 vLLM 会使用 FlashAttention/FlashInfer/Triton 等专门 attention backend 来支持 GQA 和 paged KV。:chatgpt-content-reference{index="15"} **但它只能解释公共 transient，不能解释 compressed/full 之间那一倍左右的墙差；那部分现在仍然是未解释问题。**