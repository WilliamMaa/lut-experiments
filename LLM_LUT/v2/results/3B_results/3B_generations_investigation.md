对，**这绝对不是好事**。它有趣，但不能当成“replacement 成功”。你现在的判断应该是：

> 这个 replacement 在 PPL / Acc 上看起来还行，但它没有保持 behavioral equivalence。英文 prompt 变中文输出，属于明显的 instruction/style fidelity failure。

所以我们现在不要急着庆祝，也不要急着否定整个方向。应该把它拆成两个问题：

1. **为什么会发生 language prior drift？**
2. **怎么把它压回去，同时保留 LUT replacement 的收益？**

---

## 我觉得最可能的原因

### 1. Layer 9 Group 27 可能本来就参与 language routing

Qwen2.5-3B 的模型不算特别深，Layer 9 已经可能是中层了。中层 hidden state 里很可能已经开始出现：

* answer language；
* instruction style；
* response format；
* assistant persona；
* locale prior；
* code/natural language mode。

所以你替换 Layer 9 Group 27，不只是替换某个“计算块”，而是在 residual stream 里动到了一个控制输出风格/语言的方向。

这也解释了为什么它不是纯粹变差。
它仍然会算 17×6，仍然知道东京，仍然会写 `s[::-1]`，但是**回答语言偏了**。你 generation samples 里就是这种情况：语义大多还在，但英文 prompt 下大量转中文，甚至出现 “To计算”“This表达式” 这种 code-switching。

---

### 2. LUT bucket 可能学到了“平均化”的中文方向

你的 LUT 是 64×64 bins。这个结构会把连续 activation 压到离散 bucket 里。
如果这个 group 的 activation space 里英文/中文本来是细微分开的，而 bucket 太粗，可能会出现：

> 英文 hidden state 被映射到更接近中文 response manifold 的 bucket center。

也就是原本：

```text
English prompt hidden state -> English answer region
Chinese prompt hidden state -> Chinese answer region
```

替换后变成：

```text
English prompt hidden state -> mixed / Chinese-biased region
Chinese prompt hidden state -> Chinese answer region
```

这就是典型的 quantization / approximation 导致的 latent boundary shift。

---

### 3. Calibration 数据可能有语言分布偏差

如果你 LUT pretrain / bucket calibration 的数据里中文比例偏高，或者用了中英混合数据，Layer 9 Group 27 就可能把中文样本的局部均值学得更强。

但即使数据不是中文偏多，也可能发生。因为 Qwen 本身中文能力强，中文 token manifold 在某些区域可能更“容易被吸过去”。

所以这个问题不一定是数据错误，也可能是 group 本身敏感。

---

## 现在该怎么办：不要直接放弃，先做定位

我建议你接下来分三步做。

---

## Step 1：确认这是 language prior drift，不是 prompt 偶然

你现在 sample 只有 6 个 prompt，而且每个 prompt 10 次输出完全一致，说明 sampling 设置可能比较 deterministic。
下一步需要做一个小型 language diagnostic set。

设计 4 类 prompt：

### A. 英文 prompt，无语言约束

```text
What is the capital of Japan? Answer with one word.
```

### B. 英文 prompt，强制英文

```text
What is the capital of Japan? Answer in English only. Use one word.
```

### C. 中文 prompt，强制英文

```text
日本的首都是哪里？Answer in English only. Use one word.
```

### D. 英文 prompt，强制中文

```text
What is the capital of Japan? Answer in Chinese only. Use one word.
```

然后统计：

| Metric                      | Meaning             |
| --------------------------- | ------------------- |
| Chinese output rate         | 输出中是否含中文            |
| English-only violation rate | 明确要求英文时是否仍输出中文      |
| Code-switching rate         | 是否中英混杂              |
| Exact correctness           | 答案是否语义正确            |
| First-token language        | 第一个内容 token 是中文还是英文 |

这里最关键的是 B 和 C。

如果 B/C 里 replacement 还能听懂 “English only”，那说明只是 soft prior drift，可以修。
如果 B/C 仍然中文，那说明 Layer 9 Group 27 的 replacement 破坏了 language instruction control，这个 group 就不适合直接替。

---

## Step 2：做 logit-level 检查

不要只看 generation。你要直接看第一步 token logits。

例如 prompt：

```text
What is the capital of Japan? Answer with one word.
```

比较 original vs replacement 的 top-20 first-token logits。

你要看：

```text
Original:
Tokyo
The
Japan
...

Replacement:
东京
Tokyo
日本
...
```

如果 replacement 后中文 token 排名明显上升，那就说明：

> LUT replacement 在输出分布层面注入了中文 prior。

这个证据比 generation samples 更硬。

还可以做一个简单指标：

```text
Chinese token logit mass / English token logit mass
```

不用特别复杂，先人工定义一个 token set 就行：

```python
english_tokens = ["Tokyo", "The", "To", "Certainly", "It"]
chinese_tokens = ["东京", "中国", "这", "我们", "因此"]
```

粗糙也没事，先验证方向。

---

## Step 3：尝试修复，不是马上换层

我会优先尝试下面几个修复路径。

---

# 修复方案 1：加 language-balanced calibration

这是最直接的。

你现在需要确保 LUT calibration / pretrain 数据里有明确的语言平衡：

```text
English-only prompts: 50%
Chinese-only prompts: 25%
Mixed bilingual prompts: 25%
```

更重要的是要加这种 pair：

```text
English prompt -> English answer
Chinese prompt -> Chinese answer
Chinese prompt -> English answer
English prompt -> Chinese answer
```

因为你要让 replacement 学会的是：

> 输出语言由 instruction 决定，不由 hidden state 的默认 prior 决定。

尤其是这两类必须加：

```text
请用英文回答：What is the capital of Japan?
```

```text
Answer in Chinese: What is the capital of Japan?
```

这能让 LUT 不要把“prompt language”和“answer language”绑死。

---

# 修复方案 2：给 LUT 加一个 output delta constraint

现在 replacement 可能只是在 bucket-level 拟合原输出，但没有强约束输出方向。
你可以在 LUT pretrain 的 loss 里加：

```text
MSE(replacement_output, original_output)
+ λ * cosine_distance(replacement_output, original_output)
```

甚至对这个 group 的输出加 residual-preserving loss：

```text
L = MSE(y_lut, y_original)
  + λ * (1 - cos(y_lut, y_original))
  + μ * MSE(h_after_lut, h_after_original)
```

核心是不要只让 bucket approximate 数值，而要让它保持方向。
language prior 很可能就是某个方向偏了。

---

# 修复方案 3：加 language probe regularization

这个更高级，但很有用。

训练一个很小的 probe 判断 hidden state 对应的 expected output language：

```text
English-answer / Chinese-answer / mixed
```

然后 replacement 时加一个 regularization：

```text
Probe(h_replaced) ≈ Probe(h_original)
```

这可以显式压住 language drift。

当然这一步不一定马上做，先做前两个就够了。

---

# 修复方案 4：避开 Layer 9 Group 27，选 KL 更低的 group

如果你只是想证明“3B 也能替换”，那 Layer 9 Group 27 可能不是最合适的第一目标。
因为它太敏感了。

从你的表看，Layer 27 的几个 group 更安全：

| Layer | Group | KL Bucket | PPL Bucket | Acc Bucket |
| ----- | ----: | --------: | ---------: | ---------: |
| 27    |    29 |    0.0475 |      29.64 |     0.4977 |
| 27    |     4 |    0.0380 |      30.94 |     0.4977 |
| 27    |    15 |    0.0384 |      30.58 |     0.5055 |
| 27    |    27 |    0.0418 |      30.37 |     0.5039 |

这些 group 的 KL 很低，更可能保持 behavior fidelity。

但这里要小心：
它们可能只是“不重要”，所以替换成功的意义弱一点。

所以我建议分成两条线：

### Safety line

选 Layer 27 的低 KL group，目标是证明：

> LUT replacement can be behavior-preserving on Qwen2.5-3B.

### Interesting line

保留 Layer 9 Group 27，目标是研究：

> high-impact group replacement causes language/style drift.

这两条线都很有价值，但不要混在一起说。

---

## 我现在对这个现象的定性

我会把它定义为：

> **A high-impact replacement with preserved semantic competence but failed language fidelity.**

也就是：

```text
PPL: ok
Acc: ok
Semantic correctness: mostly ok
Language fidelity: failed
Instruction fidelity: uncertain
```

所以它不是“坏掉”，但也不是“成功”。
它是一个 **partial success with behavioral drift**。

---

## 最推荐的下一步

我建议你下一轮就做一个很小的 diagnostic，不要大规模训练：

1. 用原模型和 replacement 模型跑 100 条英文 prompt；
2. 每条 prompt 加三种版本：

   * no language constraint；
   * English only；
   * Chinese only；
3. 统计 Chinese rate / English-only violation / code-switch rate；
4. 同时保存 first-token top-20 logits；
5. 对 Layer 27 Group 29 也跑同样测试，作为 low-risk comparison。

最后你会得到一个特别清楚的对比：

| Model       |   Group | PPL Δ | Acc Δ |   KL | Chinese Drift | English-only Violation |
| ----------- | ------: | ----: | ----: | ---: | ------------: | ---------------------: |
| Replacement |  L9 G27 |   low |     + | high |          high |                      ? |
| Replacement | L27 G29 |   low |     ~ |  low |           low |                      ? |

这张表会直接告诉你：

* 是不是所有 replacement 都会中文漂移；
* 还是只有 Layer 9 Group 27 这种中层高影响 group 会漂移；
* 强语言指令能不能把它拉回来。

---

我的直觉是：**这不是整个 LUT 路线的问题，而是你选中的 group 太接近 language/style control 了。**

所以现在最稳的策略是：

> 不把 L9G27 当作最终替换目标，而是把它当作一个 diagnostic case；真正展示 scaling feasibility 时，先用 L27 的低 KL group。
> 等机制讲清楚后，再把 L9G27 作为“behavior drift case study”。

这样你们的 report 会更稳，也更像研究，而不是“实验出现奇怪 bug”。
