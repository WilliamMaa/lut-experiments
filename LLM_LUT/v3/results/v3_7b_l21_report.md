# V3 Phase 1 Validation: Qwen/Qwen2.5-7B-Instruct

- Layer: 21, Groups: [4, 7, 26, 40, 50, 51]
- Group size: 64, Bins: 64×64

## Metrics

| Version | KL | PPL | Acc |
|---------|-----|-----|-----|
| Baseline | 0.0000 | 19.56 | 0.5195 |
| V3 Partial | 0.1150 | 19.74 | 0.5179 |

**Δ PPL**: +0.18  **Δ Acc**: -0.0016  **KL**: 0.1150

## Generation Comparison

### Prompt 1: What is the capital of Japan? Answer with one word.

**Baseline**

1. Tokyo

2. Tokyo

3. Tokyo

4. Tokyo

5. Tokyo

**V3 Partial**

1. Tokyo

2. Tokyo

3. Tokyo

4. Tokyo

5. Tokyo

---

### Prompt 2: A train travels at 60 km/h for 2 hours. How far does it go?

**Baseline**

1. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

2. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

3. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

4. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

5. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

**V3 Partial**

1. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{h} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

2. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{h} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

3. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{h} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

4. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{h} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

5. To find out how far the train goes, you can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given that the speed of the train is 60 km/h and the time it travels is 2 hours, you can calculate the distance as follows:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{h} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

---

### Prompt 3: Write a Python one-liner to reverse a string s.

**Baseline**

1. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

2. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

3. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

4. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

5. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

**V3 Partial**

1. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

2. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

3. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

4. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

5. Here's a Python one-liner to reverse a string `s`:

```python
s[::-1]
```

---

### Prompt 4: Summarize the theory of evolution in one sentence.

**Baseline**

1. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and other mechanisms, leading to the diverse array of life we see today.

2. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and other mechanisms, leading to the diverse array of life we see today.

3. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and other mechanisms, leading to the diverse array of life we see today.

4. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and other mechanisms, leading to the diverse array of life we see today.

5. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and other mechanisms, leading to the diverse array of life we see today.

**V3 Partial**

1. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and environmental adaptation, leading to the diverse array of life forms we see today.

2. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and environmental adaptation, leading to the diverse array of life forms we see today.

3. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and environmental adaptation, leading to the diverse array of life forms we see today.

4. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and environmental adaptation, leading to the diverse array of life forms we see today.

5. The theory of evolution posits that species change over generations through the processes of natural selection, genetic mutation, and environmental adaptation, leading to the diverse array of life forms we see today.

---

### Prompt 5: What is 17 times 6?

**Baseline**

1. 17 times 6 is equal to 102.

2. 17 times 6 is equal to 102.

3. 17 times 6 is equal to 102.

4. 17 times 6 is equal to 102.

5. 17 times 6 is equal to 102.

**V3 Partial**

1. 17 times 6 is equal to 102.

2. 17 times 6 is equal to 102.

3. 17 times 6 is equal to 102.

4. 17 times 6 is equal to 102.

5. 17 times 6 is equal to 102.

---

### Prompt 6: 请用一句话介绍长城。

**Baseline**

1. 长城是中国古代的军事防御工程，也是世界文化遗产之一。

2. 长城是中国古代的军事防御工程，也是世界文化遗产之一。

3. 长城是中国古代的军事防御工程，也是世界文化遗产之一。

4. 长城是中国古代的军事防御工程，也是世界文化遗产之一。

5. 长城是中国古代的军事防御工程，也是世界文化遗产之一。

**V3 Partial**

1. 长城是中国古代的防御工程，也是世界文化遗产之一。

2. 长城是中国古代的防御工程，也是世界文化遗产之一。

3. 长城是中国古代的防御工程，也是世界文化遗产之一。

4. 长城是中国古代的防御工程，也是世界文化遗产之一。

5. 长城是中国古代的防御工程，也是世界文化遗产之一。

---

