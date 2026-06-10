# Same-Layer Multi-Group: Qwen/Qwen2.5-3B-Instruct

- Layer: 27, Groups: [29, 4, 15]
- Group size: 64, Bins: 64×64

## Metrics

| Version | KL | PPL | Acc |
|---------|-----|-----|-----|
| Original | 0.0000 | 30.75 | 0.4945 |
| Single G29 | 0.0510 | 30.51 | 0.4899 |
| Multi [29, 4, 15] | 0.1033 | 31.09 | 0.4945 |

## Generation Samples

### Prompt 1: What is the capital of Japan? Answer with one word.

**Original (no hook)**

1. Tokyo

2. Tokyo

3. Tokyo

4. Tokyo

5. Tokyo

6. Tokyo

7. Tokyo

8. Tokyo

9. Tokyo

10. Tokyo

**Single-group L27 G29**

1. Tokyo

2. Tokyo

3. Tokyo

4. Tokyo

5. Tokyo

6. Tokyo

7. Tokyo

8. Tokyo

9. Tokyo

10. Tokyo

**Multi-group L27 [29, 4, 15]**

1. Tokyo

2. Tokyo

3. Tokyo

4. Tokyo

5. Tokyo

6. Tokyo

7. Tokyo

8. Tokyo

9. Tokyo

10. Tokyo

---

### Prompt 2: A train travels at 60 km/h for 2 hours. How far does it go?

**Original (no hook)**

1. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

2. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

3. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

4. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

5. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

6. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

7. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

8. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

9. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

10. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

**Single-group L27 G29**

1. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

2. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

3. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

4. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

5. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

6. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

7. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

8. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

9. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

10. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

**Multi-group L27 [29, 4, 15]**

1. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

2. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

3. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

4. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

5. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

6. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

7. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

8. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

9. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

10. To find out how far the train goes, we can use the formula:

\[ \text{Distance} = \text{Speed} \times \text{Time} \]

Given:
- Speed = 60 km/h
- Time = 2 hours

Plugging in these values:

\[ \text{Distance} = 60 \, \text{km/h} \times 2 \, \text{hours} = 120 \, \text{km} \]

So, the train travels 120 kilometers.

---

### Prompt 3: Write a Python one-liner to reverse a string s.

**Original (no hook)**

1. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

2. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

3. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

4. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

5. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

6. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

7. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

8. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

9. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

10. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

**Single-group L27 G29**

1. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

2. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

3. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

4. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

5. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

6. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

7. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

8. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

9. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

10. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

**Multi-group L27 [29, 4, 15]**

1. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

2. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

3. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

4. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

5. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

6. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

7. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

8. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

9. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

10. Certainly! You can reverse a string `s` in Python using slicing:

```python
s[::-1]
```

This is a one-liner that accomplishes the task. Here's how it works: 

- The slice notation `[::-1]` means start at the end of the string and move backwards until you reach the beginning, effectively reversing the string.

---

### Prompt 4: Summarize the theory of evolution in one sentence.

**Original (no hook)**

1. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

2. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

3. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

4. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

5. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

6. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

7. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

8. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

9. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

10. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

**Single-group L27 G29**

1. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

2. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

3. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

4. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

5. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

6. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

7. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

8. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

9. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

10. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

**Multi-group L27 [29, 4, 15]**

1. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

2. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

3. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

4. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

5. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

6. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

7. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

8. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

9. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

10. The theory of evolution posits that species evolve over generations through genetic variation, natural selection, and adaptation to their environments.

---

### Prompt 5: What is 17 times 6?

**Original (no hook)**

1. 17 times 6 equals 102.

2. 17 times 6 equals 102.

3. 17 times 6 equals 102.

4. 17 times 6 equals 102.

5. 17 times 6 equals 102.

6. 17 times 6 equals 102.

7. 17 times 6 equals 102.

8. 17 times 6 equals 102.

9. 17 times 6 equals 102.

10. 17 times 6 equals 102.

**Single-group L27 G29**

1. 17 times 6 equals 102.

2. 17 times 6 equals 102.

3. 17 times 6 equals 102.

4. 17 times 6 equals 102.

5. 17 times 6 equals 102.

6. 17 times 6 equals 102.

7. 17 times 6 equals 102.

8. 17 times 6 equals 102.

9. 17 times 6 equals 102.

10. 17 times 6 equals 102.

**Multi-group L27 [29, 4, 15]**

1. 17 times 6 equals 102.

2. 17 times 6 equals 102.

3. 17 times 6 equals 102.

4. 17 times 6 equals 102.

5. 17 times 6 equals 102.

6. 17 times 6 equals 102.

7. 17 times 6 equals 102.

8. 17 times 6 equals 102.

9. 17 times 6 equals 102.

10. 17 times 6 equals 102.

---

### Prompt 6: 请用一句话介绍长城。

**Original (no hook)**

1. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

2. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

3. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

4. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

5. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

6. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

7. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

8. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

9. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

10. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

**Single-group L27 G29**

1. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

2. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

3. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

4. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

5. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

6. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

7. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

8. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

9. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

10. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

**Multi-group L27 [29, 4, 15]**

1. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

2. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

3. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

4. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

5. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

6. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

7. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

8. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

9. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

10. 长城，是中国古代伟大的军事防御工程，也是世界文化遗产之一，全长超过2万公里。

---

