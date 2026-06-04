# LLM-LUT v0: Multi-Level Sensitivity Scan for Lookup-Based Approximation

## 1. Objective

The purpose of this experiment is to identify suitable entry points for LUT-based approximation in instruction-tuned large language models.

Unlike the YOLO experiment, where the target layers are relatively clear 1x1 convolution modules, LLMs contain many possible intervention levels, including Transformer blocks, attention heads, MLP projections, MLP intermediate neurons, and residual-stream channel groups. Therefore, before training any LUT replacement module, we first need to run a systematic sensitivity scan.

The goal of this stage is not to prove that LUT can already replace a language-model layer. The goal is to build a candidate map:

```
Which LLM components are low-sensitivity,
compute-relevant,
and structurally suitable for LUT-based approximation?
```

The output of this stage should be a ranked list of candidate structures for the next LUT prefit experiment.

---

## 2. Model Choice

The first experiment should use:

```
Qwen2.5-0.5B-Instruct
```

Reasons:

```
1. It is small enough for fast iteration.
2. It is instruction-tuned, so basic dialogue behavior can be tested.
3. It has a standard decoder-only Transformer architecture.
4. It contains 24 layers, which is enough for early/middle/late layer comparison.
5. If the pipeline works, the same scan can later be repeated on Qwen2.5-1.5B-Instruct.
```

The 0.5B model is suitable for candidate discovery. The 1.5B model can be used later for confirmation.

---

## 3. Core Principle

The LLM-LUT direction should avoid adding new heavy computation.

For this stage, the acceptable future LUT replacement path should be close to:

```
existing activation read
→ scalar / bucket address
→ small LUT lookup
→ vector addition
```

The following operations should be avoided as core replacement mechanisms:

```
extra linear projection
extra low-rank matrix multiplication
extra MLP
large dynamic routing module
full dense recomputation
```

Therefore, this scan should not only measure accuracy sensitivity. It should also estimate whether a candidate component can realistically be replaced using lookup plus simple addition.

---

## 4. Candidate Levels to Scan

### 4.1 MLP Down-Projection Output Groups

Target:

```
model.layers[i].mlp.down_proj output hidden groups
```

Idea:

The output of `down_proj` is divided into hidden-dimension groups. We test whether some output groups are less sensitive than others.

If a selected output group can be approximated by LUT, then the corresponding rows of `down_proj` could potentially be removed.

Potential compute saving:

```
Replacing p% of output groups can theoretically remove roughly p% of down_proj row computation.
```

This is the easiest candidate to implement and should be included in the first scan.

---

### 4.2 MLP Intermediate Contribution Groups

Target:

```
z = act(gate_proj(x)) * up_proj(x)
```

and its contribution through:

```
down_proj(z)
```

Idea:

The MLP intermediate dimension is divided into groups. Each group contributes to the final MLP output through the corresponding columns of `down_proj`.

For a group `g`:

```
z_g = intermediate activation group
contribution_g = W_down[:, g] z_g
```

Instead of only replacing `z_g`, a more meaningful LUT target is:

```
contribution_g ≈ LUT(address)
```

If successful, this could remove a coupled structure:

```
part of gate_proj output
part of up_proj output
corresponding down_proj input columns
```

This candidate is more difficult than down-projection output groups, but it may provide larger compute savings.

---

### 4.3 Attention Head Outputs

Target:

```
selected attention head output
```

Idea:

Attention heads are natural structured units. Some heads may be less important, especially in certain layers. We should test whether selected head outputs can be perturbed or approximated without causing severe model degradation.

However, attention is context-dependent, so this candidate is likely harder to approximate with simple LUTs than MLP components.

This scan should include attention heads, but they should not be the first LUT replacement target unless sensitivity is clearly low.

---

### 4.4 MLP Residual Delta Channel Groups

Target:

```
mlp_delta = MLP(x)
x_next = x + mlp_delta
```

Idea:

Instead of replacing internal MLP projections, we can test the sensitivity of different channel groups in the MLP residual output.

This provides a clean interface:

```
input: block hidden state
output: MLP residual delta
```

However, replacing the entire MLP delta is likely too hard. Therefore, only partial channel groups should be tested in v0.

---

### 4.5 Block-Level MLP / Attention Sublayer

Target:

```
entire MLP sublayer
entire attention sublayer
```

Idea:

This is not intended for immediate LUT replacement. Instead, it provides diagnostic information about which blocks are more sensitive.

For example:

```
block 4 MLP
block 12 MLP
block 20 MLP
block 4 attention
block 12 attention
block 20 attention
```

This helps determine whether early, middle, or late layers are more suitable for fine-grained LUT experiments.

---

## 5. Layer Sampling Strategy

Since Qwen2.5-0.5B has 24 layers, the first scan does not need to test all layers exhaustively.

Use a depth-based sampling strategy:

```
early layers:   3, 6
middle layers:  10, 12, 14
late layers:    18, 21
```

Minimum version:

```
layers = [6, 12, 18]
```

Recommended first full version:

```
layers = [3, 6, 10, 12, 14, 18, 21]
```

The minimum version is enough to validate the pipeline. The full version gives a better sensitivity map.

---

## 6. Grouping Strategy

For channel-level scans, divide hidden or intermediate dimensions into groups.

Suggested group sizes:

```
hidden output groups:
group_size = 32 or 64

MLP intermediate groups:
group_size = 64 or 128

attention head groups:
natural head granularity
```

Start with larger groups to reduce scan cost:

```
hidden group_size = 64
intermediate group_size = 128
```

If the results show interesting low-sensitivity regions, rerun a finer scan:

```
hidden group_size = 32
intermediate group_size = 64
```

---

## 7. Perturbation Methods

The v0 scan should not train LUTs yet. Instead, use cheap perturbations to estimate sensitivity.

### 7.1 Zero Ablation

Replace the selected component output with zero.

Example:

```
selected_group = 0
```

Purpose:

```
Measures how much the model depends on this component.
```

This is the strongest and roughest perturbation.

---

### 7.2 Mean Replacement

Replace the selected component output with a calibration-set mean.

Example:

```
selected_group = mean_selected_group
```

Purpose:

```
Tests whether the component mainly contributes a stable bias-like signal.
```

If mean replacement performs much better than zero ablation, the component may be predictable and suitable for lightweight approximation.

---

### 7.3 Noise Perturbation

Add Gaussian noise to the selected component.

Example:

```
selected_group = selected_group + sigma * std_group * noise
```

Suggested sigma values:

```
sigma = 0.05
sigma = 0.10
sigma = 0.20
```

Purpose:

```
Tests robustness to approximation error.
```

If a group tolerates noise well, it may tolerate LUT approximation error.

---

### 7.4 Bucket / LUT-Like Replacement

This is the most important pre-LUT test.

Procedure:

```
1. Choose one existing scalar activation as address.
2. Quantize the scalar into bins.
3. For each bin, compute the average target output group on calibration data.
4. During evaluation, replace the target group with the bin average.
```

This approximates a non-trained LUT:

```
address → bin → average output group
```

Purpose:

```
Tests whether the component is LUT-addressable.
```

If bucket replacement performs clearly better than mean replacement, then the selected component has activation-dependent structure that a LUT may capture.

---

## 8. Metrics

Each perturbation should be evaluated at two levels.

### 8.1 Local / Layer-Level Metrics

When teacher outputs are available, record:

```
MSE between original component output and perturbed/replaced output
cosine similarity
relative error reduction compared with zero ablation
```

For bucket replacement:

```
bucket MSE
bucket coverage
number of empty bins
per-bin variance
```

These metrics help determine whether the component has stable LUT-addressable structure.

---

### 8.2 Model-Level Metrics

For the full model, record:

```
logits KL divergence between original model and perturbed model
perplexity increase on a fixed text set
next-token accuracy change if available
generation collapse rate
repetition rate
format-following sanity check
```

The most important fast metric is:

```
teacher-student logits KL
```

Perplexity can be measured after the pipeline is stable.

Dialogue sanity checks should use a small fixed prompt set.

---

## 9. Evaluation Data

Use two small datasets.

### 9.1 Calibration Data

Purpose:

```
collect activation statistics
compute means
build bucket tables
```

Suggested size:

```
512 to 2,000 short sequences
```

The calibration data should include:

```
general English text
general Chinese text
instruction-style prompts
short reasoning prompts
```

### 9.2 Evaluation Data

Purpose:

```
measure sensitivity after perturbation
```

Suggested size:

```
128 to 512 sequences for fast scan
1,000+ sequences for confirmation
```

Use fixed seeds and fixed sequence length.

Recommended sequence length for v0:

```
512 tokens
```

Long-context evaluation should be postponed.

---

## 10. Practical Implementation Plan

### Step 1: Load Model and Tokenizer

Use Hugging Face Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

For the first scan, disable gradient computation:

```python
torch.no_grad()
```

---

### Step 2: Build a Small Prompt/Text Set

Prepare two files:

```
calib.jsonl
eval.jsonl
```

Each line:

```json
{"text": "..."}
```

For instruction prompts, apply the chat template before tokenization.

Keep the first version small. The goal is fast iteration, not final benchmark quality.

---

### Step 3: Register Hooks

Use hooks to capture or modify module outputs.

Important hook points:

```
model.model.layers[i].mlp
model.model.layers[i].mlp.down_proj
model.model.layers[i].self_attn
specific attention head output if accessible
```

For simple v0 implementation, start with module-output hooks:

```
down_proj output
MLP output
attention output
```

Head-level hooks may require modifying the attention forward function or wrapping the attention module, so they can be second priority.

---

### Step 4: Collect Baseline Outputs

For each evaluation batch, run the original model and record:

```
logits
selected layer outputs
selected component outputs
```

Avoid storing all activations for the entire dataset if memory is limited.

Instead:

```
process batch by batch
compute metrics online
save only summary statistics
```

---

### Step 5: Run Perturbation Scan

For each candidate:

```
for layer_id in selected_layers:
    for candidate_type in candidate_types:
        for group_id in groups:
            apply perturbation
            run evaluation batch
            compute metrics
            remove perturbation
```

Suggested first candidate types:

```
down_proj output groups
MLP residual delta groups
attention output groups
```

After the pipeline works, add:

```
MLP intermediate contribution groups
attention head outputs
```

---

### Step 6: Build Bucket Replacement Tables

For a candidate group:

```
address = selected existing scalar activation
bin_id = quantize(address)
target = original output group
table[bin_id] = mean(target values in this bin)
```

Recommended first version:

```
num_bins = 64
```

Then test:

```
num_bins = 128
num_bins = 256
```

Address choices:

```
1. first channel in the group
2. channel with highest activation variance
3. channel with highest correlation to group norm
```

Avoid learned address projection in v0.

---

### Step 7: Rank Candidates

For each candidate, compute:

```
sensitivity_score
compute_saving_score
addressability_score
```

Suggested ranking formula:

```
final_score =
    compute_saving_score
  + addressability_score
  - sensitivity_penalty
```

A candidate is good if:

```
1. perturbation causes small logits KL / PPL increase
2. bucket replacement is much better than mean replacement
3. replacing the structure would remove real dense computation
4. the address can be obtained without extra heavy computation
```

---

## 11. Candidate Result Table

Use the following table format:

| Layer | Candidate Type | Group | Perturbation | Local MSE | Logits KL | PPL Δ | Estimated MAC Saving | Addressability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | down output group | 3 | zero |  |  |  |  |  |  |
| 6 | down output group | 3 | mean |  |  |  |  |  |  |
| 6 | down output group | 3 | bucket |  |  |  |  |  |  |
| 12 | MLP delta group | 5 | zero |  |  |  |  |  |  |
| 18 | attention output group | 2 | noise |  |  |  |  |  |  |

The key comparison is:

```
zero vs mean vs bucket
```

If bucket replacement significantly improves over mean replacement, the candidate is likely suitable for LUT-based approximation.

---

## 12. First-Round Scope

To keep the first run manageable, use this minimum scope:

```
Model:
Qwen2.5-0.5B-Instruct

Layers:
[6, 12, 18]

Candidate types:
1. down_proj output groups
2. MLP residual delta groups
3. attention output groups

Group size:
64 for hidden groups

Perturbations:
zero
mean
bucket

Evaluation:
logits KL
local MSE
small prompt sanity check
```

Do not include full PPL in the very first debug run. Add PPL after the hook and perturbation system is stable.

---

## 13. Second-Round Scope

After the first run works, expand to:

```
Layers:
[3, 6, 10, 12, 14, 18, 21]

Candidate types:
1. down_proj output groups
2. MLP residual delta groups
3. MLP intermediate contribution groups
4. attention output groups
5. attention heads

Perturbations:
zero
mean
noise
bucket

Evaluation:
local MSE
cosine similarity
logits KL
PPL increase
generation sanity
```

Then rank the top candidates and select 1–2 for actual LUT prefit.

---

## 14. Success Criteria for v0

This scan is successful if it produces:

```
1. A working hook-based sensitivity scan pipeline.
2. A table of candidate sensitivity across layer depth and structure type.
3. Evidence that at least one structure is both low-sensitivity and LUT-addressable.
4. A clear next target for LUT prefit.
```

A candidate is promising if:

```
1. zero ablation is not catastrophic,
2. mean replacement is better than zero,
3. bucket replacement is better than mean,
4. logits KL remains controlled,
5. estimated compute saving is non-trivial.
```

---

## 15. Expected Next Step

After v0, the next stage is:

```
LLM-LUT v1: Layer-wise LUT Prefit for Top Candidate Structures
```

In v1, we will train actual LUT tables instead of using bucket averages.

The v1 workflow will follow:

```
select top candidate
capture teacher input/output
train LUT table to approximate target component
replace selected component with LUT module
evaluate local recovery and model-level degradation
expand from single group to multiple groups
```

Only after v1 shows a positive signal should we discuss multi-layer LUT replacement.

# LLM-LUT v0: Multi-Level Sensitivity Scan for Lookup-Based Approximation

## 1. Objective

The purpose of this experiment is to identify suitable entry points for LUT-based approximation in instruction-tuned large language models.

Unlike the YOLO experiment, where the target layers are relatively clear 1x1 convolution modules, LLMs contain many possible intervention levels, including Transformer blocks, attention heads, MLP projections, MLP intermediate neurons, and residual-stream channel groups. Therefore, before training any LUT replacement module, we first need to run a systematic sensitivity scan.

The goal of this stage is not to prove that LUT can already replace a language-model layer. The goal is to build a candidate map:

```
Which LLM components are low-sensitivity,
compute-relevant,
and structurally suitable for LUT-based approximation?
```

The output of this stage should be a ranked list of candidate structures for the next LUT prefit experiment.

---

## 2. Model Choice

The first experiment should use:

```
Qwen2.5-0.5B-Instruct
```

Reasons:

```
1. It is small enough for fast iteration.
2. It is instruction-tuned, so basic dialogue behavior can be tested.
3. It has a standard decoder-only Transformer architecture.
4. It contains 24 layers, which is enough for early/middle/late layer comparison.
5. If the pipeline works, the same scan can later be repeated on Qwen2.5-1.5B-Instruct.
```

The 0.5B model is suitable for candidate discovery. The 1.5B model can be used later for confirmation.

---

## 3. Core Principle

The LLM-LUT direction should avoid adding new heavy computation.

For this stage, the acceptable future LUT replacement path should be close to:

```
existing activation read
→ scalar / bucket address
→ small LUT lookup
→ vector addition
```

The following operations should be avoided as core replacement mechanisms:

```
extra linear projection
extra low-rank matrix multiplication
extra MLP
large dynamic routing module
full dense recomputation
```

Therefore, this scan should not only measure accuracy sensitivity. It should also estimate whether a candidate component can realistically be replaced using lookup plus simple addition.

---

## 4. Candidate Levels to Scan

### 4.1 MLP Down-Projection Output Groups

Target:

```
model.layers[i].mlp.down_proj output hidden groups
```

Idea:

The output of `down_proj` is divided into hidden-dimension groups. We test whether some output groups are less sensitive than others.

If a selected output group can be approximated by LUT, then the corresponding rows of `down_proj` could potentially be removed.

Potential compute saving:

```
Replacing p% of output groups can theoretically remove roughly p% of down_proj row computation.
```

This is the easiest candidate to implement and should be included in the first scan.

---

### 4.2 MLP Intermediate Contribution Groups

Target:

```
z = act(gate_proj(x)) * up_proj(x)
```

and its contribution through:

```
down_proj(z)
```

Idea:

The MLP intermediate dimension is divided into groups. Each group contributes to the final MLP output through the corresponding columns of `down_proj`.

For a group `g`:

```
z_g = intermediate activation group
contribution_g = W_down[:, g] z_g
```

Instead of only replacing `z_g`, a more meaningful LUT target is:

```
contribution_g ≈ LUT(address)
```

If successful, this could remove a coupled structure:

```
part of gate_proj output
part of up_proj output
corresponding down_proj input columns
```

This candidate is more difficult than down-projection output groups, but it may provide larger compute savings.

---

### 4.3 Attention Head Outputs

Target:

```
selected attention head output
```

Idea:

Attention heads are natural structured units. Some heads may be less important, especially in certain layers. We should test whether selected head outputs can be perturbed or approximated without causing severe model degradation.

However, attention is context-dependent, so this candidate is likely harder to approximate with simple LUTs than MLP components.

This scan should include attention heads, but they should not be the first LUT replacement target unless sensitivity is clearly low.

---

### 4.4 MLP Residual Delta Channel Groups

Target:

```
mlp_delta = MLP(x)
x_next = x + mlp_delta
```

Idea:

Instead of replacing internal MLP projections, we can test the sensitivity of different channel groups in the MLP residual output.

This provides a clean interface:

```
input: block hidden state
output: MLP residual delta
```

However, replacing the entire MLP delta is likely too hard. Therefore, only partial channel groups should be tested in v0.

---

### 4.5 Block-Level MLP / Attention Sublayer

Target:

```
entire MLP sublayer
entire attention sublayer
```

Idea:

This is not intended for immediate LUT replacement. Instead, it provides diagnostic information about which blocks are more sensitive.

For example:

```
block 4 MLP
block 12 MLP
block 20 MLP
block 4 attention
block 12 attention
block 20 attention
```

This helps determine whether early, middle, or late layers are more suitable for fine-grained LUT experiments.

---

## 5. Layer Sampling Strategy

Since Qwen2.5-0.5B has 24 layers, the first scan does not need to test all layers exhaustively.

Use a depth-based sampling strategy:

```
early layers:   3, 6
middle layers:  10, 12, 14
late layers:    18, 21
```

Minimum version:

```
layers = [6, 12, 18]
```

Recommended first full version:

```
layers = [3, 6, 10, 12, 14, 18, 21]
```

The minimum version is enough to validate the pipeline. The full version gives a better sensitivity map.

---

## 6. Grouping Strategy

For channel-level scans, divide hidden or intermediate dimensions into groups.

Suggested group sizes:

```
hidden output groups:
group_size = 32 or 64

MLP intermediate groups:
group_size = 64 or 128

attention head groups:
natural head granularity
```

Start with larger groups to reduce scan cost:

```
hidden group_size = 64
intermediate group_size = 128
```

If the results show interesting low-sensitivity regions, rerun a finer scan:

```
hidden group_size = 32
intermediate group_size = 64
```

---

## 7. Perturbation Methods

The v0 scan should not train LUTs yet. Instead, use cheap perturbations to estimate sensitivity.

### 7.1 Zero Ablation

Replace the selected component output with zero.

Example:

```
selected_group = 0
```

Purpose:

```
Measures how much the model depends on this component.
```

This is the strongest and roughest perturbation.

---

### 7.2 Mean Replacement

Replace the selected component output with a calibration-set mean.

Example:

```
selected_group = mean_selected_group
```

Purpose:

```
Tests whether the component mainly contributes a stable bias-like signal.
```

If mean replacement performs much better than zero ablation, the component may be predictable and suitable for lightweight approximation.

---

### 7.3 Noise Perturbation

Add Gaussian noise to the selected component.

Example:

```
selected_group = selected_group + sigma * std_group * noise
```

Suggested sigma values:

```
sigma = 0.05
sigma = 0.10
sigma = 0.20
```

Purpose:

```
Tests robustness to approximation error.
```

If a group tolerates noise well, it may tolerate LUT approximation error.

---

### 7.4 Bucket / LUT-Like Replacement

This is the most important pre-LUT test.

Procedure:

```
1. Choose one existing scalar activation as address.
2. Quantize the scalar into bins.
3. For each bin, compute the average target output group on calibration data.
4. During evaluation, replace the target group with the bin average.
```

This approximates a non-trained LUT:

```
address → bin → average output group
```

Purpose:

```
Tests whether the component is LUT-addressable.
```

If bucket replacement performs clearly better than mean replacement, then the selected component has activation-dependent structure that a LUT may capture.

---

## 8. Metrics

Each perturbation should be evaluated at two levels.

### 8.1 Local / Layer-Level Metrics

When teacher outputs are available, record:

```
MSE between original component output and perturbed/replaced output
cosine similarity
relative error reduction compared with zero ablation
```

For bucket replacement:

```
bucket MSE
bucket coverage
number of empty bins
per-bin variance
```

These metrics help determine whether the component has stable LUT-addressable structure.

---

### 8.2 Model-Level Metrics

For the full model, record:

```
logits KL divergence between original model and perturbed model
perplexity increase on a fixed text set
next-token accuracy change if available
generation collapse rate
repetition rate
format-following sanity check
```

The most important fast metric is:

```
teacher-student logits KL
```

Perplexity can be measured after the pipeline is stable.

Dialogue sanity checks should use a small fixed prompt set.

---

## 9. Evaluation Data

Use two small datasets.

### 9.1 Calibration Data

Purpose:

```
collect activation statistics
compute means
build bucket tables
```

Suggested size:

```
512 to 2,000 short sequences
```

The calibration data should include:

```
general English text
general Chinese text
instruction-style prompts
short reasoning prompts
```

### 9.2 Evaluation Data

Purpose:

```
measure sensitivity after perturbation
```

Suggested size:

```
128 to 512 sequences for fast scan
1,000+ sequences for confirmation
```

Use fixed seeds and fixed sequence length.

Recommended sequence length for v0:

```
512 tokens
```

Long-context evaluation should be postponed.

---

## 10. Practical Implementation Plan

### Step 1: Load Model and Tokenizer

Use Hugging Face Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

For the first scan, disable gradient computation:

```python
torch.no_grad()
```

---

### Step 2: Build a Small Prompt/Text Set

Prepare two files:

```
calib.jsonl
eval.jsonl
```

Each line:

```json
{"text": "..."}
```

For instruction prompts, apply the chat template before tokenization.

Keep the first version small. The goal is fast iteration, not final benchmark quality.

---

### Step 3: Register Hooks

Use hooks to capture or modify module outputs.

Important hook points:

```
model.model.layers[i].mlp
model.model.layers[i].mlp.down_proj
model.model.layers[i].self_attn
specific attention head output if accessible
```

For simple v0 implementation, start with module-output hooks:

```
down_proj output
MLP output
attention output
```

Head-level hooks may require modifying the attention forward function or wrapping the attention module, so they can be second priority.

---

### Step 4: Collect Baseline Outputs

For each evaluation batch, run the original model and record:

```
logits
selected layer outputs
selected component outputs
```

Avoid storing all activations for the entire dataset if memory is limited.

Instead:

```
process batch by batch
compute metrics online
save only summary statistics
```

---

### Step 5: Run Perturbation Scan

For each candidate:

```
for layer_id in selected_layers:
    for candidate_type in candidate_types:
        for group_id in groups:
            apply perturbation
            run evaluation batch
            compute metrics
            remove perturbation
```

Suggested first candidate types:

```
down_proj output groups
MLP residual delta groups
attention output groups
```

After the pipeline works, add:

```
MLP intermediate contribution groups
attention head outputs
```

---

### Step 6: Build Bucket Replacement Tables

For a candidate group:

```
address = selected existing scalar activation
bin_id = quantize(address)
target = original output group
table[bin_id] = mean(target values in this bin)
```

Recommended first version:

```
num_bins = 64
```

Then test:

```
num_bins = 128
num_bins = 256
```

Address choices:

```
1. first channel in the group
2. channel with highest activation variance
3. channel with highest correlation to group norm
```

Avoid learned address projection in v0.

---

### Step 7: Rank Candidates

For each candidate, compute:

```
sensitivity_score
compute_saving_score
addressability_score
```

Suggested ranking formula:

```
final_score =
    compute_saving_score
  + addressability_score
  - sensitivity_penalty
```

A candidate is good if:

```
1. perturbation causes small logits KL / PPL increase
2. bucket replacement is much better than mean replacement
3. replacing the structure would remove real dense computation
4. the address can be obtained without extra heavy computation
```

---

## 11. Candidate Result Table

Use the following table format:

| Layer | Candidate Type | Group | Perturbation | Local MSE | Logits KL | PPL Δ | Estimated MAC Saving | Addressability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | down output group | 3 | zero |  |  |  |  |  |  |
| 6 | down output group | 3 | mean |  |  |  |  |  |  |
| 6 | down output group | 3 | bucket |  |  |  |  |  |  |
| 12 | MLP delta group | 5 | zero |  |  |  |  |  |  |
| 18 | attention output group | 2 | noise |  |  |  |  |  |  |

The key comparison is:

```
zero vs mean vs bucket
```

If bucket replacement significantly improves over mean replacement, the candidate is likely suitable for LUT-based approximation.

---

## 12. First-Round Scope

To keep the first run manageable, use this minimum scope:

```
Model:
Qwen2.5-0.5B-Instruct

Layers:
[6, 12, 18]

Candidate types:
1. down_proj output groups
2. MLP residual delta groups
3. attention output groups

Group size:
64 for hidden groups

Perturbations:
zero
mean
bucket

Evaluation:
logits KL
local MSE
small prompt sanity check
```

Do not include full PPL in the very first debug run. Add PPL after the hook and perturbation system is stable.

---

## 13. Second-Round Scope

After the first run works, expand to:

```
Layers:
[3, 6, 10, 12, 14, 18, 21]

Candidate types:
1. down_proj output groups
2. MLP residual delta groups
3. MLP intermediate contribution groups
4. attention output groups
5. attention heads

Perturbations:
zero
mean
noise
bucket

Evaluation:
local MSE
cosine similarity
logits KL
PPL increase
generation sanity
```

Then rank the top candidates and select 1–2 for actual LUT prefit.

---

## 14. Success Criteria for v0

This scan is successful if it produces:

```
1. A working hook-based sensitivity scan pipeline.
2. A table of candidate sensitivity across layer depth and structure type.
3. Evidence that at least one structure is both low-sensitivity and LUT-addressable.
4. A clear next target for LUT prefit.
```

A candidate is promising if:

```
1. zero ablation is not catastrophic,
2. mean replacement is better than zero,
3. bucket replacement is better than mean,
4. logits KL remains controlled,
5. estimated compute saving is non-trivial.
```

---

## 15. Expected Next Step

After v0, the next stage is:

```
LLM-LUT v1: Layer-wise LUT Prefit for Top Candidate Structures
```

In v1, we will train actual LUT tables instead of using bucket averages.

The v1 workflow will follow:

```
select top candidate
capture teacher input/output
train LUT table to approximate target component
replace selected component with LUT module
evaluate local recovery and model-level degradation
expand from single group to multiple groups
```

Only after v1 shows a positive signal should we discuss multi-layer LUT replacement.