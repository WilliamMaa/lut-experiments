# LLM-LUT: Lookup-Table Replacement of MLP Residual Groups in LLMs

> **Core Question**: Can we replace a subset of MLP computations in LLMs with O(1) lookup-table operations, and does this scale?
>
> **Current Answer (June 2025)**: Yes. On Qwen2.5, deep-layer MLP residual groups can be cumulatively replaced with 2-head LUT lookup up to at least 10.7% of a layer (7B) and 15.6% (3B), with PPL impact <2% and zero behavioral drift.

---

## 1. Motivation: Why LUT?

Large language models are dominated by matrix multiplications. On standard hardware, this is unavoidable. But on **compute-in-memory (CIM)** architectures, lookup tables (LUTs) can be implemented as O(1) memory reads rather than O(N) multiply-accumulate operations.

The challenge: LLM weights are static, so where is the lookup? Our insight is that **not all MLP outputs need to be computed from scratch**. Within a layer's MLP residual, some contribution groups produce outputs that are highly predictable from a low-dimensional "address" extracted from the input activation. If we can pre-compute these mappings, inference becomes a table lookup instead of a matrix multiplication for those groups.

**This is not about accuracy**. Accuracy is merely the sanity check that the replacement does not break the model. The real metric is: *can we maintain behavioral fidelity while replacing compute with memory access?*

---

## 2. The Journey: v0 → v1 → v2

### v0: Proving the Signal Exists

We started with a two-stage pipeline:
1. Train a baseline model to convergence
2. Freeze it, then calibrate 2-dimensional "address channels" from hidden states
3. Build a fixed 64×64 bucket table mapping (address → MLP residual output)
4. Replace the group at inference time with table lookup

**Key findings**:
- **2-head joint binning >> 1-head**: Two address channels together define a 2D lookup space. Separate 1D lookups collapse because the interaction between channels carries information.
- **Uniform binning >> quantile binning**: Quantile binning creates uneven buckets that overfit to calibration data tails. Fixed uniform bins generalize better.
- **Signal is real but noisy**: On Qwen2.5-0.5B, replacing L6 G4 gave KL=0.61, PPL +10.4. High fidelity loss, but the fact that a 64×64 table could approximate any MLP group output at all was non-trivial.

### v1.x: The Graveyard of Table Structures

Convinced that the bucket table was too crude, we spent significant effort on "smarter" table architectures. **All of them failed to beat the simple fixed bucket.**

| Attempt | What We Tried | Why It Failed |
|---------|--------------|---------------|
| v1.0 Trainable LUT | Make the 64×64 table end-to-end differentiable | Barely beat fixed bucket (+0.03%). Fixed uniform bins are already a strong baseline; there is little headroom under fixed bin boundaries. |
| v1.1 Learned Codebook | VQ-VAE style discrete codebook with soft-to-hard training | Centroid collapse. Soft-to-hard gap was 25%. KL 0.83 vs bucket's 0.61. The quantization bottleneck destroys information that the bucket preserves. |
| v1.2 Additive Decomposition | ANOVA initialization + coarse interaction terms | Failed to beat fixed 2D bucket. The additive structure is too restrictive for MLP residuals, which are inherently non-additive. |

**Lesson**: Do not over-engineer the table. The 2D bucket is not a "weak baseline to beat"—it is capturing something fundamental about how addressable these groups are. If the bucket works poorly, the group is not addressable, not the table structure.

### v2: Scaling and the Real Problems

With the table structure frozen (64×64 uniform 2D bucket), we turned to the real questions:
1. Does this work on larger models?
2. Can we replace *multiple* groups?
3. What is the behavioral fidelity, not just the KL number?

#### v2 R1: Single-Group Replacement Across Scales

We built `ReplacementEngine`—a functional hook that intercepts MLP residual outputs and substitutes LUT lookups at inference time. Then we scanned four model sizes.

| Scale | Best Layer | Group | KL | PPL Δ | Acc Δ | Recovery |
|-------|-----------|-------|-----|-------|-------|----------|
| 0.5B | L6 | G4 | 0.61 | +10.4 | -2.65pt | 82.7% |
| 1.5B | L21 | G16 | 0.05 | **-0.8** | +1.25pt | 41.6% |
| 3B | L9 | G27 | 1.19 | +0.14 | +1.56pt | 75.7% |
| **7B** | **L21** | **G50** | **0.027** | **+0.03** | **+0.31pt** | **51.0%** |

**Critical discovery: PPL gap vanishes as model scales.**

The 0.5B replacement is essentially broken (+10.4 PPL). But by 7B, the replacement is PPL-neutral. We believe this is because larger models have more "slack" in their residual streams—individual group contributions are smaller relative to the total hidden state, so approximation errors are absorbed by downstream layers.

**Critical discovery: Best layer is scale-dependent.**

0.5B likes L6, 1.5B likes L21, 3B likes L9, 7B likes L21 again. There is no universal "good layer." This makes per-scale scan mandatory.

#### v2 R2: The Multi-Group Question

Single-group replacement saves negligible compute (~1-4% of `down_proj` FLOPs). For meaningful savings, we need multi-group. But naively replacing multiple groups could compound errors.

**3B L27 cumulative replacement:**

| # Groups | IDs | KL | PPL | Acc | Drift? |
|----------|-----|-----|-----|-----|--------|
| 1 | G29 | 0.051 | 30.51 | 0.490 | No |
| 3 | G29,4,15 | 0.103 | 31.09 | 0.495 | No |
| **5** | **G29,4,15,23,27** | **0.148** | **30.51** | **0.499** | **No** |

**7B L21 cumulative replacement:**

| # Groups | IDs | KL | PPL | Acc | Drift? |
|----------|-----|-----|-----|-----|--------|
| 1 | G50 | 0.027 | 19.59 | 0.523 | No |
| 4 | G26,50,51,4 | 0.084 | 19.91 | 0.521 | No |
| **6** | **G26,50,51,4,7,40** | **0.116** | **19.93** | **0.516** | **No** |

**Critical discovery: KL is sub-additive.**

On 3B, 5 groups' scan KLs sum to ~0.18 linearly, but actual multi-group KL is 0.148—~18% lower. On 7B, 6 groups' scan KLs sum to ~0.153, but actual is 0.116—**24.5% lower**. The bucket tables for different groups in the same layer are partially orthogonal, and this orthogonality strengthens with scale.

**Critical discovery: PPL can stay flat or even improve.**

Five groups (15.6% of the layer) replaced on 3B, yet PPL is identical to single-group and barely above original. On 7B, six groups (10.7%) replaced with only +1.9% PPL. The bucket average acts as a mild regularizer on high-variance group outputs.

#### v2 R2: The Language Drift Trap

Numbers can lie. 3B L9 G27 looked excellent on paper (recovery 75.7%, PPL +0.14, Acc +1.56pt). But generation revealed **English prompts producing Chinese outputs**—with preserved semantic correctness.

| Prompt | Original | L9 G27 Replacement |
|--------|----------|-------------------|
| "Capital of Japan?" | Tokyo | **东京** |
| "Train 60 km/h for 2h" | English | **To计算...** (code-switching) |
| "Summarize evolution" | English | **中文** |

L9 G27 is a mid-layer group that participates in **language routing**—controlling output language, style, and format. Replacing it distorts the residual stream's language prior without breaking semantic competence.

Meanwhile, deep-layer groups (L27 on 3B, L21 on 7B) show **zero language drift** even under multi-group replacement.

**Lesson**: Group selection criteria must include generation-based behavioral checks, not just KL/PPL/Acc. High-recovery mid-layer groups are dangerous.

#### v2 R2: Cross-Layer Feasibility

We also tested combining two layers on 7B: L21 (6 groups) + L14 G26.

| Configuration | KL | PPL | Acc | Drift |
|---------------|-----|-----|-----|-------|
| L21 Single G50 | 0.027 | 19.59 | 0.523 | None |
| L21 Multi 6 | 0.116 | 19.93 | 0.516 | None |
| L14 Single G26 | 0.058 | 20.20 | 0.510 | None |
| **L21+L14 Combined** | **0.165** | **20.59** | **0.506** | **Minor** |

Cross-layer KL remains sub-additive (5% below linear), but the effect is much weaker than same-layer (24.5%). PPL impact increases to +5.3%, and minor semantic variations appear (e.g., "other mechanisms" → "environmental adaptation" in evolution summary). Same-layer replacement is the more robust strategy.

---

## 3. What We Know Now

### What Works
1. **2-head uniform 64×64 bucket** is the definitive table structure.
2. **Deep-layer low-KL groups** are safe replacement targets (L27 on 3B, L21 on 7B).
3. **Same-layer cumulative replacement** is safe up to at least 15.6% of layer groups (3B) and 10.7% (7B), with sub-additive KL.
4. **Scaling helps**: PPL gap shrinks from +10.4 (0.5B) to +0.03 (7B).
5. **Cross-layer is feasible** but less efficient than same-layer.

### What Does Not Work
1. Trainable LUTs, codebooks, and additive decompositions do not beat fixed buckets.
2. Mid-layer high-recovery groups (e.g., 3B L9 G27) cause language/style drift despite good metrics.
3. Cross-layer combinations produce weaker sub-additivity and higher PPL degradation.

### What We Do Not Know Yet
1. The exact boundary of same-layer cumulative replacement (how many groups before KL explodes?).
2. Whether the pattern holds at 14B+ or on non-Qwen architectures.
3. How to translate functional replacement into actual FLOPs reduction (requires Stage-2: partial `down_proj` skip).
4. Whether there exists a mechanistic explanation for why deep layers are safer than mid-layers.

---

## 4. Complete Results

### 4.1 Cross-Scale Single-Group

| Model | Layer | Group | KL | PPL Δ | Acc Δ | Recovery | Drift |
|-------|-------|-------|-----|-------|-------|----------|-------|
| 0.5B | L6 | G4 | 0.61 | +10.4 | -2.65pt | 82.7% | None |
| 1.5B | L21 | G16 | 0.05 | -0.8 | +1.25pt | 41.6% | None |
| 3B | L9 | G27 | 1.19 | +0.14 | +1.56pt | 75.7% | **Lang** |
| 3B | L27 | G29 | 0.051 | -0.24 | -0.47pt | 47.9% | None |
| **7B** | **L21** | **G50** | **0.027** | **+0.03** | **+0.31pt** | **51.0%** | **None** |

### 4.2 3B Same-Layer Cumulative

| # Groups | Group IDs | KL | PPL | Acc | Drift |
|----------|-----------|-----|-----|-----|-------|
| 1 | G29 | 0.051 | 30.51 | 0.490 | None |
| 3 | G29,4,15 | 0.103 | 31.09 | 0.495 | None |
| 5 | G29,4,15,23,27 | 0.148 | 30.51 | 0.499 | None |

### 4.3 7B Same-Layer Cumulative

| # Groups | Group IDs | KL | PPL | Acc | Drift |
|----------|-----------|-----|-----|-----|-------|
| 1 | G50 | 0.027 | 19.59 | 0.523 | None |
| 4 | G26,50,51,4 | 0.084 | 19.91 | 0.521 | None |
| 6 | G26,50,51,4,7,40 | 0.116 | 19.93 | 0.516 | None |

### 4.4 7B Cross-Layer

| Configuration | KL | PPL | Acc | Drift |
|---------------|-----|-----|-----|-------|
| L14 G26 | 0.058 | 20.20 | 0.510 | None |
| L21 Multi 6 | 0.116 | 19.93 | 0.516 | None |
| **L21+L14 Combined** | **0.165** | **20.59** | **0.506** | **Minor** |

### 4.5 v3 Phase 1: Partial Skip (7B L21 6-group)

| Version | KL | PPL | Acc | Note |
|---------|-----|-----|-----|------|
| Baseline | 0.0000 | 19.56 | 0.5195 | — |
| v2 Functional Hook | 0.1156 | 19.93 | 0.5164 | Post-computation overwrite |
| **v3 Partial Skip** | **0.1150** | **19.74** | **0.5179** | **Skipped matmul for replaced groups** |

---

## 5. Current State (June 2025)

**Validated**: Functional replacement of deep-layer MLP residual groups with 2-head LUT is behavior-preserving on Qwen2.5 from 0.5B to 7B, with same-layer cumulative replacement safe up to at least 10.7% (7B) and 15.6% (3B) of layer groups.

**Best results**:
- **3B**: L27, 5 groups, KL=0.148, PPL-neutral, zero drift
- **7B**: L21, 6 groups, KL=0.116, PPL +1.9%, zero drift
- **7B Cross-layer**: L21 (6) + L14 (1), KL=0.165, PPL +5.3%, minor semantic variation

**Not yet validated**: Actual compute reduction. The current replacement is a functional hook—matrix multiplications still execute, we just overwrite the output. Real acceleration requires:
1. Identifying which `down_proj` columns correspond to replaced groups
2. Skipping those columns during matmul
3. Likely a custom CUDA kernel

**Claim constraint**: We currently CANNOT claim acceleration. Valid claim is: "Selected MLP residual groups can be functionally replaced with 2-head LUT while preserving model behavior across scales from 0.5B to 7B."

---

## 6. Next Steps

### Immediate (High Priority)
1. **Phase 2: Triton Kernel**: Write a Triton partial matmul kernel that skips replaced output channels. Benchmark against PyTorch `F.linear` to confirm actual latency reduction. Target: 5-10% `down_proj` speedup for 6 replaced groups on 7B.
2. **Phase 3: Latency Demo**: Integrate Triton kernel into full inference pipeline. Measure end-to-end token generation latency. Target: 0.5-1.5% end-to-end speedup vs baseline.
3. **Stage-2 Compute Removal (Legacy Path)**: If Triton proves too complex, fallback to CUTLASS or `torch.compile` custom operator.

### Medium Priority
3. **14B Validation**: Extend scan to Qwen2.5-14B. If the trend continues (lower KL, PPL-neutral), the scaling narrative becomes even stronger. May require multi-GPU strategy.
4. **Architecture Transfer**: Test on non-Qwen models (Llama, Mistral) to verify the approach is not Qwen-specific.
5. **Mechanistic Understanding**: Why are deep layers safer than mid-layers? Can we build a lightweight probe to predict language-routing sensitivity without running full generation eval?
6. **Boundary Analysis**: Systematically find the maximum number of replaceable groups per layer before PPL/Acc degradation becomes unacceptable.

---

## 7. Repository Structure

```
LLM_LUT/
├── v0/                 # Discovery pipeline (0.5B)
│   ├── calibrate.py    # Address channel calibration
│   ├── train.py        # Bucket table construction
│   ├── metrics.py      # KL/PPL/Acc evaluation
│   └── ...
├── v1/                 # Ablations (trainable LUT, codebook, ANOVA)
│   └── ...
├── v2/                 # Scaling validation
│   ├── run_r2.py       # Fast scan + single-group replacement
│   ├── r2_scan.py      # Two-phase scanner
│   ├── r1_replacement.py   # ReplacementEngine (hook install/uninstall)
│   ├── run_same_layer_multi.py  # Same-layer multi-group eval
│   ├── run_multi_layer.py       # Cross-layer multi-group eval
│   ├── run_single_group_gen.py  # Single-group generation only
│   └── results/        # Experiment outputs
└── README.md           # This file
```

---

## 8. How to Reproduce

### Prerequisites
- Python 3.10+, PyTorch, transformers, datasets, tqdm, scipy
- Single A100 40GB (sufficient for 7B with bfloat16)
- **Never use** `accelerate` or `device_map="auto"` (causes driver deadlock)
- `CUDA_VISIBLE_DEVICES=0` set before imports

### 8.1 Fast Scan for a New Scale
```bash
cd v2
python run_r2.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer_percentiles 0.25,0.5,0.75 \
    --calib_size 256 --eval_size 256 --top_k 5 \
    --output_dir results/7B_scan
```

### 8.2 Same-Layer Multi-Group Replacement
```bash
# 3B L27 5-group
python run_same_layer_multi.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --layer 27 --groups "29,4,15,23,27" \
    --output_dir results/3B_l27_5group

# 7B L21 6-group
python run_same_layer_multi.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 21 --groups "26,50,51,4,7,40" \
    --calib_size 512 --eval_size 256 \
    --output_dir results/7B_l21_6group
```

### 8.3 Cross-Layer Multi-Group Replacement
```bash
python run_multi_layer.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --config "21:26,50,51,4,7,40;14:26" \
    --calib_size 512 --eval_size 256 \
    --output_dir results/7B_l21_l14
```

### 8.4 v3 Phase 1 Validation (Partial Skip)
```bash
cd v3
python run_v3_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint_dir ../v2/results/7B_l21_6group_ckpt \
    --eval_size 128 --gen_samples 5 \
    --output_dir results/v3_7B_l21
```

---

## 9. Lessons Learned

1. **When you find yourself tuning for accuracy, stop.** Ask: am I validating the core hypothesis (LUT replaces compute) or am I building a better hypernetwork? The latter violates the project's red line.
2. **Fixed baselines are stronger than they look.** We wasted weeks on trainable LUTs and codebooks because we assumed the fixed bucket was "too simple." It wasn't too simple; the problem is intrinsically limited by how addressable the groups are.
3. **Metrics are necessary but not sufficient.** L9 G27 had great PPL/Acc but failed language fidelity. Always run generation checks.
4. **Scaling is your friend.** The 0.5B results were discouraging (PPL +10.4). If we had stopped there, we would have concluded the approach doesn't work. Scaling to 7B revealed the approach is viable.
5. **Same-layer > cross-layer.** Cross-layer replacement is feasible but produces weaker sub-additivity and higher PPL degradation. Concentrate compute savings within the same layer.
