# V3 Plan: From Functional Replacement to Real Compute Reduction

> **Goal**: Skip the matrix multiplication for replaced MLP residual groups, replacing it with O(1) LUT lookup. Measure actual latency reduction.

---

## 1. The Gap Between v2 and v3

**v2 achieves**: Functional replacement via forward hooks. The original `gate_proj → up_proj → SwiGLU → down_proj` pipeline still executes in full. We overwrite the output post-computation.

**v3 targets**: Partial `down_proj` skip. For replaced groups, we bypass the matmul entirely and inject the LUT output directly.

**Why this is hard**:
- LUT address must be extracted from `normed_x` (MLP input), but `down_proj` only sees `SwiGLU(normed_x)`.
- LUT stores `delta = mlp(normed_x) - normed_x`, but `down_proj` produces `mlp(normed_x)`. We need `normed_x_group + lut_delta`.
- GPU tensor cores are optimized for large dense matmuls; partial output skipping requires custom kernel work to realize actual speedup.

---

## 2. Compute Savings Analysis

### 2.1 Where is the saving?

Qwen2.5 MLP per layer:
```
normed_x = layernorm(hidden)
gate = gate_proj(normed_x)      # [B,S] x [hidden, intermediate]
up   = up_proj(normed_x)        # [B,S] x [hidden, intermediate]
swiglu = silu(gate) * up        # elementwise
down = down_proj(swiglu)        # [B,S] x [intermediate, hidden]
out = hidden + down             # residual
```

Only `down_proj` can be partially skipped. `gate_proj` and `up_proj` still run because SwiGLU output is needed for the active (non-replaced) channels.

### 2.2 Theoretical FLOPs reduction

For 7B (hidden=3584, intermediate≈14336):
- `down_proj` FLOPs: `B × S × intermediate × hidden`
- Replaced 6 groups = 6 × 64 = 384 channels
- Skipped FLOPs: `384/3584 = 10.7%` of `down_proj`
- `down_proj` is ~1/3 of total MLP FLOPs
- **MLP-level saving: ~3.6%**
- **Layer-level saving: ~1.5%** (attention dominates)
- **End-to-end saving: ~1-2%** (depending on prefill vs decode)

For 3B (hidden=2048):
- Replaced 5 groups = 5 × 64 = 320 channels
- Skipped FLOPs: `320/2048 = 15.6%` of `down_proj`
- **MLP-level saving: ~5.2%**

**Key insight**: The saving is modest per-layer but scales linearly with number of replaced groups. If we can push to 15-20 groups per layer, MLP-level savings become 10-15%.

### 2.3 Why bother with "only" 3.6%?

Because this is a **proof of concept** for a new compute paradigm:
1. On CIM hardware, LUT read is O(1) memory access vs O(N) MAC.
2. The ratio flips: memory bandwidth becomes cheaper than compute.
3. Even 3.6% FLOPs reduction, if realized as actual latency reduction, validates the hardware-software co-design thesis.

---

## 3. Three-Phase Implementation

### Phase 1: PyTorch Partial Linear (Validation)

**Goal**: Prove numerical equivalence between functional hook and partial matmul. No latency claims yet.

**Approach**:
- Replace `down_proj.forward` with a patched version.
- Compute `active_out = F.linear(hidden, active_weight, active_bias)` for non-replaced channels.
- Compute `replaced_out = normed_x_group + lut_delta` for replaced channels.
- Assemble full output tensor.

**Validation criteria**:
- PPL/KL/Acc exactly match functional hook (bitwise if possible).
- Generation outputs identical to v2 results.

**Expected outcome**: Numerical parity confirmed. Latency may be neutral or slightly worse due to PyTorch overhead.

### Phase 2: Triton Custom Kernel (Real Speedup)

**Goal**: Achieve measurable latency reduction on replaced groups.

**Approach**:
- Write a Triton kernel that performs partial matmul: `hidden [B*S, intermediate] @ W_down.T[active, :] → [B*S, active]`.
- Fuse LUT lookup and output assembly into the same kernel to minimize memory round-trips.
- Benchmark against `torch.nn.functional.linear`.

**Validation criteria**:
- Correctness: matches Phase 1 outputs within floating-point tolerance.
- Performance: `down_proj` latency reduced by at least 50% of theoretical FLOPs saving (e.g., 6 groups on 7B → ~5% `down_proj` speedup).

**Expected outcome**: Measurable `down_proj` speedup. End-to-end token latency improvement of 0.5-1.5%.

### Phase 3: End-to-End Latency Demo

**Goal**: Demonstrate that LUT-replaced model generates tokens faster than baseline.

**Approach**:
- Integrate Triton kernel into full inference pipeline.
- Measure end-to-end latency on standard prompts (128-token generation, batch=1).
- Compare three configurations:
  1. Baseline (no replacement)
  2. v2 Functional Hook (0% compute saved, full matmul + overwrite)
  3. v3 Partial Skip (actual FLOPs reduction)

**Validation criteria**:
- v3 is measurably faster than both baseline and v2 functional hook.
- Output quality (PPL/Acc/Generation) matches v2.

**Expected outcome**: First demo of "LUT lookup is faster than matrix multiplication for selected MLP groups."

---

## 4. Key Technical Challenges

### 4.1 Address Extraction Timing

**Problem**: LUT address must be computed from `normed_x` (MLP input), but `down_proj` receives `SwiGLU(normed_x)`.

**Solution**: Two-hook architecture:
1. **MLP pre-hook**: captures `normed_x`, computes bin indices for all replaced groups, caches them.
2. **Down-proj patched forward**: reads cache, performs partial matmul + LUT fill.

Execution order (guaranteed by PyTorch forward flow):
```
MLP.forward(normed_x) starts
  → MLP pre-hook fires (cache normed_x, compute bins)
  → gate_proj, up_proj, SwiGLU execute
  → down_proj.forward(swiglu_out) called
    → patched forward reads cache, partial matmul + LUT
  → residual add
```

### 4.2 LUT Output Conversion

**Problem**: v2 LUT stores `delta_group = mlp(normed_x)_group - normed_x_group`. `down_proj` produces `mlp(normed_x)_group`.

**Solution**: In patched forward, compute:
```python
replaced_out = normed_x_group + lut_delta
```

This is algebraically equivalent to what the functional hook achieves.

### 4.3 Multi-Group Assembly

**Problem**: Replaced groups may be non-contiguous (e.g., G4, G26, G50).

**Solution**: Build `active_mask` and `replaced_mask` over `hidden_size` channels. Use `index_copy_` or advanced indexing to assemble output. For Triton kernel, use a column-skip list.

### 4.4 Numerical Equivalence Trap

**Risk**: PyTorch partial linear may introduce tiny numerical differences (order 1e-6) due to different matmul accumulation order. These could compound across layers.

**Mitigation**: Compare logits at layer 21 between functional hook and partial skip. Maximum absolute diff should be < 1e-5.

---

## 5. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PyTorch partial matmul is slower than full matmul | High | Phase 1 delay | Expected; Phase 1 is for validation only, not speed |
| Triton kernel development takes >2 weeks | Medium | Phase 2 delay | Start with simple fused kernel; CUTLASS as fallback |
| End-to-end speedup <0.5% (below measurement noise) | Medium | Phase 3 failure | Increase replaced groups to 10+; optimize kernel launch overhead |
| Numerical diff causes PPL degradation | Low | Phase 1 failure | Rigorous per-layer logit comparison; debug accumulation order |
| Qwen2.5 architecture detail blocks implementation | Low | All phases | Read source code carefully; test on smallest scale first |

---

## 6. Success Criteria

| Milestone | Criterion | Timeline |
|-----------|-----------|----------|
| Phase 1 Complete | PyTorch partial linear achieves PPL ±0.1 vs v2 hook | 1 week |
| Phase 2 Complete | Triton kernel shows >50% of theoretical down_proj speedup | 2-3 weeks |
| Phase 3 Complete | End-to-end latency improvement >0.5% vs baseline | 1 week |
| **v3 Claim** | "Selected MLP groups can be replaced with LUT lookup, achieving measurable inference speedup without behavioral degradation." | — |

---

## 7. Immediate Next Step

Implement `v3/partial_linear.py`: a PyTorch-only `V3PartialEngine` that:
1. Reuses existing v2 calibration data (addr_idx, addr_mean, addr_std, table).
2. Patches `down_proj.forward` to skip replaced groups.
3. Validates numerical equivalence against v2 functional hook on 7B L21 6-group.

If Phase 1 passes, proceed to Triton kernel.
