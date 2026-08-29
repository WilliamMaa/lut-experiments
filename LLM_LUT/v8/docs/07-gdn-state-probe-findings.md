# GDN Recurrent State Compressibility Probe — Findings

## What we measured

- Model: `Qwen3.6-35B-A3B`
- 40 layers, 30 of them are `linear_attention` (Gated DeltaNet, GDN), 10 are `full_attention`
- GDN recurrent state per layer: `(B, 32, 128, 128)` FP32 = **2 MB/layer**
- 30 GDN layers total: **60 MB** of recurrent state (vs. ~0 for KV cache in GDN layers)
- Sampled 50 eval-texts at positions 256 and 512 for layers 14, 20, 26, 34

## Key results

### 1. Each head state is strongly low-rank

| layer | pos | 50% energy rank | 80% | 90% | 95% | 99% | rank-32 rel err | rank-64 rel err |
|------|-----|----------------|-----|-----|-----|-----|----------------|----------------|
| 14   | 256 | 1              | 8   | 18  | 28  | 55  | 1.7%           | 0.23%          |
| 14   | 512 | 1              | 8   | 18  | 29  | 58  | 1.6%           | 0.24%          |
| 20   | 256 | 1              | 4   | 12  | 22  | 51  | 1.1%           | 0.17%          |
| 20   | 512 | 1              | 5   | 13  | 23  | 55  | 1.1%           | 0.19%          |
| 26   | 256 | 1              | 8   | 16  | 26  | 52  | 1.4%           | 0.16%          |

- A single singular value captures 50% of energy.
- **Rank 32/128 (~25% of full rank) already gives ~1% reconstruction error.**
- **Rank 64/128 (~50% of full rank) gives ~0.2% error.**

This points to **Branch A: low-rank factorization** of the recurrent state.

### 2. State changes are NOT sparse over time

Delta between position 256 and 512 (same prompts):

| layer | relative Frobenius mean | relative Frobenius max | mean abs element delta | max abs element delta |
|------|------------------------|------------------------|------------------------|----------------------|
| 14   | 0.70                   | 3.25                   | 0.00249                | 2.92                 |
| 20   | 0.60                   | 3.43                   | 0.00231                | 3.20                 |

Interpretation:
- The state changes by **60–70% in Frobenius norm** over the next 256 tokens.
- Max sample delta can exceed 3x the original state norm.
- So the state is **not a slowly varying, sparse-delta object**. Branch B (sparse delta update) is unlikely.

### 3. Head energy is somewhat non-uniform

- Top-1 head carries only ~1.4–2.5% of total spectral energy.
- Top-4 heads carry ~26–44%.
- Top-8 heads carry ~48–60%.
- A few heads (e.g., head 23/24 in layer 14/26) are much more active than others.

This suggests future refinement: **per-head rank allocation** instead of one global rank.

## Conclusion: go with Branch A first

The recurrent state `S_t` is a low-rank matrix per head. The immediate next step is:

1. Build a **low-rank state patch** that keeps `S_t ≈ U_t V_t^T`.
2. Run generation/PPL/logit-KL eval to see if ~rank-32 propagation stays within acceptable degradation.
3. If it works, explore per-head rank allocation and quantization of the factors.

If low-rank propagation degrades too much, fall back to Branch C (approximate the transition operator rather than the state snapshot).
