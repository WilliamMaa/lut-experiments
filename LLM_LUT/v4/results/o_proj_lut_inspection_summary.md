# o_proj LUT inspection summary

## Full-output run

- Output: `LLM_LUT/v4/results/o_proj_lut_inspection.json`
- Settings: group_size=64, num_bins=64, table = **full o_proj output**

| layer | relative_mse | rmse | group_rmse_mean | group_rmse_max |
|------:|-------------:|-----:|----------------:|---------------:|
| 17 | 0.3925 | 0.1367 | 0.1360 | 0.2039 |
| 15 | 0.4572 | 0.1462 | 0.1451 | 0.2349 |
| 16 | 0.4979 | 0.1595 | 0.1583 | 0.2580 |
| 20 | 0.5178 | 0.1674 | 0.1668 | 0.2531 |
| 23 | 0.5840 | 0.3258 | 0.3235 | 0.6032 |
| 18 | 0.5952 | 0.1816 | 0.1812 | 0.2424 |
| 25 | 0.6254 | 0.4178 | 0.4132 | 0.8593 |
| 24 | 0.6402 | 0.3135 | 0.3072 | 0.7661 |
| 22 | 0.6459 | 0.3024 | 0.3013 | 0.4695 |
| 26 | 0.6699 | 0.6071 | 0.5905 | 1.6317 |
| 19 | 0.6731 | 0.2164 | 0.2141 | 0.3771 |
| 21 | 0.7431 | 0.2371 | 0.2348 | 0.4491 |
| 27 | **inf** | **inf** | **inf** | **inf** |

## Residual run (`--residual`)

- Output: `LLM_LUT/v4/results/o_proj_lut_inspection_residual.json`
- Settings: group_size=64, num_bins=64, table = **o_proj output − o_proj input**

| layer | relative_mse | rmse | group_rmse_mean | group_rmse_max | est. output_var |
|------:|-------------:|-----:|----------------:|---------------:|----------------:|
| 27 | 0.1836 | 3.1123 | 2.0310 | 14.3372 | 52.7500 |
| 24 | 1.0040 | 0.3927 | 0.3849 | 0.6238 | 0.1536 |
| 17 | 1.0582 | 0.2244 | 0.2215 | 0.2923 | 0.0476 |
| 15 | 1.1116 | 0.2279 | 0.2236 | 0.3331 | 0.0467 |
| 22 | 1.1307 | 0.4001 | 0.3972 | 0.5308 | 0.1416 |
| 23 | 1.1375 | 0.4547 | 0.4468 | 0.7594 | 0.1818 |
| 25 | 1.1948 | 0.5774 | 0.5638 | 0.9618 | 0.2791 |
| 20 | 1.3250 | 0.2678 | 0.2636 | 0.3874 | 0.0541 |
| 16 | 1.3320 | 0.2609 | 0.2569 | 0.4071 | 0.0511 |
| 18 | 1.3372 | 0.2722 | 0.2696 | 0.3475 | 0.0554 |
| 26 | 1.3397 | 0.8586 | 0.8415 | 1.6816 | 0.5503 |
| 21 | 1.6620 | 0.3547 | 0.3454 | 0.5490 | 0.0757 |
| 19 | 1.7099 | 0.3449 | 0.3365 | 0.5914 | 0.0696 |

### Key comparison

| layer | full-output rel_mse | residual rel_mse | change |
|------:|--------------------:|-----------------:|-------:|
| 17 | 0.3925 | 1.0582 | **+0.6657** |
| 15 | 0.4572 | 1.1116 | **+0.6544** |
| 16 | 0.4979 | 1.3320 | **+0.8341** |
| 20 | 0.5178 | 1.3250 | **+0.8072** |
| 27 | inf | 0.1836 | fixed, but still spiky |

## Interpretation

**Residual mode made o_proj reconstruction worse, not better.**

For `down_proj`, residual mode works because the output is close to the module input (the MLP residual stream), so the LUT only has to learn a small correction.

For `o_proj`, the output is **not close to the input**; subtracting the input actually increases the target variance. Every layer except L27 ends up with `relative_mse > 1`, meaning the LUT reconstructs the residual worse than simply predicting zero.

L27 looks like an outlier (`relative_mse = 0.18`) only because its output variance is enormous (~52.7), masking the fact that some groups still have RMSE spikes > 14. It is not a safe candidate.

## Conclusion

Under the current 2-channel-address / 64-bin 2D LUT design:

- **Full-output o_proj** is too lossy (best layer ~0.39 relative MSE).
- **Residual-output o_proj** is even worse (>1.0 relative MSE for 12/13 layers).
- **o_proj is not a viable next axis with this LUT scheme.**

The quality gap is structural: `o_proj` performs a non-local attention aggregation, and its output cannot be reliably indexed from just two input channels with this table size.

## Recommended next steps

1. **Stop pushing o_proj with this design.** The numbers do not justify adding it on top of the down_proj stack.
2. **Pivot back to down_proj optimization**, where we already have a working Pareto point (PPL=29.25, Acc=0.470, MAC↓2.78%).
3. Concrete down_proj next axes:
   - **Quantize LUT tables to INT8** to fit more groups / layers.
   - **Try all 28 layers** with smaller group counts (e.g. group 4-8) instead of 13 layers at group 12.
   - **Per-group address selection** for down_proj to push group12 quality higher.
   - **Non-uniform layer configs** from the sensitivity scans (some layers tolerate more replacement).

If you still want one more o_proj probe, the only remaining idea is to use the **true residual stream** (pre-layer-norm input) as the LUT base instead of the o_proj input. But given these numbers, the expected payoff is low.
