"""
Triton kernels for v3 partial LUT fill.

Provides fused multi-group LUT lookup + delta computation.
Falls back to PyTorch if Triton is unavailable.
"""

import torch
import torch.nn.functional as F

# Graceful fallback if Triton is not installed
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel: multi-group LUT fill
# ---------------------------------------------------------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def _lut_fill_multi_group_kernel(
        bin_idx_ptr,       # [M, num_groups, 2]   int32
        tables_ptr,        # [num_groups, 64, 64, 64]  same dtype as out
        normed_x_ptr,      # [M, hidden_size]     same dtype as out
        out_ptr,           # [M, num_groups*64]   output
        group_starts_ptr,  # [num_groups]         int32
        addr_mean_ptr,     # [num_groups, 64]
        addr_std_ptr,      # [num_groups, 64]
        num_groups,
        M,
        stride_bin_m, stride_bin_g, stride_bin_h,
        stride_table_g, stride_table_b1, stride_table_b2, stride_table_c,
        stride_normed_m, stride_normed_n,
        stride_out_m, stride_out_n,
        stride_mean_g, stride_mean_c,
        BLOCK_M: tl.constexpr,
    ):
        GROUP_SIZE: tl.constexpr = 64
        NUM_BINS: tl.constexpr = 64

        pid_m = tl.program_id(0)
        pid_group = tl.program_id(1)

        if pid_group >= num_groups:
            return

        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M

        # Group start channel in normed_x
        g_start = tl.load(group_starts_ptr + pid_group).to(tl.int32)

        # Load bin indices for this group
        b1_offs = m_offs * stride_bin_m + pid_group * stride_bin_g
        b2_offs = m_offs * stride_bin_m + pid_group * stride_bin_g + stride_bin_h
        b1 = tl.load(bin_idx_ptr + b1_offs, mask=m_mask)
        b2 = tl.load(bin_idx_ptr + b2_offs, mask=m_mask)

        # Table base for this group
        table_base = tables_ptr + pid_group * stride_table_g

        # Address stats for this group
        mean_base = addr_mean_ptr + pid_group * stride_mean_g
        std_base  = addr_std_ptr  + pid_group * stride_mean_c

        for c in tl.range(0, GROUP_SIZE):
            # ---- LUT lookup ----
            # lut_val = table[pid_group, b1, b2, c]
            table_idx = (b1 * stride_table_b1 +
                         b2 * stride_table_b2 +
                         c  * stride_table_c)
            lut_val = tl.load(table_base + table_idx, mask=m_mask)

            # ---- Denormalize ----
            # lut_delta = lut_val * addr_std + addr_mean
            addr_mean_c = tl.load(mean_base + c * stride_mean_c)
            addr_std_c  = tl.load(std_base  + c * stride_mean_c)
            lut_delta = lut_val * addr_std_c + addr_mean_c

            # ---- Load normed_x ----
            x_idx = m_offs * stride_normed_m + (g_start + c) * stride_normed_n
            x_val = tl.load(normed_x_ptr + x_idx, mask=m_mask)

            # ---- Store output ----
            out_idx = m_offs * stride_out_m + (pid_group * GROUP_SIZE + c) * stride_out_n
            tl.store(out_ptr + out_idx, x_val + lut_delta, mask=m_mask)


def triton_lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std):
    """
    Fused multi-group LUT fill.

    Args:
        bin_idx:      [M, num_groups, 2]   int64 or int32
        tables:       [num_groups, 64, 64, 64]  float16/float32
        normed_x:     [M, hidden_size]     float16/float32
        group_starts: [num_groups]         int64/int32
        addr_mean:    [num_groups, 64]
        addr_std:     [num_groups, 64]

    Returns:
        out: [M, num_groups*64]  float16/float32
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install with: pip install triton")

    M, num_groups, _ = bin_idx.shape
    group_size = 64
    device = normed_x.device
    dtype = normed_x.dtype

    out = torch.empty(M, num_groups * group_size, device=device, dtype=dtype)

    # Ensure contiguous and correct dtypes
    bin_idx = bin_idx.contiguous().to(torch.int32)
    tables = tables.contiguous().to(dtype)
    normed_x = normed_x.contiguous()
    group_starts = group_starts.contiguous().to(torch.int32)
    addr_mean = addr_mean.contiguous().to(dtype)
    addr_std = addr_std.contiguous().to(dtype)

    # Block size tuning
    BLOCK_M = 128

    grid = (triton.cdiv(M, BLOCK_M), num_groups)

    _lut_fill_multi_group_kernel[grid](
        bin_idx, tables, normed_x, out, group_starts,
        addr_mean, addr_std,
        num_groups, M,
        bin_idx.stride(0), bin_idx.stride(1), bin_idx.stride(2),
        tables.stride(0), tables.stride(1), tables.stride(2), tables.stride(3),
        normed_x.stride(0), normed_x.stride(1),
        out.stride(0), out.stride(1),
        addr_mean.stride(0), addr_mean.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return out


# ---------------------------------------------------------------------------
# PyTorch fallback: same semantics, slower (many small kernel launches)
# ---------------------------------------------------------------------------
def pytorch_lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std):
    """PyTorch fallback for LUT fill (per-group loop)."""
    M, num_groups, _ = bin_idx.shape
    group_size = 64
    device = normed_x.device
    dtype = normed_x.dtype
    out = torch.empty(M, num_groups * group_size, device=device, dtype=dtype)

    for g in range(num_groups):
        g_start = int(group_starts[g].item())
        b1 = bin_idx[:, g, 0].flatten()
        b2 = bin_idx[:, g, 1].flatten()
        lut_delta = tables[g, b1, b2]                     # [M, 64]
        lut_delta = lut_delta * addr_std[g:g+1] + addr_mean[g:g+1]
        x_group = normed_x[:, g_start:g_start + group_size]
        out[:, g * group_size:(g + 1) * group_size] = x_group + lut_delta

    return out


def lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std):
    """Unified entry: Triton if available, else PyTorch fallback."""
    if TRITON_AVAILABLE:
        try:
            return triton_lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std)
        except Exception as e:
            print(f"[Triton] Kernel failed ({e}), falling back to PyTorch.")
            return pytorch_lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std)
    else:
        return pytorch_lut_fill(bin_idx, tables, normed_x, group_starts, addr_mean, addr_std)
