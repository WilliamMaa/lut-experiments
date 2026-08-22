#!/usr/bin/env python3
"""
build_lut_ffn_output.py
"""

import os
import gc
import glob
import json
import math
import time
import itertools
import argparse
from pathlib import Path
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# 1. Teacher 专家模型
# =============================================================================
class QwenMoEExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def load_real_teacher(pt_path, device):
    print(f"\nLoading teacher weights: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu")
    gate_key = next(k for k in state_dict.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = state_dict[gate_key].shape
    expert = QwenMoEExpert(hidden_size, intermediate_size)
    clean_state_dict = {k.split("expert.")[-1] if "expert." in k else k: v for k, v in state_dict.items()}
    expert.load_state_dict(clean_state_dict, strict=False)
    expert.to(device).eval()
    return expert, hidden_size, intermediate_size


# =============================================================================
# 2. LUT 核心：tree address + 均值表（从 v5 移植，保持自包含）
# =============================================================================
class _TreeNode:
    __slots__ = (
        "node_id", "channel_idx", "signs", "threshold",
        "left", "right", "is_leaf", "leaf_index",
    )

    def __init__(self, channel_idx=None, signs=None, threshold=None,
                 left=None, right=None, is_leaf=False, leaf_index=None, node_id=None):
        self.node_id = node_id
        self.channel_idx = channel_idx
        self.signs = signs
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_index = leaf_index


class AddressGreedyTree:
    """
    Data-dependent decision-tree address.
    离线构建、在线 O(1) 遍历，无训练参数、无最近邻搜索。
    num_bits 个内部节点 => 最多 2^num_bits 个叶子/表项。
    """

    def __init__(self, input_dim: int, num_bits: int, channels_per_bit: int = 4,
                 seed: int = 0, tree_state: Optional[dict] = None):
        self.input_dim = input_dim
        self.num_bits = num_bits
        self.channels_per_bit = channels_per_bit
        self.num_entries = 2 ** num_bits
        self.num_tables = 1
        self.gen = torch.Generator().manual_seed(seed)
        self.root = None
        self._leaf_counter = 0
        if tree_state is not None:
            self.root = self._deserialize(tree_state)
            self._build_lookup_arrays()

    def build(self, x: torch.Tensor, target: torch.Tensor,
              num_candidates: int = 64, min_samples: int = 32,
              max_samples: int = 65536, device: torch.device = None,
              candidate_chunk_size: int = 32, min_relative_gain: float = 0.0):
        N = x.shape[0]
        if N > max_samples:
            perm = torch.randperm(N, device=x.device)[:max_samples]
            x = x[perm]
            target = target[perm]
        if device is not None:
            x = x.to(device)
            target = target.to(device)
        self._leaf_counter = 0
        self.min_relative_gain = min_relative_gain
        self.root = self._build_node(x, target, depth=0,
                                     num_candidates=num_candidates,
                                     min_samples=min_samples,
                                     candidate_chunk_size=candidate_chunk_size)
        self._build_lookup_arrays()

    def _build_node(self, x, target, depth, num_candidates, min_samples, candidate_chunk_size):
        N = x.shape[0]
        if depth >= self.num_bits or N < 2 * min_samples:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        parent_var = target.var(dim=0, unbiased=False).sum().item()
        if parent_var < 1e-12:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        gs = target.shape[1]
        total_sum = target.sum(dim=0)          # [gs]
        total_sum_sq = (target ** 2).sum(dim=0) # [gs]

        best_reduction = -1.0
        best_ch = None
        best_signs = None
        best_threshold = None
        best_left_mask = None

        # vectorized candidate evaluation in chunks to control memory
        for start in range(0, num_candidates, candidate_chunk_size):
            end = min(start + candidate_chunk_size, num_candidates)
            csize = end - start

            ch = torch.randint(
                0, self.input_dim, (csize, self.channels_per_bit), generator=self.gen
            ).to(x.device)
            signs = (
                torch.randint(0, 2, (csize, self.channels_per_bit), generator=self.gen).float() * 2 - 1
            ).to(x.device)

            # projections: [N, csize]
            selected = x[:, ch]  # [N, csize, channels_per_bit]
            proj = (selected * signs.to(x.dtype)).sum(dim=-1)

            # threshold: median per candidate
            thresholds = proj.median(dim=0).values  # [csize]

            # left masks: [csize, N]
            left_mask = (proj <= thresholds.unsqueeze(0)).t().contiguous()
            n_l = left_mask.sum(dim=1).float()
            n_r = N - n_l
            valid = (n_l >= min_samples) & (n_r >= min_samples)
            if not valid.any():
                continue

            # batched variance via scatter-add-like matmul
            left_mask_f = left_mask.float()
            left_sum = torch.matmul(left_mask_f, target.float())              # [csize, gs]
            left_sum_sq = torch.matmul(left_mask_f, (target ** 2).float())    # [csize, gs]
            right_sum = total_sum.float().unsqueeze(0) - left_sum
            right_sum_sq = total_sum_sq.float().unsqueeze(0) - left_sum_sq

            n_l_safe = n_l.clamp(min=1.0).unsqueeze(1)
            n_r_safe = n_r.clamp(min=1.0).unsqueeze(1)
            left_var = (left_sum_sq / n_l_safe) - (left_sum / n_l_safe) ** 2
            right_var = (right_sum_sq / n_r_safe) - (right_sum / n_r_safe) ** 2
            left_var.clamp_(min=0.0)
            right_var.clamp_(min=0.0)

            reductions = parent_var - (n_l * left_var.sum(dim=1) + n_r * right_var.sum(dim=1)) / N
            reductions = torch.where(valid, reductions, torch.full_like(reductions, -float('inf')))

            best_idx = int(torch.argmax(reductions).item())
            if reductions[best_idx].item() > best_reduction:
                best_reduction = reductions[best_idx].item()
                best_ch = ch[best_idx]
                best_signs = signs[best_idx]
                best_threshold = thresholds[best_idx].item()
                best_left_mask = left_mask[best_idx]

        # Check absolute and relative gain
        relative_gain = best_reduction / (parent_var + 1e-12) if parent_var > 0 else 0
        if best_reduction <= 0 or best_left_mask is None:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf
        if hasattr(self, 'min_relative_gain') and self.min_relative_gain > 0:
            if relative_gain < self.min_relative_gain:
                leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
                self._leaf_counter += 1
                return leaf

        left = self._build_node(x[best_left_mask], target[best_left_mask],
                                depth + 1, num_candidates, min_samples,
                                candidate_chunk_size)
        right = self._build_node(x[~best_left_mask], target[~best_left_mask],
                                 depth + 1, num_candidates, min_samples,
                                 candidate_chunk_size)
        return _TreeNode(
            channel_idx=best_ch,
            signs=best_signs,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def _build_lookup_arrays(self):
        if self.root is None:
            raise ValueError("Tree root is None")
        queue = deque([self.root])
        node_list = []
        counter = 0
        while queue:
            node = queue.popleft()
            node.node_id = counter
            counter += 1
            node_list.append(node)
            if not node.is_leaf:
                queue.append(node.left)
                queue.append(node.right)
        num_nodes = counter
        self.num_nodes = num_nodes
        self.node_is_leaf = torch.zeros(num_nodes, dtype=torch.bool)
        self.node_leaf_index = torch.full((num_nodes,), -1, dtype=torch.long)
        self.node_channel_idx = torch.zeros(num_nodes, self.channels_per_bit, dtype=torch.long)
        self.node_signs = torch.zeros(num_nodes, self.channels_per_bit, dtype=torch.float32)
        self.node_threshold = torch.zeros(num_nodes, dtype=torch.float32)
        self.node_left = torch.zeros(num_nodes, dtype=torch.long)
        self.node_right = torch.zeros(num_nodes, dtype=torch.long)
        for node in node_list:
            i = node.node_id
            self.node_is_leaf[i] = node.is_leaf
            if node.is_leaf:
                self.node_leaf_index[i] = node.leaf_index
            else:
                self.node_channel_idx[i] = node.channel_idx
                self.node_signs[i] = node.signs
                self.node_threshold[i] = node.threshold
                self.node_left[i] = node.left.node_id
                self.node_right[i] = node.right.node_id

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, input_dim]
        Returns:
            indices: [B, S, 1]
        """
        B, S, _ = x.shape
        device = x.device
        N = B * S
        x_flat = x.view(N, self.input_dim)
        out = torch.empty(N, dtype=torch.long, device=device)
        active = torch.ones(N, dtype=torch.bool, device=device)
        node_ids = torch.zeros(N, dtype=torch.long, device=device)

        node_is_leaf = self.node_is_leaf.to(device)
        node_leaf_index = self.node_leaf_index.to(device)
        node_channel_idx = self.node_channel_idx.to(device)
        node_signs = self.node_signs.to(device, dtype=x.dtype)
        node_threshold = self.node_threshold.to(device)
        node_left = self.node_left.to(device)
        node_right = self.node_right.to(device)

        for _ in range(self.num_bits + 1):
            if not active.any():
                break
            cur_ids = node_ids[active]
            is_leaf = node_is_leaf[cur_ids]
            if is_leaf.any():
                active_idx = torch.where(active)[0]
                leaf_idx = active_idx[is_leaf]
                out[leaf_idx] = node_leaf_index[cur_ids[is_leaf]]
                active[leaf_idx] = False
            if not active.any():
                break
            cur_ids = node_ids[active]
            active_idx = torch.where(active)[0]
            ch = node_channel_idx[cur_ids]
            signs = node_signs[cur_ids]
            selected = x_flat[active_idx.unsqueeze(1), ch]
            proj = (selected * signs).sum(dim=-1)
            go_left = proj <= node_threshold[cur_ids]
            node_ids[active] = torch.where(go_left, node_left[cur_ids], node_right[cur_ids])
        return out.view(B, S, 1)

    def serialize(self) -> dict:
        return {"tree": self._serialize_node(self.root)}

    def _serialize_node(self, node: _TreeNode):
        if node.is_leaf:
            return {"leaf_index": node.leaf_index}
        return {
            "channel_idx": node.channel_idx.cpu().tolist(),
            "signs": node.signs.cpu().tolist(),
            "threshold": float(node.threshold),
            "left": self._serialize_node(node.left),
            "right": self._serialize_node(node.right),
        }

    def _deserialize(self, state: dict):
        return self._deserialize_node(state["tree"])

    def _deserialize_node(self, d: dict):
        if "leaf_index" in d:
            return _TreeNode(is_leaf=True, leaf_index=d["leaf_index"])
        return _TreeNode(
            channel_idx=torch.tensor(d["channel_idx"], dtype=torch.long),
            signs=torch.tensor(d["signs"], dtype=torch.float32),
            threshold=d["threshold"],
            left=self._deserialize_node(d["left"]),
            right=self._deserialize_node(d["right"]),
        )


class Address2D:
    """Original v3/v4 style: 2 selected channels -> 2D bins -> flattened index."""

    def __init__(self, addr_idx: torch.Tensor, addr_mean: torch.Tensor,
                 addr_std: torch.Tensor, num_bins: int = 64, addr_clip: float = 3.0):
        self.addr_idx = addr_idx.cpu().long()
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu()
        self.num_bins = num_bins
        self.addr_clip = addr_clip
        self.num_entries = num_bins * num_bins
        self.num_tables = 1

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
        Returns:
            indices: [B, S, 1] flattened 2D bin index
        """
        B, S, _ = x.shape
        device = x.device
        idx = self.addr_idx.to(device)
        addr = x.index_select(-1, idx)  # [B, S, 2]

        mean = self.addr_mean.to(device, x.dtype).view(1, 1, -1)
        std = self.addr_std.to(device, x.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        b = torch.round(qf).long().clamp(0, self.num_bins - 1)
        flat = b[:, :, 0] * self.num_bins + b[:, :, 1]
        return flat.unsqueeze(-1)  # [B, S, 1]


class AddressHighOrderRandom:
    """
    Fixed random high-order address.
    For each of M tables and B bits, randomly select K input channels and random
    signs, project x, standardize, and threshold to get a binary bit.
    The B bits form an integer index in [0, 2^B).
    """

    def __init__(self, input_dim: int, num_tables: int, num_bits: int,
                 channels_per_bit: int = 4, seed: int = 0,
                 addr_mean: Optional[torch.Tensor] = None,
                 addr_std: Optional[torch.Tensor] = None):
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.channels_per_bit = channels_per_bit
        self.num_entries = 2 ** num_bits

        gen = torch.Generator().manual_seed(seed)
        self.channel_idx = torch.randint(
            0, input_dim, (num_tables, num_bits, channels_per_bit), generator=gen
        ).long()
        self.signs = (torch.randint(0, 2, (num_tables, num_bits, channels_per_bit), generator=gen).float() * 2 - 1)

        if addr_mean is None:
            addr_mean = torch.zeros(num_tables, num_bits)
        if addr_std is None:
            addr_std = torch.ones(num_tables, num_bits)
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu().clamp_min(1e-6)
        self.powers = (2 ** torch.arange(num_bits)).long()

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, input_dim]
        Returns:
            indices: [B, S, num_tables] integer index per table
        """
        B, S, _ = x.shape
        device = x.device
        N = B * S
        x_flat = x.view(N, self.input_dim)
        idx = self.channel_idx.to(device)
        selected = x_flat[:, idx]  # [N, num_tables, num_bits, channels_per_bit]
        signs = self.signs.to(device, x.dtype)
        proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]

        mean = self.addr_mean.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        std = self.addr_std.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        z = (proj - mean) / std
        bits = (z > 0).long()  # [N, num_tables, num_bits]

        powers = self.powers.to(device).view(1, 1, self.num_bits)
        indices = (bits * powers).sum(dim=-1)  # [N, num_tables]
        return indices.view(B, S, self.num_tables)

    def fit_calibration(self, x: torch.Tensor):
        """Re-compute addr_mean/std from calibration data."""
        with torch.no_grad():
            B, S, _ = x.shape
            N = B * S
            x_flat = x.view(N, self.input_dim)
            idx = self.channel_idx.to(x_flat.device)
            selected = x_flat[:, idx]
            signs = self.signs.to(x_flat.device, x_flat.dtype)
            proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]
            self.addr_mean = proj.mean(dim=0).cpu()
            self.addr_std = proj.std(dim=0).cpu().clamp_min(1e-6)


def select_2d_address(calib_x, target, group_size, num_bins, num_candidates=8):
    """Pick a good 2-channel address pair using the first group target."""
    hidden_size = calib_x.shape[-1]
    channel_var = calib_x.var(dim=0)
    top_channels = torch.topk(channel_var, k=min(num_candidates * 2, hidden_size)).indices.tolist()
    pairs = list(itertools.combinations(top_channels[:num_candidates], 2))[:20]

    out_group = target[:, :group_size]
    best_rmse = float("inf")
    best_pair = None
    for c1, c2 in pairs:
        addr_idx = torch.tensor([c1, c2], device=calib_x.device)
        addr = Address2D(addr_idx,
                         calib_x[:, addr_idx].mean(dim=0),
                         calib_x[:, addr_idx].std(dim=0),
                         num_bins=num_bins)
        indices = addr.compute_indices(calib_x.unsqueeze(0)).view(-1, 1)
        lut = LUTGroup(1, addr.num_entries, group_size, device=calib_x.device)
        lut.initialize_from_calibration(indices, out_group)
        rec = lut(indices).squeeze(1)
        rmse = F.mse_loss(rec, out_group.float(), reduction="mean").item() ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_pair = (c1, c2)
    if best_pair is None:
        best_pair = (0, 1)
    return torch.tensor(best_pair, device=calib_x.device), best_rmse


class LUTGroup(nn.Module):
    """
    可训练（或离线初始化）的 LUT 表。
    形状 [num_tables, num_entries, group_size]，forward 时按地址查表并求和。
    """

    def __init__(self, num_tables: int, num_entries: int, group_size: int,
                 init_table: torch.Tensor = None, device: torch.device = None):
        super().__init__()
        self.num_tables = num_tables
        self.num_entries = num_entries
        self.group_size = group_size

        if init_table is not None:
            table = init_table.float().clone()
        else:
            table = torch.zeros(num_tables, num_entries, group_size)
        if device is not None:
            table = table.to(device)
        self.table = nn.Parameter(table)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert indices.shape[-1] == self.num_tables
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, self.num_tables).to(self.table.device)
        N = flat_idx.shape[0]
        outs = []
        for m in range(self.num_tables):
            t = self.table[m]
            idx_m = flat_idx[:, m].clamp(0, self.num_entries - 1)
            outs.append(t[idx_m])
        out = torch.stack(outs, dim=1).sum(dim=1)
        if len(orig_shape) == 2:
            return out
        return out.view(orig_shape[0], orig_shape[1], self.group_size)

    def initialize_from_calibration(self, indices: torch.Tensor, targets: torch.Tensor):
        """每个表项用 calibration target 的均值初始化。"""
        with torch.no_grad():
            M = self.num_tables
            E = self.num_entries
            gs = self.group_size
            device = self.table.device
            indices = indices.to(device)
            targets = targets.to(device)
            new_table = torch.zeros(M, E, gs, device=device, dtype=torch.float32)
            counts = torch.zeros(M, E, device=device, dtype=torch.float32)

            for m in range(M):
                idx_m = indices[:, m].clamp(0, E - 1)
                idx_exp = idx_m.unsqueeze(1).expand(-1, gs)
                new_table[m].scatter_add_(0, idx_exp, targets.float())
                counts[m].scatter_add_(0, idx_m, torch.ones_like(idx_m, dtype=torch.float32))

            counts = counts.clamp_min(1.0).unsqueeze(-1)
            new_table = new_table / counts
            self.table.copy_(new_table)


# =============================================================================
# 3. 数据流
# =============================================================================
def collect_calibration_and_eval(
    train_input_files, test_input_files, teacher,
    calib_size: int, eval_size: int, batch_size: int, device: torch.device,
    train_output_files=None, test_output_files=None,
):
    """
    从 train_input_files 里取 calib_size 条、test_input_files 里取 eval_size 条。
    如果 train/test_output_files 提供，则直接读取预计算的 FFN 输出；
    否则通过 Teacher 前向得到真实 FFN 输出。
    """
    weight_dtype = next(teacher.parameters()).dtype
    use_precomputed_outputs = train_output_files is not None

    def collect_from_files(input_paths, output_paths, needed, desc):
        inputs = []
        targets = []
        collected = 0
        pbar = tqdm(total=needed, desc=desc, unit="sample")

        x_buffer = []
        y_buffer = []
        buffer_len = 0

        def load_tensor_2d(path, name):
            tensor = torch.load(path, map_location="cpu")
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() != 2:
                raise ValueError(
                    f"{name} {path} has shape {tensor.shape}, expected [N, hidden] or [hidden]"
                )
            return tensor

        for idx, in_path in enumerate(sorted(input_paths)):
            if collected >= needed:
                break
            try:
                x_tensor = load_tensor_2d(in_path, "input")
            except Exception as e:
                print(f"  skip {in_path}: {e}")
                continue

            if use_precomputed_outputs:
                out_path = output_paths[idx]
                try:
                    y_tensor = load_tensor_2d(out_path, "output")
                except Exception as e:
                    print(f"  skip {out_path}: {e}")
                    continue
                if y_tensor.shape != x_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch: {in_path} {x_tensor.shape} vs {out_path} {y_tensor.shape}"
                    )
            else:
                y_tensor = None

            x_buffer.append(x_tensor)
            if y_tensor is not None:
                y_buffer.append(y_tensor)
            buffer_len += x_tensor.shape[0]

            while buffer_len >= batch_size:
                cat_x = torch.cat(x_buffer, dim=0)
                n_take = min(batch_size, cat_x.shape[0])
                x_batch = cat_x[:n_take].to(device, dtype=weight_dtype)
                if y_tensor is not None:
                    cat_y = torch.cat(y_buffer, dim=0)
                    y_batch = cat_y[:n_take].float().cpu()
                else:
                    with torch.no_grad():
                        y_batch = teacher(x_batch).float().cpu()
                x_batch = x_batch.float().cpu()
                inputs.append(x_batch)
                targets.append(y_batch)
                collected += n_take
                pbar.update(n_take)

                leftover_x = cat_x[n_take:]
                leftover_y = cat_y[n_take:] if y_tensor is not None else None
                x_buffer = [leftover_x] if leftover_x.shape[0] > 0 else []
                y_buffer = [leftover_y] if (leftover_y is not None and leftover_y.shape[0] > 0) else []
                buffer_len = leftover_x.shape[0] if leftover_x.shape[0] > 0 else 0

        if buffer_len > 0:
            x_batch = torch.cat(x_buffer, dim=0).to(device, dtype=weight_dtype)
            if y_buffer:
                y_batch = torch.cat(y_buffer, dim=0).float().cpu()
            else:
                with torch.no_grad():
                    y_batch = teacher(x_batch).float().cpu()
            x_batch = x_batch.float().cpu()
            inputs.append(x_batch)
            targets.append(y_batch)
            collected += x_batch.shape[0]
            pbar.update(x_batch.shape[0])

        pbar.close()
        if collected == 0:
            raise RuntimeError(f"No valid samples found in {input_paths[:3]}...")
        return torch.cat(inputs, dim=0)[:needed], torch.cat(targets, dim=0)[:needed]

    calib_x, calib_y = collect_from_files(train_input_files, train_output_files, calib_size, desc="calibration")
    eval_x, eval_y = collect_from_files(test_input_files, test_output_files, eval_size, desc="eval")
    return calib_x, calib_y, eval_x, eval_y


# =============================================================================
# 4. 构建与评估
# =============================================================================
@torch.no_grad()
def _build_single_address_lut(calib_x, group_target_for_lut, group_id, group_size, args, num_bits, seed_offset, desc, device=None):
    """Helper: build one address + LUT."""
    seed = args.seed * 10000 + group_id * 100 + seed_offset

    if args.address_mode == "2d":
        addr_idx, _ = select_2d_address(calib_x, group_target_for_lut, group_size, args.num_bins)
        address = Address2D(
            addr_idx,
            calib_x[:, addr_idx].mean(dim=0),
            calib_x[:, addr_idx].std(dim=0),
            num_bins=args.num_bins,
        )
    elif args.address_mode == "high_order":
        address = AddressHighOrderRandom(
            input_dim=calib_x.shape[-1],
            num_tables=args.num_tables,
            num_bits=num_bits,
            channels_per_bit=args.channels_per_bit,
            seed=seed,
        )
        address.fit_calibration(calib_x.unsqueeze(0))
    else:  # tree
        address = AddressGreedyTree(
            input_dim=calib_x.shape[-1],
            num_bits=num_bits,
            channels_per_bit=args.channels_per_bit,
            seed=seed,
        )
        if desc:
            print(f"  building {desc} tree for group {group_id} ...")
        t0 = time.time()
        address.build(
            calib_x, group_target_for_lut,
            num_candidates=args.tree_candidates,
            min_samples=args.tree_min_samples,
            max_samples=args.tree_max_samples,
            device=device,
            candidate_chunk_size=args.tree_candidate_chunk_size,
            min_relative_gain=getattr(args, 'tree_min_relative_gain', 0.0),
        )
        if desc:
            print(f"  {desc} tree built in {time.time() - t0:.2f}s, leaves={address._leaf_counter}")

    indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
    lut = LUTGroup(
        num_tables=address.num_tables,
        num_entries=address.num_entries,
        group_size=group_size,
        device=calib_x.device,
    )
    lut.initialize_from_calibration(indices, group_target_for_lut)
    return address, lut


@torch.no_grad()
def build_lut_for_group(calib_x, calib_y, group_id, group_size, args, device):
    g_start = group_id * group_size
    g_end = g_start + group_size
    group_target = calib_y[:, g_start:g_end]

    group_mean = None
    if args.target_mode == "residual_input":
        baseline = calib_x[:, g_start:g_end]
        group_target_for_lut = group_target - baseline
    elif args.target_mode == "residual_mean":
        group_mean = group_target.mean(dim=0)
        group_target_for_lut = group_target - group_mean
    else:
        group_target_for_lut = group_target

    address, lut = _build_single_address_lut(
        calib_x, group_target_for_lut, group_id, group_size, args,
        num_bits=args.num_bits, seed_offset=0, desc="single", device=device,
    )
    return [address], [lut], group_mean


@torch.no_grad()
def build_coarse_residual_luts_for_group(calib_x, calib_y, group_id, group_size, args, device):
    """Build two LUTs: coarse predicts the main target, residual predicts the error of coarse."""
    g_start = group_id * group_size
    g_end = g_start + group_size
    group_target = calib_y[:, g_start:g_end]

    group_mean = None
    if args.target_mode == "residual_input":
        baseline = calib_x[:, g_start:g_end]
        group_target_for_lut = group_target - baseline
    elif args.target_mode == "residual_mean":
        group_mean = group_target.mean(dim=0)
        group_target_for_lut = group_target - group_mean
    else:
        group_target_for_lut = group_target

    # Coarse LUT
    coarse_address, coarse_lut = _build_single_address_lut(
        calib_x, group_target_for_lut, group_id, group_size, args,
        num_bits=args.coarse_num_bits, seed_offset=0, desc="coarse", device=device,
    )

    # Compute coarse prediction on calibration data
    coarse_indices = coarse_address.compute_indices(calib_x.unsqueeze(0)).view(-1, coarse_address.num_tables)
    coarse_pred = coarse_lut(coarse_indices).squeeze(1)  # [N, group_size]

    # Residual target = remaining error after coarse
    residual_target = group_target_for_lut - coarse_pred

    # Residual LUT
    residual_address, residual_lut = _build_single_address_lut(
        calib_x, residual_target, group_id, group_size, args,
        num_bits=args.residual_num_bits, seed_offset=1, desc="residual", device=device,
    )

    # Triple residual: third LUT for remaining error after coarse + residual
    if args.triple_residual and args.third_num_bits is not None:
        # Compute coarse + residual prediction
        residual_indices = residual_address.compute_indices(calib_x.unsqueeze(0)).view(-1, residual_address.num_tables)
        residual_pred = residual_lut(residual_indices).squeeze(1)
        combined_pred = coarse_pred + residual_pred

        # Third target = remaining error
        third_target = group_target_for_lut - combined_pred

        # Third LUT
        third_address, third_lut = _build_single_address_lut(
            calib_x, third_target, group_id, group_size, args,
            num_bits=args.third_num_bits, seed_offset=2, desc="third", device=device,
        )
        return [coarse_address, residual_address, third_address], [coarse_lut, residual_lut, third_lut], group_mean

    return [coarse_address, residual_address], [coarse_lut, residual_lut], group_mean


@torch.no_grad()
def evaluate_group(addresses, luts, group_mean, target_mode, eval_x, eval_y, group_id, group_size, device):
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)

    pred_group = None
    for address, lut in zip(addresses, luts):
        lut = lut.to(device)
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
        out = lut(indices).squeeze(1)
        pred_group = out if pred_group is None else pred_group + out

    g_start = group_id * group_size
    g_end = g_start + group_size

    if target_mode == "residual_input":
        pred_group = pred_group + eval_x[:, g_start:g_end]
    elif target_mode == "residual_mean":
        pred_group = pred_group + group_mean.to(device)

    true_group = eval_y[:, g_start:g_end]

    mse = F.mse_loss(pred_group, true_group).item()
    rmse = math.sqrt(mse)
    var = true_group.var().item()
    rel_mse = mse / (var + 1e-8)
    rel_l2 = torch.norm(pred_group - true_group).item() / (torch.norm(true_group).item() + 1e-8)
    cos_sim = F.cosine_similarity(pred_group, true_group, dim=-1).mean().item()
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
    }


@torch.no_grad()
def evaluate_full_output(addresses_per_group, luts_per_group, group_means, target_mode, group_ids, eval_x, eval_y, group_size, device):
    """
    把所有被替换 group 拼回完整 FFN 输出，再和真实输出比。
    当 group_ids 覆盖所有输出通道时，等价于完整 FFN 替换。
    """
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    hidden_size = eval_y.shape[-1]

    pred_y = torch.empty_like(eval_y)
    covered = torch.zeros(hidden_size, dtype=torch.bool, device=device)

    for gid in group_ids:
        addresses = addresses_per_group[gid]
        luts = luts_per_group[gid]
        pred_group = None
        for address, lut in zip(addresses, luts):
            lut = lut.to(device)
            indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
            out = lut(indices).squeeze(1)
            pred_group = out if pred_group is None else pred_group + out

        g_start = gid * group_size
        g_end = g_start + group_size

        if target_mode == "residual_input":
            pred_group = pred_group + eval_x[:, g_start:g_end]
        elif target_mode == "residual_mean":
            pred_group = pred_group + group_means[gid].to(device)

        pred_y[:, g_start:g_end] = pred_group
        covered[g_start:g_end] = True

    if not covered.all():
        pred_y[:, ~covered] = eval_y[:, ~covered]
    else:
        assert covered.all(), "Full FFN replacement must cover every output channel"

    mse = F.mse_loss(pred_y, eval_y).item()
    rmse = math.sqrt(mse)
    var = eval_y.var().item()
    rel_mse = mse / (var + 1e-8)
    rel_l2 = torch.norm(pred_y - eval_y).item() / (torch.norm(eval_y).item() + 1e-8)
    cos_sim = F.cosine_similarity(pred_y, eval_y, dim=-1)
    cos_mean = cos_sim.mean().item()
    cos_p10 = torch.quantile(cos_sim, 0.10).item()
    cos_p50 = torch.quantile(cos_sim, 0.50).item()
    cos_p90 = torch.quantile(cos_sim, 0.90).item()

    pred_norm = torch.norm(pred_y, dim=-1)
    true_norm = torch.norm(eval_y, dim=-1)
    norm_ratio = pred_norm / (true_norm + 1e-8)
    norm_ratio_mean = norm_ratio.mean().item()
    norm_ratio_p10 = torch.quantile(norm_ratio, 0.10).item()
    norm_ratio_p50 = torch.quantile(norm_ratio, 0.50).item()
    norm_ratio_p90 = torch.quantile(norm_ratio, 0.90).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_mean,
        "cosine_similarity_p10": cos_p10,
        "cosine_similarity_p50": cos_p50,
        "cosine_similarity_p90": cos_p90,
        "norm_ratio": norm_ratio_mean,
        "norm_ratio_p10": norm_ratio_p10,
        "norm_ratio_p50": norm_ratio_p50,
        "norm_ratio_p90": norm_ratio_p90,
    }


def finetune_luts(addresses, luts, group_means, target_mode, calib_x, calib_y,
                  group_ids, group_size, device, args):
    """
    离线 finetune：地址固定，只优化 LUT 表值。
    在 calibration 数据上最小化所有被替换 group 的 MSE。
    """
    epochs = args.finetune_epochs
    if epochs <= 0:
        return

    print(f"\n[Finetune] optimizing LUT tables for {epochs} epochs (lr={args.finetune_lr}, batch={args.finetune_batch_size}, loss={args.finetune_loss_mode}) ...")
    params = []
    for gid in group_ids:
        for lut in luts[gid]:
            lut.to(device)
            params.append(lut.table)

    optimizer = torch.optim.Adam(params, lr=args.finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    calib_x = calib_x  # keep on CPU, move batch to device
    calib_y = calib_y
    n_samples = calib_x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0}
        n_batches = 0
        for start in range(0, n_samples, args.finetune_batch_size):
            idx = perm[start:start + args.finetune_batch_size]
            xb = calib_x[idx].to(device)
            yb = calib_y[idx].to(device)

            optimizer.zero_grad()

            # Reconstruct full replaced output for all groups
            pred_y = torch.zeros_like(yb)
            for gid in group_ids:
                g_start = gid * group_size
                g_end = g_start + group_size

                pred_group = None
                for address, lut in zip(addresses[gid], luts[gid]):
                    indices = address.compute_indices(xb.unsqueeze(0)).view(-1, address.num_tables)
                    out = lut(indices).squeeze(1)
                    pred_group = out if pred_group is None else pred_group + out

                if target_mode == "residual_input":
                    pred_group = pred_group + xb[:, g_start:g_end]
                elif target_mode == "residual_mean":
                    pred_group = pred_group + group_means[gid].to(device)

                pred_y[:, g_start:g_end] = pred_group

            mse = F.mse_loss(pred_y, yb)
            cos = F.cosine_similarity(pred_y, yb, dim=-1).mean()

            if args.finetune_loss_mode == "mse":
                loss = mse
            elif args.finetune_loss_mode == "cosine":
                loss = 1 - cos
            else:  # mse+cosine
                loss = mse + args.finetune_cosine_alpha * (1 - cos)

            loss.backward()
            optimizer.step()
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mse"] += mse.item()
            epoch_metrics["cos"] += cos.item()
            n_batches += 1

        scheduler.step()
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)
        print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_metrics['loss']:.6e}, mse={epoch_metrics['mse']:.6e}, cos={epoch_metrics['cos']:.6f}")
    print("[Finetune] done")


def compute_lut_storage_bytes(num_entries, group_size, num_tables=1):
    """按 FP16 表值估算。"""
    return num_tables * num_entries * group_size * 2


def table_storage_bytes_for_group(addresses, group_size):
    """支持 coarse+residual 等多张 LUT 的存储估算。"""
    total = 0
    for addr in addresses:
        total += compute_lut_storage_bytes(addr.num_entries, group_size, addr.num_tables)
    return total


def count_leaves_for_group(addresses):
    """对每组所有 address 统计叶子 / 表项数。"""
    total = 0
    for addr in addresses:
        if isinstance(addr, AddressGreedyTree):
            total += addr._leaf_counter
        else:
            total += addr.num_entries
    return total


def estimate_eval_files_needed(input_files, eval_size):
    """
    从末尾预留足够多的 .pt 文件，确保 eval 集至少能收集到 eval_size 条样本。
    先 probe 最后 100 个文件估算每个文件平均样本数。
    """
    n = len(input_files)
    if n <= 1:
        return n

    probe = min(100, n)
    counts = []
    for path in input_files[-probe:]:
        try:
            t = torch.load(path, map_location="cpu")
        except Exception:
            counts.append(1)
            continue
        if t.dim() == 1:
            counts.append(1)
        elif t.dim() == 2:
            counts.append(t.shape[0])
        else:
            counts.append(1)

    avg = sum(counts) / len(counts) if counts else 1.0
    n_eval = int(math.ceil(eval_size / avg))
    return min(n_eval, n - 1)


def parse_group_ids(s: str, max_group: int):
    """解析 group_ids，支持逗号分隔和连字符范围，例如 '0,1,2' 或 '0-7' 或 '0-3,8,10-15'。"""
    ids = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    ids = sorted(ids)
    if not ids:
        raise ValueError("--group_ids must contain at least one integer")
    for gid in ids:
        if not (0 <= gid < max_group):
            raise ValueError(f"group_id {gid} out of range [0, {max_group})")
    return ids


# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build single-expert FFN output group LUT (v6 corrected direction). "
                    "Supports single LUT or coarse+residual LUT."
    )
    parser.add_argument("--teacher_weight_path", required=True, help="Path to expert .pt weights")
    parser.add_argument("--dataset_dir", required=True, help="Dir containing .pt input tensors")
    parser.add_argument("--output_dataset_dir", default=None, help="Optional dir containing precomputed .pt FFN output tensors")
    parser.add_argument("--output_root", required=True, help="Output directory for checkpoints and summary")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--group_ids", type=str, default="0-3",
                        help="Output group ids to replace, e.g. '0,1,2,3' or '0-7' or '0-3,8,10-15'")
    parser.add_argument("--num_bits", type=int, default=12,
                        help="Tree depth or bits per table => 2^num_bits entries per table")
    parser.add_argument("--coarse_num_bits", type=int, default=None,
                        help="Coarse LUT bits. If set together with --residual_num_bits, use coarse+residual mode.")
    parser.add_argument("--residual_num_bits", type=int, default=None,
                        help="Residual LUT bits. If set together with --coarse_num_bits, use coarse+residual mode.")
    parser.add_argument("--triple_residual", action="store_true",
                        help="Enable triple residual mode: coarse + residual + third LUT.")
    parser.add_argument("--third_num_bits", type=int, default=None,
                        help="Third LUT bits for triple residual mode. Requires --triple_residual.")
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--address_mode", type=str, default="tree", choices=["tree", "high_order", "2d"],
                        help="Address generator mode")
    parser.add_argument("--num_tables", type=int, default=1,
                        help="Number of tables for high_order address (total entries = num_tables * 2^num_bits)")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Bins per axis for 2d address (entries = num_bins^2). Used for both coarse and residual in 2d mode.")
    parser.add_argument("--tree_candidates", type=int, default=64)
    parser.add_argument("--tree_min_samples", type=int, default=16)
    parser.add_argument("--tree_max_samples", type=int, default=65536,
                        help="Subsample calibration data for tree building")
    parser.add_argument("--tree_candidate_chunk_size", type=int, default=32,
                        help="Number of tree candidates evaluated in one GPU batch (lower = less memory)")
    parser.add_argument("--tree_min_relative_gain", type=float, default=0.0,
                        help="Minimum relative variance gain to continue splitting (0 = disabled, try 1e-3 or 5e-4 for noisy residuals)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints in output_root/checkpoints/")
    parser.add_argument("--target_mode", type=str, default="direct", choices=["direct", "residual_mean", "residual_input"],
                        help="direct: LUT stores full FFN output; residual_mean: LUT stores output - group_mean; residual_input: LUT stores output - input_residual")
    parser.add_argument("--calib_size", type=int, default=65536)
    parser.add_argument("--eval_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Number of offline LUT table value finetune epochs after mean initialization. 0 = no finetune.")
    parser.add_argument("--finetune_lr", type=float, default=1e-3,
                        help="Learning rate for offline LUT finetune")
    parser.add_argument("--finetune_batch_size", type=int, default=1024,
                        help="Batch size for offline LUT finetune")
    parser.add_argument("--finetune_loss_mode", type=str, default="mse",
                        choices=["mse", "mse+cosine", "cosine"],
                        help="Loss for LUT finetune. mse+cosine optimizes both magnitude and direction.")
    parser.add_argument("--finetune_cosine_alpha", type=float, default=1.0,
                        help="Weight for cosine term when finetune_loss_mode=mse+cosine")
    args = parser.parse_args()

    if args.tree_max_samples <= 0:
        raise ValueError(f"--tree_max_samples must be positive, got {args.tree_max_samples}")
    if args.tree_candidate_chunk_size <= 0:
        raise ValueError(f"--tree_candidate_chunk_size must be positive, got {args.tree_candidate_chunk_size}")
    if args.tree_min_samples <= 0:
        raise ValueError(f"--tree_min_samples must be positive, got {args.tree_min_samples}")
    if args.calib_size <= 0:
        raise ValueError(f"--calib_size must be positive, got {args.calib_size}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")

    coarse_residual_provided = (args.coarse_num_bits is not None) or (args.residual_num_bits is not None)
    if coarse_residual_provided and not ((args.coarse_num_bits is not None) and (args.residual_num_bits is not None)):
        raise ValueError("--coarse_num_bits and --residual_num_bits must be provided together or not at all")
    if args.coarse_num_bits is not None and args.coarse_num_bits <= 0:
        raise ValueError(f"--coarse_num_bits must be positive, got {args.coarse_num_bits}")
    if args.residual_num_bits is not None and args.residual_num_bits <= 0:
        raise ValueError(f"--residual_num_bits must be positive, got {args.residual_num_bits}")
    if args.triple_residual:
        if args.coarse_num_bits is None or args.residual_num_bits is None:
            raise ValueError("--triple_residual requires --coarse_num_bits and --residual_num_bits")
        if args.third_num_bits is None:
            raise ValueError("--triple_residual requires --third_num_bits")
        if args.third_num_bits <= 0:
            raise ValueError(f"--third_num_bits must be positive, got {args.third_num_bits}")
    if args.num_bits <= 0:
        raise ValueError(f"--num_bits must be positive, got {args.num_bits}")
    if args.finetune_epochs < 0:
        raise ValueError(f"--finetune_epochs must be non-negative, got {args.finetune_epochs}")
    if args.finetune_lr <= 0:
        raise ValueError(f"--finetune_lr must be positive, got {args.finetune_lr}")
    if args.finetune_batch_size <= 0:
        raise ValueError(f"--finetune_batch_size must be positive, got {args.finetune_batch_size}")
    if args.finetune_loss_mode not in ("mse", "mse+cosine", "cosine"):
        raise ValueError(f"--finetune_loss_mode must be mse/mse+cosine/cosine, got {args.finetune_loss_mode}")
    if args.finetune_cosine_alpha <= 0:
        raise ValueError(f"--finetune_cosine_alpha must be positive, got {args.finetune_cosine_alpha}")

    use_coarse_residual = args.coarse_num_bits is not None

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # group_ids will be parsed after we know max_group from teacher dimensions

    # -------------------------------------------------------------------------
    # 输入 / 输出数据配对
    # -------------------------------------------------------------------------
    input_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {args.dataset_dir}")

    if args.output_dataset_dir:
        output_files = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(args.output_dataset_dir, "*.pt"))
        }
        paired = [(inp, output_files.get(os.path.basename(inp))) for inp in input_files]
        paired = [pair for pair in paired if pair[1] is not None]
        if not paired:
            raise FileNotFoundError(
                f"No matching .pt files between {args.dataset_dir} and {args.output_dataset_dir}"
            )
        input_files = [p[0] for p in paired]
        output_files = [p[1] for p in paired]
        print(f"Found {len(paired)} paired input/output .pt files")
        n_eval = estimate_eval_files_needed(input_files, args.eval_size)
        train_input_files = input_files[:-n_eval]
        test_input_files = input_files[-n_eval:]
        train_output_files = output_files[:-n_eval]
        test_output_files = output_files[-n_eval:]
        print(f"  using {len(train_input_files)} files for calibration, {len(test_input_files)} files for eval")
    else:
        n_eval = estimate_eval_files_needed(input_files, args.eval_size)
        print(f"Found {len(input_files)} .pt input files, "
              f"using {len(input_files) - n_eval} files for calibration, {n_eval} files for eval")
        train_input_files = input_files[:-n_eval]
        test_input_files = input_files[-n_eval:]
        train_output_files = None
        test_output_files = None

    teacher, hidden_size, intermediate_size = load_real_teacher(args.teacher_weight_path, device)
    print(f"Teacher: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    max_group = hidden_size // args.group_size
    group_ids = parse_group_ids(args.group_ids, max_group)
    print(f"Replacing {len(group_ids)} groups: {group_ids}")

    print("\nCollecting calibration / evaluation samples ...")
    calib_x, calib_y, eval_x, eval_y = collect_calibration_and_eval(
        train_input_files, test_input_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
        train_output_files, test_output_files,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Check for resume
    addresses = {}
    luts = {}
    group_means = {}
    completed_groups = set()

    if args.resume:
        print("\n[Resume] checking for existing checkpoints...")
        for gid in group_ids:
            ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
            if ckpt_path.exists():
                print(f"  Found checkpoint for group {gid}, loading...")
                ckpt = torch.load(ckpt_path, map_location="cpu")
                addresses[gid] = ckpt["addresses"]
                luts[gid] = [LUTGroup(num_tables=a.num_tables, num_entries=a.num_entries,
                                      group_size=args.group_size, init_table=t, device=device)
                            for a, t in zip(ckpt["addresses"], ckpt["lut_tables"])]
                group_means[gid] = ckpt.get("group_mean", None)
                if group_means[gid] is not None:
                    group_means[gid] = group_means[gid].to(device)
                completed_groups.add(gid)
        print(f"[Resume] {len(completed_groups)}/{len(group_ids)} groups already completed")

    group_metrics = []

    for gid in group_ids:
        if gid in completed_groups:
            print(f"\n[Group {gid}] skipping (already completed)")
            continue

        print(f"\n[Group {gid}] building LUT ...")
        if use_coarse_residual:
            address_list, lut_list, group_mean = build_coarse_residual_luts_for_group(
                calib_x, calib_y, gid, args.group_size, args, device
            )
        else:
            address_list, lut_list, group_mean = build_lut_for_group(
                calib_x, calib_y, gid, args.group_size, args, device
            )
        addresses[gid] = address_list
        luts[gid] = lut_list
        group_means[gid] = group_mean

        # Save immediately after building this group
        ckpt = {
            "group_id": gid,
            "group_size": args.group_size,
            "num_bits": args.num_bits,
            "coarse_num_bits": args.coarse_num_bits,
            "residual_num_bits": args.residual_num_bits,
            "third_num_bits": args.third_num_bits if args.triple_residual else None,
            "num_bins": args.num_bins,
            "channels_per_bit": args.channels_per_bit,
            "address_mode": args.address_mode,
            "num_tables": args.num_tables,
            "target_mode": args.target_mode,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "finetune_batch_size": args.finetune_batch_size,
            "finetune_loss_mode": args.finetune_loss_mode,
            "finetune_cosine_alpha": args.finetune_cosine_alpha,
            "addresses": address_list,
            "lut_tables": [lut.table.detach().cpu().half() for lut in lut_list],
            "group_mean": group_mean.detach().cpu().half() if group_mean is not None else None,
        }
        if len(address_list) == 1:
            ckpt["address"] = address_list[0]
            ckpt["lut_table"] = ckpt["lut_tables"][0]

        ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"  [Group {gid}] checkpoint saved: {ckpt_path}")

        # Release memory after saving
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [Group {gid}] memory released")

    if args.finetune_epochs > 0:
        finetune_luts(addresses, luts, group_means, args.target_mode, calib_x, calib_y, group_ids, args.group_size, device, args)

    for gid in group_ids:
        ckpt = {
            "group_id": gid,
            "group_size": args.group_size,
            "num_bits": args.num_bits,
            "coarse_num_bits": args.coarse_num_bits,
            "residual_num_bits": args.residual_num_bits,
            "num_bins": args.num_bins,
            "channels_per_bit": args.channels_per_bit,
            "address_mode": args.address_mode,
            "num_tables": args.num_tables,
            "target_mode": args.target_mode,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "finetune_batch_size": args.finetune_batch_size,
            "finetune_loss_mode": args.finetune_loss_mode,
            "finetune_cosine_alpha": args.finetune_cosine_alpha,
            "addresses": addresses[gid],
            "lut_tables": [lut.table.detach().cpu().half() for lut in luts[gid]],  # FP16 for storage
            "group_mean": group_means[gid].detach().cpu().half() if group_means[gid] is not None else None,
        }
        # 保持单 LUT 模式下旧字段名兼容
        if len(addresses[gid]) == 1:
            ckpt["address"] = addresses[gid][0]
            ckpt["lut_table"] = ckpt["lut_tables"][0]

        ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

    print("\n[Group evaluation] ...")
    for gid in group_ids:
        metrics = evaluate_group(addresses[gid], luts[gid], group_means[gid], args.target_mode, eval_x, eval_y, gid, args.group_size, device)
        print(f"  group {gid}: cos_sim={metrics['cosine_similarity']:.4f}, "
              f"rel_l2={metrics['relative_l2']:.2%}, rel_mse={metrics['relative_mse']:.4f}")
        group_metrics.append({"group_id": gid, "num_luts": len(addresses[gid]), **metrics})

    print("\n[Full output] evaluating all replaced groups together ...")
    full_metrics = evaluate_full_output(addresses, luts, group_means, args.target_mode, group_ids, eval_x, eval_y, args.group_size, device)
    print(f"  full output: cos_sim={full_metrics['cosine_similarity']:.4f}, "
          f"rel_l2={full_metrics['relative_l2']:.2%}, "
          f"norm_ratio={full_metrics['norm_ratio']:.4f}, "
          f"cos_p50={full_metrics['cosine_similarity_p50']:.4f}")

    replaced_channels = len(group_ids) * args.group_size

    total_ffn_mac = 3 * hidden_size * intermediate_size
    if replaced_channels >= hidden_size:
        # 完整 FFN 替换：跳过 gate_proj / up_proj / down_proj / SiLU
        saved_mac = total_ffn_mac
        mac_reduction_ratio = 1.0
        full_ffn_replacement = True
    else:
        saved_mac = replaced_channels * intermediate_size  # skip down_proj slice for replaced groups
        mac_reduction_ratio = saved_mac / total_ffn_mac
        full_ffn_replacement = False

    table_bytes_per_group = {str(gid): table_storage_bytes_for_group(addresses[gid], args.group_size) for gid in group_ids}
    total_table_bytes = sum(table_bytes_per_group.values())
    num_entries_per_group = {str(gid): sum(addr.num_entries * addr.num_tables for addr in addresses[gid]) for gid in group_ids}

    summary = {
        "teacher_weight_path": args.teacher_weight_path,
        "dataset_dir": args.dataset_dir,
        "output_dataset_dir": args.output_dataset_dir,
        "output_root": args.output_root,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "group_size": args.group_size,
        "group_ids": group_ids,
        "address_mode": args.address_mode,
        "num_tables": args.num_tables,
        "num_bins": args.num_bins,
        "num_bits": args.num_bits,
        "coarse_num_bits": args.coarse_num_bits,
        "residual_num_bits": args.residual_num_bits,
        "channels_per_bit": args.channels_per_bit,
        "tree_candidates": args.tree_candidates,
        "tree_min_samples": args.tree_min_samples,
        "tree_max_samples": args.tree_max_samples,
        "target_mode": args.target_mode,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_loss_mode": args.finetune_loss_mode,
        "finetune_cosine_alpha": args.finetune_cosine_alpha,
        "num_entries_per_group": num_entries_per_group,
        "actual_leaves_per_group": {str(gid): count_leaves_for_group(addresses[gid]) for gid in group_ids},
        "table_bytes_per_group": table_bytes_per_group,
        "total_table_bytes": total_table_bytes,
        "total_table_mib": total_table_bytes / (1024 * 1024),
        "replaced_channels": replaced_channels,
        "replaced_ratio": replaced_channels / hidden_size,
        "full_ffn_replacement": full_ffn_replacement,
        "saved_mac_per_token": saved_mac,
        "total_ffn_mac_per_token": total_ffn_mac,
        "mac_reduction_ratio": mac_reduction_ratio,
        "group_metrics": group_metrics,
        "full_output_metrics": full_metrics,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
    print(f"Total LUT storage: {summary['total_table_mib']:.2f} MiB")
    if full_ffn_replacement:
        print(f"Full FFN replacement: bypassing {saved_mac:,} MAC/token (100.00%)")
    else:
        print(f"Theoretical MAC reduction: {mac_reduction_ratio:.2%}")


if __name__ == "__main__":
    main()
