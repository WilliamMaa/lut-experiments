#!/usr/bin/env python3
"""
build_lut_ffn_output.py

v6 纠偏后的第一个最小实验：单层单专家 FFN output group LUT。

- 不最近邻搜索
- 不 JVP / Jacobian
- 固定 O(1) tree address 查表
- 先验证 LUT 对真实 FFN 输出的近似能力

对应设计文档：LLM_LUT/v6/docs/00-ideas.md
"""

import os
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
              max_samples: int = 65536):
        N = x.shape[0]
        if N > max_samples:
            perm = torch.randperm(N, device=x.device)[:max_samples]
            x = x[perm]
            target = target[perm]
        self._leaf_counter = 0
        self.root = self._build_node(x, target, depth=0,
                                     num_candidates=num_candidates,
                                     min_samples=min_samples)
        self._build_lookup_arrays()

    def _build_node(self, x, target, depth, num_candidates, min_samples):
        N = x.shape[0]
        if depth >= self.num_bits or N < 2 * min_samples:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        parent_var = target.var(dim=0).sum().item()
        if parent_var < 1e-12:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        best_reduction = -1.0
        best_ch = None
        best_signs = None
        best_threshold = None
        best_left_mask = None

        for _ in range(num_candidates):
            ch = torch.randint(0, self.input_dim, (self.channels_per_bit,), generator=self.gen).to(x.device)
            signs = (torch.randint(0, 2, (self.channels_per_bit,), generator=self.gen).float() * 2 - 1).to(x.device)
            proj = (x[:, ch] * signs.to(x.dtype)).sum(dim=-1)
            threshold = proj.median().item()
            left_mask = proj <= threshold
            right_mask = ~left_mask
            n_l = int(left_mask.sum().item())
            n_r = N - n_l
            if n_l < min_samples or n_r < min_samples:
                continue
            var_l = target[left_mask].var(dim=0).sum().item()
            var_r = target[right_mask].var(dim=0).sum().item()
            reduction = parent_var - (n_l * var_l + n_r * var_r) / N
            if reduction > best_reduction:
                best_reduction = reduction
                best_ch = ch
                best_signs = signs
                best_threshold = threshold
                best_left_mask = left_mask

        if best_reduction <= 0 or best_left_mask is None:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        left = self._build_node(x[best_left_mask], target[best_left_mask],
                                depth + 1, num_candidates, min_samples)
        right = self._build_node(x[~best_left_mask], target[~best_left_mask],
                                 depth + 1, num_candidates, min_samples)
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
# 3. 数据流：复用 exp11.py 的 .pt 输入格式
#    可选：如果目录里已有预计算的 FFN 输出，直接读 output，不再 forward Teacher
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

    seed = args.seed * 10000 + group_id

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
            num_bits=args.num_bits,
            channels_per_bit=args.channels_per_bit,
            seed=seed,
        )
        address.fit_calibration(calib_x.unsqueeze(0))
    else:  # tree
        address = AddressGreedyTree(
            input_dim=calib_x.shape[-1],
            num_bits=args.num_bits,
            channels_per_bit=args.channels_per_bit,
            seed=seed,
        )
        print(f"  building tree for group {group_id} ...")
        t0 = time.time()
        address.build(
            calib_x, group_target_for_lut,
            num_candidates=args.tree_candidates,
            min_samples=args.tree_min_samples,
            max_samples=args.tree_max_samples,
        )
        print(f"  tree built in {time.time() - t0:.2f}s, leaves={address._leaf_counter}")

    indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
    lut = LUTGroup(
        num_tables=address.num_tables,
        num_entries=address.num_entries,
        group_size=group_size,
        device=calib_x.device,
    )
    lut.initialize_from_calibration(indices, group_target_for_lut)
    return address, lut, group_mean


@torch.no_grad()
def evaluate_group(address, lut, group_mean, target_mode, eval_x, eval_y, group_id, group_size, device):
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    lut = lut.to(device)
    indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
    pred_group = lut(indices).squeeze(1)

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
def evaluate_full_output(addresses, luts, group_means, target_mode, group_ids, eval_x, eval_y, group_size, device):
    """把所有被替换 group 拼回完整 FFN 输出，再和真实输出比。"""
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    pred_y = eval_y.clone()

    for gid in group_ids:
        address = addresses[gid]
        lut = luts[gid].to(device)
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
        pred_group = lut(indices).squeeze(1)

        g_start = gid * group_size
        g_end = g_start + group_size

        if target_mode == "residual_input":
            pred_group = pred_group + eval_x[:, g_start:g_end]
        elif target_mode == "residual_mean":
            pred_group = pred_group + group_means[gid].to(device)

        pred_y[:, g_start:g_end] = pred_group

    mse = F.mse_loss(pred_y, eval_y).item()
    rmse = math.sqrt(mse)
    var = eval_y.var().item()
    rel_mse = mse / (var + 1e-8)
    rel_l2 = torch.norm(pred_y - eval_y).item() / (torch.norm(eval_y).item() + 1e-8)
    cos_sim = F.cosine_similarity(pred_y, eval_y, dim=-1).mean().item()
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
    }


def compute_lut_storage_bytes(num_entries, group_size, num_tables=1):
    """按 FP16 表值估算。"""
    return num_tables * num_entries * group_size * 2


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
        description="Build single-expert FFN output group LUT (v6 corrected direction)."
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
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--address_mode", type=str, default="tree", choices=["tree", "high_order", "2d"],
                        help="Address generator mode")
    parser.add_argument("--num_tables", type=int, default=1,
                        help="Number of tables for high_order address (total entries = num_tables * 2^num_bits)")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Bins per axis for 2d address (entries = num_bins^2)")
    parser.add_argument("--tree_candidates", type=int, default=64)
    parser.add_argument("--tree_min_samples", type=int, default=16)
    parser.add_argument("--tree_max_samples", type=int, default=65536,
                        help="Subsample calibration data for tree building")
    parser.add_argument("--target_mode", type=str, default="direct", choices=["direct", "residual_mean", "residual_input"],
                        help="direct: LUT stores full FFN output; residual_mean: LUT stores output - group_mean; residual_input: LUT stores output - input_residual")
    parser.add_argument("--calib_size", type=int, default=65536)
    parser.add_argument("--eval_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tree_max_samples <= 0:
        raise ValueError(f"--tree_max_samples must be positive, got {args.tree_max_samples}")
    if args.tree_min_samples <= 0:
        raise ValueError(f"--tree_min_samples must be positive, got {args.tree_min_samples}")
    if args.calib_size <= 0:
        raise ValueError(f"--calib_size must be positive, got {args.calib_size}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")

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
        train_input_files = input_files[:-100]
        test_input_files = input_files[-100:]
        train_output_files = output_files[:-100]
        test_output_files = output_files[-100:]
    else:
        print(f"Found {len(input_files)} .pt input files, "
              f"using first {len(input_files)-100} for calibration, last 100 for eval")
        train_input_files = input_files[:-100]
        test_input_files = input_files[-100:]
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

    addresses = {}
    luts = {}
    group_means = {}
    group_metrics = []

    for gid in group_ids:
        print(f"\n[Group {gid}] building LUT ...")
        address, lut, group_mean = build_lut_for_group(calib_x, calib_y, gid, args.group_size, args, device)
        metrics = evaluate_group(address, lut, group_mean, args.target_mode, eval_x, eval_y, gid, args.group_size, device)
        print(f"  group {gid}: cos_sim={metrics['cosine_similarity']:.4f}, "
              f"rel_l2={metrics['relative_l2']:.2%}, rel_mse={metrics['relative_mse']:.4f}")

        addresses[gid] = address
        luts[gid] = lut
        group_means[gid] = group_mean
        group_metrics.append({"group_id": gid, **metrics})

        ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
        torch.save({
            "group_id": gid,
            "group_size": args.group_size,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "target_mode": args.target_mode,
            "address": address,
            "lut_table": lut.table.detach().cpu().half(),  # FP16 for storage
            "group_mean": group_mean.detach().cpu().half() if group_mean is not None else None,
        }, ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

    print("\n[Full output] evaluating all replaced groups together ...")
    full_metrics = evaluate_full_output(addresses, luts, group_means, args.target_mode, group_ids, eval_x, eval_y, args.group_size, device)
    print(f"  full output: cos_sim={full_metrics['cosine_similarity']:.4f}, "
          f"rel_l2={full_metrics['relative_l2']:.2%}")

    num_entries = 2 ** args.num_bits
    num_tables = addresses[group_ids[0]].num_tables
    table_bytes_per_group = compute_lut_storage_bytes(num_entries, args.group_size, num_tables=num_tables)
    total_table_bytes = table_bytes_per_group * len(group_ids)
    replaced_channels = len(group_ids) * args.group_size

    total_ffn_mac = 3 * hidden_size * intermediate_size
    saved_mac = replaced_channels * intermediate_size  # skip down_proj slice for replaced groups
    mac_reduction_ratio = saved_mac / total_ffn_mac

    def count_leaves(addr):
        if isinstance(addr, AddressGreedyTree):
            return addr._leaf_counter
        return addr.num_entries

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
        "num_tables": num_tables,
        "num_bins": args.num_bins,
        "num_bits": args.num_bits,
        "channels_per_bit": args.channels_per_bit,
        "tree_candidates": args.tree_candidates,
        "tree_min_samples": args.tree_min_samples,
        "tree_max_samples": args.tree_max_samples,
        "target_mode": args.target_mode,
        "num_entries_per_group": num_entries * num_tables,
        "actual_leaves_per_group": {str(gid): count_leaves(addresses[gid]) for gid in group_ids},
        "table_bytes_per_group": table_bytes_per_group,
        "total_table_bytes": total_table_bytes,
        "total_table_mib": total_table_bytes / (1024 * 1024),
        "replaced_channels": replaced_channels,
        "replaced_ratio": replaced_channels / hidden_size,
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
    print(f"Theoretical MAC reduction: {mac_reduction_ratio:.2%}")


if __name__ == "__main__":
    main()
