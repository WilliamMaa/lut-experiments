#!/usr/bin/env python3
"""
build_lut_ffn_output_v3_lowrank.py
结构版：共享 global coarse (2048维) + 分组 residual (64维) + per-coarse-leaf 低秩修正

在 v3 shared_coarse 基础上，为每个 coarse leaf 增加一个低秩线性修正：

    y_full = y_lut + A_ell(x) * V^T * x

其中 V ∈ R^(2048×r) 为共享输入投影，A_ell ∈ R^(2048×r) 为每个 coarse leaf 的输出修正。
这能捕捉 leaf 内的连续变化和跨组交互的共享低维结构。

核心设计：
- 完整 2048 维 FFN 输出作为 coarse 目标
- 一棵 shared global coarse tree，一张 2048 维 coarse table
- 计算完整 residual 后，32 组各自独立建 residual tree
- 用全局 residual PCA 初始化 V，用 ridge regression 初始化 A_ell
- 最后联合 finetune（coarse + residual + lowrank）

默认运行指令：
python -u build_lut_ffn_output_v3_shared_coarse.py \
    --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
    --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/output \
    --output_root ./outputs_ffn_lut_layer39_full_moe_v3_shared \
    --group_size 64 \
    --group_ids "0-31" \
    --coarse_num_bits 14 \
    --residual_num_bits 16 \
    --coarse_finetune_epochs 10 \
    --residual_finetune_epochs 10 \
    --finetune_epochs 50 \
    --finetune_loss_mode multi \
    --finetune_cosine_alpha 1.0 \
    --finetune_residual_cosine_alpha 0.5 \
    --finetune_norm_alpha 0.01 \
    --tree_max_samples 400000 \
    --tree_min_samples 4 \
    --tree_candidates 256 \
    --calib_size 400000 \
    --eval_size 100000 \
    --device cuda:5 \
    > v3_shared.log 2>&1 &
"""

import os
import gc
import glob
import json
import math
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict, Tuple

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
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    gate_key = next(k for k in state_dict.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = state_dict[gate_key].shape
    expert = QwenMoEExpert(hidden_size, intermediate_size)
    clean_state_dict = {k.split("expert.")[-1] if "expert." in k else k: v for k, v in state_dict.items()}
    expert.load_state_dict(clean_state_dict, strict=False)
    expert.to(device).eval()
    return expert, hidden_size, intermediate_size


# =============================================================================
# 2. LUT 核心：tree address + 均值表
# =============================================================================
class _TreeNode:
    __slots__ = ("node_id", "channel_idx", "signs", "threshold",
                 "left", "right", "is_leaf", "leaf_index", "parent")

    def __init__(self, channel_idx=None, signs=None, threshold=None,
                 left=None, right=None, is_leaf=False, leaf_index=None, node_id=None, parent=None):
        self.node_id = node_id
        self.channel_idx = channel_idx
        self.signs = signs
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_index = leaf_index
        self.parent = parent


class AddressGreedyTree:
    """Data-dependent decision-tree address。"""

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
        self.root = self._build_node(x, target, depth=0, parent=None,
                                     num_candidates=num_candidates,
                                     min_samples=min_samples,
                                     candidate_chunk_size=candidate_chunk_size)
        self._build_lookup_arrays()

    def _build_node(self, x, target, depth, parent, num_candidates, min_samples, candidate_chunk_size):
        N = x.shape[0]
        if depth >= self.num_bits or N < 2 * min_samples:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter, parent=parent)
            self._leaf_counter += 1
            return leaf

        parent_var = target.var(dim=0, unbiased=False).sum().item()
        if parent_var < 1e-12:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter, parent=parent)
            self._leaf_counter += 1
            return leaf

        gs = target.shape[1]
        total_sum = target.sum(dim=0)
        total_sum_sq = (target ** 2).sum(dim=0)

        best_reduction = -1.0
        best_ch = None
        best_signs = None
        best_threshold = None
        best_left_mask = None

        for start in range(0, num_candidates, candidate_chunk_size):
            end = min(start + candidate_chunk_size, num_candidates)
            csize = end - start

            ch = torch.randint(
                0, self.input_dim, (csize, self.channels_per_bit), generator=self.gen
            ).to(x.device)
            signs = (
                torch.randint(0, 2, (csize, self.channels_per_bit), generator=self.gen).float() * 2 - 1
            ).to(x.device)

            selected = x[:, ch]
            proj = (selected * signs.to(x.dtype)).sum(dim=-1)
            thresholds = proj.median(dim=0).values
            left_mask = (proj <= thresholds.unsqueeze(0)).t().contiguous()
            n_l = left_mask.sum(dim=1).float()
            n_r = N - n_l
            valid = (n_l >= min_samples) & (n_r >= min_samples)
            if not valid.any():
                continue

            left_mask_f = left_mask.float()
            left_sum = torch.matmul(left_mask_f, target.float())
            left_sum_sq = torch.matmul(left_mask_f, (target ** 2).float())
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

        relative_gain = best_reduction / (parent_var + 1e-12) if parent_var > 0 else 0
        if best_reduction <= 0 or best_left_mask is None:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter, parent=parent)
            self._leaf_counter += 1
            return leaf
        if self.min_relative_gain > 0 and relative_gain < self.min_relative_gain:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter, parent=parent)
            self._leaf_counter += 1
            return leaf

        left = self._build_node(x[best_left_mask], target[best_left_mask],
                                depth + 1, parent, num_candidates, min_samples, candidate_chunk_size)
        right = self._build_node(x[~best_left_mask], target[~best_left_mask],
                                 depth + 1, parent, num_candidates, min_samples, candidate_chunk_size)
        node = _TreeNode(
            channel_idx=best_ch,
            signs=best_signs,
            threshold=best_threshold,
            left=left,
            right=right,
        )
        left.parent = node
        right.parent = node
        return node

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


class LUTGroup(nn.Module):
    """可训练 LUT 表。"""

    def __init__(self, num_tables: int, num_entries: int, output_dim: int,
                 init_table: torch.Tensor = None, device: torch.device = None):
        super().__init__()
        self.num_tables = num_tables
        self.num_entries = num_entries
        self.output_dim = output_dim

        if init_table is not None:
            table = init_table.float().clone()
        else:
            table = torch.zeros(num_tables, num_entries, output_dim)
        if device is not None:
            table = table.to(device)
        self.table = nn.Parameter(table)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert indices.shape[-1] == self.num_tables
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, self.num_tables).to(self.table.device)
        outs = []
        for m in range(self.num_tables):
            t = self.table[m]
            idx_m = flat_idx[:, m].clamp(0, self.num_entries - 1)
            outs.append(t[idx_m])
        out = torch.stack(outs, dim=1).sum(dim=1)
        if len(orig_shape) == 2:
            return out
        return out.view(orig_shape[0], orig_shape[1], self.output_dim)

    def initialize_from_calibration(self, indices: torch.Tensor, targets: torch.Tensor):
        """普通均值初始化，只遍历实际使用的叶子。"""
        with torch.no_grad():
            M = self.num_tables
            E = self.num_entries
            out_dim = self.output_dim
            device = self.table.device
            indices = indices.to(device)
            targets = targets.to(device)

            new_table = torch.zeros(M, E, out_dim, device=device, dtype=torch.float32)
            counts = torch.zeros(M, E, device=device, dtype=torch.float32)

            for m in range(M):
                idx_m = indices[:, m].clamp(0, E - 1)
                idx_exp = idx_m.unsqueeze(1).expand(-1, out_dim)
                new_table[m].scatter_add_(0, idx_exp, targets.float())
                counts[m].scatter_add_(0, idx_m, torch.ones_like(idx_m, dtype=torch.float32))

            # 只遍历有样本的叶子
            occupied = (counts > 0).nonzero(as_tuple=False)
            for m, e in occupied:
                m, e = int(m), int(e)
                new_table[m, e] /= counts[m, e]

            self.table.copy_(new_table)


# =============================================================================
# Low-rank correction helpers
# =============================================================================
def _predict_base(coarse_lut, coarse_address, residual_luts, residual_addresses,
                  x, group_ids, group_size, device):
    """Compute base prediction = coarse + per-group residual for x [N, hidden]."""
    x = x.to(device)
    coarse_indices = coarse_address.compute_indices(x.unsqueeze(0)).view(-1, coarse_address.num_tables)
    coarse_full = coarse_lut(coarse_indices).to(device)
    pred_y = torch.zeros_like(coarse_full)
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        coarse_group = coarse_full[:, g_start:g_end]
        residual_indices = residual_addresses[gid].compute_indices(x.unsqueeze(0)).view(
            -1, residual_addresses[gid].num_tables
        )
        residual_group = residual_luts[gid](residual_indices).to(device)
        pred_y[:, g_start:g_end] = coarse_group + residual_group
    return pred_y


def init_lowrank_correction(
    coarse_address, coarse_lut, residual_luts, residual_addresses,
    calib_x, calib_y, group_ids, group_size,
    rank: int, reg: float, min_samples: int, max_samples: int, device: torch.device,
):
    """
    Initialize shared projection V and per-coarse-leaf output correction A_l.
    V from global residual PCA; A_l from ridge regression per leaf.
    Returns (V, A_table) as trainable Parameters.
    """
    print(f"\n[Low-rank Init] rank={rank}, reg={reg}, min_samples={min_samples}, max_samples={max_samples}")
    N, hidden = calib_x.shape
    if N > max_samples:
        perm = torch.randperm(N)[:max_samples]
        calib_x = calib_x[perm]
        calib_y = calib_y[perm]
        N = max_samples
    calib_x = calib_x.to(device)
    calib_y = calib_y.to(device)

    with torch.no_grad():
        base_pred = _predict_base(
            coarse_lut, coarse_address, residual_luts, residual_addresses,
            calib_x, group_ids, group_size, device
        )
        residual = calib_y - base_pred  # [N, hidden]
        print(f"  residual norm mean={residual.abs().mean():.6f}")

    # Global PCA on residual -> V
    residual_f = residual.float().cpu()
    residual_centered = residual_f - residual_f.mean(dim=0, keepdim=True)
    U, S, Vh = torch.svd(residual_centered)
    V_tensor = Vh[:, :rank].to(device=device, dtype=calib_x.dtype)  # [hidden, rank]
    explained = S[:rank].pow(2).sum() / S.pow(2).sum()
    print(f"  PCA rank-{rank} explained variance: {explained:.4f}")

    # Per-leaf ridge regression for A_l
    with torch.no_grad():
        z = calib_x @ V_tensor  # [N, rank]
        leaf_ids = coarse_address.compute_indices(calib_x.unsqueeze(0)).view(-1)  # [N]
        num_leaves = coarse_address.num_entries

        A_table = torch.zeros(num_leaves, hidden, rank, device=device, dtype=calib_x.dtype)
        counts = torch.zeros(num_leaves, device=device, dtype=torch.long)
        for i in range(N):
            leaf = int(leaf_ids[i].item())
            counts[leaf] += 1

        # Process leaves that have enough samples; small leaves stay zero (fall back to no correction)
        for leaf in range(num_leaves):
            if counts[leaf] < min_samples:
                continue
            mask = (leaf_ids == leaf)
            z_leaf = z[mask].float()      # [n, rank]
            r_leaf = residual[mask].float()  # [n, hidden]
            ztz = z_leaf.t() @ z_leaf    # [rank, rank]
            ztz += reg * torch.eye(rank, device=ztz.device, dtype=z_leaf.dtype)
            ztr = z_leaf.t() @ r_leaf    # [rank, hidden]
            A_leaf = torch.linalg.solve(ztz, ztr).t().to(calib_x.dtype)  # [hidden, rank]
            A_table[leaf] = A_leaf

        n_fitted = (counts >= min_samples).sum().item()
        print(f"  Fitted A_l for {n_fitted}/{num_leaves} leaves (>= {min_samples} samples)")

    V = nn.Parameter(V_tensor, requires_grad=True)
    A_table = nn.Parameter(A_table, requires_grad=True)
    return V, A_table


def apply_lowrank_correction(x, coarse_address, V, A_table):
    """
    Apply per-coarse-leaf low-rank correction.
    x: [N, hidden] or [B, S, hidden]
    Returns correction of same shape.
    """
    if V is None or A_table is None:
        return 0.0
    orig_shape = x.shape
    orig_dtype = x.dtype
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, S, hidden = x.shape
    x_flat = x.reshape(B * S, hidden).float()
    V_f = V.float()
    A_f = A_table.float()
    z = x_flat @ V_f  # [B*S, rank]
    leaf_ids = coarse_address.compute_indices(x).view(-1)  # [B*S]
    A_ell = A_f[leaf_ids]  # [B*S, hidden, rank]
    correction = torch.bmm(A_ell, z.unsqueeze(-1)).squeeze(-1)  # [B*S, hidden]
    return correction.view(orig_shape).to(dtype=orig_dtype)


# =============================================================================
# 3. 数据流
# =============================================================================
def collect_calibration_and_eval(
    input_files, output_files, teacher,
    calib_size: int, eval_size: int, batch_size: int, device: torch.device,
):
    """
    收集所有文件数据，加载到内存后随机分割。
    适用于文件数量少但每个文件样本多的情况。
    """
    weight_dtype = next(teacher.parameters()).dtype
    use_precomputed_outputs = output_files is not None
    
    total_needed = calib_size + eval_size
    
    all_inputs = []
    all_targets = []
    collected = 0
    
    pbar = tqdm(total=total_needed, desc="loading data", unit="sample")
    
    for idx, in_path in enumerate(sorted(input_files)):
        if collected >= total_needed:
            break
            
        try:
            x_tensor = torch.load(in_path, map_location="cpu", weights_only=False)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            elif x_tensor.dim() != 2:
                print(f"  skip {in_path}: shape {x_tensor.shape}")
                continue
        except Exception as e:
            print(f"  skip {in_path}: {e}")
            continue

        if use_precomputed_outputs:
            out_path = output_files[idx]
            try:
                y_tensor = torch.load(out_path, map_location="cpu", weights_only=False)
                if y_tensor.dim() == 1:
                    y_tensor = y_tensor.unsqueeze(0)
                elif y_tensor.dim() != 2:
                    print(f"  skip {out_path}: shape {y_tensor.shape}")
                    continue
            except Exception as e:
                print(f"  skip {out_path}: {e}")
                continue
            if y_tensor.shape != x_tensor.shape:
                print(f"  skip pair: shape mismatch {x_tensor.shape} vs {y_tensor.shape}")
                continue
        else:
            y_tensor = None

        # 分批处理大文件
        n_samples = x_tensor.shape[0]
        for start in range(0, n_samples, batch_size):
            if collected >= total_needed:
                break
            end = min(start + batch_size, n_samples)
            
            x_batch = x_tensor[start:end].to(device, dtype=weight_dtype)
            
            if y_tensor is not None:
                y_batch = y_tensor[start:end].float()
            else:
                with torch.no_grad():
                    y_batch = teacher(x_batch).float()
            
            x_batch = x_batch.float().cpu()
            
            all_inputs.append(x_batch)
            all_targets.append(y_batch)
            
            n_take = end - start
            collected += n_take
            pbar.update(n_take)
    
    pbar.close()
    
    if collected < total_needed:
        print(f"[Warning] Only collected {collected} samples, need {total_needed}")
        if collected < calib_size:
            raise RuntimeError(f"Not enough samples: {collected} < calib_size {calib_size}")
    
    # 合并所有数据
    full_x = torch.cat(all_inputs, dim=0)
    full_y = torch.cat(all_targets, dim=0)
    
    # 随机打乱
    perm = torch.randperm(full_x.shape[0])
    full_x = full_x[perm]
    full_y = full_y[perm]
    
    # 分割
    calib_x = full_x[:calib_size]
    calib_y = full_y[:calib_size]
    eval_x = full_x[calib_size:calib_size + eval_size]
    eval_y = full_y[calib_size:calib_size + eval_size]
    
    print(f"\nData loaded: calibration {calib_x.shape}, eval {eval_x.shape}")
    return calib_x, calib_y, eval_x, eval_y


def parse_group_ids(s: str, max_group: int):
    """解析 group_ids，支持逗号分隔和连字符范围。"""
    ids = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
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
# 4. 构建与评估
# =============================================================================
def finetune_coarse_global(coarse_lut, coarse_address, calib_x, calib_y, 
                           device, epochs, lr, batch_size, loss_mode="mse+cosine"):
    """全局 coarse finetune：使用完整 2048 维计算 loss。"""
    if epochs <= 0:
        return

    print(f"\n[Finetune Global Coarse] {epochs} epochs (lr={lr}, loss={loss_mode}) ...")
    coarse_lut.to(device)
    optimizer = torch.optim.Adam([coarse_lut.table], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = calib_x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0}
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = calib_x[idx].to(device)
            yb = calib_y[idx].to(device)  # 完整 2048 维

            optimizer.zero_grad()

            indices = coarse_address.compute_indices(xb.unsqueeze(0)).view(-1, coarse_address.num_tables)
            pred = coarse_lut(indices)

            mse = F.mse_loss(pred, yb)
            cos = F.cosine_similarity(pred, yb, dim=-1).mean()

            if loss_mode == "mse":
                loss = mse
            elif loss_mode == "cosine":
                loss = 1 - cos
            else:
                loss = mse + (1 - cos)

            loss.backward()
            optimizer.step()

            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mse"] += mse.item()
            epoch_metrics["cos"] += cos.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_metrics['loss']/max(n_batches,1):.6e}, "
                  f"cos={epoch_metrics['cos']/max(n_batches,1):.4f}")


def finetune_residual_groups(residual_luts, residual_addresses, calib_x, full_residual, 
                             group_ids, group_size, device, epochs, lr, batch_size, 
                             loss_mode="mse+cosine"):
    """每组 residual 单独 finetune：只使用当前 group 的 64 维计算 local loss。"""
    if epochs <= 0:
        return

    for gid in group_ids:
        print(f"\n[Finetune Residual Group {gid}] {epochs} epochs ...")
        g_start = gid * group_size
        g_end = g_start + group_size
        
        group_residual = full_residual[:, g_start:g_end]
        residual_lut = residual_luts[gid]
        residual_address = residual_addresses[gid]
        
        residual_lut.to(device)
        optimizer = torch.optim.Adam([residual_lut.table], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        n_samples = calib_x.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb = calib_x[idx].to(device)
                yb = group_residual[idx].to(device)

                optimizer.zero_grad()
                indices = residual_address.compute_indices(xb.unsqueeze(0)).view(-1, residual_address.num_tables)
                pred = residual_lut(indices)

                mse = F.mse_loss(pred, yb)
                cos = F.cosine_similarity(pred, yb, dim=-1).mean()

                if loss_mode == "mse":
                    loss = mse
                elif loss_mode == "cosine":
                    loss = 1 - cos
                else:
                    loss = mse + (1 - cos)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_loss/max(n_batches,1):.6e}")


def finetune_joint_all(coarse_lut, coarse_address, residual_luts, residual_addresses,
                       calib_x, calib_y, group_ids, group_size, device, args,
                       lowrank_V=None, lowrank_A=None):
    """
    最终联合 finetune：coarse (2048d) + 所有 residual (64d x 32) + 可选低秩修正 一起优化。
    coarse 只查一次，然后切片给各 group。
    """
    epochs = args.finetune_epochs
    if epochs <= 0:
        return

    print(f"\n[Joint Finetune All] {epochs} epochs ...")
    if lowrank_V is not None:
        print(f"  with lowrank correction (rank={lowrank_V.shape[1]})")
    
    # 正确收集参数：coarse 一次 + 所有 residual + 可选 lowrank
    params = [coarse_lut.table]
    for gid in group_ids:
        params.append(residual_luts[gid].table)
        residual_luts[gid].to(device)
    coarse_lut.to(device)
    if lowrank_V is not None:
        params.append(lowrank_V)
        if lowrank_A is not None:
            params.append(lowrank_A)
        lowrank_V = lowrank_V.to(device)
        if lowrank_A is not None:
            lowrank_A = lowrank_A.to(device)

    optimizer = torch.optim.Adam(params, lr=args.finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = calib_x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "res_cos": 0.0, "norm_ratio": 0.0}
        n_batches = 0

        for start in range(0, n_samples, args.finetune_batch_size):
            idx = perm[start:start + args.finetune_batch_size]
            xb = calib_x[idx].to(device)
            yb = calib_y[idx].to(device)

            optimizer.zero_grad()

            # coarse 只查一次，得到完整 2048 维
            coarse_indices = coarse_address.compute_indices(xb.unsqueeze(0)).view(-1, coarse_address.num_tables)
            coarse_full = coarse_lut(coarse_indices)  # [batch, 2048]

            # 重建完整输出：coarse_slice + residual
            pred_y = torch.zeros_like(yb)
            for gid in group_ids:
                g_start = gid * group_size
                g_end = g_start + group_size

                coarse_group = coarse_full[:, g_start:g_end]
                residual_indices = residual_addresses[gid].compute_indices(xb.unsqueeze(0)).view(-1, residual_addresses[gid].num_tables)
                residual_group = residual_luts[gid](residual_indices)
                
                pred_y[:, g_start:g_end] = coarse_group + residual_group

            # 加上低秩修正
            if lowrank_V is not None:
                pred_y = pred_y + apply_lowrank_correction(xb, coarse_address, lowrank_V, lowrank_A)

            # 完整维度的 loss
            mse = F.mse_loss(pred_y, yb)
            cos_output = F.cosine_similarity(pred_y, yb, dim=-1).mean()
            
            # Residual stream
            pred_residual = xb + pred_y
            true_residual = xb + yb
            cos_residual = F.cosine_similarity(pred_residual, true_residual, dim=-1).mean()

            # Log norm ratio
            pred_norm = torch.norm(pred_y, dim=-1)
            true_norm = torch.norm(yb, dim=-1)
            log_norm_loss = (torch.log((pred_norm + 1e-6) / (true_norm + 1e-6)) ** 2).mean()

            # 组合 loss
            if args.finetune_loss_mode == "mse":
                loss = mse
            elif args.finetune_loss_mode == "cosine":
                loss = 1 - cos_output
            elif args.finetune_loss_mode == "mse+cosine":
                loss = mse + args.finetune_cosine_alpha * (1 - cos_output)
            elif args.finetune_loss_mode == "multi":
                loss = (mse + 
                        args.finetune_cosine_alpha * (1 - cos_output) +
                        args.finetune_residual_cosine_alpha * (1 - cos_residual) +
                        args.finetune_norm_alpha * log_norm_loss)

            loss.backward()
            optimizer.step()

            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mse"] += mse.item()
            epoch_metrics["cos"] += cos_output.item()
            epoch_metrics["res_cos"] += cos_residual.item()
            epoch_metrics["norm_ratio"] += (pred_norm / (true_norm + 1e-6)).mean().item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: "
                  f"cos={epoch_metrics['cos']/max(n_batches,1):.4f}, "
                  f"res_cos={epoch_metrics['res_cos']/max(n_batches,1):.4f}, "
                  f"norm_ratio={epoch_metrics['norm_ratio']/max(n_batches,1):.4f}")


@torch.no_grad()
def evaluate_full_output(coarse_lut, coarse_address, residual_luts, residual_addresses,
                         eval_x, eval_y, group_ids, group_size, device,
                         lowrank_V=None, lowrank_A=None):
    """评估完整输出。coarse 只查一次，然后切片，可选低秩修正。"""
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    
    # coarse 只查一次
    coarse_lut.to(device)
    coarse_indices = coarse_address.compute_indices(eval_x.unsqueeze(0)).view(-1, coarse_address.num_tables)
    coarse_full = coarse_lut(coarse_indices)  # [batch, 2048]
    
    # 组装各 group
    pred_y = torch.zeros_like(eval_y)
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size

        coarse_group = coarse_full[:, g_start:g_end]
        residual_luts[gid].to(device)
        residual_indices = residual_addresses[gid].compute_indices(eval_x.unsqueeze(0)).view(-1, residual_addresses[gid].num_tables)
        residual_group = residual_luts[gid](residual_indices)
        
        pred_y[:, g_start:g_end] = coarse_group + residual_group

    # 低秩修正
    if lowrank_V is not None:
        pred_y = pred_y + apply_lowrank_correction(eval_x, coarse_address, lowrank_V, lowrank_A)

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
    norm_ratio = (pred_norm / (true_norm + 1e-6)).mean().item()

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_mse": rel_mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_mean,
        "cosine_similarity_p10": cos_p10,
        "cosine_similarity_p50": cos_p50,
        "cosine_similarity_p90": cos_p90,
        "norm_ratio": norm_ratio,
    }


# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build LUT FFN (v3_shared_coarse): shared global coarse + group residuals"
    )
    parser.add_argument("--teacher_weight_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dataset_dir", default=None)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--group_ids", type=str, default="0-31")
    parser.add_argument("--coarse_num_bits", type=int, default=14)
    parser.add_argument("--residual_num_bits", type=int, default=16)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--tree_candidates", type=int, default=64)
    parser.add_argument("--tree_min_samples", type=int, default=4)
    parser.add_argument("--tree_max_samples", type=int, default=400000)
    parser.add_argument("--tree_candidate_chunk_size", type=int, default=32)
    parser.add_argument("--tree_min_relative_gain", type=float, default=0.0)
    
    parser.add_argument("--coarse_finetune_epochs", type=int, default=10)
    parser.add_argument("--coarse_finetune_lr", type=float, default=1e-3)
    parser.add_argument("--coarse_finetune_batch_size", type=int, default=1024)
    
    parser.add_argument("--residual_finetune_epochs", type=int, default=10)
    parser.add_argument("--residual_finetune_lr", type=float, default=1e-3)
    parser.add_argument("--residual_finetune_batch_size", type=int, default=1024)
    
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_batch_size", type=int, default=1024)
    parser.add_argument("--finetune_loss_mode", type=str, default="multi",
                        choices=["mse", "cosine", "mse+cosine", "multi"])
    parser.add_argument("--finetune_cosine_alpha", type=float, default=1.0)
    parser.add_argument("--finetune_residual_cosine_alpha", type=float, default=0.5)
    parser.add_argument("--finetune_norm_alpha", type=float, default=0.01)
    
    parser.add_argument("--calib_size", type=int, default=400000)
    parser.add_argument("--eval_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base_checkpoint_dir", type=str, default=None,
                      help="Load existing coarse/residual base checkpoints from this directory (e.g. v3 output) instead of building them. Saves low-rank results to --output_root.")

    # Per-coarse-leaf low-rank correction
    parser.add_argument("--lowrank_rank", type=int, default=8,
                      help="Rank of per-coarse-leaf low-rank correction. 0 to disable.")
    parser.add_argument("--lowrank_reg", type=float, default=1e-3,
                      help="Ridge regularization for initializing A_l.")
    parser.add_argument("--lowrank_min_samples", type=int, default=32,
                      help="Minimum samples per coarse leaf for independent A_l.")
    parser.add_argument("--lowrank_max_samples", type=int, default=200000,
                      help="Max samples used for PCA and per-leaf ridge regression.")
    parser.add_argument("--lowrank_finetune_epochs", type=int, default=10,
                      help="Epochs to jointly finetune low-rank parameters.")
    parser.add_argument("--lowrank_finetune_lr", type=float, default=1e-3,
                      help="Learning rate for low-rank parameters.")
    parser.add_argument("--lowrank_freeze_V", action="store_true",
                      help="Freeze shared projection V after PCA initialization.")
    parser.add_argument("--lowrank_loss_mode", type=str, default="multi",
                        choices=["mse", "multi"],
                        help="Loss mode for lowrank-only finetune.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load data
    input_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {args.dataset_dir}")

    output_files = None
    if args.output_dataset_dir:
        output_files_map = {os.path.basename(p): p for p in glob.glob(os.path.join(args.output_dataset_dir, "*.pt"))}
        output_files = [output_files_map.get(os.path.basename(f)) for f in input_files]
        # 过滤掉没有配对的
        paired_indices = [i for i, o in enumerate(output_files) if o is not None]
        input_files = [input_files[i] for i in paired_indices]
        output_files = [output_files[i] for i in paired_indices]
        print(f"Found {len(input_files)} paired input/output files")

    teacher, hidden_size, intermediate_size = load_real_teacher(args.teacher_weight_path, device)
    print(f"Teacher: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    max_group = hidden_size // args.group_size
    group_ids = parse_group_ids(args.group_ids, max_group)
    print(f"Replacing {len(group_ids)} groups: {group_ids}")

    print("\nCollecting calibration / evaluation samples ...")
    calib_x, calib_y, eval_x, eval_y = collect_calibration_and_eval(
        input_files, output_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # 检查 resume / base checkpoint dir
    base_ckpt_dir = Path(args.base_checkpoint_dir) if args.base_checkpoint_dir else ckpt_dir
    shared_coarse_ckpt = ckpt_dir / "shared_coarse.pt"
    base_shared_coarse_ckpt = base_ckpt_dir / "shared_coarse.pt"
    base_residual_ckpts_exist = all((base_ckpt_dir / f"residual_g{gid}.pt").exists() for gid in group_ids)
    
    lowrank_V = None
    lowrank_A = None

    if (args.resume or args.base_checkpoint_dir) and base_shared_coarse_ckpt.exists() and base_residual_ckpts_exist:
        print(f"\n[Resume] Loading base checkpoints from {base_ckpt_dir} ...")
        coarse_ckpt = torch.load(base_shared_coarse_ckpt, map_location="cpu", weights_only=False)
        coarse_address = coarse_ckpt["address"]
        coarse_lut = LUTGroup(
            num_tables=coarse_address.num_tables,
            num_entries=coarse_address.num_entries,
            output_dim=hidden_size,
            init_table=coarse_ckpt["table"],
            device=device
        )

        # Load low-rank params if present (from base dir, otherwise they will be initialized)
        if "lowrank_V" in coarse_ckpt and args.lowrank_rank > 0:
            lowrank_V = coarse_ckpt["lowrank_V"].to(device)
            lowrank_A = coarse_ckpt["lowrank_A"].to(device)
            print(f"[Resume] Loaded lowrank correction (rank={lowrank_V.shape[1]})")
        elif args.lowrank_rank > 0:
            print("[Resume] No lowrank params found in base checkpoint; will initialize from data.")
        
        residual_addresses = {}
        residual_luts = {}
        for gid in group_ids:
            res_ckpt = torch.load(base_ckpt_dir / f"residual_g{gid}.pt", map_location="cpu", weights_only=False)
            residual_addresses[gid] = res_ckpt["address"]
            residual_luts[gid] = LUTGroup(
                num_tables=residual_addresses[gid].num_tables,
                num_entries=residual_addresses[gid].num_entries,
                output_dim=args.group_size,
                init_table=res_ckpt["table"],
                device=device
            )
        print("[Resume] All base checkpoints loaded.")
    else:
        # =============================================================================
        # Phase 1: Build SHARED GLOBAL coarse tree (2048维)
        # =============================================================================
        print(f"\n{'='*60}")
        print("[Phase 1] Building SHARED GLOBAL coarse tree (2048-dim output)")
        print(f"{'='*60}")
        
        coarse_seed = args.seed * 10000
        coarse_address = AddressGreedyTree(
            input_dim=calib_x.shape[-1],
            num_bits=args.coarse_num_bits,
            channels_per_bit=args.channels_per_bit,
            seed=coarse_seed,
        )
        print(f"  Building {args.coarse_num_bits}-bit coarse tree on full 2048-dim output...")
        t0 = time.time()
        coarse_address.build(
            calib_x, calib_y,
            num_candidates=args.tree_candidates,
            min_samples=args.tree_min_samples,
            max_samples=args.tree_max_samples,
            device=device,
            candidate_chunk_size=args.tree_candidate_chunk_size,
            min_relative_gain=args.tree_min_relative_gain,
        )
        print(f"  Coarse tree built in {time.time() - t0:.2f}s, leaves={coarse_address._leaf_counter}")

        coarse_indices = coarse_address.compute_indices(calib_x.unsqueeze(0)).view(-1, coarse_address.num_tables)
        coarse_lut = LUTGroup(
            num_tables=coarse_address.num_tables,
            num_entries=coarse_address.num_entries,
            output_dim=hidden_size,
            device=calib_x.device,
        )
        coarse_lut.initialize_from_calibration(coarse_indices, calib_y)
        print(f"  Coarse LUT initialized: {coarse_lut.table.shape}")

        # =============================================================================
        # Phase 2: Finetune global coarse ALONE
        # =============================================================================
        finetune_coarse_global(
            coarse_lut, coarse_address, calib_x, calib_y,
            device, args.coarse_finetune_epochs, args.coarse_finetune_lr,
            args.coarse_finetune_batch_size, loss_mode="mse+cosine"
        )

        # Save shared coarse
        torch.save({
            "address": coarse_address,
            "table": coarse_lut.table.detach().cpu().half(),
        }, shared_coarse_ckpt)
        print(f"  Saved shared coarse: {shared_coarse_ckpt}")

        # =============================================================================
        # Phase 3: Compute residual and build group-specific residuals
        # =============================================================================
        print(f"\n{'='*60}")
        print("[Phase 3] Computing residual and building group-specific residual trees")
        print(f"{'='*60}")
        
        with torch.no_grad():
            coarse_pred = coarse_lut(coarse_indices.to(device)).cpu()
            full_residual = calib_y - coarse_pred
            print(f"  Residual computed: mean_abs={full_residual.abs().mean():.6f}")

        residual_addresses = {}
        residual_luts = {}
        
        for gid in group_ids:
            print(f"\n[Group {gid}] Building residual tree...")
            g_start = gid * args.group_size
            g_end = g_start + args.group_size
            group_residual = full_residual[:, g_start:g_end]

            residual_seed = args.seed * 10000 + gid * 100 + 1
            residual_address = AddressGreedyTree(
                input_dim=calib_x.shape[-1],
                num_bits=args.residual_num_bits,
                channels_per_bit=args.channels_per_bit,
                seed=residual_seed,
            )
            
            t0 = time.time()
            residual_address.build(
                calib_x, group_residual,
                num_candidates=args.tree_candidates,
                min_samples=args.tree_min_samples,
                max_samples=args.tree_max_samples,
                device=device,
                candidate_chunk_size=args.tree_candidate_chunk_size,
                min_relative_gain=args.tree_min_relative_gain,
            )
            print(f"  Residual tree built in {time.time() - t0:.2f}s, leaves={residual_address._leaf_counter}")

            residual_indices = residual_address.compute_indices(calib_x.unsqueeze(0)).view(-1, residual_address.num_tables)
            residual_lut = LUTGroup(
                num_tables=residual_address.num_tables,
                num_entries=residual_address.num_entries,
                output_dim=args.group_size,
                device=calib_x.device,
            )
            residual_lut.initialize_from_calibration(residual_indices, group_residual)

            residual_addresses[gid] = residual_address
            residual_luts[gid] = residual_lut

            # Save per-group residual
            torch.save({
                "group_id": gid,
                "address": residual_address,
                "table": residual_lut.table.detach().cpu().half(),
            }, ckpt_dir / f"residual_g{gid}.pt")
            print(f"  Saved residual: {ckpt_dir / f'residual_g{gid}.pt'}")

            gc.collect()
            torch.cuda.empty_cache()

        # =============================================================================
        # Phase 4: Finetune each residual group
        # =============================================================================
        finetune_residual_groups(
            residual_luts, residual_addresses, calib_x, full_residual, 
            group_ids, args.group_size,
            device, args.residual_finetune_epochs, args.residual_finetune_lr,
            args.residual_finetune_batch_size, loss_mode="mse+cosine"
        )

    # =============================================================================
    # Phase 4.5: Initialize per-coarse-leaf low-rank correction
    # =============================================================================
    if args.lowrank_rank > 0 and (lowrank_V is None or lowrank_A is None):
        lowrank_V, lowrank_A = init_lowrank_correction(
            coarse_address, coarse_lut, residual_luts, residual_addresses,
            calib_x, calib_y, group_ids, args.group_size,
            rank=args.lowrank_rank,
            reg=args.lowrank_reg,
            min_samples=args.lowrank_min_samples,
            max_samples=args.lowrank_max_samples,
            device=device,
        )

    # Ensure loaded/initialized low-rank tensors require grad for finetuning
    if lowrank_V is not None:
        lowrank_V.requires_grad_(not args.lowrank_freeze_V)
    if lowrank_A is not None:
        lowrank_A.requires_grad_(True)

    # =============================================================================
    # Phase 5: Joint finetune all
    # =============================================================================
    if args.finetune_epochs > 0:
        print(f"\n{'='*60}")
        print("[Phase 5] Global joint finetune")
        print(f"{'='*60}")
        finetune_joint_all(
            coarse_lut, coarse_address, residual_luts, residual_addresses,
            calib_x, calib_y, group_ids, args.group_size, device, args,
            lowrank_V=lowrank_V, lowrank_A=lowrank_A
        )

    # =============================================================================
    # Phase 5.5: Optional lowrank-only finetune
    # =============================================================================
    if args.lowrank_rank > 0 and args.lowrank_finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"[Phase 5.5] Lowrank-only finetune ({args.lowrank_finetune_epochs} epochs)")
        print(f"{'='*60}")
        params_lr = []
        if lowrank_V is not None and not args.lowrank_freeze_V:
            params_lr.append({"params": [lowrank_V], "lr": args.lowrank_finetune_lr})
        if lowrank_A is not None:
            params_lr.append({"params": [lowrank_A], "lr": args.lowrank_finetune_lr})
        if not params_lr:
            print("  No lowrank params to finetune (skip)")
        else:
            optimizer = torch.optim.Adam(params_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lowrank_finetune_epochs)
            n_samples = calib_x.shape[0]
            for epoch in range(args.lowrank_finetune_epochs):
                perm = torch.randperm(n_samples)
                epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "res_cos": 0.0, "norm_ratio": 0.0}
                n_batches = 0
                for start in range(0, n_samples, args.finetune_batch_size):
                    idx = perm[start:start + args.finetune_batch_size]
                    xb = calib_x[idx].to(device)
                    yb = calib_y[idx].to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        pred_y = _predict_base(coarse_lut, coarse_address, residual_luts, residual_addresses,
                                               xb, group_ids, args.group_size, device)
                    pred_y = pred_y + apply_lowrank_correction(xb, coarse_address, lowrank_V, lowrank_A)

                    mse = F.mse_loss(pred_y, yb)
                    cos_output = F.cosine_similarity(pred_y, yb, dim=-1).mean()
                    pred_residual = xb + pred_y
                    true_residual = xb + yb
                    cos_residual = F.cosine_similarity(pred_residual, true_residual, dim=-1).mean()
                    pred_norm = torch.norm(pred_y, dim=-1)
                    true_norm = torch.norm(yb, dim=-1)
                    log_norm_loss = (torch.log((pred_norm + 1e-6) / (true_norm + 1e-6)) ** 2).mean()

                    if args.lowrank_loss_mode == "mse":
                        loss = mse
                    else:
                        loss = (mse +
                                args.finetune_cosine_alpha * (1 - cos_output) +
                                args.finetune_residual_cosine_alpha * (1 - cos_residual) +
                                args.finetune_norm_alpha * log_norm_loss)

                    loss.backward()
                    optimizer.step()
                    epoch_metrics["loss"] += loss.item()
                    epoch_metrics["mse"] += mse.item()
                    epoch_metrics["cos"] += cos_output.item()
                    epoch_metrics["res_cos"] += cos_residual.item()
                    epoch_metrics["norm_ratio"] += (pred_norm / (true_norm + 1e-6)).mean().item()
                    n_batches += 1
                scheduler.step()
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  epoch {epoch + 1}/{args.lowrank_finetune_epochs}: "
                          f"loss={epoch_metrics['loss']/max(n_batches,1):.6e}, "
                          f"cos={epoch_metrics['cos']/max(n_batches,1):.4f}, "
                          f"res_cos={epoch_metrics['res_cos']/max(n_batches,1):.4f}, "
                          f"norm_ratio={epoch_metrics['norm_ratio']/max(n_batches,1):.4f}")

    # Save final checkpoints
    save_dict = {
        "address": coarse_address,
        "table": coarse_lut.table.detach().cpu().half(),
    }
    if lowrank_V is not None:
        save_dict["lowrank_V"] = lowrank_V.detach().cpu().half()
        save_dict["lowrank_A"] = lowrank_A.detach().cpu().half()
        save_dict["lowrank_rank"] = args.lowrank_rank
    torch.save(save_dict, shared_coarse_ckpt)
    for gid in group_ids:
        torch.save({
            "group_id": gid,
            "address": residual_addresses[gid],
            "table": residual_luts[gid].table.detach().cpu().half(),
        }, ckpt_dir / f"residual_g{gid}.pt")

    # =============================================================================
    # Evaluation
    # =============================================================================
    print("\n[Final Evaluation] ...")
    metrics = evaluate_full_output(
        coarse_lut, coarse_address, residual_luts, residual_addresses,
        eval_x, eval_y, group_ids, args.group_size, device,
        lowrank_V=lowrank_V, lowrank_A=lowrank_A
    )
    print(f"  Full output: cos_sim={metrics['cosine_similarity']:.4f}, "
          f"rel_l2={metrics['relative_l2']:.2%}, "
          f"norm_ratio={metrics['norm_ratio']:.4f}")

    # Summary
    coarse_bytes = coarse_address.num_entries * hidden_size * 2
    residual_bytes = sum(
        residual_addresses[gid].num_entries * args.group_size * 2
        for gid in group_ids
    )
    lowrank_bytes = 0
    if lowrank_V is not None:
        lowrank_bytes = (
            lowrank_V.numel() * 2 +
            lowrank_A.numel() * 2
        )
    total_bytes = coarse_bytes + residual_bytes + lowrank_bytes
    
    summary = {
        "teacher_weight_path": args.teacher_weight_path,
        "hidden_size": hidden_size,
        "group_size": args.group_size,
        "group_ids": group_ids,
        "coarse_num_bits": args.coarse_num_bits,
        "residual_num_bits": args.residual_num_bits,
        "shared_coarse": True,
        "lowrank_rank": args.lowrank_rank,
        "lowrank_frozen_V": args.lowrank_freeze_V,
        "coarse_table_mib": coarse_bytes / (1024 * 1024),
        "residual_tables_mib": residual_bytes / (1024 * 1024),
        "lowrank_tables_mib": lowrank_bytes / (1024 * 1024),
        "total_table_mib": total_bytes / (1024 * 1024),
        "full_output_metrics": metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")
    print(f"  Coarse table: {summary['coarse_table_mib']:.2f} MiB")
    print(f"  Residual tables: {summary['residual_tables_mib']:.2f} MiB")
    print(f"  Lowrank tables: {summary['lowrank_tables_mib']:.2f} MiB")
    print(f"  Total: {summary['total_table_mib']:.2f} MiB")


if __name__ == "__main__":
    main()
