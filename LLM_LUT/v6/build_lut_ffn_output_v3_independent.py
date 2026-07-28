#!/usr/bin/env python3
"""
build_lut_ffn_output_v3_independent.py
修复版：正确的两阶段构建，local loss，全局 joint finetune
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
from typing import Optional, List

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
                 "left", "right", "is_leaf", "leaf_index")

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
    """Data-dependent decision-tree address."""

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
              candidate_chunk_size: int = 32):
        N = x.shape[0]
        if N > max_samples:
            perm = torch.randperm(N, device=x.device)[:max_samples]
            x = x[perm]
            target = target[perm]
        if device is not None:
            x = x.to(device)
            target = target.to(device)
        self._leaf_counter = 0
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

        if best_reduction <= 0 or best_left_mask is None:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        left = self._build_node(x[best_left_mask], target[best_left_mask],
                                depth + 1, num_candidates, min_samples, candidate_chunk_size)
        right = self._build_node(x[~best_left_mask], target[~best_left_mask],
                                 depth + 1, num_candidates, min_samples, candidate_chunk_size)
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
        """普通均值初始化，只遍历实际使用的叶子。"""
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

            # 只遍历有样本的叶子
            occupied = (counts > 0).nonzero(as_tuple=False)
            for m, e in occupied:
                m, e = int(m), int(e)
                new_table[m, e] /= counts[m, e]

            self.table.copy_(new_table)


# =============================================================================
# 3. 数据流
# =============================================================================
def collect_calibration_and_eval(
    train_input_files, test_input_files, teacher,
    calib_size: int, eval_size: int, batch_size: int, device: torch.device,
    train_output_files=None, test_output_files=None,
):
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
            tensor = torch.load(path, map_location="cpu", weights_only=False)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() != 2:
                raise ValueError(f"{name} {path} has shape {tensor.shape}")
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
                    raise ValueError(f"Shape mismatch: {in_path} vs {out_path}")
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
            raise RuntimeError(f"No valid samples found")
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


def finetune_single_lut(lut, address, calib_x, calib_y_group, group_id, group_size, 
                        device, epochs, lr, batch_size, loss_mode="mse"):
    """
    单表 finetune，只使用当前 group 的 64 维计算 local loss。
    """
    if epochs <= 0:
        return

    print(f"\n[Finetune Group {group_id}] {epochs} epochs (lr={lr}, loss={loss_mode}) ...")
    lut.to(device)
    optimizer = torch.optim.Adam([lut.table], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = calib_x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = calib_x[idx].to(device)
            yb = calib_y_group[idx].to(device)

            optimizer.zero_grad()

            # 只预测当前 group
            indices = address.compute_indices(xb.unsqueeze(0)).view(-1, address.num_tables)
            pred = lut(indices).squeeze(1)

            # Local loss: 只在这个 64 维上计算
            mse = F.mse_loss(pred, yb)
            cos = F.cosine_similarity(pred, yb, dim=-1).mean()

            if loss_mode == "mse":
                loss = mse
            elif loss_mode == "cosine":
                loss = 1 - cos
            else:  # mse+cosine
                loss = mse + (1 - cos)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_loss / max(n_batches, 1):.6e}")


def finetune_luts_joint(addresses, luts, group_means, target_mode, calib_x, calib_y,
                        group_ids, group_size, device, args):
    """
    全局 joint finetune：所有 group 一起优化，使用完整 2048 维计算 loss。
    这才可以用 residual stream cosine 和 log norm loss。
    """
    epochs = args.finetune_epochs
    if epochs <= 0:
        return

    print(f"\n[Joint Finetune] All groups together for {epochs} epochs ...")
    print(f"  loss_mode={args.finetune_loss_mode}")

    params = []
    for gid in group_ids:
        for lut in luts[gid]:
            lut.to(device)
            params.append(lut.table)

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

            # 重建完整 2048 维输出
            pred_y = torch.zeros_like(yb)
            for gid in group_ids:
                g_start = gid * group_size
                g_end = g_start + group_size

                pred_group = None
                for address, lut in zip(addresses[gid], luts[gid]):
                    indices = address.compute_indices(xb.unsqueeze(0)).view(-1, address.num_tables)
                    out = lut(indices).squeeze(1)
                    pred_group = out if pred_group is None else pred_group + out

                pred_y[:, g_start:g_end] = pred_group

            # 完整维度的 loss
            mse = F.mse_loss(pred_y, yb)
            cos_output = F.cosine_similarity(pred_y, yb, dim=-1).mean()
            
            # Residual stream
            pred_residual = xb + pred_y
            true_residual = xb + yb
            cos_residual = F.cosine_similarity(pred_residual, true_residual, dim=-1).mean()

            # Log norm ratio loss
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
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: "
                  f"loss={epoch_metrics['loss']:.6e}, "
                  f"mse={epoch_metrics['mse']:.6e}, "
                  f"cos={epoch_metrics['cos']:.4f}, "
                  f"res_cos={epoch_metrics['res_cos']:.4f}, "
                  f"norm_ratio={epoch_metrics['norm_ratio']:.4f}")


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
    }


def compute_lut_storage_bytes(num_entries, group_size, num_tables=1):
    return num_tables * num_entries * group_size * 2


def table_storage_bytes_for_group(addresses, group_size):
    total = 0
    for addr in addresses:
        total += compute_lut_storage_bytes(addr.num_entries, group_size, addr.num_tables)
    return total


def count_leaves_for_group(addresses):
    total = 0
    for addr in addresses:
        if isinstance(addr, AddressGreedyTree):
            total += addr._leaf_counter
        else:
            total += addr.num_entries
    return total


def estimate_eval_files_needed(input_files, eval_size):
    n = len(input_files)
    if n <= 1:
        return n

    probe = min(100, n)
    counts = []
    for path in input_files[-probe:]:
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
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
        description="Build LUT FFN (v3_independent): correct two-stage building"
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
    parser.add_argument("--coarse_finetune_epochs", type=int, default=10,
                        help="Epochs to finetune coarse LUT alone before building residual")
    parser.add_argument("--coarse_finetune_lr", type=float, default=1e-3)
    parser.add_argument("--coarse_finetune_batch_size", type=int, default=1024)
    parser.add_argument("--finetune_epochs", type=int, default=50,
                        help="Final joint finetune epochs for all groups together")
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_batch_size", type=int, default=1024)
    parser.add_argument("--finetune_loss_mode", type=str, default="multi",
                        choices=["mse", "cosine", "mse+cosine", "multi"])
    parser.add_argument("--finetune_cosine_alpha", type=float, default=1.0)
    parser.add_argument("--finetune_residual_cosine_alpha", type=float, default=0.5)
    parser.add_argument("--finetune_norm_alpha", type=float, default=0.01)
    parser.add_argument("--target_mode", type=str, default="direct",
                        choices=["direct", "residual_mean", "residual_input"])
    parser.add_argument("--calib_size", type=int, default=400000)
    parser.add_argument("--eval_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load data
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
        input_files = [p[0] for p in paired]
        output_files = [p[1] for p in paired]
        n_eval = estimate_eval_files_needed(input_files, args.eval_size)
        train_input_files = input_files[:-n_eval]
        test_input_files = input_files[-n_eval:]
        train_output_files = output_files[:-n_eval]
        test_output_files = output_files[-n_eval:]
    else:
        n_eval = estimate_eval_files_needed(input_files, args.eval_size)
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

    # Resume logic
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
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    # =============================================================================
    # Phase 1 & 2: 每组单独构建 coarse + finetune + residual
    # =============================================================================
    for gid in group_ids:
        if gid in completed_groups:
            print(f"\n[Group {gid}] skipping (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"[Group {gid}] Two-stage building ...")
        print(f"{'='*60}")

        g_start = gid * args.group_size
        g_end = g_start + args.group_size
        group_target = calib_y[:, g_start:g_end]

        # Stage 1: Build coarse LUT
        print(f"\n[Stage 1] Building coarse LUT ({args.coarse_num_bits} bits) ...")
        coarse_address, coarse_lut = _build_single_address_lut(
            calib_x, group_target, gid, args.group_size, args,
            num_bits=args.coarse_num_bits, seed_offset=0, desc="coarse", device=device,
        )

        # Stage 1b: Finetune coarse LUT alone (使用 local loss)
        finetune_single_lut(
            coarse_lut, coarse_address, calib_x, group_target, gid, args.group_size,
            device, args.coarse_finetune_epochs, args.coarse_finetune_lr,
            args.coarse_finetune_batch_size, loss_mode="mse+cosine"
        )

        # Stage 2: 根据 finetune 后的 coarse 重新计算 residual
        print(f"\n[Stage 2] Computing residual and building residual LUT ({args.residual_num_bits} bits) ...")
        with torch.no_grad():
            coarse_indices = coarse_address.compute_indices(calib_x.unsqueeze(0)).view(-1, coarse_address.num_tables)
            coarse_pred = coarse_lut(coarse_indices).squeeze(1).cpu()  # 移回 CPU
            residual_target = group_target - coarse_pred

        residual_address, residual_lut = _build_single_address_lut(
            calib_x, residual_target, gid, args.group_size, args,
            num_bits=args.residual_num_bits, seed_offset=1, desc="residual", device=device,
        )

        addresses[gid] = [coarse_address, residual_address]
        luts[gid] = [coarse_lut, residual_lut]
        group_means[gid] = None

        # Save checkpoint
        ckpt = {
            "group_id": gid,
            "group_size": args.group_size,
            "coarse_num_bits": args.coarse_num_bits,
            "residual_num_bits": args.residual_num_bits,
            "addresses": addresses[gid],
            "lut_tables": [lut.table.detach().cpu().half() for lut in luts[gid]],
        }
        ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"  [Group {gid}] checkpoint saved: {ckpt_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # =============================================================================
    # Phase 3: 全局 joint finetune (所有 32 组一起，使用完整 2048 维 loss)
    # =============================================================================
    if args.finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"[Phase 3] Global joint finetune: all groups together")
        print(f"{'='*60}")
        finetune_luts_joint(addresses, luts, group_means, args.target_mode,
                            calib_x, calib_y, group_ids, args.group_size, device, args)

        # 保存 finetune 后的最终 checkpoint
        for gid in group_ids:
            ckpt = {
                "group_id": gid,
                "group_size": args.group_size,
                "coarse_num_bits": args.coarse_num_bits,
                "residual_num_bits": args.residual_num_bits,
                "addresses": addresses[gid],
                "lut_tables": [lut.table.detach().cpu().half() for lut in luts[gid]],
            }
            ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
            torch.save(ckpt, ckpt_path)

    # Final evaluation
    print("\n[Final Evaluation] ...")
    for gid in group_ids:
        metrics = evaluate_group(addresses[gid], luts[gid], group_means[gid],
                                 args.target_mode, eval_x, eval_y, gid, args.group_size, device)
        print(f"  group {gid}: cos_sim={metrics['cosine_similarity']:.4f}, "
              f"rel_l2={metrics['relative_l2']:.2%}")
        group_metrics.append({"group_id": gid, "num_luts": 2, **metrics})

    full_metrics = evaluate_full_output(addresses, luts, group_means, args.target_mode,
                                        group_ids, eval_x, eval_y, args.group_size, device)
    print(f"\n[Full Output] cos_sim={full_metrics['cosine_similarity']:.4f}, "
          f"rel_l2={full_metrics['relative_l2']:.2%}, "
          f"norm_ratio={full_metrics['norm_ratio']:.4f}")

    # Summary
    total_table_bytes = sum(table_storage_bytes_for_group(addresses[gid], args.group_size) for gid in group_ids)
    summary = {
        "teacher_weight_path": args.teacher_weight_path,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "group_size": args.group_size,
        "group_ids": group_ids,
        "coarse_num_bits": args.coarse_num_bits,
        "residual_num_bits": args.residual_num_bits,
        "total_table_mib": total_table_bytes / (1024 * 1024),
        "group_metrics": group_metrics,
        "full_output_metrics": full_metrics,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
