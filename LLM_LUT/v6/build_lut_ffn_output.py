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
# =============================================================================
def collect_calibration_and_eval(
    train_files, test_files, teacher,
    calib_size: int, eval_size: int, batch_size: int, device: torch.device,
):
    """
    从 train_files 里取 calib_size 条、test_files 里取 eval_size 条。
    输入 tensor 形状 [N, hidden_size]，脚本通过 Teacher 得到真实 FFN 输出。
    """
    weight_dtype = next(teacher.parameters()).dtype

    def collect_from_files(file_paths, needed, desc):
        inputs = []
        targets = []
        collected = 0
        pbar = tqdm(total=needed, desc=desc, unit="sample")
        for fpath in sorted(file_paths):
            if collected >= needed:
                break
            try:
                tensor = torch.load(fpath, map_location="cpu")
            except Exception as e:
                print(f"  skip {fpath}: {e}")
                continue
            if tensor.dim() != 2:
                raise ValueError(f"{fpath} has shape {tensor.shape}, expected [N, hidden]")
            for i in range(0, tensor.shape[0], batch_size):
                if collected >= needed:
                    break
                x = tensor[i:i + batch_size].to(device, dtype=weight_dtype)
                with torch.no_grad():
                    y = teacher(x).float().cpu()
                x = x.float().cpu()
                inputs.append(x)
                targets.append(y)
                collected += x.shape[0]
                pbar.update(x.shape[0])
            if collected >= needed:
                break
        pbar.close()
        if collected == 0:
            raise RuntimeError(f"No valid samples found in {file_paths[:3]}...")
        return torch.cat(inputs, dim=0)[:needed], torch.cat(targets, dim=0)[:needed]

    calib_x, calib_y = collect_from_files(train_files, calib_size, desc="calibration")
    eval_x, eval_y = collect_from_files(test_files, eval_size, desc="eval")
    return calib_x, calib_y, eval_x, eval_y


# =============================================================================
# 4. 构建与评估
# =============================================================================
@torch.no_grad()
def build_lut_for_group(calib_x, calib_y, group_id, group_size, args, device):
    g_start = group_id * group_size
    g_end = g_start + group_size
    group_target = calib_y[:, g_start:g_end]

    seed = args.seed * 10000 + group_id
    address = AddressGreedyTree(
        input_dim=calib_x.shape[-1],
        num_bits=args.num_bits,
        channels_per_bit=args.channels_per_bit,
        seed=seed,
    )
    print(f"  building tree for group {group_id} ...")
    t0 = time.time()
    address.build(
        calib_x, group_target,
        num_candidates=args.tree_candidates,
        min_samples=args.tree_min_samples,
        max_samples=args.tree_max_samples,
    )
    print(f"  tree built in {time.time() - t0:.2f}s, leaves={address._leaf_counter}")

    indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, 1)
    lut = LUTGroup(
        num_tables=1,
        num_entries=address.num_entries,
        group_size=group_size,
        device=calib_x.device,
    )
    lut.initialize_from_calibration(indices, group_target)
    return address, lut


@torch.no_grad()
def evaluate_group(address, lut, eval_x, eval_y, group_id, group_size, device):
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    lut = lut.to(device)
    indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, 1)
    pred_group = lut(indices).squeeze(1)
    g_start = group_id * group_size
    g_end = g_start + group_size
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
def evaluate_full_output(addresses, luts, group_ids, eval_x, eval_y, group_size, device):
    """把所有被替换 group 拼回完整 FFN 输出，再和真实输出比。"""
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    pred_y = eval_y.clone()

    for gid in group_ids:
        address = addresses[gid]
        lut = luts[gid].to(device)
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, 1)
        pred_group = lut(indices).squeeze(1)
        g_start = gid * group_size
        g_end = g_start + group_size
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


# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build single-expert FFN output group LUT (v6 corrected direction)."
    )
    parser.add_argument("--teacher_weight_path", required=True, help="Path to expert .pt weights")
    parser.add_argument("--dataset_dir", required=True, help="Dir containing .pt input tensors")
    parser.add_argument("--output_root", required=True, help="Output directory for checkpoints and summary")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--group_ids", type=str, default="0,1,2,3",
                        help="Comma-separated output group ids to replace, e.g. '0,1,2,3'")
    parser.add_argument("--num_bits", type=int, default=12,
                        help="Tree depth => 2^num_bits entries per group")
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--tree_candidates", type=int, default=64)
    parser.add_argument("--tree_min_samples", type=int, default=16)
    parser.add_argument("--tree_max_samples", type=int, default=65536,
                        help="Subsample calibration data for tree building")
    parser.add_argument("--calib_size", type=int, default=65536)
    parser.add_argument("--eval_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    group_ids = [int(x.strip()) for x in args.group_ids.split(",") if x.strip()]
    if not group_ids:
        raise ValueError("--group_ids must contain at least one integer")

    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {args.dataset_dir}")
    print(f"Found {len(all_files)} .pt files, using first {len(all_files)-100} for calibration, last 100 for eval")
    train_files = all_files[:-100]
    test_files = all_files[-100:]

    teacher, hidden_size, intermediate_size = load_real_teacher(args.teacher_weight_path, device)
    print(f"Teacher: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    max_group = hidden_size // args.group_size
    for gid in group_ids:
        if not (0 <= gid < max_group):
            raise ValueError(f"group_id {gid} out of range [0, {max_group})")

    print("\nCollecting calibration / evaluation samples ...")
    calib_x, calib_y, eval_x, eval_y = collect_calibration_and_eval(
        train_files, test_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    addresses = {}
    luts = {}
    group_metrics = []

    for gid in group_ids:
        print(f"\n[Group {gid}] building LUT ...")
        address, lut = build_lut_for_group(calib_x, calib_y, gid, args.group_size, args, device)
        metrics = evaluate_group(address, lut, eval_x, eval_y, gid, args.group_size, device)
        print(f"  group {gid}: cos_sim={metrics['cosine_similarity']:.4f}, "
              f"rel_l2={metrics['relative_l2']:.2%}, rel_mse={metrics['relative_mse']:.4f}")

        addresses[gid] = address
        luts[gid] = lut
        group_metrics.append({"group_id": gid, **metrics})

        ckpt_path = ckpt_dir / f"replacement_g{gid}.pt"
        torch.save({
            "group_id": gid,
            "group_size": args.group_size,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "address_type": "tree",
            "tree_state": address.serialize(),
            "lut_table": lut.table.detach().cpu().half(),  # FP16 for storage
        }, ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

    print("\n[Full output] evaluating all replaced groups together ...")
    full_metrics = evaluate_full_output(addresses, luts, group_ids, eval_x, eval_y, args.group_size, device)
    print(f"  full output: cos_sim={full_metrics['cosine_similarity']:.4f}, "
          f"rel_l2={full_metrics['relative_l2']:.2%}")

    num_entries = 2 ** args.num_bits
    table_bytes_per_group = compute_lut_storage_bytes(num_entries, args.group_size, num_tables=1)
    total_table_bytes = table_bytes_per_group * len(group_ids)
    replaced_channels = len(group_ids) * args.group_size

    total_ffn_mac = 3 * hidden_size * intermediate_size
    saved_mac = replaced_channels * intermediate_size  # skip down_proj slice for replaced groups
    mac_reduction_ratio = saved_mac / total_ffn_mac

    summary = {
        "teacher_weight_path": args.teacher_weight_path,
        "dataset_dir": args.dataset_dir,
        "output_root": args.output_root,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "group_size": args.group_size,
        "group_ids": group_ids,
        "num_bits": args.num_bits,
        "channels_per_bit": args.channels_per_bit,
        "tree_candidates": args.tree_candidates,
        "tree_min_samples": args.tree_min_samples,
        "tree_max_samples": args.tree_max_samples,
        "num_entries_per_group": num_entries,
        "actual_leaves_per_group": {str(gid): addresses[gid]._leaf_counter for gid in group_ids},
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
