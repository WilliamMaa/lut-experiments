#!/usr/bin/env python3
"""
build_pairwise_correction_v3.py

在 v3 shared-coarse + per-group residual 基础上，为 top-K group pairs 增加一个
factorized pairwise correction：

    y = y_LUT + sum_{(g,h) in pairs} U_gh (E_g[leaf_g] .* E_h[leaf_h])

其中：
  - leaf_g / leaf_h 来自对应 group 的 residual tree address；
  - E_g, E_h 是每个 group 所有 leaf 的 rank-r embedding；
  - U_gh 把交互向量映射到 2048 维输出修正。

这是 doc 12-reflection 里提到的 cheap factorized pairwise 方案。
"""

import os
import sys
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut_ffn_output_v3_shared_coarse as v3
from build_lut_ffn_output_v3_lowrank import (
    load_real_teacher,
    collect_calibration_and_eval,
    _predict_base,
    parse_group_ids,
)


def _inject_v3_classes_into_main():
    import __main__ as _main_mod
    for _name in ("AddressGreedyTree", "_TreeNode", "LUTGroup", "QwenMoEExpert"):
        _cls = getattr(v3, _name, None)
        if _cls is not None and not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, _cls)


class PairwiseCorrections(nn.Module):
    """Factorized pairwise correction for a set of group pairs."""

    def __init__(self, hidden_size: int, pairs, num_leaves: int, rank: int = 8, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.pairs = list(pairs)
        self.num_leaves = num_leaves

        involved = sorted(set(g for p in self.pairs for g in p))
        for gid in involved:
            self.register_parameter(f"E_{gid}", nn.Parameter(torch.randn(num_leaves, rank, dtype=dtype) * 0.01))
        for g, h in self.pairs:
            self.register_parameter(f"U_{g}_{h}", nn.Parameter(torch.randn(hidden_size, rank, dtype=dtype) * 0.01))

    def forward(self, x: torch.Tensor, residual_addresses: dict):
        """
        x: [N, hidden] or [B, S, hidden]
        residual_addresses: {gid: AddressGreedyTree}
        Returns correction of same shape as x.
        """
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, S, H = x.shape
        device = x.device
        x_flat = x.reshape(B * S, H)
        out = torch.zeros(B * S, H, device=device, dtype=torch.float32)

        for g, h in self.pairs:
            leaf_g = residual_addresses[g].compute_indices(x).view(-1)
            leaf_h = residual_addresses[h].compute_indices(x).view(-1)

            E_g = getattr(self, f"E_{g}").to(device, dtype=torch.float32)
            E_h = getattr(self, f"E_{h}").to(device, dtype=torch.float32)
            e_g = E_g[leaf_g]
            e_h = E_h[leaf_h]
            interaction = e_g * e_h

            U = getattr(self, f"U_{g}_{h}").to(device, dtype=torch.float32)
            out = out + (interaction @ U.t())

        return out.view(orig_shape)


def apply_pairwise_correction(x, residual_addresses, pairwise_module):
    if pairwise_module is None:
        return 0.0
    return pairwise_module(x, residual_addresses).to(dtype=x.dtype)


def parse_pair_ids(s: str):
    """Parse 'g1,h1;g2,h2;...' into list of (g,h) tuples."""
    pairs = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        g, h = part.split(",")
        pairs.append((int(g.strip()), int(h.strip())))
    return pairs


def train_pairwise(
    coarse_lut, coarse_address, residual_luts, residual_addresses,
    calib_x, calib_y, group_ids, group_size, device, pairwise_module,
    epochs=20, lr=1e-3, batch_size=1024,
    cosine_alpha=1.0, residual_cosine_alpha=0.5, norm_alpha=0.01,
):
    print(f"\n[Pairwise Training] {len(pairwise_module.pairs)} pairs, rank={pairwise_module.rank}, "
          f"epochs={epochs}, lr={lr}")

    optimizer = torch.optim.Adam(pairwise_module.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_samples = calib_x.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "res_cos": 0.0, "norm_ratio": 0.0}
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = calib_x[idx].to(device)
            yb = calib_y[idx].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                base_pred = _predict_base(
                    coarse_lut, coarse_address, residual_luts, residual_addresses,
                    xb, group_ids, group_size, device
                )

            pred_y = base_pred + apply_pairwise_correction(xb, residual_addresses, pairwise_module)

            mse = F.mse_loss(pred_y, yb)
            cos_output = F.cosine_similarity(pred_y, yb, dim=-1).mean()
            pred_residual = xb + pred_y
            true_residual = xb + yb
            cos_residual = F.cosine_similarity(pred_residual, true_residual, dim=-1).mean()
            pred_norm = torch.norm(pred_y, dim=-1)
            true_norm = torch.norm(yb, dim=-1)
            log_norm_loss = (torch.log((pred_norm + 1e-6) / (true_norm + 1e-6)) ** 2).mean()

            loss = (mse +
                    cosine_alpha * (1 - cos_output) +
                    residual_cosine_alpha * (1 - cos_residual) +
                    norm_alpha * log_norm_loss)

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
                  f"loss={epoch_metrics['loss']/max(n_batches,1):.6e}, "
                  f"cos={epoch_metrics['cos']/max(n_batches,1):.4f}, "
                  f"res_cos={epoch_metrics['res_cos']/max(n_batches,1):.4f}, "
                  f"norm_ratio={epoch_metrics['norm_ratio']/max(n_batches,1):.4f}")


@torch.no_grad()
def evaluate_pairwise(
    coarse_lut, coarse_address, residual_luts, residual_addresses,
    eval_x, eval_y, group_ids, group_size, device, pairwise_module,
):
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)

    base_pred = _predict_base(
        coarse_lut, coarse_address, residual_luts, residual_addresses,
        eval_x, group_ids, group_size, device
    )
    pred_y = base_pred + apply_pairwise_correction(eval_x, residual_addresses, pairwise_module)

    mse = F.mse_loss(pred_y, eval_y).item()
    rel_l2 = torch.norm(pred_y - eval_y).item() / (torch.norm(eval_y).item() + 1e-8)
    cos_sim = F.cosine_similarity(pred_y, eval_y, dim=-1)
    cos_mean = cos_sim.mean().item()
    cos_p10 = torch.quantile(cos_sim, 0.10).item()
    norm_ratio = (torch.norm(pred_y, dim=-1) / (torch.norm(eval_y, dim=-1) + 1e-6)).mean().item()

    return {
        "mse": mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_mean,
        "cosine_similarity_p10": cos_p10,
        "norm_ratio": norm_ratio,
    }


def load_v3_base(ckpt_dir: Path, hidden_size: int, group_size: int, device: torch.device):
    _inject_v3_classes_into_main()
    ckpt_dir = Path(ckpt_dir)
    print(f"Loading v3 base from {ckpt_dir}")

    coarse_ckpt = torch.load(ckpt_dir / "shared_coarse.pt", map_location="cpu", weights_only=False)
    coarse_address = coarse_ckpt["address"]
    coarse_lut = v3.LUTGroup(
        num_tables=coarse_address.num_tables,
        num_entries=coarse_address.num_entries,
        output_dim=hidden_size,
        init_table=coarse_ckpt["table"],
        device=device,
    )

    residual_addresses = {}
    residual_luts = {}
    max_group = hidden_size // group_size
    for gid in range(max_group):
        residual_path = ckpt_dir / f"residual_g{gid}.pt"
        if not residual_path.exists():
            continue
        res_ckpt = torch.load(residual_path, map_location="cpu", weights_only=False)
        residual_addresses[gid] = res_ckpt["address"]
        residual_luts[gid] = v3.LUTGroup(
            num_tables=residual_addresses[gid].num_tables,
            num_entries=residual_addresses[gid].num_entries,
            output_dim=group_size,
            init_table=res_ckpt["table"],
            device=device,
        )

    group_ids = sorted(residual_luts.keys())
    print(f"  loaded coarse entries={coarse_address.num_entries}, groups={len(group_ids)}")
    return coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build factorized pairwise correction on top of v3 shared-coarse base."
    )
    parser.add_argument("--teacher_weight_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dataset_dir", required=True)
    parser.add_argument("--base_checkpoint_dir", required=True,
                        help="Directory containing v3 shared_coarse.pt and residual_g*.pt")
    parser.add_argument("--output_pairwise_path", default=None,
                        help="Output path for pairwise.pt. Default: base_checkpoint_dir/pairwise.pt")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--group_ids", type=str, default="0-31")
    parser.add_argument("--pair_ids", type=str, default="19,25;19,28;19,13;19,9",
                        help="Semicolon-separated list of group pairs, e.g. '19,25;19,28'")
    parser.add_argument("--rank", type=int, default=8,
                        help="Embedding rank for each pair.")
    parser.add_argument("--calib_size", type=int, default=550000)
    parser.add_argument("--eval_size", type=int, default=68000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_batch_size", type=int, default=1024)
    parser.add_argument("--finetune_cosine_alpha", type=float, default=1.0)
    parser.add_argument("--finetune_residual_cosine_alpha", type=float, default=0.5)
    parser.add_argument("--finetune_norm_alpha", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    _inject_v3_classes_into_main()

    teacher, hidden_size, intermediate_size = load_real_teacher(args.teacher_weight_path, device)
    print(f"Teacher: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    group_ids = parse_group_ids(args.group_ids, hidden_size // args.group_size)
    pairs = parse_pair_ids(args.pair_ids)
    print(f"Groups: {group_ids}")
    print(f"Pairwise pairs: {pairs}")

    input_files = sorted(Path(args.dataset_dir).glob("*.pt"))
    output_files = sorted(Path(args.output_dataset_dir).glob("*.pt"))
    if not input_files:
        raise FileNotFoundError(f"No .pt files in {args.dataset_dir}")

    calib_x, calib_y, eval_x, eval_y = collect_calibration_and_eval(
        input_files, output_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    coarse_address, coarse_lut, residual_addresses, residual_luts, base_group_ids = load_v3_base(
        args.base_checkpoint_dir, hidden_size, args.group_size, device
    )
    if base_group_ids != group_ids:
        print(f"[Warning] base group_ids {base_group_ids} != requested {group_ids}; using base groups")
        group_ids = base_group_ids

    num_leaves = residual_addresses[group_ids[0]].num_entries
    pairwise_module = PairwiseCorrections(hidden_size, pairs, num_leaves, rank=args.rank).to(device)

    train_pairwise(
        coarse_lut, coarse_address, residual_luts, residual_addresses,
        calib_x, calib_y, group_ids, args.group_size, device, pairwise_module,
        epochs=args.epochs, lr=args.lr, batch_size=args.finetune_batch_size,
        cosine_alpha=args.finetune_cosine_alpha,
        residual_cosine_alpha=args.finetune_residual_cosine_alpha,
        norm_alpha=args.finetune_norm_alpha,
    )

    print("\n[Pairwise Evaluation]")
    metrics = evaluate_pairwise(
        coarse_lut, coarse_address, residual_luts, residual_addresses,
        eval_x, eval_y, group_ids, args.group_size, device, pairwise_module,
    )
    print(f"  cos={metrics['cosine_similarity']:.4f}, p10={metrics['cosine_similarity_p10']:.4f}, "
          f"rel_l2={metrics['relative_l2']:.4f}, norm_ratio={metrics['norm_ratio']:.4f}")

    if args.output_pairwise_path:
        out_path = Path(args.output_pairwise_path)
    else:
        # Default: keep pairwise.pt outside the base checkpoint dir to avoid overwriting it
        out_path = Path(args.base_checkpoint_dir).parent / "pairwise" / "pairwise.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hidden_size": hidden_size,
        "pairs": pairs,
        "num_leaves": num_leaves,
        "rank": args.rank,
        "state_dict": pairwise_module.state_dict(),
        "metrics": metrics,
    }, out_path)
    print(f"\nSaved pairwise correction: {out_path}")


if __name__ == "__main__":
    main()
