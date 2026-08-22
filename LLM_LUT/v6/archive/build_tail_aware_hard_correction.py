#!/usr/bin/env python3
"""
build_tail_aware_hard_correction.py

Experiment B：在 v3 shared coarse + grouped residual 的 checkpoint 上，
增加一个 tail-aware shared hard correction 表（12-13bit，2048维），
最终导出为兼容 v6_replacement_engine.py 的 per-group checkpoint。

核心设计：
1. 加载 v3 base（shared coarse + per-group residual）
2. 在 calibration 数据上计算 base_pred = coarse + residual
3. 计算 hard_target = y - base_pred
4. 计算逐样本权重 weight = sigmoid((tau - base_cos) / T)，聚焦 p10 困难样本
5. 建立一棵共享 2048 维 hard correction tree
6. 用加权 loss 训练 hard correction LUT
7. 导出 per-group checkpoint：coarse_slice + residual + hard_slice

使用示例：
python -u build_tail_aware_hard_correction.py \
  --base_checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
  --output_dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/output \
  --output_root ./outputs_ffn_lut_layer39_full_moe_v4_tail \
  --group_size 64 \
  --group_ids "0-31" \
  --hard_num_bits 13 \
  --hard_tau 0.80 \
  --hard_temperature 0.05 \
  --hard_finetune_epochs 30 \
  --tree_max_samples 200000 \
  --tree_min_samples 4 \
  --tree_candidates 256 \
  --calib_size 400000 \
  --eval_size 100000 \
  --device cuda:0 \
  > v4_tail.log 2>&1 &
"""

import os
import glob
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import build_lut_ffn_output_v3_shared_coarse as v3


def _inject_v3_classes_into_main():
    """Inject v3 classes into __main__ so torch.load can resolve __main__-pickled checkpoints."""
    import __main__ as _main_mod
    for _name in ("AddressGreedyTree", "_TreeNode", "LUTGroup", "QwenMoEExpert"):
        _cls = getattr(v3, _name, None)
        if _cls is not None and not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, _cls)


def _v3_load(path, **kwargs):
    """Load a v3 checkpoint that may have been pickled as __main__ classes."""
    _inject_v3_classes_into_main()
    return torch.load(path, weights_only=False, **kwargs)


@torch.no_grad()
def load_v3_base(ckpt_dir: Path, hidden_size: int, group_size: int, device: torch.device):
    """Load shared coarse + per-group residual from v3 checkpoint dir."""
    ckpt_dir = Path(ckpt_dir)
    print(f"\nLoading v3 base from {ckpt_dir}")

    coarse_ckpt = _v3_load(ckpt_dir / "shared_coarse.pt", map_location="cpu")
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
        res_ckpt = _v3_load(residual_path, map_location="cpu")
        residual_addresses[gid] = res_ckpt["address"]
        residual_luts[gid] = v3.LUTGroup(
            num_tables=residual_addresses[gid].num_tables,
            num_entries=residual_addresses[gid].num_entries,
            output_dim=group_size,
            init_table=res_ckpt["table"],
            device=device,
        )

    print(f"  loaded coarse: entries={coarse_address.num_entries}, groups={len(residual_luts)}")
    return coarse_address, coarse_lut, residual_addresses, residual_luts


@torch.no_grad()
def predict_base(coarse_address, coarse_lut, residual_addresses, residual_luts,
                 group_ids, group_size, x, device):
    """Compute base prediction = shared coarse + per-group residual."""
    x = x.to(device)
    coarse_lut.to(device)
    coarse_indices = coarse_address.compute_indices(x.unsqueeze(0)).view(-1, coarse_address.num_tables)
    coarse_full = coarse_lut(coarse_indices)  # [N, hidden]

    pred_y = torch.zeros_like(x)
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        residual_luts[gid].to(device)
        residual_indices = residual_addresses[gid].compute_indices(x.unsqueeze(0)).view(
            -1, residual_addresses[gid].num_tables
        )
        residual_group = residual_luts[gid](residual_indices)
        pred_y[:, g_start:g_end] = coarse_full[:, g_start:g_end] + residual_group

    return pred_y


def compute_hard_weights(base_pred, true_y, tau, temperature):
    """Tail-aware weights: focus on samples where base cosine is low."""
    cos = F.cosine_similarity(base_pred, true_y, dim=-1)
    weights = torch.sigmoid((tau - cos) / temperature)
    return weights, cos


def weighted_initialize(lut, indices, targets, weights):
    """Initialize LUT table with weighted mean per entry."""
    with torch.no_grad():
        M, E, out_dim = lut.table.shape
        device = lut.table.device
        new_table = torch.zeros_like(lut.table)
        counts = torch.zeros(M, E, device=device, dtype=torch.float32)

        for m in range(M):
            idx_m = indices[:, m].clamp(0, E - 1)
            idx_exp = idx_m.unsqueeze(1).expand(-1, out_dim)
            new_table[m].scatter_add_(0, idx_exp, targets.float() * weights.unsqueeze(1))
            counts[m].scatter_add_(0, idx_m, weights)

        occupied = (counts > 0).nonzero(as_tuple=False)
        for m, e in occupied:
            m, e = int(m), int(e)
            new_table[m, e] /= counts[m, e]

        lut.table.copy_(new_table)
    return lut


def build_hard_correction(calib_x, hard_target, weights, args, device):
    """Build shared 2048-dim hard correction tree and initialize weighted table."""
    print(f"\n[Hard Correction] Building {args.hard_num_bits}-bit shared tree (2048-dim)")
    hard_address = v3.AddressGreedyTree(
        input_dim=calib_x.shape[-1],
        num_bits=args.hard_num_bits,
        channels_per_bit=args.channels_per_bit,
        seed=args.seed * 30000 + 7,
    )
    t0 = time.time()
    hard_address.build(
        calib_x, hard_target,
        num_candidates=args.tree_candidates,
        min_samples=args.tree_min_samples,
        max_samples=args.tree_max_samples,
        device=device,
    )
    print(f"  Hard tree built in {time.time() - t0:.2f}s, leaves={hard_address._leaf_counter}")

    hard_indices = hard_address.compute_indices(calib_x.unsqueeze(0)).view(-1, hard_address.num_tables)
    hard_lut = v3.LUTGroup(
        num_tables=hard_address.num_tables,
        num_entries=hard_address.num_entries,
        output_dim=calib_x.shape[-1],
        device=calib_x.device,
    )
    hard_lut = weighted_initialize(hard_lut, hard_indices, hard_target, weights)
    print(f"  Hard LUT initialized: {hard_lut.table.shape}")
    return hard_address, hard_lut


def finetune_hard_correction(hard_lut, hard_address, calib_x, hard_target, base_pred, true_y,
                             weights, device, args):
    """Finetune hard correction with weighted loss."""
    epochs = args.hard_finetune_epochs
    if epochs <= 0:
        return

    print(f"\n[Hard Finetune] {epochs} epochs (lr={args.hard_finetune_lr}, batch={args.hard_finetune_batch_size}, loss={args.hard_finetune_loss_mode})")
    hard_lut.to(device)
    optimizer = torch.optim.Adam([hard_lut.table], lr=args.hard_finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_samples = calib_x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"loss": 0.0, "mse": 0.0, "cos": 0.0}
        n_batches = 0

        for start in range(0, n_samples, args.hard_finetune_batch_size):
            idx = perm[start:start + args.hard_finetune_batch_size]
            xb = calib_x[idx].to(device)
            hb = hard_target[idx].to(device)
            wb = weights[idx].to(device)
            base_b = base_pred[idx].to(device)
            true_b = true_y[idx].to(device)

            optimizer.zero_grad()
            pred_hard = hard_lut(hard_address.compute_indices(xb.unsqueeze(0)).view(-1, hard_address.num_tables))
            full_pred = base_b + pred_hard

            mse_hard = F.mse_loss(pred_hard, hb, reduction="none").mean(dim=-1)
            cos_full = F.cosine_similarity(full_pred, true_b, dim=-1)

            if args.hard_finetune_loss_mode == "mse":
                loss = (wb * mse_hard).sum() / wb.sum()
            elif args.hard_finetune_loss_mode == "cosine":
                loss = (wb * (1 - cos_full)).sum() / wb.sum()
            else:  # mse+cosine
                loss = (wb * (mse_hard + (1 - cos_full))).sum() / wb.sum()

            loss.backward()
            optimizer.step()

            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mse"] += mse_hard.mean().item()
            epoch_metrics["cos"] += cos_full.mean().item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_metrics['loss']/max(n_batches,1):.6e}, "
                  f"mse={epoch_metrics['mse']/max(n_batches,1):.6e}, cos={epoch_metrics['cos']/max(n_batches,1):.4f}")


@torch.no_grad()
def evaluate_with_base_and_hard(coarse_address, coarse_lut, residual_addresses, residual_luts,
                                hard_address, hard_lut, eval_x, eval_y, group_ids, group_size, device):
    """Evaluate base and base+hard metrics."""
    base_pred = predict_base(coarse_address, coarse_lut, residual_addresses, residual_luts,
                             group_ids, group_size, eval_x, device)

    hard_indices = hard_address.compute_indices(eval_x.to(device).unsqueeze(0)).view(-1, hard_address.num_tables)
    hard_lut.to(device)
    hard_corr = hard_lut(hard_indices)
    full_pred = base_pred + hard_corr

    def _metrics(pred, true):
        mse = F.mse_loss(pred, true).item()
        rel_mse = mse / (true.var().item() + 1e-12)
        rel_l2 = torch.norm(pred - true).item() / (torch.norm(true).item() + 1e-12)
        cos = F.cosine_similarity(pred, true, dim=-1)
        return {
            "mse": mse,
            "relative_mse": rel_mse,
            "relative_l2": rel_l2,
            "cosine_similarity": cos.mean().item(),
            "cosine_similarity_p10": torch.quantile(cos, 0.10).item(),
            "cosine_similarity_p50": torch.quantile(cos, 0.50).item(),
            "cosine_similarity_p90": torch.quantile(cos, 0.90).item(),
            "norm_ratio": (torch.norm(pred, dim=-1) / (torch.norm(true, dim=-1) + 1e-6)).mean().item(),
        }

    base_metrics = _metrics(base_pred, eval_y.to(device))
    full_metrics = _metrics(full_pred, eval_y.to(device))
    return base_metrics, full_metrics


def export_per_group_checkpoints(output_dir, group_ids, group_size, hidden_size,
                                 coarse_address, coarse_lut,
                                 residual_addresses, residual_luts,
                                 hard_address, hard_lut):
    """Export final per-group checkpoints compatible with v6_replacement_engine.py."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    coarse_table = coarse_lut.table.detach().cpu().half()
    hard_table = hard_lut.table.detach().cpu().half()

    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size

        addresses = [coarse_address, residual_addresses[gid], hard_address]
        tables = [
            coarse_table[:, :, g_start:g_end],
            residual_luts[gid].table.detach().cpu().half(),
            hard_table[:, :, g_start:g_end],
        ]

        ckpt = {
            "group_id": gid,
            "group_size": group_size,
            "num_bits": hard_address.num_bits,
            "coarse_num_bits": coarse_address.num_bits,
            "residual_num_bits": residual_addresses[gid].num_bits,
            "hard_num_bits": hard_address.num_bits,
            "address_mode": "tail_aware_v4",
            "target_mode": "direct",
            "addresses": addresses,
            "lut_tables": tables,
            "group_mean": None,
        }
        torch.save(ckpt, ckpt_dir / f"replacement_g{gid}.pt")

    print(f"Exported {len(group_ids)} per-group checkpoints to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build tail-aware shared hard correction on top of v3 base")
    parser.add_argument("--base_checkpoint_dir", required=True, help="v3 checkpoint dir with shared_coarse.pt and residual_g*.pt")
    parser.add_argument("--teacher_weight_path", required=True, help="expert .pt for hidden/intermediate size")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dataset_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--group_ids", type=str, default="0-31")
    parser.add_argument("--hard_num_bits", type=int, default=13)
    parser.add_argument("--hard_tau", type=float, default=0.80)
    parser.add_argument("--hard_temperature", type=float, default=0.05)
    parser.add_argument("--hard_finetune_epochs", type=int, default=30)
    parser.add_argument("--hard_finetune_lr", type=float, default=1e-3)
    parser.add_argument("--hard_finetune_batch_size", type=int, default=1024)
    parser.add_argument("--hard_finetune_loss_mode", type=str, default="mse+cosine",
                        choices=["mse", "cosine", "mse+cosine"])
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--tree_candidates", type=int, default=256)
    parser.add_argument("--tree_min_samples", type=int, default=4)
    parser.add_argument("--tree_max_samples", type=int, default=200000)
    parser.add_argument("--calib_size", type=int, default=400000)
    parser.add_argument("--eval_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    teacher, hidden_size, intermediate_size = v3.load_real_teacher(args.teacher_weight_path, device)
    print(f"Teacher: hidden_size={hidden_size}, intermediate_size={intermediate_size}")

    max_group = hidden_size // args.group_size
    group_ids = v3.parse_group_ids(args.group_ids, max_group)
    print(f"Processing {len(group_ids)} groups: {group_ids}")

    input_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {args.dataset_dir}")

    output_files_map = {os.path.basename(p): p for p in glob.glob(os.path.join(args.output_dataset_dir, "*.pt"))}
    output_files = [output_files_map.get(os.path.basename(f)) for f in input_files]
    paired_indices = [i for i, o in enumerate(output_files) if o is not None]
    input_files = [input_files[i] for i in paired_indices]
    output_files = [output_files[i] for i in paired_indices]
    print(f"Found {len(input_files)} paired input/output files")

    print("\nCollecting calibration / evaluation samples ...")
    calib_x, calib_y, eval_x, eval_y = v3.collect_calibration_and_eval(
        input_files, output_files, teacher,
        args.calib_size, args.eval_size, args.batch_size, device,
    )
    print(f"Calibration: {calib_x.shape}, Eval: {eval_x.shape}")

    # Load v3 base
    coarse_address, coarse_lut, residual_addresses, residual_luts = load_v3_base(
        args.base_checkpoint_dir, hidden_size, args.group_size, device
    )

    # Compute base predictions
    print("\nComputing base predictions ...")
    base_pred_calib = predict_base(coarse_address, coarse_lut, residual_addresses, residual_luts,
                                     group_ids, args.group_size, calib_x, device)
    hard_target_calib = calib_y.to(device) - base_pred_calib
    weights, base_cos = compute_hard_weights(base_pred_calib, calib_y.to(device), args.hard_tau, args.hard_temperature)
    print(f"  base cos: mean={base_cos.mean().item():.4f}, p10={torch.quantile(base_cos, 0.10).item():.4f}")
    print(f"  hard weight: mean={weights.mean().item():.4f}, p10={torch.quantile(weights, 0.10).item():.4f}")

    # Build hard correction
    hard_address, hard_lut = build_hard_correction(calib_x, hard_target_calib.cpu(), weights.cpu(), args, device)

    # Finetune hard correction
    finetune_hard_correction(
        hard_lut, hard_address, calib_x, hard_target_calib.cpu(), base_pred_calib.cpu(), calib_y,
        weights.cpu(), device, args
    )

    # Evaluate
    print("\nEvaluating ...")
    base_metrics, full_metrics = evaluate_with_base_and_hard(
        coarse_address, coarse_lut, residual_addresses, residual_luts,
        hard_address, hard_lut, eval_x, eval_y, group_ids, args.group_size, device
    )
    print(f"  base only:    cos={base_metrics['cosine_similarity']:.4f}, p10={base_metrics['cosine_similarity_p10']:.4f}")
    print(f"  + hard corr:  cos={full_metrics['cosine_similarity']:.4f}, p10={full_metrics['cosine_similarity_p10']:.4f}")

    # Export
    print("\nExporting checkpoints ...")
    export_per_group_checkpoints(
        args.output_root, group_ids, args.group_size, hidden_size,
        coarse_address, coarse_lut, residual_addresses, residual_luts,
        hard_address, hard_lut,
    )

    # Summary
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_checkpoint_dir": str(args.base_checkpoint_dir),
        "output_root": str(args.output_root),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "group_size": args.group_size,
        "group_ids": group_ids,
        "hard_num_bits": args.hard_num_bits,
        "hard_tau": args.hard_tau,
        "hard_temperature": args.hard_temperature,
        "hard_finetune_epochs": args.hard_finetune_epochs,
        "hard_finetune_lr": args.hard_finetune_lr,
        "hard_finetune_batch_size": args.hard_finetune_batch_size,
        "hard_finetune_loss_mode": args.hard_finetune_loss_mode,
        "base_metrics": base_metrics,
        "full_metrics": full_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
