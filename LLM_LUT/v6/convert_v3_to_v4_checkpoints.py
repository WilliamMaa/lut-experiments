#!/usr/bin/env python3
"""
convert_v3_to_v4_checkpoints.py

把 v3 shared-coarse + per-group residual checkpoint 导出为 v6 engine 兼容的
replacement_g*.pt 格式（不添加 hard correction）。

用法：
  python -u convert_v3_to_v4_checkpoints.py \
    --v3_checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
    --output_root ./outputs_ffn_lut_layer39_full_moe_v3_as_v4
"""

import argparse
import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut_ffn_output_v3_shared_coarse as v3


def _inject_v3_classes_into_main():
    import __main__ as _main_mod
    for _name in ("AddressGreedyTree", "_TreeNode", "LUTGroup", "QwenMoEExpert"):
        _cls = getattr(v3, _name, None)
        if _cls is not None and not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, _cls)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_checkpoint_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    _inject_v3_classes_into_main()

    # 用第一个 residual checkpoint 的 group_id 推断 hidden_size
    ckpt_dir = Path(args.v3_checkpoint_dir)
    first_residual = next(ckpt_dir.glob("residual_g*.pt"))
    res_ckpt = torch.load(first_residual, map_location="cpu", weights_only=False)
    group_size = res_ckpt["table"].shape[-1]
    max_group = int(first_residual.stem.split("g")[-1]) + 1
    while (ckpt_dir / f"residual_g{max_group}.pt").exists():
        max_group += 1
    hidden_size = max_group * group_size
    print(f"Inferred hidden_size={hidden_size}, group_size={group_size}, num_groups={max_group}")

    coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids = load_v3_base(
        args.v3_checkpoint_dir, hidden_size, group_size, device
    )

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out_dir = out_dir / "checkpoints"
    ckpt_out_dir.mkdir(exist_ok=True)

    coarse_table = coarse_lut.table.detach().cpu().half()

    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size

        addresses = [coarse_address, residual_addresses[gid]]
        tables = [
            coarse_table[:, :, g_start:g_end],
            residual_luts[gid].table.detach().cpu().half(),
        ]

        ckpt = {
            "group_id": gid,
            "group_size": group_size,
            "num_bits": residual_addresses[gid].num_bits,
            "coarse_num_bits": coarse_address.num_bits,
            "residual_num_bits": residual_addresses[gid].num_bits,
            "hard_num_bits": 0,
            "address_mode": "v3_shared_coarse",
            "target_mode": "direct",
            "addresses": addresses,
            "lut_tables": tables,
            "group_mean": None,
        }
        torch.save(ckpt, ckpt_out_dir / f"replacement_g{gid}.pt")

    print(f"Exported {len(group_ids)} groups to {ckpt_out_dir}")


if __name__ == "__main__":
    main()
