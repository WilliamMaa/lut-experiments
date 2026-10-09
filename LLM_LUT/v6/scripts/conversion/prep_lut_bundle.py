#!/usr/bin/env python3
"""prep_lut_bundle.py — pack a v6 `_as_v4/checkpoints` dir into ONE
bundle.pt that the v8 vLLM plugin can load for its shared_expert LUT
forward (v8/docs/43 P1.3).

The plugin's triton kernel wants plain stacked tensors, not pickled tree
objects, so this tool flattens the 32 replacement_g*.pt files into:
  coarse_table  [2^coarse_bits, hidden]  fp16  (cat of per-group slices)
  resid_table   [G, 2^resid_bits, gs]    fp16  (stack of per-group tables)
  tree params   padded, self-looping node arrays (leaves AND pad rows
                point at themselves, so a fixed-depth loop supports
                early-stopped trees)

Padding rule (fixed point): a pad row has ch=0, sign=+1, thr=+inf, so
proj <= thr is always true and left=self keeps the walker in place;
real leaf rows get left/right rewritten to their own id the same way.

Output math (matches the v6 engine, but accumulated in fp32 — the v6
engine adds in fp16, fp32 can only be closer to the teacher):
  out = coarse_table[leaf_c] + resid_table[g, leaf_g]

Usage (remote, lut_py310):
  python v6/scripts/conversion/prep_lut_bundle.py \
      --checkpoint_dir outputs_ffn_lut_layer39_full_moe_v3_as_v4/checkpoints \
      --output bundles/layer39.pt
"""

import argparse
import sys
import types
from pathlib import Path

import torch

# The pickled tree objects reference classes from the training script
# (build_lut_ffn_output_v3_shared_coarse). We only read their attributes,
# so inject a stub module that fabricates any requested class on demand —
# this avoids importing the real training script and all of its deps.
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


for _mod_name in ("build_lut_ffn_output_v3_shared_coarse",
                  "build_lut_ffn_output",
                  "build_lut_ffn_output_v3_lowrank",
                  "build_pairwise_correction_v3"):
    sys.modules.setdefault(_mod_name, _StubModule(_mod_name))


def _load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _padded_tree_arrays(tree, n_max: int) -> dict:
    """Flatten one pickled tree into padded self-looping node tensors.

    leaf rows: left/right rewritten to their own id (a walker that
    reaches a leaf stays there for the remaining depth steps).
    rows in [n, n_max): pad rows, left/right = their own id, ch=0,
    sign=+1, thr=+inf so proj <= thr always goes left = self.
    """
    n = int(tree.node_channel_idx.shape[0])
    if n > n_max:
        raise ValueError(f"tree has {n} nodes > n_max {n_max}")
    ch = tree.node_channel_idx.cpu().to(torch.long)
    signs = tree.node_signs.cpu().to(torch.float32)
    thr = tree.node_threshold.cpu().to(torch.float32)
    leaf = tree.node_leaf_index.cpu().to(torch.long)
    left = tree.node_left.cpu().to(torch.long)
    right = tree.node_right.cpu().to(torch.long)
    is_leaf = leaf >= 0
    idx = torch.arange(n, dtype=torch.long)
    left = torch.where(is_leaf, idx, left)
    right = torch.where(is_leaf, idx, right)
    pad = n_max - n

    def grow(t: torch.Tensor, fill) -> torch.Tensor:
        if pad == 0:
            return t
        g = torch.full((pad,) + tuple(t.shape[1:]), fill, dtype=t.dtype)
        return torch.cat([t, g], dim=0)

    ch = grow(ch, 0)
    signs = grow(signs, 1.0)
    thr = grow(thr, float("inf"))
    leaf = grow(leaf, 0)
    left = grow(left, 0).clone()
    right = grow(right, 0).clone()
    if pad > 0:
        self_id = torch.arange(n, n_max, dtype=torch.long)
        left[n:] = self_id
        right[n:] = self_id
    return {"ch": ch, "signs": signs, "thr": thr, "leaf": leaf,
            "left": left, "right": right}


def _bundle_size_mib(bundle: dict) -> float:
    total = 0
    for v in bundle.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            total += sum(t.numel() * t.element_size()
                         for t in v.values() if isinstance(t, torch.Tensor))
    return total / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True,
                        help="a layer's _as_v4/checkpoints dir holding "
                             "replacement_g*.pt")
    parser.add_argument("--output", required=True, help="bundle .pt path")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        sys.exit(f"[prep_lut_bundle] not a directory: {ckpt_dir}")

    files = {}
    for p in sorted(ckpt_dir.glob("replacement_g*.pt")):
        gid = int(p.stem.split("g")[-1])
        files[gid] = p
    if not files:
        sys.exit(f"[prep_lut_bundle] no replacement_g*.pt under {ckpt_dir}")
    gids = sorted(files)
    if gids != list(range(len(gids))):
        sys.exit(f"[prep_lut_bundle] non-contiguous group ids {gids} "
                 f"(expected 0..{len(gids) - 1})")

    ckpts = {}
    for gid in gids:
        ckpts[gid] = _load_ckpt(files[gid])

    ref = ckpts[gids[0]]
    group_size = int(ref["group_size"])
    num_groups = len(gids)
    hidden = num_groups * group_size
    coarse_num_bits = int(ref["coarse_num_bits"])
    residual_num_bits = int(ref["residual_num_bits"])
    print(f"[prep_lut_bundle] hidden={hidden} group_size={group_size} "
          f"num_groups={num_groups} coarse_bits={coarse_num_bits} "
          f"residual_bits={residual_num_bits}")

    coarse_entries = 2 ** coarse_num_bits
    resid_entries = 2 ** residual_num_bits

    # --- validation across all 32 files ---
    coarse_slices = []
    resid_tables = []
    coarse_tree = ref["addresses"][0]
    resid_trees = {}
    for gid in gids:
        ck = ckpts[gid]
        if ck.get("target_mode") != "direct":
            sys.exit(f"[prep_lut_bundle] {files[gid].name}: target_mode="
                     f"{ck.get('target_mode')!r} (expected 'direct')")
        if int(ck["group_size"]) != group_size:
            sys.exit(f"[prep_lut_bundle] {files[gid].name}: group_size "
                     f"{ck['group_size']} != {group_size}")
        if int(ck["coarse_num_bits"]) != coarse_num_bits or \
                int(ck["residual_num_bits"]) != residual_num_bits:
            sys.exit(f"[prep_lut_bundle] {files[gid].name}: bit counts "
                     f"differ from group {gids[0]}")
        addresses = ck["addresses"]
        tables = ck["lut_tables"]
        if addresses[0].node_channel_idx.shape[0] != \
                coarse_tree.node_channel_idx.shape[0]:
            sys.exit(f"[prep_lut_bundle] {files[gid].name}: coarse tree "
                     f"node count differs from group {gids[0]}")
        coarse_slice = tables[0].detach().cpu().half().reshape(
            coarse_entries, group_size)
        resid_table = tables[1].detach().cpu().half().reshape(
            resid_entries, group_size)
        if tuple(coarse_slice.shape) != (coarse_entries, group_size) or \
                tuple(resid_table.shape) != (resid_entries, group_size):
            sys.exit(f"[prep_lut_bundle] {files[gid].name}: bad table "
                     f"shape coarse={tuple(coarse_slice.shape)} "
                     f"resid={tuple(resid_table.shape)}")
        coarse_slices.append(coarse_slice)
        resid_tables.append(resid_table)
        resid_trees[gid] = addresses[1]

    coarse_table = torch.cat(coarse_slices, dim=1).contiguous()  # [E_c, H]
    resid_table = torch.stack(resid_tables, dim=0).contiguous()  # [G,E_r,gs]

    # --- trees -> padded stacked tensors ---
    n_c = int(coarse_tree.node_channel_idx.shape[0])
    coarse = _padded_tree_arrays(coarse_tree, n_c)
    n_r = {gid: int(resid_trees[gid].node_channel_idx.shape[0])
           for gid in gids}
    n_r_max = max(n_r.values())
    resid = {}
    for key in ("ch", "signs", "thr", "leaf", "left", "right"):
        resid[key] = torch.stack(
            [_padded_tree_arrays(resid_trees[gid], n_r[gid])[key]
             for gid in gids], dim=0).contiguous()  # [G, n_r_max, ...]
    print(f"[prep_lut_bundle] tree nodes: coarse={n_c} "
          f"resid max={n_r_max} (full would be "
          f"{2 ** (residual_num_bits + 1) - 1})")

    has_lowrank = (ckpt_dir / "lowrank.pt").exists()
    has_pairwise = (ckpt_dir / "pairwise.pt").exists()
    if has_lowrank:
        print("[prep_lut_bundle] WARNING: lowrank.pt present but the "
              "v8 plugin does not implement it (docs/43 P1.3 v1) — the "
              "correction is DROPPED from the bundle")
    if has_pairwise:
        print("[prep_lut_bundle] WARNING: pairwise.pt present but the "
              "v8 plugin does not implement it (docs/43 P1.3 v1) — the "
              "correction is DROPPED from the bundle")

    bundle = {
        "hidden": hidden,
        "group_size": group_size,
        "num_groups": num_groups,
        "coarse_num_bits": coarse_num_bits,
        "residual_num_bits": residual_num_bits,
        "coarse": coarse,
        "resid": resid,
        "coarse_table": coarse_table,
        "resid_table": resid_table,
        "has_lowrank": has_lowrank,
        "has_pairwise": has_pairwise,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"[prep_lut_bundle] wrote {out_path} "
          f"({_bundle_size_mib(bundle):.1f} MiB)")


if __name__ == "__main__":
    main()
