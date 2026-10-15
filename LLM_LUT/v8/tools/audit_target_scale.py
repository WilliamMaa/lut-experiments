#!/usr/bin/env python3
"""audit_target_scale.py — 裁决 L37/L38 叶值膨胀是在数据侧还是 serve 侧。

判读规则：
  metadata.json 的 hook_path 如果是 "...mlp"（完整 MoE block）而不是
  "...mlp.shared_expert"，采集的 y 就是 shared_expert + 全部 routed experts
  之和，幅值天然是 shared expert 的数倍 -> 膨胀在训练数据里，serve 无罪。
  dataset output 文件的逐行 norm 若 ~1200-2900（对比 serve 实测 dense |ydf|
  L37~190 / L38~215），同样坐实数据侧膨胀。

用法（在 lut_py310 / vllm_py310 任意环境）：
  python tools/audit_target_scale.py
"""

import json
import glob
import os

import torch

ROOTS = ["/home/u/mmy", "/data/mamingyu"]
LAYERS = [
    ("layer37_shared_expert_v3_onpolicy", 190.0),
    ("layer38_shared_expert_v3_onpolicy", 215.0),
    ("layer39_shared_expert_v3_onpolicy_multilayer", 1377.0),
]

def find_ds(name):
    for root in ROOTS:
        # 单层布局: <ds>/metadata.json；多层布局: <ds>/layer{L}/metadata.json
        pats = [os.path.join(root, "datasets", name, "metadata.json"),
                os.path.join(root, "datasets", name, "layer*", "metadata.json")]
        for pat in pats:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[0]
    return None

def find_outputs(meta_path):
    d = os.path.dirname(meta_path)
    outs = sorted(glob.glob(os.path.join(d, "output", "*.pt")))
    if not outs:
        outs = sorted(glob.glob(os.path.join(d, "*.pt")))
    return outs

for name, dense_norm in LAYERS:
    print(f"\n=== {name}  (serve dense |ydf| ~ {dense_norm}) ===")
    meta = find_ds(name)
    if meta is None:
        print("  [skip] metadata.json not found in either root")
        continue
    with open(meta) as f:
        m = json.load(f)
    hook = m.get("hook_path", "?")
    print(f"  hook_path = {hook}")
    print(f"  -> {'SUSPECT (full mlp, y includes routed experts!)' if hook.endswith('mlp') else 'ok (shared_expert)'}")
    outs = find_outputs(meta)
    print(f"  output files: {len(outs)}")
    for f in outs[:3]:
        y = torch.load(f, map_location="cpu", weights_only=True)
        y = y.float()
        if y.dim() == 3:
            y = y.reshape(-1, y.shape[-1])
        n = y.norm(dim=-1)
        print(f"    {os.path.basename(f)}: rows={y.shape[0]} "
              f"|y| mean={n.mean():.1f} p10={n.quantile(0.1):.1f} p90={n.quantile(0.9):.1f}")

# bundle 表行 norm（与上面 dataset norm 对比）
print("\n=== bundle coarse_table row norms ===")
for layer in (37, 38, 39):
    for root in ROOTS:
        p = os.path.join(root, "lut_bundles", f"layer{layer}.pt")
        if os.path.exists(p):
            b = torch.load(p, map_location="cpu", weights_only=True)
            ct = b["coarse_table"].float()
            rt = b["resid_table"].float()
            cn = ct.norm(dim=-1)
            occupied = cn > 0
            print(f"  layer{layer}: coarse rows={tuple(ct.shape)} "
                  f"|row| occupied_mean={cn[occupied].mean():.1f} "
                  f"resid |row| mean={rt.norm(dim=-1).mean():.2f}")
            break
    else:
        print(f"  layer{layer}: bundle not found")
