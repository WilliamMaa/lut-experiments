#!/usr/bin/env python3
"""audit_teacher_weights.py — 对比 teacher .pt 与真实模型 shared_expert 权重。

裁决对象：L37/L38 serve dense 比训练数据小 7~13 倍，而 bundle 表忠实匹配
训练数据 -> 怀疑 qwen_35b_shared_expert_l{37,38}.pt 不是从当前 serve 的
模型提取的（或提取源被错误缩放）。

模型是 trust_remote_code 自定义架构，key 名不保证是标准
"model.layers.N.mlp.shared_expert.*"，所以两边都按子串模糊匹配。

判读：三个投影的 norm 比值 teacher/model 若 ~7.4 (L37)、~10 (L38)、~1 (L39)，
直接定罪 teacher 权重来源错误。

用法：
  python tools/audit_teacher_weights.py
"""

import glob
import json
import os
import re

import torch
from safetensors import safe_open

MODEL_PATH = "/home/u/downloads/models/Qwen3.6-35B-A3B"
PT_ROOTS = ["/data/mamingyu", "/home/u/mmy"]

PROJ_SHORT = {
    "gate_proj": "gate_proj.weight",
    "up_proj": "up_proj.weight",
    "down_proj": "down_proj.weight",
}

def find_pt(layer_idx):
    for root in PT_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, f"qwen_35b_shared_expert_l{layer_idx}*.pt")))
        if hits:
            return hits[0]
    return None

def collect_model_keys(layer_idx):
    """返回 {short_key: (fname, full_key)}，按 shared_expert 子串匹配。"""
    pat = re.compile(rf"layers[.\[]?{layer_idx}[.\]]?.*shared_expert", re.IGNORECASE)
    out = {}
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    files = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            wm = json.load(f)["weight_map"]
        for full, fname in wm.items():
            if pat.search(full):
                for tag, short in PROJ_SHORT.items():
                    if tag in full:
                        out[short] = (fname, full)
                if "shared_expert_gate" in full:
                    out["shared_expert_gate.weight"] = (fname, full)
    else:
        for st in sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))):
            files.append(st)
        with safe_open(files[0], framework="pt") as f:
            all_keys = f.keys()
        for full in all_keys:
            if pat.search(full):
                for tag, short in PROJ_SHORT.items():
                    if tag in full:
                        out[short] = (files[0], full)
    return out

for layer_idx in (37, 38, 39):
    print(f"\n=== layer {layer_idx} ===")
    pt_path = find_pt(layer_idx)
    if pt_path is None:
        print("  teacher .pt not found")
        continue
    print(f"  teacher: {pt_path}")
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    print(f"  teacher keys: {sorted(sd.keys())}")

    model_map = collect_model_keys(layer_idx)
    if not model_map:
        print("  [warn] no shared_expert keys matched in model index; "
              "sample index keys:")
        with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        sample = [k for k in wm if f"layers" in k][:10]
        for k in sample:
            print(f"    {k}")
        continue

    for short, (fname, full) in sorted(model_map.items()):
        m = None
        with safe_open(os.path.join(MODEL_PATH, fname), framework="pt") as f:
            m = f.get_tensor(full).float()
        t = sd.get(short)
        if t is None:
            print(f"  {short}: teacher key missing (model has {full}, "
                  f"norm={m.norm():.2f})")
            continue
        t = t.float()
        if t.shape != m.shape:
            print(f"  {short}: SHAPE MISMATCH pt={tuple(t.shape)} model={tuple(m.shape)}")
            continue
        ratio = (t.norm() / m.norm()).item()
        cos = torch.nn.functional.cosine_similarity(t.flatten(), m.flatten(), dim=0).item()
        print(f"  {short}: norm_ratio(teacher/model)={ratio:.3f} cos={cos:.4f} "
              f"(model norm={m.norm():.2f})")
        if "gate" in short:
            sg = torch.sigmoid(t)
            print(f"    gate sigmoid: mean={sg.mean():.4f} min={sg.min():.4f} "
                  f"max={sg.max():.4f}")
