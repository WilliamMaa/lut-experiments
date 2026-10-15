#!/usr/bin/env python3
"""audit_teacher_weights.py — 对比 teacher .pt 与真实模型 shared_expert 权重。

裁决对象：L37/L38 serve dense 比训练数据小 7~13 倍，而 bundle 表忠实匹配
训练数据 -> 怀疑 qwen_35b_shared_expert_l{37,38}.pt 不是从当前 serve 的
模型提取的（或提取源被错误缩放）。

判读：三个投影的 norm 比值 teacher/model 若 ~7.4 (L37)、~10 (L38)、~1 (L39)，
直接定罪 teacher 权重来源错误。注意只看比值，不逐元素对比（提取时 bf16）。

用法：
  python tools/audit_teacher_weights.py
"""

import glob
import json
import os

import torch
from safetensors import safe_open

MODEL_PATH = "/home/u/downloads/models/Qwen3.6-35B-A3B"
PT_ROOTS = ["/home/u/mmy", "/data/mamingyu"]

def load_model_layer_weights(layer_idx):
    """从 safetensors（含 index）或单文件加载 shared_expert 三个投影。"""
    want = {
        f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight": "gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight": "up_proj.weight",
        f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight": "down_proj.weight",
    }
    out = {}
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        file_for = {}
        for k in want:
            if k not in index:
                print(f"  [warn] {k} not in model index")
                continue
            file_for.setdefault(index[k], []).append(k)
        for fname, keys in file_for.items():
            with safe_open(os.path.join(MODEL_PATH, fname), framework="pt") as f:
                for k in keys:
                    out[want[k]] = f.get_tensor(k).float()
    else:
        for st in sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))):
            with safe_open(st, framework="pt") as f:
                for k, short in want.items():
                    if k in out:
                        continue
                    try:
                        out[short] = f.get_tensor(k).float()
                    except KeyError:
                        pass
    return out

def find_pt(layer_idx):
    for root in PT_ROOTS:
        for name in (f"qwen_35b_shared_expert_l{layer_idx}.pt",
                     f"qwen_35b_shared_expert_l{layer_idx}_*.pt"):
            hits = sorted(glob.glob(os.path.join(root, name)))
            if hits:
                return hits[0]
    return None

for layer_idx in (37, 38, 39):
    print(f"\n=== layer {layer_idx} ===")
    pt_path = find_pt(layer_idx)
    if pt_path is None:
        print("  teacher .pt not found")
        continue
    print(f"  teacher: {pt_path}")
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    mw = load_model_layer_weights(layer_idx)
    for key in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        t = sd.get(key)
        m = mw.get(key)
        if t is None or m is None:
            print(f"  {key}: MISSING (pt={t is not None}, model={m is not None})")
            continue
        t, m = t.float(), m.float()
        ratio = t.norm() / m.norm()
        cos = torch.nn.functional.cosine_similarity(
            t.flatten(), m.flatten(), dim=0)
        print(f"  {key}: shape pt={tuple(t.shape)} model={tuple(m.shape)} "
              f"norm_ratio={ratio:.3f} cos={cos:.4f}")
    if "shared_expert_gate.weight" in sd:
        print(f"  shared_expert_gate.weight: {tuple(sd['shared_expert_gate.weight'].shape)} "
              f"(sigmoid={torch.sigmoid(sd['shared_expert_gate.weight'].float()).item():.4f})")
