#!/usr/bin/env python3
"""audit_teacher_vs_data.py — 用提取的 teacher expert 重放数据集 (x, y)。

裁决：训练数据 y ~15.5/token，serve vLLM dense ~2.1/token，权重已验相同、
x 尺度已验相同。两边必有一边不是"真 dense"。

  teacher(x) ≈ y（cos>0.99, norm 匹配）-> 数据集自洽，膨胀在 vLLM 端口
    （HF 与 vLLM 对 shared_expert 的实现缩放不一致，LUT 匹配 HF ->
     装到 vLLM 上就是 7x 投毒，HF 链路因此健康）。
  teacher(x) ≈ y/7 -> 采集时保存的 y 被放大，collector 有罪。

纯 CPU，不加载模型。用法：
  python tools/audit_teacher_vs_data.py
"""

import glob
import os

import torch

PT_ROOTS = ["/data/mamingyu", "/home/u/mmy"]
DS_ROOTS = ["/home/u/mmy", "/data/mamingyu"]
LAYERS = [
    (37, "layer37_shared_expert_v3_onpolicy"),
    (38, "layer38_shared_expert_v3_onpolicy"),
    (39, "layer39_shared_expert_v3_onpolicy_multilayer"),
]

class TeacherExpert(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.gate_proj = torch.nn.Linear(sd["gate_proj.weight"].shape[1],
                                         sd["gate_proj.weight"].shape[0],
                                         bias=False)
        self.up_proj = torch.nn.Linear(sd["up_proj.weight"].shape[1],
                                       sd["up_proj.weight"].shape[0],
                                       bias=False)
        self.down_proj = torch.nn.Linear(sd["down_proj.weight"].shape[1],
                                         sd["down_proj.weight"].shape[0],
                                         bias=False)
        self.gate_proj.weight.data = sd["gate_proj.weight"].float()
        self.up_proj.weight.data = sd["up_proj.weight"].float()
        self.down_proj.weight.data = sd["down_proj.weight"].float()
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

def find_pt(layer_idx):
    for root in PT_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, f"qwen_35b_shared_expert_l{layer_idx}*.pt")))
        if hits:
            return hits[0]
    return None

def find_ds(name):
    for root in DS_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, "datasets", name, "input", "*.pt")))
        if hits:
            return os.path.join(root, "datasets", name)
    return None

for layer_idx, ds_name in LAYERS:
    print(f"\n=== layer {layer_idx} ===")
    pt = find_pt(layer_idx)
    ds = find_ds(ds_name)
    if pt is None or ds is None:
        print(f"  missing: pt={pt is not None} ds={ds is not None}")
        continue
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    teacher = TeacherExpert(sd).eval()

    in_files = sorted(glob.glob(os.path.join(ds, "input", "*.pt")))[:3]
    for inf in in_files:
        outf = inf.replace("/input/", "/output/")
        if not os.path.exists(outf):
            print(f"  no matching output for {inf}")
            continue
        x = torch.load(inf, map_location="cpu", weights_only=True).float()
        y = torch.load(outf, map_location="cpu", weights_only=True).float()
        with torch.no_grad():
            y_hat = teacher(x)
        cos = torch.nn.functional.cosine_similarity(
            y_hat, y, dim=-1).mean().item()
        print(f"  {os.path.basename(inf)}: rows={x.shape[0]} "
              f"|y_file|={y.norm(dim=-1).mean():.2f} "
              f"|teacher(x)|={y_hat.norm(dim=-1).mean():.2f} "
              f"ratio(file/teacher)={(y.norm(dim=-1).mean()/y_hat.norm(dim=-1).mean()):.3f} "
              f"cos={cos:.4f}")
