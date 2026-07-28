"""
LUT 替换 1x1 Conv - v10
========================
目标：用 spatial group multi-head small LUT delta 替换 YOLOv8n backbone 的 6.cv1 / 8.cv1。

核心约束：
  - 推理阶段仍然完全是 LUT 查表，替代指定 1x1 Conv。
  - 不引入额外 Conv / Linear / 大矩阵乘。
  - 使用多个小 LUT 表，每个表只服务一个 channel group，避免单表过大。
  - 训练阶段允许 teacher、raw distill、output distill、QAT、staged training。

v10 相比 v9 的主要变化：
  P0: 支持 --target_mode 6 / 8 / 68，分别跑 6.cv1、8.cv1、6+8。
  P1: image-level gamma/beta -> spatial token-level group delta。
  P2: one wide LUT -> many small group/head LUTs。
  P3: address 从全局 abs(mean) 改为 per-spatial scalar quantization。
  P4: Phase0 拟合 old.conv(x) raw output，而不是 Conv+BN+Act 后输出。
  P5: Phase1 加 raw conv distill + module output distill。
  P6: Phase2 对 LUT table 做 per-group/head fake quant。

推荐用法：
  # Phase0 prefit
  python lut_coco_v10.py --phase 0 --target_mode 6  --device 3 --prefit_images 5000 --prefit_epochs 10
  python lut_coco_v10.py --phase 0 --target_mode 8  --device 3 --prefit_images 5000 --prefit_epochs 10
  python lut_coco_v10.py --phase 0 --target_mode 68 --device 3 --prefit_images 5000 --prefit_epochs 10

  # Phase1 detection fine-tune
  python lut_coco_v10.py --phase 1 --target_mode 6  --device 3 --epochs 50
  python lut_coco_v10.py --phase 1 --target_mode 8  --device 3 --epochs 50
  python lut_coco_v10.1.py --phase 1 --target_mode 68 --device 3 --epochs 50

  # Phase2 QAT
  python lut_coco_v10.py --phase 2 --target_mode 6  --device 3 --epochs 40
  python lut_coco_v10.py --phase 2 --target_mode 8  --device 3 --epochs 40
  python lut_coco_v10.py --phase 2 --target_mode 68 --device 3 --epochs 40

注意：
  - 验证/部署阶段只保留 student LUT，不需要 teacher。
  - teacher 只用于 Phase0 prefit 和 Phase1/2 training loss。
  - 默认 COCO 路径为 /data1/datasets/coco。
"""

import os
import sys
import glob
import json
import yaml
import math
import copy
import types
import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.models.yolo.detect import DetectionTrainer


# ============================================================
# Config
# ============================================================

TARGET_MODES = {
    "6": ("6.cv1",),
    "8": ("8.cv1",),
    "68": ("6.cv1", "8.cv1"),
}

DEFAULT_COCO_ROOT = "/data1/datasets/coco"
PROJECT = "runs/detect/runs/lut_v10"
BASELINE_MAP = 0.373


# ============================================================
# Basic tools
# ============================================================

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def get_submodule_by_path(root: nn.Module, path: str):
    cur = root
    for part in path.split('.'):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def set_submodule_by_path(root: nn.Module, path: str, new_module: nn.Module):
    parts = path.split('.')
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def get_target_paths(target_mode: str):
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Invalid target_mode={target_mode}, choose from {list(TARGET_MODES)}")
    return TARGET_MODES[target_mode]


def get_prefit_ckpt(target_mode: str):
    return f"{PROJECT}/v10_{target_mode}_phase0_prefit.pt"


def get_phase1_ckpt(target_mode: str):
    return f"{PROJECT}/v10_{target_mode}_phase1/weights/best.pt"


def quantize_ste_scaled_per_table(w, qmax, eps=1e-8):
    """Per [group, head] scaled fake quant.

    w shape: [G, heads, lut_size, group_size]
    scale shape: [G, heads, 1, 1]
    """
    scale = w.detach().abs().amax(dim=(2, 3), keepdim=True).clamp_min(eps) / qmax
    q = RoundSTE.apply((w / scale).clamp(-qmax, qmax))
    return q * scale


def get_quant_bits(channels: int):
    # 对两个目标层默认用 8-bit table；如需极端压缩可以命令行改。
    return 8


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# ============================================================
# Dataset config and image loader
# ============================================================

def prepare_dataset_config(root=DEFAULT_COCO_ROOT):
    if not os.path.exists(root):
        print(f"[ERROR] COCO root not found: {root}")
        sys.exit(1)
    cfg = {
        'path': root,
        'train': 'images/train2017',
        'val': 'images/val2017',
        'nc': 80,
        'names': {
            0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
            6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',11:'stop sign',
            12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',
            19:'cow',20:'elephant',21:'bear',22:'zebra',23:'giraffe',24:'backpack',
            25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
            31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',
            36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
            41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',
            48:'sandwich',49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',
            54:'donut',55:'cake',56:'chair',57:'couch',58:'potted plant',59:'bed',
            60:'dining table',61:'toilet',62:'tv',63:'laptop',64:'mouse',65:'remote',
            66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',71:'sink',
            72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',
            78:'hair drier',79:'toothbrush'
        }
    }
    with open("my_coco_config.yaml", 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return "my_coco_config.yaml"


class ImageOnlyDataset(Dataset):
    """Phase0 用的轻量图片 loader。

    注意：这里仍然是简单 resize，不完全等价于 YOLO train transform。
    v10 的重点是 raw feature prefit。如果后续还要提升，可以再接入 Ultralytics dataset transform。
    """
    def __init__(self, root=DEFAULT_COCO_ROOT, split="train2017", imgsz=640, limit=1000, seed=0):
        img_dir = Path(root) / "images" / split
        paths = sorted([str(p) for p in img_dir.glob("*.jpg")])
        if not paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        random.Random(seed).shuffle(paths)
        self.paths = paths[:limit]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        if im is None:
            raise RuntimeError(f"Failed to read image: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        return im


# ============================================================
# v10 LUT module
# ============================================================

class SpatialGroupMultiHeadLUTDelta(nn.Module):
    """Small-table spatial group multi-head LUT delta replacement for Conv1x1.

    It approximates old.conv(x) using:
        raw_g = x_g + alpha_g * mean_h LUT[g,h][addr_g,h(x)]
        out = old_act(old_bn(raw))

    Inference compute is scalar quantization + LUT lookup + add.
    No Conv/Linear/matrix multiplication is introduced in this replacement module.
    """
    def __init__(self, old: Conv, group_size=16, heads=2, lut_size=128,
                 quant_bits=None, qnoise=0.0, addr_idx=None, addr_mean=None,
                 addr_std=None, addr_clip=3.0, init_alpha=0.25):
        super().__init__()
        if not (isinstance(old, Conv) and old.conv.kernel_size == (1, 1)):
            raise TypeError(f"old must be Ultralytics Conv 1x1, got {type(old)}")
        ci, co = old.conv.in_channels, old.conv.out_channels
        if ci != co:
            raise ValueError(f"v10 currently requires ci == co, got {ci}->{co}")
        if ci % group_size != 0:
            raise ValueError(f"channels={ci} must be divisible by group_size={group_size}")

        self.channels = ci
        self.group_size = int(group_size)
        self.groups = ci // self.group_size
        self.heads = int(heads)
        self.lut_size = int(lut_size)
        self.quant_bits = quant_bits
        self.qnoise = float(qnoise)
        self.addr_clip = float(addr_clip)

        # Many small tables: [G, H, L, group_size]
        self.tables = nn.Parameter(torch.zeros(self.groups, self.heads, self.lut_size, self.group_size))
        nn.init.normal_(self.tables, mean=0.0, std=0.01)

        # Residual strength. tanh(alpha_raw) is used in forward.
        # Start non-zero so gradients are not too weak, but still stable.
        alpha_init = torch.ones(self.groups) * math.atanh(max(min(init_alpha, 0.95), -0.95))
        self.alpha_raw = nn.Parameter(alpha_init)

        # Copy BN/Act from old Conv. This keeps output distribution closer to teacher.
        self.bn = copy.deepcopy(old.bn)
        self.act = copy.deepcopy(old.act)

        if addr_idx is None:
            # Default local address: for each group/head choose deterministic channels in that group.
            idx = []
            for g in range(self.groups):
                base = g * self.group_size
                idx.append([base + ((h * max(1, self.group_size // self.heads)) % self.group_size) for h in range(self.heads)])
            addr_idx = torch.tensor(idx, dtype=torch.long)
        else:
            addr_idx = torch.as_tensor(addr_idx, dtype=torch.long)
            if addr_idx.shape != (self.groups, self.heads):
                raise ValueError(f"addr_idx shape must be {(self.groups, self.heads)}, got {tuple(addr_idx.shape)}")

        if addr_mean is None:
            addr_mean = torch.zeros(self.groups, self.heads)
        else:
            addr_mean = torch.as_tensor(addr_mean, dtype=torch.float32)
        if addr_std is None:
            addr_std = torch.ones(self.groups, self.heads)
        else:
            addr_std = torch.as_tensor(addr_std, dtype=torch.float32).clamp_min(1e-6)

        self.register_buffer("addr_idx", addr_idx.clone())
        self.register_buffer("addr_mean", addr_mean.clone())
        self.register_buffer("addr_std", addr_std.clone())

        # Debug hooks for distill.
        self.last_raw = None
        self.last_out = None

    def _lookup_tables(self):
        if self.quant_bits is None:
            return self.tables
        qmax = 2 ** (self.quant_bits - 1) - 1
        q_tables = quantize_ste_scaled_per_table(self.tables, qmax)
        if self.training:
            return (1.0 - self.qnoise) * self.tables + self.qnoise * q_tables
        return q_tables

    def _quant_float_all(self, z):
        """
        z: [B, K, H, W], K = groups * heads
        """
        G, Hd = self.groups, self.heads
        mean = self.addr_mean.reshape(1, G * Hd, 1, 1).to(z.device, z.dtype)
        std = self.addr_std.reshape(1, G * Hd, 1, 1).to(z.device, z.dtype).clamp_min(1e-6)

        z_norm = (z - mean) / std
        z_clip = z_norm.clamp(-self.addr_clip, self.addr_clip)
        qf = (z_clip + self.addr_clip) / (2.0 * self.addr_clip) * (self.lut_size - 1)
        return qf

    def _interp_lookup_all(self, tables, qf):
        """
        tables: [G, heads, L, group_size]
        qf:     [B, K, H, W], K = G * heads

        returns:
            d: [B, G, heads, group_size, H, W]
        """
        B, K, H, W = qf.shape
        G, Hd, L, GS = self.groups, self.heads, self.lut_size, self.group_size

        tables_flat = tables.reshape(K, L, GS)  # [K, L, GS]

        q_low = torch.floor(qf).long().clamp(0, L - 1)
        q_high = (q_low + 1).clamp(0, L - 1)
        w = (qf - q_low.float()).unsqueeze(-1)  # [B, K, H, W, 1]

        k_idx = torch.arange(K, device=qf.device).view(1, K, 1, 1).expand(B, K, H, W)

        lo = tables_flat[k_idx, q_low]   # [B, K, H, W, GS]
        hi = tables_flat[k_idx, q_high]  # [B, K, H, W, GS]

        d = (1.0 - w) * lo + w * hi      # [B, K, H, W, GS]

        d = d.view(B, G, Hd, H, W, GS)
        d = d.permute(0, 1, 2, 5, 3, 4).contiguous()  # [B, G, heads, GS, H, W]
        return d

    def forward_raw(self, x):
        B, C, H, W = x.shape
        assert C == self.channels

        tables = self._lookup_tables()

        # [G, heads] -> [K]
        addr_flat = self.addr_idx.reshape(-1).to(x.device)

        # Gather all address channels at once: [B, K, H, W]
        z = x.index_select(1, addr_flat)

        # Quantize all addresses at once
        qf = self._quant_float_all(z)

        # Lookup all groups/heads at once
        d = self._interp_lookup_all(tables, qf)  # [B, G, heads, GS, H, W]

        # Fuse heads
        delta = d.mean(dim=2)  # [B, G, GS, H, W]

        # Residual group delta
        xg = x.view(B, self.groups, self.group_size, H, W)
        alpha = torch.tanh(self.alpha_raw).to(x.dtype).view(1, self.groups, 1, 1, 1)

        raw = xg + alpha * delta
        raw = raw.reshape(B, C, H, W).contiguous()

        self.last_raw = raw
        return raw

    def forward(self, x):
        raw = self.forward_raw(x)
        if raw.dtype != x.dtype:
            raw = raw.to(x.dtype)
        out = self.act(self.bn(raw))
        self.last_out = out
        return out


# ============================================================
# Surgery / quant controls / reports
# ============================================================

def set_lut_qbits(module, enabled=True, qbits=8):
    for m in module.modules():
        if isinstance(m, SpatialGroupMultiHeadLUTDelta):
            m.quant_bits = qbits if enabled else None


def set_lut_qnoise(module, qnoise):
    for m in module.modules():
        if isinstance(m, SpatialGroupMultiHeadLUTDelta):
            m.qnoise = float(qnoise)


def set_lut_bn_trainable(module, enabled: bool):
    for m in module.modules():
        if isinstance(m, SpatialGroupMultiHeadLUTDelta):
            for p in m.bn.parameters():
                p.requires_grad_(enabled)


def set_only_lut_trainable(model, bn_trainable=False):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, SpatialGroupMultiHeadLUTDelta):
            m.tables.requires_grad_(True)
            m.alpha_raw.requires_grad_(True)
            for p in m.bn.parameters():
                p.requires_grad_(bn_trainable)


def set_all_trainable(model):
    for p in model.parameters():
        p.requires_grad_(True)


def collect_lut_report(replaced, quant=False, qbits=8):
    fp_total = 0.0
    q_total = 0.0
    for r in replaced:
        G, Hh, L, gs = r['groups'], r['heads'], r['lut_size'], r['group_size']
        fp_kb = G * Hh * L * gs * 4 / 1024
        q_kb = G * Hh * L * gs * qbits / 8 / 1024 if quant else fp_kb
        r['storage_fp_kb'] = fp_kb
        r['storage_q_kb'] = q_kb
        r['quant_bits'] = qbits if quant else None
        fp_total += fp_kb
        q_total += q_kb
    return fp_total, q_total


def load_prefit_ckpt(target_mode: str):
    path = get_prefit_ckpt(target_mode)
    if not os.path.exists(path):
        print(f"  [WARN] prefit ckpt not found: {path}")
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


def replace_target_convs_with_lut(model_seq, target_paths, prefitted=None,
                                  group_size=16, heads=2, lut_size=128, addr_clip=3.0):
    replaced = []
    for path in target_paths:
        old = get_submodule_by_path(model_seq, path)
        if not (isinstance(old, Conv) and old.conv.kernel_size == (1, 1)):
            raise TypeError(f"Target {path} is not Ultralytics Conv 1x1: {type(old)}")
        ci, co = old.conv.in_channels, old.conv.out_channels
        if ci != co:
            raise ValueError(f"Target {path} is not ci==co: {ci}->{co}")
        if ci % group_size != 0:
            raise ValueError(f"Target {path} channels={ci} not divisible by group_size={group_size}")

        layer_prefit = None
        if prefitted and path in prefitted.get('lut_state', {}):
            layer_prefit = prefitted

        addr_idx = layer_prefit['addr_idx'][path] if layer_prefit else None
        addr_mean = layer_prefit['addr_mean'][path] if layer_prefit else None
        addr_std = layer_prefit['addr_std'][path] if layer_prefit else None

        lut = SpatialGroupMultiHeadLUTDelta(
            old=old,
            group_size=group_size,
            heads=heads,
            lut_size=lut_size,
            addr_idx=addr_idx,
            addr_mean=addr_mean,
            addr_std=addr_std,
            addr_clip=addr_clip,
        )
        if layer_prefit:
            missing, unexpected = lut.load_state_dict(layer_prefit['lut_state'][path], strict=False)
            print(f"  [PREFIT] loaded {path}, missing={len(missing)} unexpected={len(unexpected)}")

        set_submodule_by_path(model_seq, path, lut)
        replaced.append({
            'path': path,
            'c_in': ci,
            'c_out': co,
            'groups': ci // group_size,
            'group_size': group_size,
            'heads': heads,
            'lut_size': lut_size,
        })
        per_table_kb = lut_size * group_size * 4 / 1024
        print(f"  [SURGERY] {path}: Conv({ci}->{co}) -> SpatialGroupMultiHeadLUTDelta "
              f"G={ci//group_size} H={heads} L={lut_size} gs={group_size} table={per_table_kb:.1f}KB")
    return replaced


def load_weights_flexible(yolo_model, weights):
    if not (weights and os.path.exists(weights)):
        print(f"  [WARN] ckpt not found: {weights}")
        return False
    print(f"  Loading weights: {weights}")
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        m, u = yolo_model.model.load_state_dict(ckpt['model_state'], strict=False)
        print(f"  Loaded model_state. Missing:{len(m)} Unexpected:{len(u)}")
        return True

    state = None
    if isinstance(ckpt, dict):
        for k in ['ema', 'model']:
            raw = ckpt.get(k)
            if raw is not None:
                state = raw.state_dict() if hasattr(raw, 'state_dict') else (raw if isinstance(raw, dict) else None)
                if state:
                    break
    if state:
        m, u = yolo_model.model.load_state_dict(state, strict=False)
        print(f"  Loaded YOLO ckpt. Missing:{len(m)} Unexpected:{len(u)}")
        return True
    print("  [WARN] Could not extract state_dict")
    return False


# ============================================================
# Phase0 calibration and prefit
# ============================================================

@torch.no_grad()
def calibrate_v10_addr(teacher_model, target_paths, loader, device,
                       group_size=16, heads=2, addr_clip=3.0, max_batches=999999):
    """Calibrate per-group/head address channels and quant stats.

    For each target path:
      - head 0: channel with largest activation variance inside group.
      - head 1: channel with largest correlation proxy to residual magnitude inside group.
        residual magnitude proxy is mean_abs(old.conv(x)-x) for that group.
      - more heads alternate between these sorted candidates.
    """
    teacher_model.eval()
    results = {}

    for path in target_paths:
        old = get_submodule_by_path(teacher_model.model, path)
        C = old.conv.in_channels
        G = C // group_size

        sum_x = torch.zeros(C, device=device)
        sum_x2 = torch.zeros(C, device=device)
        n_x = torch.zeros(C, device=device)

        # corr proxy per group/channel: E[abs(x_c) * residual_mag_group]
        corr_sum = torch.zeros(G, group_size, device=device)
        corr_n = torch.zeros(G, group_size, device=device)

        captures = {}
        handle = old.register_forward_pre_hook(lambda mod, inp: captures.setdefault('x', inp[0].detach()))
        for bi, imgs in enumerate(loader):
            if bi >= max_batches:
                break
            captures.clear()
            imgs = imgs.to(device, non_blocking=True)
            _ = teacher_model(imgs)
            x = captures.get('x')
            if x is None:
                continue
            raw = old.conv(x)
            residual = raw - x
            B, _, H, W = x.shape
            flat = x.float().permute(1, 0, 2, 3).reshape(C, -1)
            sum_x += flat.sum(dim=1)
            sum_x2 += (flat ** 2).sum(dim=1)
            n_x += flat.shape[1]

            xg = x.float().view(B, G, group_size, H, W)
            rg = residual.float().view(B, G, group_size, H, W)
            rmag = rg.abs().mean(dim=2, keepdim=True)  # [B,G,1,H,W]
            proxy = (xg.abs() * rmag).mean(dim=(0, 3, 4))  # [G, group_size]
            corr_sum += proxy
            corr_n += 1
        handle.remove()

        mean = sum_x / n_x.clamp_min(1)
        var = sum_x2 / n_x.clamp_min(1) - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        corr = corr_sum / corr_n.clamp_min(1)

        addr_idx = torch.zeros(G, heads, dtype=torch.long)
        addr_mean = torch.zeros(G, heads, dtype=torch.float32)
        addr_std = torch.ones(G, heads, dtype=torch.float32)

        for g in range(G):
            base = g * group_size
            local_var = var[base:base + group_size]
            local_corr = corr[g]
            var_rank = torch.argsort(local_var, descending=True)
            corr_rank = torch.argsort(local_corr, descending=True)
            chosen = []
            for h in range(heads):
                rank = var_rank if h % 2 == 0 else corr_rank
                pick = None
                for cand in rank.tolist():
                    if cand not in chosen:
                        pick = cand
                        break
                if pick is None:
                    pick = rank[0].item()
                chosen.append(pick)
                ch = base + pick
                addr_idx[g, h] = ch
                addr_mean[g, h] = mean[ch].detach().cpu()
                addr_std[g, h] = std[ch].detach().cpu().clamp_min(1e-6)

        results[path] = {
            'addr_idx': addr_idx.cpu(),
            'addr_mean': addr_mean.cpu(),
            'addr_std': addr_std.cpu(),
        }
        print(f"  [ADDR] {path}: G={G}, heads={heads}, first groups={addr_idx[:min(4,G)].tolist()}")
    return results


def raw_distill_loss(pred_raw, target_raw, stat_weight=0.05, cos_weight=0.10):
    l_raw = F.smooth_l1_loss(pred_raw, target_raw)
    # cosine over sampled flattened tokens/channels to avoid huge overhead
    p = pred_raw.float().flatten(2).transpose(1, 2).reshape(-1, pred_raw.shape[1])
    t = target_raw.float().flatten(2).transpose(1, 2).reshape(-1, target_raw.shape[1])
    if p.shape[0] > 8192:
        idx = torch.randperm(p.shape[0], device=p.device)[:8192]
        p = p[idx]
        t = t[idx]
    l_cos = 1.0 - F.cosine_similarity(p, t, dim=1).mean()
    l_mean = F.mse_loss(pred_raw.mean(dim=[0, 2, 3]), target_raw.mean(dim=[0, 2, 3]))
    l_std = F.mse_loss(pred_raw.std(dim=[0, 2, 3]), target_raw.std(dim=[0, 2, 3]))
    return l_raw + cos_weight * l_cos + stat_weight * (l_mean + l_std), {
        'raw': safe_float(l_raw.detach()),
        'cos': safe_float(l_cos.detach()),
        'stat': safe_float((l_mean + l_std).detach()),
    }


def prefit_luts_v10(target_mode='68', device=0, root=DEFAULT_COCO_ROOT, imgsz=640,
                    prefit_images=5000, prefit_epochs=10, batch=4, lr=3e-4,
                    group_size=16, heads=2, lut_size=128, addr_clip=3.0):
    os.makedirs(PROJECT, exist_ok=True)
    target_paths = get_target_paths(target_mode)
    print("=" * 70)
    print(f"v10 Phase0: raw-conv prefit | target_mode={target_mode} paths={target_paths}")
    print("=" * 70)

    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    teacher = YOLO('yolov8n.pt').model.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ds = ImageOnlyDataset(root=root, split="train2017", imgsz=imgsz, limit=prefit_images, seed=0)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    print("\n[1/3] Calibrating spatial address channels and quant stats ...")
    cal = calibrate_v10_addr(
        teacher_model=teacher,
        target_paths=target_paths,
        loader=loader,
        device=dev,
        group_size=group_size,
        heads=heads,
        addr_clip=addr_clip,
    )

    print("\n[2/3] Building standalone v10 LUT modules ...")
    luts = nn.ModuleDict()
    old_modules = {}
    for path in target_paths:
        old = get_submodule_by_path(teacher.model, path)
        lut = SpatialGroupMultiHeadLUTDelta(
            old=old,
            group_size=group_size,
            heads=heads,
            lut_size=lut_size,
            addr_idx=cal[path]['addr_idx'],
            addr_mean=cal[path]['addr_mean'],
            addr_std=cal[path]['addr_std'],
            addr_clip=addr_clip,
        ).to(dev).train()
        # Phase0 focuses on raw table fitting. Keep copied BN irrelevant but harmless.
        old_modules[path] = old
        luts[path.replace('.', '_')] = lut
        print(f"  [LUT] {path}: C={old.conv.in_channels}, G={old.conv.in_channels//group_size}, H={heads}, L={lut_size}")

    opt = torch.optim.AdamW(luts.parameters(), lr=lr, weight_decay=1e-4)

    captures = {}
    handles = []
    for path in target_paths:
        old = old_modules[path]
        def make_hook(pth):
            def hook(mod, inp):
                captures[pth] = inp[0].detach()
            return hook
        handles.append(old.register_forward_pre_hook(make_hook(path)))

    print("\n[3/3] Prefitting LUT raw output to old.conv(x) ...")
    last_metrics = {}
    for ep in range(prefit_epochs):
        total = 0.0
        n = 0
        detail_last = []
        for imgs in loader:
            imgs = imgs.to(dev, non_blocking=True)
            captures.clear()
            with torch.no_grad():
                _ = teacher(imgs)

            opt.zero_grad(set_to_none=True)
            loss = imgs.new_tensor(0.0)
            metrics_ep = {}
            for path in target_paths:
                x = captures[path]
                old = old_modules[path]
                with torch.no_grad():
                    y_raw = old.conv(x)
                lut = luts[path.replace('.', '_')]
                pred_raw = lut.forward_raw(x)
                l, m = raw_distill_loss(pred_raw, y_raw)
                loss = loss + l
                metrics_ep[path] = m
                detail_last.append(f"{path}:raw={m['raw']:.4f},cos={m['cos']:.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(luts.parameters(), 5.0)
            opt.step()
            total += safe_float(loss.detach())
            n += 1
            last_metrics = metrics_ep

        print(f"  [PREFIT] epoch={ep+1}/{prefit_epochs} loss={total/max(n,1):.5f} {' '.join(detail_last[-len(target_paths):])}")

    for h in handles:
        h.remove()

    ckpt = {
        'version': 'v10',
        'target_mode': target_mode,
        'target_paths': target_paths,
        'group_size': group_size,
        'heads': heads,
        'lut_size': lut_size,
        'addr_clip': addr_clip,
        'addr_idx': {p: cal[p]['addr_idx'].cpu() for p in target_paths},
        'addr_mean': {p: cal[p]['addr_mean'].cpu() for p in target_paths},
        'addr_std': {p: cal[p]['addr_std'].cpu() for p in target_paths},
        'lut_state': {p: luts[p.replace('.', '_')].cpu().state_dict() for p in target_paths},
        'phase0_layer_metrics': last_metrics,
        'prefit_images': prefit_images,
        'prefit_epochs': prefit_epochs,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    out = get_prefit_ckpt(target_mode)
    torch.save(ckpt, out)
    print(f"\n  Saved prefit ckpt: {out}")


# ============================================================
# Distillation trainer
# ============================================================

class LUTV10DistillTrainer(DetectionTrainer):
    """DetectionTrainer with teacher raw/output distillation for v10 LUT layers."""
    teacher_model = None
    distill_paths = ()
    patched = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        return weights

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not self.patched:
            self._install_distill_loss_patch()
            self.patched = True
        return batch

    def _set_stage_trainability(self):
        e = getattr(self, 'epoch', 0)

        if getattr(self, '_last_stage_epoch', None) == e:
            return

        self._last_stage_epoch = e

        if e < 5:
            set_only_lut_trainable(self.model, bn_trainable=False)
            stage = 'LUT-only'
        elif e < 10:
            set_only_lut_trainable(self.model, bn_trainable=True)
            stage = 'LUT+BN'
        else:
            set_all_trainable(self.model)
            stage = 'full'

        print(f"  [STAGE] epoch={e} trainability={stage}")

    def _install_distill_loss_patch(self):
        student = self.model
        device = next(student.parameters()).device
        teacher = YOLO('yolov8n.pt').model.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher_model = teacher

        student_out = {}
        teacher_in = {}
        teacher_out = {}
        teacher_raw_mods = {}

        def make_stu_hook(path):
            def hook(mod, inp, out):
                student_out[path] = out
            return hook

        def make_tea_pre_hook(path):
            def hook(mod, inp):
                teacher_in[path] = inp[0].detach()
            return hook

        def make_tea_out_hook(path):
            def hook(mod, inp, out):
                teacher_out[path] = out.detach()
            return hook

        for path in self.distill_paths:
            sm = get_submodule_by_path(student.model, path)
            tm = get_submodule_by_path(teacher.model, path)
            if not isinstance(sm, SpatialGroupMultiHeadLUTDelta):
                print(f"  [WARN] student path {path} is not v10 LUT: {type(sm)}")
            sm.register_forward_hook(make_stu_hook(path))
            tm.register_forward_pre_hook(make_tea_pre_hook(path))
            tm.register_forward_hook(make_tea_out_hook(path))
            teacher_raw_mods[path] = tm.conv

        original_loss = student.loss
        trainer_ref = self

        def loss_with_v10_distill(self_model, batch, preds=None):
            trainer_ref._set_stage_trainability()
            student_out.clear()
            teacher_in.clear()
            teacher_out.clear()

            det_loss, loss_items = original_loss(batch, preds)
            imgs = batch['img']
            with torch.no_grad():
                _ = teacher(imgs)

            raw_loss = imgs.new_tensor(0.0)
            out_loss = imgs.new_tensor(0.0)
            stat_loss = imgs.new_tensor(0.0)
            for path in trainer_ref.distill_paths:
                sm = get_submodule_by_path(student.model, path)
                if not isinstance(sm, SpatialGroupMultiHeadLUTDelta):
                    continue
                if path not in teacher_in or sm.last_raw is None:
                    continue
                with torch.no_grad():
                    t_raw = teacher_raw_mods[path](teacher_in[path])
                s_raw = sm.last_raw.float()
                raw_loss = raw_loss + F.smooth_l1_loss(s_raw, t_raw.float())

                if path in student_out and path in teacher_out:
                    s_out = student_out[path].float()
                    t_out = teacher_out[path].float()
                    att = t_out.detach().abs().mean(dim=1, keepdim=True)
                    att = att / (att.mean(dim=[2, 3], keepdim=True) + 1e-6)
                    out_loss = out_loss + F.smooth_l1_loss(s_out * att, t_out * att)

                stat_loss = stat_loss + F.mse_loss(s_raw.mean(dim=[0,2,3]), t_raw.float().mean(dim=[0,2,3]))
                stat_loss = stat_loss + F.mse_loss(s_raw.std(dim=[0,2,3]), t_raw.float().std(dim=[0,2,3]))

            e = getattr(trainer_ref, 'epoch', 0)
            if e < 5:
                lam_raw, lam_out, lam_stat = 2.0, 1.0, 0.05
            elif e < 20:
                lam_raw, lam_out, lam_stat = 1.0, 0.5, 0.03
            else:
                lam_raw, lam_out, lam_stat = 0.3, 0.2, 0.01

            total = det_loss + lam_raw * raw_loss + lam_out * out_loss + lam_stat * stat_loss
            return total, loss_items

        student.loss = types.MethodType(loss_with_v10_distill, student)
        print(f"  [DISTILL] patched v10 model.loss, paths={self.distill_paths}")


# ============================================================
# Training phases
# ============================================================

def build_student_from_prefit(target_mode, device, quant=False, load_ckpt=None,
                              group_size=16, heads=2, lut_size=128, addr_clip=3.0,
                              qbits=8):
    target_paths = get_target_paths(target_mode)
    print("\n[1/3] Loading yolov8n.pt student ...")
    model = YOLO('yolov8n.pt')

    print("\n[2/3] Replacing target layers ...")
    prefitted = load_prefit_ckpt(target_mode)
    if prefitted is None:
        print("  [WARN] No Phase0 prefit ckpt. Using random v10 LUT init.")
    else:
        # Prefer ckpt hyperparams unless explicitly different code path is desired.
        group_size = int(prefitted.get('group_size', group_size))
        heads = int(prefitted.get('heads', heads))
        lut_size = int(prefitted.get('lut_size', lut_size))
        addr_clip = float(prefitted.get('addr_clip', addr_clip))

    replaced = replace_target_convs_with_lut(
        model.model.model,
        target_paths=target_paths,
        prefitted=prefitted,
        group_size=group_size,
        heads=heads,
        lut_size=lut_size,
        addr_clip=addr_clip,
    )

    if load_ckpt:
        ok = load_weights_flexible(model, load_ckpt)
        if not ok:
            raise RuntimeError(f"Failed to load ckpt: {load_ckpt}")

    set_lut_qbits(model.model, enabled=quant, qbits=qbits)
    set_lut_qnoise(model.model, 1.0 if quant else 0.0)
    return model, replaced


def train_phase_v10(phase, target_mode='68', device=0, epochs=50, lr=None, batch=16,
                    root=DEFAULT_COCO_ROOT, group_size=16, heads=2, lut_size=128,
                    addr_clip=3.0, qbits=8):
    data = prepare_dataset_config(root)
    quant = (phase == 2)
    target_paths = get_target_paths(target_mode)
    name = f"v10_{target_mode}_phase1" if phase == 1 else f"v10_{target_mode}_phase2_qat"
    load_ckpt = None
    if phase == 2:
        load_ckpt = get_phase1_ckpt(target_mode)
        if not os.path.exists(load_ckpt):
            hits = glob.glob(f"**/v10_{target_mode}_phase1/weights/best.pt", recursive=True)
            load_ckpt = hits[0] if hits else None
        if load_ckpt is None:
            raise FileNotFoundError(f"Phase2 requires v10 target_mode={target_mode} Phase1 best.pt")

    lr = lr if lr is not None else (5e-4 if phase == 1 else 1e-4)
    tag = f"v10 Phase{phase} {'QAT' if quant else 'Distill Fine-tune'} target_mode={target_mode}"
    print("=" * 70)
    print(tag)
    print("=" * 70)

    model, replaced = build_student_from_prefit(
        target_mode=target_mode,
        device=device,
        quant=quant,
        load_ckpt=load_ckpt,
        group_size=group_size,
        heads=heads,
        lut_size=lut_size,
        addr_clip=addr_clip,
        qbits=qbits,
    )

    total_fp, total_q = collect_lut_report(replaced, quant=quant, qbits=qbits)
    for r in replaced:
        per_table_kb = r['lut_size'] * r['group_size'] * 4 / 1024
        print(f"  [LAYER] {r['path']} C={r['c_in']} G={r['groups']} H={r['heads']} "
              f"L={r['lut_size']} gs={r['group_size']} table={per_table_kb:.1f}KB "
              f"FP={r['storage_fp_kb']:.1f}KB Q={r['storage_q_kb']:.1f}KB")
    print(f"  Storage: FP={total_fp:.1f}KB Q={total_q:.1f}KB")

    # Set trainer class distill paths before training.
    LUTV10DistillTrainer.distill_paths = target_paths
    LUTV10DistillTrainer.patched = False

    overrides = {
        'model': 'yolov8n.pt',
        'data': data,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch,
        'device': device,
        'project': PROJECT,
        'name': name,
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': lr,
        'weight_decay': 1e-4,
        'cos_lr': True,
        'close_mosaic': 10,
        'patience': 50,
        'verbose': True,
        'cache': False,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'workers': 8,
    }

    if quant:
        def on_epoch_end(trainer_obj):
            e = trainer_obj.epoch
            if e < 3:
                q = 0.0
            elif e < 15:
                q = (e - 3) / 12.0
            else:
                q = 1.0
            set_lut_qnoise(trainer_obj.model, q)
            if e % 5 == 0 or e in (3, 14):
                print(f"  [QNOISE] epoch={e} qnoise={q:.3f}")
        model.add_callback('on_train_epoch_end', on_epoch_end)

    print(f"\n[3/3] Train: epochs={epochs} lr={lr} quant={quant} target_mode={target_mode}")
    results = model.train(trainer=LUTV10DistillTrainer, **overrides)

    mAP = mAP50 = 0.0
    if results is not None:
        if hasattr(results, 'box') and results.box:
            mAP = safe_float(results.box.map)
            mAP50 = safe_float(results.box.map50)
        elif hasattr(results, 'results_dict'):
            mAP = safe_float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
            mAP50 = safe_float(results.results_dict.get('metrics/mAP50(B)', 0.0))

    phase0_metrics = {}
    pf = get_prefit_ckpt(target_mode)
    if os.path.exists(pf):
        try:
            ck = torch.load(pf, map_location='cpu', weights_only=False)
            phase0_metrics = ck.get('phase0_layer_metrics', {})
        except Exception:
            phase0_metrics = {}

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v10',
        'phase': phase,
        'target_mode': target_mode,
        'target_paths': target_paths,
        'module': 'SpatialGroupMultiHeadLUTDelta',
        'hyperparams': {
            'group_size': group_size,
            'heads': heads,
            'lut_size': lut_size,
            'addr_clip': addr_clip,
            'qbits': qbits,
        },
        'baseline': {'yolov8n': BASELINE_MAP},
        'result': {'mAP50-95': mAP, 'mAP50': mAP50},
        'delta_vs_baseline': mAP - BASELINE_MAP,
        'layers': replaced,
        'storage': {'fp_kb': total_fp, 'q_kb': total_q},
        'phase0_layer_metrics': phase0_metrics,
    }
    rf = f"results/report_lut_v10_mode{target_mode}_phase{phase}_{ts}.json"
    with open(rf, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"{tag} done")
    print(f"  mAP50-95 : {mAP:.4f}  delta={mAP-BASELINE_MAP:+.4f}")
    print(f"  mAP50    : {mAP50:.4f}")
    print(f"  Storage  : FP={total_fp:.1f}KB Q={total_q:.1f}KB")
    print(f"  Report   : {rf}")
    print("=" * 70)


def smoke_test(target_mode='68', device=0, root=DEFAULT_COCO_ROOT):
    train_phase_v10(phase=1, target_mode=target_mode, device=device, epochs=1, lr=1e-4, batch=4, root=root)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--phase', type=int, choices=[0, 1, 2], help='0=prefit, 1=fine-tune, 2=QAT')
    g.add_argument('--smoke', action='store_true')
    p.add_argument('--target_mode', type=str, default='68', choices=['6', '8', '68'])
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--root', type=str, default=DEFAULT_COCO_ROOT)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--prefit_images', type=int, default=5000)
    p.add_argument('--prefit_epochs', type=int, default=10)
    p.add_argument('--prefit_batch', type=int, default=4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--group_size', type=int, default=16)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--lut_size', type=int, default=128)
    p.add_argument('--addr_clip', type=float, default=3.0)
    p.add_argument('--qbits', type=int, default=8)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'smoke' if args.smoke else f'phase{args.phase}'
    log = f"logs/lut_v10_mode{args.target_mode}_{tag}_{ts}.log"
    print(f"[INFO] Log: {log} GPU arg={args.device} target_mode={args.target_mode}")

    fh = open(log, 'a', buffering=1)
    sys.stdout = fh
    sys.stderr = fh
    try:
        # CUDA_VISIBLE_DEVICES remaps selected GPU to cuda:0 inside this process.
        internal_device = 0
        if args.smoke:
            smoke_test(target_mode=args.target_mode, device=internal_device, root=args.root)
        elif args.phase == 0:
            prefit_luts_v10(
                target_mode=args.target_mode,
                device=internal_device,
                root=args.root,
                imgsz=args.imgsz,
                prefit_images=args.prefit_images,
                prefit_epochs=args.prefit_epochs,
                batch=args.prefit_batch,
                group_size=args.group_size,
                heads=args.heads,
                lut_size=args.lut_size,
                addr_clip=args.addr_clip,
            )
        else:
            default_epochs = 50 if args.phase == 1 else 40
            train_phase_v10(
                phase=args.phase,
                target_mode=args.target_mode,
                device=internal_device,
                epochs=args.epochs or default_epochs,
                lr=args.lr,
                batch=args.batch,
                root=args.root,
                group_size=args.group_size,
                heads=args.heads,
                lut_size=args.lut_size,
                addr_clip=args.addr_clip,
                qbits=args.qbits,
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        fh.close()
