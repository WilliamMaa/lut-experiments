"""
LUT 替换 1x1 Conv - v9
======================
目标：只替换 YOLOv8n backbone 的 6.cv1 + 8.cv1，并加入：
  P0: train split 小样本 LUT layer-wise prefit
  P1: 去掉 addr_scale.detach()，addr_scale 可学习
  P2: LUT-NN 风格的地址通道 calibration：按真实 feature variance 选 top-k channel
  P3: Phase1 检测微调时加入 teacher layer-output distillation
  P4: Phase2 QAT，量化 LUT weight 本身，保留 distillation

推荐用法：
  python lut_coco_v9.py --phase 0 --device 3 --prefit_images 1000 --prefit_epochs 3
  python lut_coco_v9.py --phase 1 --device 3 --epochs 50
  python lut_coco_v9.py --phase 2 --device 3 --epochs 40
  python lut_coco_v9.py --smoke --device 3

注意：
  - 验证/部署阶段仍然只有 LUT student，不需要 teacher。
  - teacher 只在 Phase0 prefit 和 Phase1/2 training loss 中使用。
  - 本代码默认 COCO 在 /data1/datasets/coco。
"""

import os, sys, glob, json, yaml, argparse, random, types
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

TARGET_REPLACE_PATHS = ("6.cv1", "8.cv1")
DEFAULT_COCO_ROOT = "/data1/datasets/coco"
PROJECT = "runs/detect/runs/lut_v9"
PREFIT_CKPT = f"{PROJECT}/v9_phase0_prefit.pt"
PHASE1_CKPT = f"{PROJECT}/v9_phase1/weights/best.pt"


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


def inv_softplus(y: float) -> float:
    y = torch.tensor(float(y))
    return float(torch.log(torch.exp(y) - 1.0))


def quantize_ste_scaled(w, qmax, eps=1e-8):
    """Per-output-dimension scaled fake quant for LUT weight.

    LUT weight shape: [lut_size, 2C]. We scale each output dimension separately.
    During export, this is still just a quantized LUT table; scale is absorbed by
    materialized values if needed.
    """
    scale = w.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
    q = RoundSTE.apply((w / scale).clamp(-qmax, qmax))
    return q * scale


def get_quant_config(c_in):
    return 6 if c_in >= 256 else 8


def get_submodule_by_path(root: nn.Module, path: str):
    cur = root
    for part in path.split('.'):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
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


# ============================================================
# LUT module v9
# ============================================================

class LUT_Conv1x1_Replacement(nn.Module):
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8,
                 quant_bits=None, qnoise=0.0, addr_idx=None):
        super().__init__()
        assert c_in == c_out, "v9 currently only replaces ci == co 1x1 convs"
        self.channels = c_in
        self.lut_size = lut_size
        self.addr_dim = min(addr_dim, c_in)
        self.quant_bits = quant_bits
        self.qnoise = qnoise

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

        # Non-zero init: otherwise tanh(0) kills LUT branch and weakens lut.weight gradient.
        self.scale_gamma = nn.Parameter(torch.ones(1) * 0.10)
        self.scale_beta = nn.Parameter(torch.ones(1) * 0.10)

        # Learnable positive address scale. Init softplus(addr_scale_raw) ~= 10.
        self.addr_scale_raw = nn.Parameter(torch.ones(1) * inv_softplus(10.0))

        if addr_idx is None:
            addr_idx = torch.arange(self.addr_dim, dtype=torch.long)
        else:
            addr_idx = torch.as_tensor(addr_idx, dtype=torch.long)[:self.addr_dim]
        self.register_buffer("addr_idx", addr_idx.clone())

        self.bn = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def _lookup_weight(self):
        if self.quant_bits is None:
            return self.lut.weight
        qm = 2 ** (self.quant_bits - 1) - 1
        if self.training:
            q_w = quantize_ste_scaled(self.lut.weight, qm)
            # qnoise ramp: 0 -> float, 1 -> full fake quant
            return (1.0 - self.qnoise) * self.lut.weight + self.qnoise * q_w
        return quantize_ste_scaled(self.lut.weight, qm)

    def forward(self, x):
        B, C, H, W = x.shape
        idx = self.addr_idx.to(x.device)
        scale = F.softplus(self.addr_scale_raw) + 1e-3

        a = x.index_select(1, idx).mean(dim=[2, 3]).abs() * scale
        f = a % self.lut_size
        fl = f.long()
        w = f - fl.float()

        lut_w = self._lookup_weight()
        o = 0.0
        for i in range(self.addr_dim):
            lo = F.embedding(fl[:, i], lut_w)
            hi = F.embedding((fl[:, i] + 1) % self.lut_size, lut_w)
            o = o + (1.0 - w[:, i:i + 1]) * lo + w[:, i:i + 1] * hi
        o = o / float(self.addr_dim)

        gf = o[:, :C] * torch.tanh(self.scale_gamma)
        bf = o[:, C:] * torch.tanh(self.scale_beta)
        out = x * (1.0 + gf.view(B, C, 1, 1)) + bf.view(B, C, 1, 1)
        out = out.to(x.dtype) if out.dtype != x.dtype else out
        return self.act(self.bn(out))


# ============================================================
# Dataset config and prefit image loader
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
# Surgery / quant controls
# ============================================================

def replace_target_convs_with_lut(model_seq, target_paths=TARGET_REPLACE_PATHS, prefitted=None):
    replaced = []
    for path in target_paths:
        old = get_submodule_by_path(model_seq, path)
        if not (isinstance(old, Conv) and old.conv.kernel_size == (1, 1)):
            raise TypeError(f"Target {path} is not Ultralytics Conv 1x1: {type(old)}")
        ci, co = old.conv.in_channels, old.conv.out_channels
        if ci != co:
            raise ValueError(f"Target {path} is not ci==co: {ci}->{co}")

        addr_idx = None
        if prefitted and path in prefitted.get("addr_idx", {}):
            addr_idx = prefitted["addr_idx"][path]
        lut = LUT_Conv1x1_Replacement(ci, co, addr_idx=addr_idx)
        if prefitted and path in prefitted.get("lut_state", {}):
            lut.load_state_dict(prefitted["lut_state"][path], strict=False)

        set_submodule_by_path(model_seq, path, lut)
        fpk = 256 * ci * 2 * 4 / 1024
        replaced.append({'path': path, 'c_in': ci, 'c_out': co, 'storage_fp_kb': fpk})
        print(f"  [SURGERY] {path}: Conv({ci}->{co}) -> LUT  FP={fpk:.0f}KB")
    return replaced


def set_lut_qbits(module, enabled=True):
    for m in module.modules():
        if isinstance(m, LUT_Conv1x1_Replacement):
            m.quant_bits = get_quant_config(m.channels) if enabled else None


def set_lut_qnoise(module, qnoise):
    for m in module.modules():
        if isinstance(m, LUT_Conv1x1_Replacement):
            m.qnoise = float(qnoise)


def collect_lut_report(replaced, quant=False):
    for r in replaced:
        bits = get_quant_config(r['c_in']) if quant else None
        r['quant_bits'] = bits
        r['storage_q_kb'] = (256 * r['c_in'] * 2 * bits / 8 / 1024) if bits else r['storage_fp_kb']
    return sum(r['storage_fp_kb'] for r in replaced), sum(r['storage_q_kb'] for r in replaced)


def load_prefit_ckpt(path=PREFIT_CKPT):
    if not os.path.exists(path):
        print(f"  [WARN] prefit ckpt not found: {path}")
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


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
# Phase0: LUT-NN-ish calibration + layer-wise prefit
# ============================================================

@torch.no_grad()
def calibrate_addr_idx(teacher_model, target_paths, loader, device, addr_dim=8, max_batches=999999):
    stats_sum = {}
    stats_sumsq = {}
    stats_n = {}
    captures = {}
    handles = []

    def make_pre_hook(path):
        def hook(mod, inp):
            x = inp[0].detach()
            v = x.mean(dim=[2, 3])  # [B, C]
            captures[path] = v
        return hook

    for path in target_paths:
        mod = get_submodule_by_path(teacher_model.model, path)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(path)))

    teacher_model.eval()
    for bi, imgs in enumerate(loader):
        if bi >= max_batches:
            break
        captures.clear()
        imgs = imgs.to(device, non_blocking=True)
        _ = teacher_model(imgs)
        for path, v in captures.items():
            v = v.float()
            if path not in stats_sum:
                C = v.shape[1]
                stats_sum[path] = torch.zeros(C, device=device)
                stats_sumsq[path] = torch.zeros(C, device=device)
                stats_n[path] = 0
            stats_sum[path] += v.sum(dim=0)
            stats_sumsq[path] += (v ** 2).sum(dim=0)
            stats_n[path] += v.shape[0]

    for h in handles:
        h.remove()

    addr_idx = {}
    for path in target_paths:
        mean = stats_sum[path] / max(stats_n[path], 1)
        var = stats_sumsq[path] / max(stats_n[path], 1) - mean ** 2
        k = min(addr_dim, var.numel())
        idx = torch.topk(var, k=k).indices.sort().values.cpu()
        addr_idx[path] = idx
        print(f"  [ADDR] {path}: selected idx={idx.tolist()}")
    return addr_idx


def prefit_luts(device=0, root=DEFAULT_COCO_ROOT, imgsz=640, prefit_images=1000,
                prefit_epochs=3, batch=8, lr=1e-3, addr_dim=8):
    os.makedirs(PROJECT, exist_ok=True)
    print("=" * 60)
    print("v9 Phase0: LUT layer-wise prefit")
    print("=" * 60)

    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_str)

    teacher = YOLO('yolov8n.pt').model.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ds = ImageOnlyDataset(root=root, split="train2017", imgsz=imgsz, limit=prefit_images, seed=0)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    print("\n[1/3] Calibrating LUT address channels from train split ...")
    addr_idx = calibrate_addr_idx(teacher, TARGET_REPLACE_PATHS, loader, dev, addr_dim=addr_dim)

    print("\n[2/3] Building standalone LUT modules ...")
    luts = nn.ModuleDict()
    for path in TARGET_REPLACE_PATHS:
        old = get_submodule_by_path(teacher.model, path)
        ci, co = old.conv.in_channels, old.conv.out_channels
        lut = LUT_Conv1x1_Replacement(ci, co, addr_dim=addr_dim, addr_idx=addr_idx[path])
        lut = lut.to(dev).train()
        luts[path.replace('.', '_')] = lut
        print(f"  [LUT] {path}: {ci}->{co}")

    opt = torch.optim.AdamW(luts.parameters(), lr=lr, weight_decay=1e-4)
    captures = {}
    handles = []

    def make_pre_hook(path):
        def hook(mod, inp):
            captures[path + "_in"] = inp[0].detach()
        return hook

    def make_out_hook(path):
        def hook(mod, inp, out):
            captures[path + "_out"] = out.detach()
        return hook

    for path in TARGET_REPLACE_PATHS:
        mod = get_submodule_by_path(teacher.model, path)
        handles.append(mod.register_forward_pre_hook(make_pre_hook(path)))
        handles.append(mod.register_forward_hook(make_out_hook(path)))

    print("\n[3/3] Prefitting LUT outputs to original Conv module outputs ...")
    for ep in range(prefit_epochs):
        total_loss = 0.0
        n = 0
        for imgs in loader:
            imgs = imgs.to(dev, non_blocking=True)
            captures.clear()
            with torch.no_grad():
                _ = teacher(imgs)

            opt.zero_grad(set_to_none=True)
            loss = 0.0
            detail = []
            for path in TARGET_REPLACE_PATHS:
                x = captures[path + "_in"]
                y = captures[path + "_out"]
                lut = luts[path.replace('.', '_')]
                pred = lut(x)
                l_main = F.smooth_l1_loss(pred, y)
                l_mean = F.mse_loss(pred.mean(dim=[0, 2, 3]), y.mean(dim=[0, 2, 3]))
                l_std = F.mse_loss(pred.std(dim=[0, 2, 3]), y.std(dim=[0, 2, 3]))
                l = l_main + 0.05 * l_mean + 0.05 * l_std
                loss = loss + l
                detail.append(f"{path}:{float(l_main.detach()):.4f}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(luts.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach())
            n += 1
        print(f"  [PREFIT] epoch={ep+1}/{prefit_epochs} loss={total_loss/max(n,1):.5f} {' '.join(detail)}")

    for h in handles:
        h.remove()

    ckpt = {
        'version': 'v9',
        'target_paths': TARGET_REPLACE_PATHS,
        'addr_idx': {p: addr_idx[p].cpu() for p in TARGET_REPLACE_PATHS},
        'lut_state': {p: luts[p.replace('.', '_')].cpu().state_dict() for p in TARGET_REPLACE_PATHS},
        'prefit_images': prefit_images,
        'prefit_epochs': prefit_epochs,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(ckpt, PREFIT_CKPT)
    print(f"\n  Saved prefit ckpt: {PREFIT_CKPT}")


# ============================================================
# Distillation trainer
# ============================================================

class LUTDistillTrainer(DetectionTrainer):
    """Trainer that reuses the already-surgeried model and patches model.loss.

    It adds layer-output distillation from original yolov8n teacher at target paths.
    The patch is installed lazily when batches are already on the target device.
    """
    teacher_model = None
    distill_paths = TARGET_REPLACE_PATHS
    lambda_lut = 1.0
    patched = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        return weights

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not self.patched:
            self._install_distill_loss_patch()
            self.patched = True
        return batch

    def _install_distill_loss_patch(self):
        student = self.model
        device = next(student.parameters()).device
        teacher = YOLO('yolov8n.pt').model.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher_model = teacher

        student_feats = {}
        teacher_feats = {}

        def make_stu_hook(path):
            def hook(mod, inp, out):
                student_feats[path] = out
            return hook

        def make_tea_hook(path):
            def hook(mod, inp, out):
                teacher_feats[path] = out.detach()
            return hook

        for path in self.distill_paths:
            get_submodule_by_path(student.model, path).register_forward_hook(make_stu_hook(path))
            get_submodule_by_path(teacher.model, path).register_forward_hook(make_tea_hook(path))

        original_loss = student.loss
        trainer_ref = self

        def loss_with_distill(self_model, batch, preds=None):
            student_feats.clear()
            teacher_feats.clear()
            det_loss, loss_items = original_loss(batch, preds)

            imgs = batch['img']
            with torch.no_grad():
                _ = teacher(imgs)

            dloss = imgs.new_tensor(0.0)
            for path in trainer_ref.distill_paths:
                if path in student_feats and path in teacher_feats:
                    s = student_feats[path].float()
                    t = teacher_feats[path].float()
                    # Attention-style weighting: emphasize strong teacher activations.
                    att = t.detach().abs().mean(dim=1, keepdim=True)
                    att = att / (att.mean(dim=[2, 3], keepdim=True) + 1e-6)
                    dloss = dloss + (F.smooth_l1_loss(s * att, t * att))
            lam = trainer_ref.lambda_lut_schedule()
            total = det_loss + lam * dloss
            return total, loss_items

        student.loss = types.MethodType(loss_with_distill, student)
        print(f"  [DISTILL] patched model.loss, paths={self.distill_paths}")

    def lambda_lut_schedule(self):
        # self.epoch is set by BaseTrainer. Strong early, weaker later.
        e = getattr(self, 'epoch', 0)
        if e < 10:
            return 2.0
        if e < 30:
            return 1.0
        return 0.3


# ============================================================
# Training phases
# ============================================================

def build_student_from_prefit(device, quant=False, load_ckpt=None):
    print("\n[1/3] Loading yolov8n.pt student ...")
    model = YOLO('yolov8n.pt')
    print("\n[2/3] Replacing target layers ...")
    prefitted = load_prefit_ckpt(PREFIT_CKPT)
    if prefitted is None:
        print("  [WARN] No Phase0 prefit ckpt. Using random LUT init.")
    replaced = replace_target_convs_with_lut(model.model.model, TARGET_REPLACE_PATHS, prefitted=prefitted)
    if load_ckpt:
        ok = load_weights_flexible(model, load_ckpt)
        if not ok:
            raise RuntimeError(f"Failed to load ckpt: {load_ckpt}")
    set_lut_qbits(model.model, enabled=quant)
    set_lut_qnoise(model.model, 1.0 if quant else 0.0)
    return model, replaced


def train_phase(phase, device=0, epochs=50, lr=None, batch=16, root=DEFAULT_COCO_ROOT):
    data = prepare_dataset_config(root)
    quant = (phase == 2)
    name = "v9_phase1" if phase == 1 else "v9_phase2_qat"
    load_ckpt = None if phase == 1 else PHASE1_CKPT
    if phase == 2 and not os.path.exists(load_ckpt):
        hits = glob.glob('**/v9_phase1/weights/best.pt', recursive=True)
        load_ckpt = hits[0] if hits else None
        if load_ckpt is None:
            raise FileNotFoundError("Phase2 requires v9 Phase1 best.pt")

    lr = lr if lr is not None else (5e-4 if phase == 1 else 1e-4)

    tag = f"v9 Phase{phase} {'QAT' if quant else 'Distill Fine-tune'}"
    print("=" * 60)
    print(tag)
    print("=" * 60)

    model, replaced = build_student_from_prefit(device, quant=quant, load_ckpt=load_ckpt)
    total_fp, total_q = collect_lut_report(replaced, quant=quant)
    for r in replaced:
        print(f"  [LAYER] {r['path']} c={r['c_in']} FP={r['storage_fp_kb']:.0f}KB Q={r['storage_q_kb']:.0f}KB bits={r['quant_bits']}")
    print(f"  Storage: FP={total_fp:.0f}KB Q={total_q:.0f}KB")

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
            # 量化阶段也做 ramp，避免一上来硬量化把 LUT 表锁死。
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

    print(f"\n[3/3] Train: epochs={epochs} lr={lr} quant={quant}")
    results = model.train(trainer=LUTDistillTrainer, **overrides)

    mAP = mAP50 = 0.0
    if results is not None:
        if hasattr(results, 'box') and results.box:
            mAP = float(results.box.map)
            mAP50 = float(results.box.map50)
        elif hasattr(results, 'results_dict'):
            mAP = float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
            mAP50 = float(results.results_dict.get('metrics/mAP50(B)', 0.0))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v9',
        'phase': phase,
        'target_paths': TARGET_REPLACE_PATHS,
        'baseline': {'yolov8n': 0.373},
        'result': {'mAP50-95': mAP, 'mAP50': mAP50},
        'delta_vs_baseline': mAP - 0.373,
        'layers': replaced,
        'storage': {'fp_kb': total_fp, 'q_kb': total_q},
    }
    rf = f"results/report_lut_v9_phase{phase}_{ts}.json"
    with open(rf, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{tag} done")
    print(f"  mAP50-95 : {mAP:.4f}  delta={mAP-0.373:+.4f}")
    print(f"  mAP50    : {mAP50:.4f}")
    print(f"  Storage  : FP={total_fp:.0f}KB Q={total_q:.0f}KB")
    print(f"  Report   : {rf}")
    print("=" * 60)


def smoke_test(device=0, root=DEFAULT_COCO_ROOT):
    # Fast build + 1 epoch check. If no prefit ckpt exists, random LUT is used.
    train_phase(phase=1, device=device, epochs=1, lr=1e-4, batch=4, root=root)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--phase', type=int, choices=[0, 1, 2], help='0=prefit, 1=fine-tune, 2=QAT')
    g.add_argument('--smoke', action='store_true')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--root', type=str, default=DEFAULT_COCO_ROOT)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--prefit_images', type=int, default=1000)
    p.add_argument('--prefit_epochs', type=int, default=3)
    p.add_argument('--prefit_batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'smoke' if args.smoke else f'phase{args.phase}'
    log = f"logs/lut_v9_{tag}_{ts}.log"
    print(f"[INFO] Log: {log} GPU arg={args.device}")

    fh = open(log, 'a', buffering=1)
    sys.stdout = fh
    sys.stderr = fh
    try:
        # Since CUDA_VISIBLE_DEVICES remaps selected GPU to cuda:0 inside this process.
        internal_device = 0
        if args.smoke:
            smoke_test(device=internal_device, root=args.root)
        elif args.phase == 0:
            prefit_luts(
                device=internal_device,
                root=args.root,
                imgsz=args.imgsz,
                prefit_images=args.prefit_images,
                prefit_epochs=args.prefit_epochs,
                batch=args.prefit_batch,
            )
        else:
            default_epochs = 50 if args.phase == 1 else 40
            train_phase(
                phase=args.phase,
                device=internal_device,
                epochs=args.epochs or default_epochs,
                lr=args.lr,
                batch=args.batch,
                root=args.root,
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        fh.close()
