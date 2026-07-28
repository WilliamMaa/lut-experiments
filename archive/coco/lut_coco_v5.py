"""
LUT 替换 1x1 Conv 完整训练 (v5 - 独立版)
=========================================
从头训练，不依赖任何 checkpoint。

用法:
  # Phase 1: 手术 + 冷启动全量训练 (50 epoch)
  python lut_coco_v5.py --phase 1 --device 1

  # Phase 2: 量化感知微调 (15 epoch, 加载 Phase 1 best.pt)
  python lut_coco_v5.py --phase 2 --device 1
"""

import torch
import torch.nn as nn
import yaml
import os
import sys
import json
import argparse
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


# =====================================================================
# 1. HGQ 工具
# =====================================================================

class RoundSTE(torch.autograd.Function):
    """前向 round，反向 identity"""
    @staticmethod
    def forward(ctx, x):        return x.round()
    @staticmethod
    def backward(ctx, g):       return g


def quantize_ste(x, qmax):
    x = (x * qmax).clamp(-qmax, qmax)
    return RoundSTE.apply(x) / qmax


# =====================================================================
# 2. LUT 替换模块
# =====================================================================

class LUT_Conv1x1_Replacement(nn.Module):
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8,
                 quant_bits=None, qnoise=0.0):
        super().__init__()
        assert c_in == c_out
        self.channels   = c_in
        self.lut_size   = lut_size
        self.addr_dim   = addr_dim
        self.quant_bits = quant_bits
        self.qnoise     = qnoise

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))
        self.bn  = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        a = x[:, :self.addr_dim].mean(dim=[2, 3]).abs() * 100.0
        f = a % self.lut_size
        fl = f.long()
        w = f - fl.float()

        o = 0
        for i in range(self.addr_dim):
            flw = self.lut(fl[:, i])
            clw = self.lut((fl[:, i] + 1) % self.lut_size)
            o = o + (1 - w[:, i:i+1]) * flw + w[:, i:i+1] * clw
        o = o / self.addr_dim

        g_fp = o[:, :C] * self.scale_gamma
        b_fp = o[:, C:] * self.scale_beta

        if self.training and self.qnoise > 0 and self.quant_bits is not None:
            qm = 2 ** (self.quant_bits - 1) - 1
            g_q = quantize_ste(g_fp, qm)
            b_q = quantize_ste(b_fp, qm)
            g = (1 - self.qnoise) * g_fp + self.qnoise * g_q.detach()
            b = (1 - self.qnoise) * b_fp + self.qnoise * b_q.detach()
        else:
            g, b = g_fp, b_fp

        out = x * (1 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1)
        return self.act(self.bn(out))


# =====================================================================
# 3. 网络手术 / qnoise 控制
# =====================================================================

def get_quant_config(c_in):
    if c_in >= 256: return 4
    if c_in >= 128: return 6
    return 8


def replace_1x1_conv_with_lut(module, prefix='', replaced=None, with_quant=False):
    if replaced is None:
        replaced = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Conv) and child.conv.kernel_size == (1, 1):
            ci, co = child.conv.in_channels, child.conv.out_channels
            if ci == co:
                bits = get_quant_config(ci) if with_quant else None
                fpk = 256 * ci * 2 * 4 / 1024
                qk  = 256 * ci * 2 * bits / 8 / 1024 if bits else fpk
                print(f"  [SURGERY] {path}: Conv({ci}->{co}) -> LUT (FP:{fpk:.0f}KB Q:{qk:.0f}KB bits={bits})")
                setattr(module, name, LUT_Conv1x1_Replacement(ci, co, quant_bits=bits))
                replaced.append({'path': path, 'c_in': ci, 'c_out': co,
                                 'storage_fp_kb': fpk, 'storage_q_kb': qk, 'quant_bits': bits})
        else:
            replace_1x1_conv_with_lut(child, path, replaced, with_quant)
    return replaced


def set_lut_qnoise(model, qnoise):
    def _set(m):
        for c in m.children():
            if isinstance(c, LUT_Conv1x1_Replacement):
                c.qnoise = qnoise
            else:
                _set(c)
    _set(model.model.model)


# =====================================================================
# 4. 数据集
# =====================================================================

def prepare_dataset_config():
    root = "/data1/datasets/coco"
    if not os.path.exists(root):
        print(f"[ERROR] {root}")
        sys.exit(1)
    cfg = {'path': root, 'train': 'images/train2017', 'val': 'images/val2017', 'nc': 80,
           'names': {
               0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',
               8:'boat',9:'traffic light',10:'fire hydrant',11:'stop sign',12:'parking meter',
               13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
               21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',
               28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',32:'sports ball',33:'kite',
               34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',
               38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',
               44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
               51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'couch',
               58:'potted plant',59:'bed',60:'dining table',61:'toilet',62:'tv',63:'laptop',
               64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',
               70:'toaster',71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',
               76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush'}}
    with open("my_coco_config.yaml", 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return "my_coco_config.yaml"


# =====================================================================
# 5. 训练
# =====================================================================

def train(phase=1, device=0):
    data = prepare_dataset_config()

    if phase == 1:
        epochs, lr = 50, 1e-3
        quant, proj, name = False, 'runs/lut_v5_train', 'v5_fulltrain'
        ckpt = None
    else:
        epochs, lr = 15, 5e-5
        quant, proj, name = True, 'runs/lut_v5_qat', 'v5_qat'
        ckpt = f"{proj}/v5_fulltrain/weights/best.pt"

    print("=" * 60)
    print(f"LUT v5 - Phase {phase} ({'Full Train' if phase==1 else 'QAT'})")
    print("=" * 60)

    # 加载
    print(f"\n[1/3] Loading yolov8n.pt...")
    model = YOLO('yolov8n.pt')

    # 手术
    print(f"\n[2/3] Surgery (quant={'yes' if quant else 'no'})...")
    replaced = replace_1x1_conv_with_lut(model.model.model, with_quant=quant)

    # 灌 Phase 1 权重 (仅 Phase 2)
    if phase == 2 and os.path.exists(ckpt):
        print(f"  Loading Phase 1 weights from: {ckpt}")
        raw = torch.load(ckpt, map_location='cpu', weights_only=False)
        state = None
        if isinstance(raw, dict):
            vals = [v for v in raw.values() if v is not None][:3]
            if vals and all(isinstance(v, torch.Tensor) for v in vals):
                state = raw
            for k in ['ema', 'model']:
                if state: break
                r = raw.get(k)
                if r is not None:
                    state = r.state_dict() if hasattr(r, 'state_dict') else (r if isinstance(r, dict) else None)
            if state is None and 'model' in raw:
                inner = raw['model']
                if hasattr(inner, 'model'):
                    state = inner.model.state_dict()
        if state:
            model.model.load_state_dict(state, strict=False)
            print(f"  Weights loaded successfully")
        else:
            print(f"  [WARN] Could not extract state_dict, skipping")

    for r in replaced:
        print(f"    {r['path']}: {r['c_in']}ch, FP={r['storage_fp_kb']:.0f}KB, Q={r['storage_q_kb']:.0f}KB")
    total_fp = sum(r['storage_fp_kb'] for r in replaced)
    total_q  = sum(r['storage_q_kb']  for r in replaced)
    print(f"  Total: FP={total_fp:.0f}KB, Q={total_q:.0f}KB")

    # qnoise 调度 (Phase 2)
    if phase == 2:
        def on_epoch_end(epoch):
            q = min(1.0, epoch / (epochs - 1)) if epochs > 1 else 0.0
            set_lut_qnoise(model, q)
        set_lut_qnoise(model, 0.0)
        model.add_callback('on_train_epoch_end', lambda e: on_epoch_end(e.epoch))

    # 训练
    print(f"\n[3/3] Train: epochs={epochs}, lr={lr}")
    kwargs = {
        'data': data, 'epochs': epochs, 'imgsz': 640, 'batch': 16,
        'device': device, 'project': proj, 'name': name, 'exist_ok': True,
        'optimizer': 'AdamW', 'lr0': lr, 'weight_decay': 1e-4,
        'cos_lr': True, 'close_mosaic': 10, 'patience': 50,
        'verbose': True, 'cache': False,
    }
    if phase == 1:
        kwargs['warmup_epochs'] = 3
        kwargs['warmup_momentum'] = 0.8

    results = model.train(**kwargs)

    mAP   = float(results.box.map)   if hasattr(results, 'box') and results.box is not None else 0.0
    mAP50 = float(results.box.map50) if hasattr(results, 'box') and results.box is not None else 0.0

    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "phase": phase,
        "baseline": {"model": "yolov8n", "mAP50-95": 0.373, "mAP50": 0.528},
        "result": {"mAP50-95": mAP, "mAP50": mAP50},
        "delta": mAP - 0.373,
        "config": {"epochs": epochs, "lr": lr, "quant": quant},
        "layers": replaced,
        "storage": {"fp_kb": total_fp, "q_kb": total_q},
    }

    rf = f"report_lut_v5_phase{phase}.json"
    with open(rf, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Phase {phase} done")
    print(f"  mAP50-95: {mAP:.4f}  (yolov8n: 0.373)")
    print(f"  Storage:  {total_fp:.0f}KB(fp) / {total_q:.0f}KB(q)")
    print(f"  Report:   {rf}")
    print("=" * 60)
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--device', type=int, default=3)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    log = f"lut_v5_phase{args.phase}.log"
    print(f"[INFO] Log: {log}  GPU: physical={args.device}")
    fh = open(log, "a", buffering=1)
    sys.stdout = fh
    sys.stderr = fh

    try:
        train(phase=args.phase, device=0)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        fh.close()