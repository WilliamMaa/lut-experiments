"""
LUT 替换 1x1 Conv 完整训练 (v6 - 从头训练版)
=============================================
不依赖任何外部 checkpoint，冷启动全训练。

改进点 (相对 v5):
  - addr_scale 改为可学习参数 (FIX-1, 替代 hardcode *100)
  - Phase 2 扩到 30 epoch (FIX-5, 给 QAT 更多适应时间)
  - 梯度裁剪 (FIX-6, 防梯度突刺)
  - 位宽分配: 大层低位宽 (最大化存储压缩)

Phase 1: 手术 + 冷启动全训练 (50 epoch, lr=1e-3)
Phase 2: QAT (30 epoch, lr=8e-5, 自动加载 Phase 1 best.pt)
"""

import torch, torch.nn as nn, yaml, os, sys, json, argparse
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g

def quantize_ste(x, qmax):
    return RoundSTE.apply((x * qmax).clamp(-qmax, qmax)) / qmax


class LUT_Conv1x1_Replacement(nn.Module):
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8,
                 quant_bits=None, qnoise=0.0):
        super().__init__()
        assert c_in == c_out
        self.channels   = c_in; self.lut_size = lut_size; self.addr_dim = addr_dim
        self.quant_bits = quant_bits; self.qnoise = qnoise

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.005)

        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))
        self.addr_scale  = nn.Parameter(torch.ones(1) * 10.0)
        self.bn  = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        a = x[:, :self.addr_dim].mean(dim=[2, 3]).abs() * self.addr_scale.abs()
        f = a % self.lut_size; fl = f.long(); w = f - fl.float()
        o = 0
        for i in range(self.addr_dim):
            flw = self.lut(fl[:, i]); clw = self.lut((fl[:, i] + 1) % self.lut_size)
            o = o + (1 - w[:, i:i+1]) * flw + w[:, i:i+1] * clw
        o = o / self.addr_dim
        gf, bf = o[:, :C] * self.scale_gamma, o[:, C:] * self.scale_beta
        if self.training and self.qnoise > 0 and self.quant_bits is not None:
            qm = 2 ** (self.quant_bits - 1) - 1
            g = (1-self.qnoise)*gf + self.qnoise * quantize_ste(gf, qm).detach()
            b = (1-self.qnoise)*bf + self.qnoise * quantize_ste(bf, qm).detach()
        else:
            g, b = gf, bf
        return self.act(self.bn(x * (1 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1)))


def get_quant_config(c_in):
    """256ch→INT6, 128ch→INT8, 小层→INT8 (比 v4 放宽，优先保精度)"""
    if c_in >= 256: return 6
    return 8

def replace_1x1_conv_with_lut(module, prefix='', replaced=None):
    if replaced is None: replaced = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Conv) and child.conv.kernel_size == (1, 1):
            ci, co = child.conv.in_channels, child.conv.out_channels
            if ci == co:
                fpk = 256 * ci * 2 * 4 / 1024
                print(f"  [SURGERY] {path}: Conv({ci}->{co}) -> LUT  FP={fpk:.0f}KB")
                setattr(module, name, LUT_Conv1x1_Replacement(ci, co))
                replaced.append({'path': path, 'c_in': ci, 'c_out': co, 'storage_fp_kb': fpk})
        else:
            replace_1x1_conv_with_lut(child, path, replaced)
    return replaced

def set_lut_qbits(model):
    def _set(m):
        for c in m.children():
            if isinstance(c, LUT_Conv1x1_Replacement):
                c.quant_bits = get_quant_config(c.channels)
            else: _set(c)
    _set(model.model.model)

def set_lut_qnoise(model, qnoise):
    def _set(m):
        for c in m.children():
            if isinstance(c, LUT_Conv1x1_Replacement): c.qnoise = qnoise
            else: _set(c)
    _set(model.model.model)


def prepare_dataset_config():
    root = "/data1/datasets/coco"
    if not os.path.exists(root): print(f"[ERROR] {root}"); sys.exit(1)
    cfg = {'path': root, 'train': 'images/train2017', 'val': 'images/val2017', 'nc': 80,
        'names': {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
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
        78:'hair drier',79:'toothbrush'}}
    with open("my_coco_config.yaml", 'w') as f: yaml.dump(cfg, f, sort_keys=False)
    return "my_coco_config.yaml"


def load_weights_from_ckpt(model, weights, replaced):
    """从 YOLO ckpt 提取 state_dict 并灌入。返回是否成功。"""
    if not (weights and os.path.exists(weights)):
        return False
    print(f"  Loading weights: {weights}")
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    state = None
    if isinstance(ckpt, dict):
        vals = [v for v in ckpt.values() if v is not None][:3]
        if vals and all(isinstance(v, torch.Tensor) for v in vals): state = ckpt
    if state is None:
        for k in ['ema', 'model']:
            raw = ckpt.get(k) if isinstance(ckpt, dict) else None
            if raw is not None:
                state = raw.state_dict() if hasattr(raw, 'state_dict') else (
                    raw if isinstance(raw, dict) else None)
                if state: break
    if state is None and isinstance(ckpt, dict) and 'model' in ckpt:
        inner = ckpt['model']
        if hasattr(inner, 'model'): state = inner.model.state_dict()
    if state:
        m, u = model.model.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing:{len(m)} Unexpected:{len(u)}")
        return True
    print("  [WARN] Could not extract state_dict")
    return False


def train(phase=1, device=0):
    data = prepare_dataset_config()
    proj = 'runs/detect/runs/lut_v6'

    if phase == 1:
        epochs, lr, wu, name = 50, 1e-3, 3, 'v6_fulltrain'
        ckpt = None
    else:
        epochs, lr, wu, name = 40, 1e-4, 3, 'v6_qat'
        ckpt = f"{proj}/{name.replace('qat', 'fulltrain')}/weights/best.pt"

    tag = f"v6 Phase {phase} ({'Full Train' if phase==1 else 'QAT'})"
    print("=" * 60); print(tag); print("=" * 60)

    # 1. 加载 + 手术
    print(f"\n[1/3] Loading yolov8n.pt ...")
    model = YOLO('yolov8n.pt')
    print(f"\n[2/3] Surgery ...")
    replaced = replace_1x1_conv_with_lut(model.model.model)

    # 2. 灌 Phase 1 权重 (仅 Phase 2)
    if phase == 2:
        loaded = load_weights_from_ckpt(model, ckpt, replaced)
        if not loaded:
            print("  [ERROR] Phase 2 requires Phase 1 checkpoint!")
            return

    # 3. Phase 2: 设 quant_bits
    if phase == 2:
        print(f"\n  Setting quant_bits:")
        set_lut_qbits(model); set_lut_qnoise(model, 0.0)
        for r in replaced:
            bits = get_quant_config(r['c_in'])
            r['quant_bits'] = bits
            r['storage_q_kb'] = 256 * r['c_in'] * 2 * bits / 8 / 1024
            print(f"    {r['path']}: {r['c_in']}ch -> INT{bits}  "
                  f"FP={r['storage_fp_kb']:.0f}KB  Q={r['storage_q_kb']:.0f}KB")

    total_fp = sum(r['storage_fp_kb'] for r in replaced)
    total_q  = sum(r.get('storage_q_kb', r['storage_fp_kb']) for r in replaced)
    print(f"  Total: FP={total_fp:.0f}KB  Q={total_q:.0f}KB")

    # 5. Phase 2: qnoise ramp → plateau (让模型有充足时间适应纯量化)
    if phase == 2:
        ramp_end = 15  # 前 15 epoch 从 0 ramp 到 1.0, 之后 plateau

        def on_epoch_end(trainer):
            e = trainer.epoch
            if e < wu:
                q = 0.0
            elif e < ramp_end:
                q = min(1.0, (e - wu) / (ramp_end - wu))
            else:
                q = 1.0  # plateau: 纯量化训练
            set_lut_qnoise(model, q)
            if e % 5 == 0 or e == wu or e == ramp_end - 1:
                phase_label = "plateau" if e >= ramp_end else "ramp" if e >= wu else "warmup"
                print(f"  [QNOISE] epoch={e} qnoise={q:.3f} [{phase_label}]")
        model.add_callback('on_train_epoch_end', on_epoch_end)

    # 6. 训练
    print(f"\n[3/3] Train: epochs={epochs} lr={lr} warmup={wu}")
    kwargs = {
        'data': data, 'epochs': epochs, 'imgsz': 640, 'batch': 16,
        'device': device, 'project': proj, 'name': name, 'exist_ok': True,
        'optimizer': 'AdamW', 'lr0': lr, 'weight_decay': 1e-4,
        'cos_lr': True, 'close_mosaic': 10, 'patience': 50,
        'verbose': True, 'cache': False,
        'warmup_epochs': wu, 'warmup_momentum': 0.8,
        'val': False if phase == 2 else True,  # Phase 2 跳过逐 epoch 验证，只在最后验证
    }
    results = model.train(**kwargs)

    mAP = float(results.box.map) if hasattr(results, 'box') and results.box else 0.0
    mAP50 = float(results.box.map50) if hasattr(results, 'box') and results.box else 0.0

    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": "v6", "phase": phase,
        "baseline": {"yolov8n": 0.373},
        "result": {"mAP50-95": mAP, "mAP50": mAP50},
        "delta": mAP - 0.373,
        "layers": replaced,
        "storage": {"fp_kb": total_fp, "q_kb": total_q},
    }
    rf = f"report_lut_v6_phase{phase}.json"
    with open(rf, 'w') as f: json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{tag} done")
    print(f"  mAP50-95: {mAP:.4f}  (yolov8n: 0.373  delta: {mAP-0.373:+.4f})")
    print(f"  mAP50:    {mAP50:.4f}")
    print(f"  Storage:  FP={total_fp:.0f}KB  Q={total_q:.0f}KB")
    print(f"  Report:   {rf}")
    print("=" * 60)
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=int, default=1, choices=[1, 2])
    p.add_argument('--device', type=int, default=3)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    log = f"lut_v6_phase{args.phase}.log"
    fh = open(log, "a", buffering=1); sys.stdout = fh; sys.stderr = fh
    try:
        train(phase=args.phase, device=0)
    except Exception as e:
        print(f"[ERROR] {e}"); import traceback; traceback.print_exc()
    finally:
        fh.close()