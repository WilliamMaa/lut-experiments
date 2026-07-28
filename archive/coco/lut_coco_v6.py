"""
LUT 替换 1x1 Conv 完整训练 (v6 - spawn fix)
============================================
修复 DataLoader 多进程 + CUDA fork 死锁：
  mp.set_start_method('spawn') 替代默认 fork，
  子进程不继承父进程 CUDA context，死锁消除。
  workers 恢复 8，训练速度不受影响。

Phase 1: 手术 + 冷启动全训练 (50 epoch, lr=1e-3)
Phase 2: QAT (40 epoch, lr=1e-4, 自动加载 Phase 1 best.pt)

用法:
  python lut_v6_spawn.py --phase 1 --device 3
  python lut_v6_spawn.py --phase 2 --device 3
  python lut_v6_spawn.py --smoke  --device 3   # 1 epoch 完整 val，验证不死锁
"""

import torch, torch.nn as nn, torch.multiprocessing as mp
import yaml, os, sys, json, argparse, glob
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


# =====================================================================
# 基础工具
# =====================================================================

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g

def quantize_ste(x, qmax):
    return RoundSTE.apply((x * qmax).clamp(-qmax, qmax)) / qmax


# =====================================================================
# LUT 模块
# =====================================================================

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
        scale = self.addr_scale.detach().abs()
        a = x[:, :self.addr_dim].mean(dim=[2, 3]).abs() * scale
        f = a % self.lut_size; fl = f.long(); w = f - fl.float()
        o = 0
        for i in range(self.addr_dim):
            flw = self.lut(fl[:, i]); clw = self.lut((fl[:, i] + 1) % self.lut_size)
            o = o + (1 - w[:, i:i+1]) * flw + w[:, i:i+1] * clw
        o = o / self.addr_dim
        gf, bf = o[:, :C] * self.scale_gamma, o[:, C:] * self.scale_beta
        if self.quant_bits is not None:
            qm = 2 ** (self.quant_bits - 1) - 1
            if self.training:
                g_q = quantize_ste(gf, qm)
                b_q = quantize_ste(bf, qm)
                g = (1 - self.qnoise) * gf + self.qnoise * g_q
                b = (1 - self.qnoise) * bf + self.qnoise * b_q
            else:
                g = (gf * qm).clamp(-qm, qm).round() / qm
                b = (bf * qm).clamp(-qm, qm).round() / qm
        else:
            g, b = gf, bf
        return self.act(self.bn(x * (1 + g.view(B, C, 1, 1)) + b.view(B, C, 1, 1)))


# =====================================================================
# 手术 & 配置
# =====================================================================

def get_quant_config(c_in):
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


# =====================================================================
# 数据集
# =====================================================================

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


# =====================================================================
# 权重加载
# =====================================================================

def load_weights_from_ckpt(model, weights, replaced):
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


# =====================================================================
# 核心训练逻辑（phase 1 / 2 / smoke 共用）
# =====================================================================

def _build_and_train(phase, epochs, lr, wu, name, proj, ckpt, device, smoke=False):
    data = prepare_dataset_config()

    print(f"\n[1/3] Loading yolov8n.pt ...")
    model = YOLO('yolov8n.pt')
    print(f"\n[2/3] Surgery ...")
    replaced = replace_1x1_conv_with_lut(model.model.model)

    need_quant = (phase == 2 or smoke)
    if need_quant:
        if ckpt:
            loaded = load_weights_from_ckpt(model, ckpt, replaced)
            if not loaded and phase == 2:
                print("  [ERROR] Failed to load Phase 1 weights!"); return None
        elif phase == 2:
            print("  [ERROR] Phase 2 requires Phase 1 checkpoint!"); return None
        else:
            print("  [WARN] No ckpt found for smoke, using random init")

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
    print(f"  Storage total: FP={total_fp:.0f}KB  Q={total_q:.0f}KB")

    if need_quant:
        ramp_end = 15

        def on_epoch_end(trainer):
            e = trainer.epoch
            if e < wu:         q = 0.0
            elif e < ramp_end: q = min(1.0, (e - wu) / (ramp_end - wu))
            else:              q = 1.0
            set_lut_qnoise(model, q)
            if e % 5 == 0 or e == wu or e == ramp_end - 1:
                label = "plateau" if e >= ramp_end else "ramp" if e >= wu else "warmup"
                print(f"  [QNOISE] epoch={e} qnoise={q:.3f} [{label}]")

        model.add_callback('on_train_epoch_end', on_epoch_end)

    print(f"\n[3/3] Train: epochs={epochs} lr={lr} warmup={wu}"
          f"{'  [SMOKE]' if smoke else ''}")
    kwargs = {
        'data':            data,
        'epochs':          epochs,
        'imgsz':           640,
        'batch':           16,
        'device':          device,
        'project':         proj,
        'name':            name,
        'exist_ok':        True,
        'optimizer':       'AdamW',
        'lr0':             lr,
        'weight_decay':    1e-4,
        'cos_lr':          True,
        'close_mosaic':    10,
        'patience':        50,
        'verbose':         True,
        'cache':           False,
        'warmup_epochs':   wu,
        'warmup_momentum': 0.8,
        'workers':         8,   # spawn 模式下多进程安全
    }

    results = model.train(**kwargs)
    return results, replaced, total_fp, total_q


# =====================================================================
# 训练入口
# =====================================================================

def train(phase=1, device=0):
    proj = 'runs/detect/runs/lut_v6'

    if phase == 1:
        epochs, lr, wu, name = 50, 1e-3, 3, 'v6_fulltrain'
        ckpt = None
    else:
        epochs, lr, wu, name = 40, 1e-4, 3, 'v6_qat'
        ckpt = f"{proj}/v6_fulltrain/weights/best.pt"
        if not os.path.exists(ckpt):
            hits = glob.glob('**/v6_fulltrain/weights/best.pt', recursive=True)
            ckpt = hits[0] if hits else None

    tag = f"v6 Phase {phase} ({'Full Train' if phase == 1 else 'QAT'})"
    print("=" * 60); print(tag); print("=" * 60)

    out = _build_and_train(phase, epochs, lr, wu, name, proj, ckpt, device, smoke=False)
    if out is None: return

    results, replaced, total_fp, total_q = out
    mAP   = float(results.box.map)   if hasattr(results, 'box') and results.box else 0.0
    mAP50 = float(results.box.map50) if hasattr(results, 'box') and results.box else 0.0

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": "v6", "phase": phase,
        "baseline": {"yolov8n": 0.373},
        "result": {"mAP50-95": mAP, "mAP50": mAP50},
        "delta": mAP - 0.373,
        "layers": replaced,
        "storage": {"fp_kb": total_fp, "q_kb": total_q},
    }
    os.makedirs("results", exist_ok=True)
    rf = f"results/report_lut_v6_phase{phase}_{ts}.json"
    with open(rf, 'w') as f: json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{tag} done")
    print(f"  mAP50-95 : {mAP:.4f}  (baseline: 0.373  delta: {mAP-0.373:+.4f})")
    print(f"  mAP50    : {mAP50:.4f}")
    print(f"  Storage  : FP={total_fp:.0f}KB  Q={total_q:.0f}KB")
    print(f"  Report   : {rf}")
    print("=" * 60)


# =====================================================================
# Smoke Test：1 epoch，完整 val2017，完全对齐 Phase 2 路径
# =====================================================================

def smoke_test(device=0):
    proj = 'runs/detect/runs/lut_v6'
    ckpt = f"{proj}/v6_fulltrain/weights/best.pt"
    if not os.path.exists(ckpt):
        hits = glob.glob('**/v6_fulltrain/weights/best.pt', recursive=True)
        ckpt = hits[0] if hits else None

    print("=" * 60)
    print("LUT v6 Smoke Test (1 epoch, 完整 val2017, spawn workers=8)")
    print(f"  ckpt  : {ckpt or 'NOT FOUND - random init'}")
    print(f"  device: cuda:{device}")
    print("=" * 60)

    out = _build_and_train(
        phase=2,
        epochs=1,
        lr=1e-4,
        wu=3,
        name='smoke_spawn',
        proj='runs/lut_v6_smoke_spawn',
        ckpt=ckpt,
        device=device,
        smoke=True,
    )
    if out is None:
        print("[SMOKE] FAILED - could not build model")
        return

    results, _, _, _ = out
    mAP   = float(results.box.map)   if hasattr(results, 'box') and results.box else 0.0
    mAP50 = float(results.box.map50) if hasattr(results, 'box') and results.box else 0.0
    print("\n" + "=" * 60)
    print("[SMOKE] PASSED - validation completed without hanging")
    print(f"  mAP50-95: {mAP:.4f}")
    print(f"  mAP50   : {mAP50:.4f}")
    print("=" * 60)


# =====================================================================
# 入口
# =====================================================================

if __name__ == '__main__':
    # spawn 替代 fork：子进程不继承父进程 CUDA context，消除死锁根因
    # 必须在 __main__ 保护下、任何 CUDA 操作之前调用
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--phase', type=int, choices=[1, 2], help='Phase 1: full train / Phase 2: QAT')
    g.add_argument('--smoke', action='store_true',      help='1 epoch smoke test (完整 val)')
    p.add_argument('--device', type=int, default=3)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("logs", exist_ok=True)
    log = f"logs/lut_v6_{'smoke' if args.smoke else f'phase{args.phase}'}_{ts}.log"
    fh = open(log, "a", buffering=1)
    sys.stdout = fh
    sys.stderr = fh
    print(f"[INFO] Log: {log}  GPU: cuda:{args.device}")

    try:
        if args.smoke:
            smoke_test(device=0)
        else:
            train(phase=args.phase, device=0)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        fh.close()