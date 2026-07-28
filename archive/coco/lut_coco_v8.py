"""
LUT 替换 1x1 Conv - v8 (Phase 2 量化修正)
==========================================
复用 v7 Phase 1 checkpoint，只跑 Phase 2。

v7 Phase 2 的问题：
  - 量化对象是 gf/bf（动态激活），不是 lut.weight（静态权重）
  - qnoise ramp 期间 STE 梯度对 lut.weight 几乎没有更新压力
  - 结果：Phase 2 几乎无提升（0.3122 → 0.3128）

v8 修正：
  - 直接量化 lut.weight（这才是存进 SRAM 的东西）
  - 去掉 qnoise ramp，Phase 2 从第一个 epoch 全量化
  - gf/bf 保持浮点，scale 用 tanh 约束

用法:
  python lut_coco_v8.py --phase 2 --device 3
  python lut_coco_v8.py --smoke  --device 3
"""

import torch, torch.nn as nn, torch.multiprocessing as mp
import torch.nn.functional as F
import yaml, os, sys, json, argparse, glob
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.models.yolo.detect import DetectionTrainer


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
# LUT 模块 (v8)
# =====================================================================

class LUT_Conv1x1_Replacement(nn.Module):
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8,
                 quant_bits=None):
        super().__init__()
        assert c_in == c_out
        self.channels   = c_in
        self.lut_size   = lut_size
        self.addr_dim   = addr_dim
        self.quant_bits = quant_bits   # None = 浮点, int = 量化位宽

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.005)

        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))
        self.addr_scale  = nn.Parameter(torch.ones(1) * 10.0)
        self.bn  = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # 地址计算
        scale = self.addr_scale.detach().abs()
        a  = x[:, :self.addr_dim].mean(dim=[2, 3]).abs() * scale
        f  = a % self.lut_size
        fl = f.long()
        w  = f - fl.float()

        # [v8] 量化 lut.weight 本身（训练时走 STE，验证时走 round）
        if self.quant_bits is not None:
            qm = 2 ** (self.quant_bits - 1) - 1
            if self.training:
                lut_w = quantize_ste(self.lut.weight, qm)
            else:
                lut_w = (self.lut.weight * qm).clamp(-qm, qm).round() / qm
        else:
            lut_w = self.lut.weight

        # 插值查表
        o = 0
        for i in range(self.addr_dim):
            flw = F.embedding(fl[:, i], lut_w)
            clw = F.embedding((fl[:, i] + 1) % self.lut_size, lut_w)
            o   = o + (1 - w[:, i:i+1]) * flw + w[:, i:i+1] * clw
        o = o / self.addr_dim

        # tanh 约束 scale，保持 gf/bf 浮点
        gf = o[:, :C] * torch.tanh(self.scale_gamma)
        bf = o[:, C:] * torch.tanh(self.scale_beta)

        out = x * (1 + gf.view(B, C, 1, 1)) + bf.view(B, C, 1, 1)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return self.act(self.bn(out))


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

def set_lut_qbits(module):
    for m in module.modules():
        if isinstance(m, LUT_Conv1x1_Replacement):
            m.quant_bits = get_quant_config(m.channels)

def clear_lut_qbits(module):
    for m in module.modules():
        if isinstance(m, LUT_Conv1x1_Replacement):
            m.quant_bits = None


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

def load_weights_from_ckpt(model, weights):
    if not (weights and os.path.exists(weights)):
        print(f"  [WARN] ckpt not found: {weights}")
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
        # BN 校验
        for name, mod in model.model.named_modules():
            if isinstance(mod, LUT_Conv1x1_Replacement):
                s = mod.bn.running_mean.abs().sum().item()
                lut_s = mod.lut.weight.abs().mean().item()
                print(f"  {name}: BN_mean_sum={s:.3f} lut_weight_abs_mean={lut_s:.5f}")
        return True
    print("  [WARN] Could not extract state_dict")
    return False


# =====================================================================
# 自定义 Trainer
# =====================================================================

class LUTTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        return weights  # 直接返回已手术的模型，阻止按 yaml 重建


# =====================================================================
# Phase 2 训练
# =====================================================================

def train_phase2(device=0, epochs=40, lr=1e-4, smoke=False):
    data    = prepare_dataset_config()
    proj    = 'runs/detect/runs/lut_v8'
    wu      = 3
    name    = 'v8_smoke' if smoke else 'v8_qat'
    epochs  = 1 if smoke else epochs

    # 找 v7 Phase 1 ckpt
    v7_ckpt = 'runs/detect/runs/lut_v7/v7_fulltrain/weights/best.pt'
    if not os.path.exists(v7_ckpt):
        hits = glob.glob('**/v7_fulltrain/weights/best.pt', recursive=True)
        v7_ckpt = hits[0] if hits else None

    tag = f"v8 Phase 2 QAT{'  [SMOKE]' if smoke else ''}"
    print("=" * 60); print(tag); print("=" * 60)
    print(f"  v7 ckpt : {v7_ckpt or 'NOT FOUND'}")

    # 1. 加载 + 手术
    print(f"\n[1/3] Loading yolov8n.pt ...")
    model = YOLO('yolov8n.pt')
    print(f"\n[2/3] Surgery ...")
    replaced = replace_1x1_conv_with_lut(model.model.model)

    # 2. 灌 v7 Phase 1 权重（包含训好的 lut.weight）
    if v7_ckpt:
        loaded = load_weights_from_ckpt(model, v7_ckpt)
        if not loaded:
            print("  [ERROR] Failed to load v7 Phase 1 weights!"); return
    else:
        print("  [ERROR] v7 Phase 1 checkpoint not found!"); return

    # 3. 设量化位宽（直接全量化，不 ramp）
    print(f"\n  Setting quant_bits (full quantization from ep0):")
    set_lut_qbits(model.model)
    for r in replaced:
        bits = get_quant_config(r['c_in'])
        r['quant_bits']    = bits
        r['storage_q_kb']  = 256 * r['c_in'] * 2 * bits / 8 / 1024
        print(f"    {r['path']}: {r['c_in']}ch -> INT{bits}  "
              f"FP={r['storage_fp_kb']:.0f}KB  Q={r['storage_q_kb']:.0f}KB")

    total_fp = sum(r['storage_fp_kb'] for r in replaced)
    total_q  = sum(r['storage_q_kb'] for r in replaced)
    print(f"  Storage: FP={total_fp:.0f}KB  Q={total_q:.0f}KB")

    # 4. 训练
    print(f"\n[3/3] Train: epochs={epochs} lr={lr} warmup={wu}")
    overrides = {
        'model':           'yolov8n.pt',
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
        'workers':         8,
    }

    results = model.train(trainer=LUTTrainer, **overrides)

    mAP = mAP50 = 0.0
    if results is not None:
        if hasattr(results, 'box') and results.box:
            mAP   = float(results.box.map)
            mAP50 = float(results.box.map50)
        elif hasattr(results, 'results_dict'):
            mAP   = float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
            mAP50 = float(results.results_dict.get('metrics/mAP50(B)', 0.0))

    if smoke:
        print("\n" + "=" * 60)
        status = "PASSED" if mAP > 0 else "PASSED (mAP=0, check log)"
        print(f"[SMOKE] {status} - validation completed without hanging")
        print(f"  mAP50-95: {mAP:.4f}  mAP50: {mAP50:.4f}")
        print("=" * 60)
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("results", exist_ok=True)
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version": "v8", "phase": 2,
        "v7_phase1_ckpt": v7_ckpt,
        "baseline": {"yolov8n": 0.373},
        "v7_phase1": {"mAP50-95": 0.3122},
        "result": {"mAP50-95": mAP, "mAP50": mAP50},
        "delta_vs_baseline": mAP - 0.373,
        "delta_vs_phase1":   mAP - 0.3122,
        "layers": replaced,
        "storage": {"fp_kb": total_fp, "q_kb": total_q},
    }
    rf = f"results/report_lut_v8_phase2_{ts}.json"
    with open(rf, 'w') as f: json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{tag} done")
    print(f"  mAP50-95 : {mAP:.4f}  (v7 Phase1: 0.3122  delta: {mAP-0.3122:+.4f})")
    print(f"  mAP50    : {mAP50:.4f}")
    print(f"  Storage  : FP={total_fp:.0f}KB  Q={total_q:.0f}KB")
    print(f"  Report   : {rf}")
    print("=" * 60)


# =====================================================================
# 入口
# =====================================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--phase', type=int, choices=[2], help='Phase 2: QAT (复用 v7 Phase 1)')
    g.add_argument('--smoke', action='store_true',   help='1 epoch smoke test')
    p.add_argument('--device', type=int, default=3)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr',     type=float, default=1e-4)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs("logs", exist_ok=True)
    log = f"logs/lut_v8_{'smoke' if args.smoke else 'phase2'}_{ts}.log"
    print(f"[INFO] Log: {log}  GPU: cuda:{args.device}")

    fh = open(log, "a", buffering=1)
    sys.stdout = fh
    sys.stderr = fh
    try:
        train_phase2(device=0, epochs=args.epochs, lr=args.lr, smoke=args.smoke)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        fh.close()