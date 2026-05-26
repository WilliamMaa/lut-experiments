"""
LUT 替换 1x1 Conv + HGQ 量化感知微调 (v4 - fixed)
==================================================
Phase 1: 加载 ckpt → 低 lr 微调
Phase 2: 加载 Phase 1 ckpt → 先恢复 BN → 再设 quant_bits → QAT

关键修复: Phase 2 手术时不带 quant_bits, 灌完权重后再手动设
          (防止新 BN 层覆盖 Phase 1 训好的 running stats)
"""

import torch
import torch.nn as nn
import yaml, os, sys, json, argparse
from datetime import datetime
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv

# =====================================================================
# 1. 工具
# =====================================================================
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g

def quantize_ste(x, qmax):
    return RoundSTE.apply((x * qmax).clamp(-qmax, qmax)) / qmax

# =====================================================================
# 2. LUT 模块
# =====================================================================
class LUT_Conv1x1_Replacement(nn.Module):
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8,
                 quant_bits=None, qnoise=0.0):
        super().__init__()
        assert c_in == c_out
        self.channels = c_in; self.lut_size = lut_size; self.addr_dim = addr_dim
        self.quant_bits = quant_bits; self.qnoise = qnoise
        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)
        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))
        self.bn  = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        a = x[:, :self.addr_dim].mean(dim=[2,3]).abs() * 100.0
        f = a % self.lut_size; fl = f.long(); w = f - fl.float()
        o = 0
        for i in range(self.addr_dim):
            flw = self.lut(fl[:,i]); clw = self.lut((fl[:,i]+1) % self.lut_size)
            o = o + (1 - w[:,i:i+1]) * flw + w[:,i:i+1] * clw
        o = o / self.addr_dim
        gf = o[:,:C] * self.scale_gamma; bf = o[:,C:] * self.scale_beta
        if self.training and self.qnoise > 0 and self.quant_bits is not None:
            qm = 2 ** (self.quant_bits-1) - 1
            g = (1-self.qnoise)*gf + self.qnoise * quantize_ste(gf,qm).detach()
            b = (1-self.qnoise)*bf + self.qnoise * quantize_ste(bf,qm).detach()
        else:
            g, b = gf, bf
        return self.act(self.bn(x * (1 + g.view(B,C,1,1)) + b.view(B,C,1,1)))

# =====================================================================
# 3. 手术 & 配置
# =====================================================================
def get_quant_config(c_in):
    if c_in >= 256: return 4
    if c_in >= 128: return 6
    return 8

def replace_1x1_conv_with_lut(module, prefix='', replaced=None):
    if replaced is None: replaced = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Conv) and child.conv.kernel_size == (1,1):
            ci, co = child.conv.in_channels, child.conv.out_channels
            if ci == co:
                fpk = 256*ci*2*4/1024
                print(f"  [SURGERY] {path}: Conv({ci}->{co}) -> LUT (FP:{fpk:.0f}KB)")
                setattr(module, name, LUT_Conv1x1_Replacement(ci, co))
                replaced.append({'path':path,'c_in':ci,'c_out':co,'storage_fp_kb':fpk})
        else:
            replace_1x1_conv_with_lut(child, path, replaced)
    return replaced

def set_lut_qbits(model):
    """Phase 2: 给已有 LUT 层设 quant_bits"""
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
# 4. 数据集
# =====================================================================
def prepare_dataset_config():
    import yaml
    root = "/data1/datasets/coco"
    if not os.path.exists(root): print(f"[ERROR] {root}"); sys.exit(1)
    cfg = {'path':root,'train':'images/train2017','val':'images/val2017','nc':80,
        'names':{0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
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
    with open("my_coco_config.yaml",'w') as f: yaml.dump(cfg,f,sort_keys=False)
    return "my_coco_config.yaml"

# =====================================================================
# 5. 训练
# =====================================================================
V3_CKPT = "runs/detect/runs/lut_replace/lut_1x1_replace/weights/best.pt"

def train(phase=1, weights=None, device=0):
    data = prepare_dataset_config()

    if phase == 1:
        epochs, lr = 20, 2e-4
        proj, name = 'runs/lut_qat_phase1', 'phase1_recovery'
        if weights is None: weights = V3_CKPT
    else:
        epochs, lr = 15, 5e-5
        proj, name = 'runs/lut_qat_phase2', 'phase2_qat'

    tag = f"Phase {phase}"
    print("="*60); print(f"LUT QAT - {tag}"); print("="*60)

    # 1. 加载 + 手术 (统一不带 quant_bits)
    print(f"\n[1/4] Loading yolov8n.pt...")
    model = YOLO('yolov8n.pt')
    print(f"\n[2/4] Surgery (no quant_bits yet)...")
    replaced = replace_1x1_conv_with_lut(model.model.model)

    # 2. 灌权重
    from_ckpt = weights and os.path.exists(weights)
    if from_ckpt:
        print(f"  Loading weights: {weights}")
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        state = None
        if isinstance(ckpt, dict):
            vals = [v for v in ckpt.values() if v is not None][:3]
            if vals and all(isinstance(v,torch.Tensor) for v in vals): state = ckpt
        if state is None:
            for k in ['ema','model']:
                raw = ckpt.get(k) if isinstance(ckpt,dict) else None
                if raw is not None:
                    state = raw.state_dict() if hasattr(raw,'state_dict') else (raw if isinstance(raw,dict) else None)
                    if state: break
        if state is None and isinstance(ckpt,dict) and 'model' in ckpt:
            inner = ckpt['model']
            if hasattr(inner,'model'): state = inner.model.state_dict()
        if state:
            m,u = model.model.load_state_dict(state, strict=False)
            print(f"  Loaded. Missing:{len(m)} Unexpected:{len(u)}")
            # 验证 BN 非零
            try:
                for r in replaced:
                    m = model.model.model
                    for p in r['path'].split('.'): m = getattr(m,p)
                    s = m.bn.running_mean.abs().sum().item()
                    print(f"  {r['path']} BN mean sum: {s:.2f} {'OK' if s>0 else 'WARN:ZERO'}")
            except: pass
        else:
            print("  [WARN] Could not extract state_dict")

    # 3. Phase 2: 设 quant_bits
    if phase == 2:
        print(f"  Setting quant_bits for QAT:")
        set_lut_qbits(model)
        set_lut_qnoise(model, 0.0)
        for r in replaced:
            bits = get_quant_config(r['c_in'])
            r['quant_bits'] = bits
            r['storage_q_kb'] = 256 * r['c_in'] * 2 * bits / 8 / 1024
            print(f"    {r['path']}: {r['c_in']}ch → INT{bits} ({r['storage_q_kb']:.0f}KB)")

    total_fp = sum(r['storage_fp_kb'] for r in replaced)
    total_q  = sum(r.get('storage_q_kb', r['storage_fp_kb']) for r in replaced)

    for r in replaced:
        q = r.get('storage_q_kb', r['storage_fp_kb'])
        print(f"    {r['path']}: {r['c_in']}ch FP={r['storage_fp_kb']:.0f}KB Q={q:.0f}KB")
    print(f"  Total: FP={total_fp:.0f}KB Q={total_q:.0f}KB")

    # 4. 训练
    print(f"\n[3/4] Train: epochs={epochs} lr={lr} qnoise={'ramped' if phase==2 else '0'}")
    kwargs = {
        'data':data,'epochs':epochs,'imgsz':640,'batch':16,'device':device,
        'project':proj,'name':name,'exist_ok':True,
        'optimizer':'AdamW','lr0':lr,'weight_decay':1e-4,
        'cos_lr':True,'close_mosaic':10,'patience':50,'verbose':True,'cache':False,
        'warmup_epochs':3,'warmup_momentum':0.8,
    }

    if phase == 2:
        def on_epoch_end(trainer):
            e = trainer.epoch
            q = 0.0 if e < 0 else min(1.0, e / (epochs-1)) if epochs>1 else 1.0
            set_lut_qnoise(model, q)
        model.add_callback('on_train_epoch_end', on_epoch_end)

    print(f"\n[4/4] Training...")
    results = model.train(**kwargs)

    mAP = float(results.box.map) if hasattr(results,'box') and results.box else 0.0
    mAP50 = float(results.box.map50) if hasattr(results,'box') and results.box else 0.0

    report = {
        "timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "phase":phase,"baseline":{"yolov8n":0.373},"result":{"mAP50-95":mAP,"mAP50":mAP50},
        "delta":mAP-0.373,"layers":replaced,"storage":{"fp_kb":total_fp,"q_kb":total_q},
    }
    rf = f"report_lut_qat_phase{phase}.json"
    with open(rf,'w') as f: json.dump(report,f,indent=4,ensure_ascii=False)

    print("\n"+("="*60))
    print(f"{tag} done | mAP: {mAP:.4f} (yolov8n: 0.373) | Storage: {total_fp:.0f}KB/{total_q:.0f}KB")
    print(f"Report: {rf}")
    print("="*60)
    return model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase',type=int,default=1,choices=[1,2])
    p.add_argument('--weights',type=str,default=None)
    p.add_argument('--device',type=int,default=3)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    log = f"lut_v4_phase{args.phase}.log"
    print(f"[INFO] Log: {log} GPU: {args.device}")
    fh = open(log,"a",buffering=1); sys.stdout=fh; sys.stderr=fh
    try:
        train(phase=args.phase, weights=args.weights, device=0)
    except Exception as e:
        print(f"[ERROR] {e}"); import traceback; traceback.print_exc()
    finally:
        fh.close()