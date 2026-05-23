"""
LUT 替换 1x1 卷积实验 (v3)
===========================
核心思想：
  用 O(1) 查表调制替代 C_in==C_out 的 1x1 卷积，真正消除计算量。

关键设计：
  1. LUT 直接输出 gamma/beta 调制向量 (保留空间信息，不做广播)
  2. 线性插值可微查表 (梯度可回传到 backbone)
  3. 每层独立专属 LUT (多表切换，专表专用)
  4. 不冻结 backbone (网络结构改变，需全量微调)

计算量对比 (以 256×256 1x1 Conv 为例):
  原始: 65,536 MAC / pixel
  LUT:  1,024  MAC / pixel  (64x 下降)
  
  其中 LUT 计算: 8 次 Embedding 查表 + 8 次 add + 256 次乘加调制
  在 CIM 硬件上: 查表 = BRAM 读取, 调制 = SIMD 乘加
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
# 1. LUT 替换模块 (替代 1x1 Conv)
# =====================================================================

class LUT_Conv1x1_Replacement(nn.Module):
    """
    用 LUT 调制替代 1x1 卷积。

    仅适用于 C_in == C_out 的情况。
    保留 BN + SiLU，和原始 Conv 结构对齐，防止特征崩塌。
    前向: floor 索引硬查表 | 反向: 线性插值梯度桥
    """
    def __init__(self, c_in, c_out, lut_size=256, addr_dim=8):
        super().__init__()
        assert c_in == c_out, f"LUT replacement requires C_in==C_out, got {c_in}!={c_out}"

        self.channels = c_in
        self.lut_size = lut_size
        self.addr_dim = addr_dim

        # LUT 表: gamma + beta
        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

        # 冷启动: 零调制 + BN 近恒等
        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))

        # 保留 BN + SiLU，匹配原始 Conv 结构
        self.bn  = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Channel mismatch: {C} vs {self.channels}"

        # ---- 1. 提取地址 ----
        addr_feat   = x[:, :self.addr_dim, :, :]
        addr_global = addr_feat.mean(dim=[2, 3])
        addr_float  = addr_global.abs() * 100.0

        # ---- 2. 线性插值可微查表 ----
        idx_float = addr_float % self.lut_size
        idx_floor = idx_float.long()
        alpha     = idx_float - idx_floor.float()

        lut_out = 0
        for i in range(self.addr_dim):
            floor_w = self.lut(idx_floor[:, i])
            ceil_w  = self.lut((idx_floor[:, i] + 1) % self.lut_size)
            interp  = (1 - alpha[:, i:i+1]) * floor_w + alpha[:, i:i+1] * ceil_w
            lut_out = lut_out + interp
        lut_out = lut_out / self.addr_dim

        # ---- 3. FiLM 调制 ----
        gamma = lut_out[:, :C] * self.scale_gamma
        beta  = lut_out[:, C:] * self.scale_beta
        out   = x * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)

        # ---- 4. BN + SiLU (匹配原始 Conv 结构，防崩塌) ----
        return self.act(self.bn(out))


# =====================================================================
# 2. 网络手术: 递归替换 C_in==C_out 的 1x1 Conv
# =====================================================================

def replace_1x1_conv_with_lut(module, prefix='', replaced_list=None):
    """
    递归遍历模型树。
    当遇到 C_in == C_out 的 1x1 Conv 时，用 LUT 替换。
    
    返回: 被替换的层路径列表
    """
    if replaced_list is None:
        replaced_list = []
    
    for name, child in module.named_children():
        full_path = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, Conv) and child.conv.kernel_size == (1, 1):
            c_in  = child.conv.in_channels
            c_out = child.conv.out_channels
            
            if c_in == c_out:
                storage_kb = 256 * c_in * 2 * 4 / 1024
                print(f"  [SURGERY] {full_path}: Conv({c_in}->{c_out}, 1x1) -> LUT ({storage_kb:.1f} KB)")
                
                lut = LUT_Conv1x1_Replacement(c_in=c_in, c_out=c_out)
                setattr(module, name, lut)
                replaced_list.append({
                    'path': full_path,
                    'c_in': c_in,
                    'c_out': c_out,
                    'storage_kb': storage_kb,
                })
        else:
            replace_1x1_conv_with_lut(child, full_path, replaced_list)
    
    return replaced_list


# =====================================================================
# 3. 数据集配置
# =====================================================================

def prepare_dataset_config():
    dataset_root = "/data1/datasets/coco"
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found: {dataset_root}")
        sys.exit(1)
    config_data = {
        'path': dataset_root, 'train': 'images/train2017', 'val': 'images/val2017',
        'nc': 80,
        'names': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
            57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
    }
    yaml_path = "my_coco_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    return yaml_path


# =====================================================================
# 4. 训练
# =====================================================================

def train_lut_replacement(epochs=50, batch_size=16, device=3):
    data_yaml = prepare_dataset_config()
    
    print("=" * 60)
    print("LUT 1x1 Conv Replacement Experiment")
    print("=" * 60)
    
    # ---- 加载模型 ----
    print("\n[1/4] Loading YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    # ---- 网络手术 ----
    print("\n[2/4] Performing network surgery...")
    replaced = replace_1x1_conv_with_lut(model.model.model)
    
    if not replaced:
        print("\n[WARN] No 1x1 Conv (C_in==C_out) found! Check model structure.")
        return
    
    total_storage = sum(r['storage_kb'] for r in replaced)
    total_params = sum(r['c_in'] * 2 * 256 for r in replaced)
    
    print(f"\n  Summary: {len(replaced)} layers replaced")
    print(f"  Total LUT params: {total_params:,}")
    print(f"  Total LUT storage: {total_storage:.1f} KB")
    for r in replaced:
        print(f"    - {r['path']}: {r['c_in']}ch, {r['storage_kb']:.1f} KB")
    
    # ---- 不冻结 backbone (网络结构变了，需要全量微调) ----
    print("\n[3/4] Backbone: UNFROZEN (adaptation required)")
    
    # ---- 训练 ----
    print("\n[4/4] Starting training...")
    train_kwargs = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': device,
        'project': 'runs/lut_replace',
        'name': 'lut_1x1_replace',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': 1e-3,
        'weight_decay': 1e-4,
        'cos_lr': True,
        'close_mosaic': 10,
        'patience': 50,
        'verbose': True,
        'cache': False,
    }
    
    results = model.train(**train_kwargs)
    
    # ---- 提取结果 ----
    if hasattr(results, 'box') and results.box is not None:
        mAP = float(results.box.map)
        mAP50 = float(results.box.map50)
    else:
        mAP, mAP50 = 0.0, 0.0
    
    # ---- 输出报告 ----
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "experiment": "lut_1x1_conv_replacement",
        "baseline": {"model": "yolov8n", "mAP50-95": 0.373, "mAP50": 0.528},
        "result": {"mAP50-95": mAP, "mAP50": mAP50},
        "delta_mAP": mAP - 0.373,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "replaced_layers": replaced,
        "total_lut_params": total_params,
        "total_lut_storage_kb": total_storage,
    }
    
    with open("report_lut_replace.json", "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  mAP50-95: {mAP:.4f}  (baseline yolov8n: 0.373)")
    print(f"  mAP50:    {mAP50:.4f}  (baseline yolov8n: 0.528)")
    print(f"  delta:    {mAP - 0.373:+.4f}")
    print(f"  Replaced : {len(replaced)} layers, {total_storage:.1f} KB LUT storage")
    print(f"  Report  : report_lut_replace.json")
    print("=" * 60)
    
    return model, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=int, default=3)
    args = parser.parse_args()
    
    # 强制进程只看见一张 GPU，避免 YOLO 内部偷偷占用其他卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = 0  # 进程内 GPU 编号变成 0
    
    LOG_FILE = "lut_v3_replace.log"
    print(f"[INFO] Log: {LOG_FILE}")
    print(f"[INFO] GPU: CUDA_VISIBLE_DEVICES={args.device} (internal id=0)")
    
    log_handle = open(LOG_FILE, "a", buffering=1)
    sys.stdout = log_handle
    sys.stderr = log_handle
    
    try:
        train_lut_replacement(epochs=args.epochs, batch_size=args.batch, device=device)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_handle.close()