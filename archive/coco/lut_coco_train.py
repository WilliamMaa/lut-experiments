"""
YOLOv8 + LUT 动态注入 (清晰日志版)
=====================================
修复点：
1. 开启详细日志 (verbose=True)，解决卡住假象。
2. 关闭缓存 (cache=False)，防止加载 11 万张图片时卡死内存。
3. 增加进度刷新，确保日志实时写入文件。
"""

import torch
import torch.nn as nn
import yaml
import os
import sys
import json
from ultralytics import YOLO

# 确保输出不被缓冲
sys.stdout.reconfigure(line_buffering=True)


# =====================================================================
# 1. LUT 注入模块
# =====================================================================

class SliceLUT_FiLM(nn.Module):
    def __init__(self, c_in=256, lut_size=256, addr_dim=8):
        super().__init__()
        self.c_in = c_in
        self.lut_size = lut_size
        self.addr_dim = addr_dim

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

        # 冷启动：全 0 初始化
        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        addr_feat = x[:, :self.addr_dim, :, :]
        addr_global = addr_feat.mean(dim=[2, 3])
        
        # 索引计算
        idx = (addr_global.abs() * 100).long() % self.lut_size

        # 查表
        lut_out = 0
        for i in range(self.addr_dim):
            lut_out = lut_out + self.lut(idx[:, i])
        lut_out = lut_out / self.addr_dim

        # FiLM
        gamma = lut_out[:, :C] * self.scale_gamma
        beta = lut_out[:, C:] * self.scale_beta
        return x * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)


# =====================================================================
# 2. 配置数据集
# =====================================================================

def prepare_dataset_config():
    dataset_root = "/data1/datasets/coco"
    if not os.path.exists(dataset_root):
        print(f"[ERROR] Cannot find dataset at: {dataset_root}")
        sys.exit(1)

    print(f"[INFO] Dataset found: {dataset_root}")

    config_data = {
        'path': dataset_root,
        'train': 'images/train2017',
        'val': 'images/val2017',
        'nc': 80,
        'names': {i: str(i) for i in range(80)} # 简化名称列表，减少日志冗余
    }
    # 补充实际类别名 (部分)
    names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    config_data['names'] = names

    yaml_path = "my_coco_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)
    
    return yaml_path


# =====================================================================
# 3. 训练逻辑
# =====================================================================

def train_coco_with_lut(
    epochs=50,
    batch_size=16,
    img_size=640,
    lut_size=256,
    inject_layer=10,
    device=3,
):
    print("="*50)
    print("Starting YOLOv8 + LUT Training")
    print("="*50)
    sys.stdout.flush()
    
    data_yaml = prepare_dataset_config()
    
    # 1. 加载模型
    print("[1/5] Loading YOLOv8 model...")
    sys.stdout.flush()
    model = YOLO('yolov8n.pt')
    
    # 2. 冻结主干
    print("[2/5] Freezing backbone...")
    sys.stdout.flush()
    for p in model.model.parameters():
        p.requires_grad = False

    # 3. 创建 LUT
    print("[3/5] Creating LUT module...")
    sys.stdout.flush()
    lut_module = SliceLUT_FiLM(c_in=256, lut_size=lut_size, addr_dim=8)
    for p in lut_module.parameters():
        p.requires_grad = True

    model.model.lut_module = lut_module
    
    # 4. 注入 Hook
    print(f"[4/5] Injecting LUT at layer {inject_layer}...")
    sys.stdout.flush()
    
    def hook_fn(module, input, output):
        return model.model.lut_module(output)
        
    layers = list(model.model.model.children())
    if inject_layer < len(layers):
        layers[inject_layer].register_forward_hook(hook_fn)
    else:
        print("[ERROR] Layer index out of range.")
        return

    storage_kb = lut_size * 512 * 4 / 1024
    print(f"[INFO] LUT Storage: {storage_kb:.1f} KB")
    print("[5/5] Starting training loop...")
    sys.stdout.flush()
    
    train_kwargs = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': 'runs/lut_coco_final',
        'name': f'lut_{lut_size}_layer{inject_layer}',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': 1e-3,
        'weight_decay': 1e-4,
        'cos_lr': True,
        'close_mosaic': 10,
        'patience': 50,
        
        # 关键修改：开启日志，关闭缓存
        'verbose': True,
        'cache': False, 
    }
    
    try:
        # 执行训练
        results = model.train(**train_kwargs)
        
        print("\n[INFO] Training finished successfully.")
        print("[INFO] Generating report...")
        
        # 保存结果到 JSON
        save_clean_report(results, train_kwargs, storage_kb)

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

def save_clean_report(results, config, storage_kb):
    try:
        # YOLO 返回的是 DetMetrics 对象，不是字典
        if hasattr(results, 'box') and results.box is not None:
            final_box_map = float(results.box.map)
            final_box_map50 = float(results.box.map50)
        elif hasattr(results, 'results_dict'):
            d = results.results_dict
            final_box_map = float(d.get('metrics/mAP50-95(B)', 0.0))
            final_box_map50 = float(d.get('metrics/mAP50(B)', 0.0))
        else:
            print("[WARN] Cannot extract mAP from results object")
            return
        
        report = {
            "status": "success",
            "final_metrics": {
                "mAP50-95": float(final_box_map),
                "mAP50": float(final_box_map50)
            },
            "config": {
                "lut_size": config.get('lut_size', 256), # Note: config is dict of kwargs
                "inject_layer": 10,
                "epochs": config.get('epochs', 50),
                "batch_size": config.get('batch_size', 16),
                "lut_storage_kb": storage_kb
            }
        }

        with open("final_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        print("="*50)
        print("FINAL RESULTS SAVED TO: final_report.json")
        print(f"  mAP50-95: {final_box_map:.4f}")
        print(f"  mAP50:    {final_box_map50:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")

if __name__ == '__main__':
    import sys
    
    LOG_FILE = "lut_coco_train_final.log"
    print(f"[INFO] Logging output to: {LOG_FILE}")
    print(f"[INFO] You can watch progress with: tail -f {LOG_FILE}")
    sys.stdout.flush()
    
    # 重定向输出
    log_handle = open(LOG_FILE, "a", buffering=1)
    sys.stdout = log_handle
    sys.stderr = log_handle

    train_coco_with_lut()
    log_handle.close()
