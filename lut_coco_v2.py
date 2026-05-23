"""
LUT 动态注入实验平台 v2
======================
核心原则：所有模式均为 O(1) 查表，不引入 MLP。

实验模式（按顺序执行）：
  1. channel_select  - 方差分析选出最优地址通道 vs 默认前8通道
  2. compress        - LUT 表压缩网格搜索 (256/128/64/32)，找 knee point
  3. multilayer      - 多层小 LUT 注入 (单层/双层/三层)
  4. joint_finetune  - 解冻主干 + LUT 联合微调

输出：
  - lut_v2_all.log          : 完整训练过程日志
  - lut_v2_results.json     : 所有实验结果汇总 (单个文件)

用法：
  python lut_coco_v2.py                           # 跑全部 4 个实验
  python lut_coco_v2.py --mode channel_select     # 只跑单个
  python lut_coco_v2.py --device 3                # 指定 GPU
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


# =====================================================================
# 1. LUT 核心模块  (O(1) 查表，无 MLP)
# =====================================================================

class SliceLUT_FiLM(nn.Module):
    """FiLM 版 LUT 动态注入。支持自定义地址通道 (addr_indices)。"""
    def __init__(self, c_in=256, lut_size=256, addr_dim=8, addr_indices=None):
        super().__init__()
        self.c_in = c_in
        self.lut_size = lut_size
        self.addr_dim = addr_dim

        if addr_indices is None:
            self.addr_indices = list(range(addr_dim))
        else:
            self.addr_indices = addr_indices[:addr_dim]
            self.addr_dim = len(self.addr_indices)

        self.lut = nn.Embedding(lut_size, c_in * 2)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)
        self.scale_gamma = nn.Parameter(torch.zeros(1))
        self.scale_beta  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        addr_feat = x[:, self.addr_indices, :, :]
        addr_global = addr_feat.mean(dim=[2, 3])
        idx = (addr_global.abs() * 100).long() % self.lut_size

        lut_out = 0
        for i in range(self.addr_dim):
            lut_out = lut_out + self.lut(idx[:, i])
        lut_out = lut_out / self.addr_dim

        gamma = lut_out[:, :C] * self.scale_gamma
        beta  = lut_out[:, C:] * self.scale_beta
        return x * (1 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)


# =====================================================================
# 2. 数据集配置
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
# 3. 通道重要性分析
# =====================================================================

def analyze_channel_importance(model, data_yaml, sample_batches=20, device=3):
    """用随机数据跑前向，统计各通道激活方差。"""
    print(f"  [channel_analysis] sampling {sample_batches} batches (random input) ...")

    # 模型搬到 GPU
    model.model = model.model.to(f'cuda:{device}')

    activations = []
    def collect_hook(module, input, output):
        activations.append(output.detach().cpu())

    layers = list(model.model.model.children())
    hook = layers[10].register_forward_hook(collect_hook)

    for _ in range(sample_batches):
        dummy = torch.randn(16, 3, 640, 640, device=f'cuda:{device}')
        with torch.no_grad():
            model.model(dummy)

    hook.remove()

    if not activations:
        print("  [channel_analysis] WARN: no activations, using default [0..7]")
        return list(range(8))

    all_acts = torch.cat(activations, dim=0)
    channel_var = all_acts.var(dim=[0, 2, 3])
    sorted_idx = channel_var.argsort(descending=True)

    print(f"  [channel_analysis] top-10 indices: {sorted_idx[:10].tolist()}")
    print(f"  [channel_analysis] top-10 variance: {[f'{channel_var[i].item():.3f}' for i in sorted_idx[:10]]}")
    return sorted_idx.tolist()


# =====================================================================
# 4. 通用训练函数
# =====================================================================

def train_single_config(
    lut_size=256,
    addr_dim=8,
    addr_indices=None,
    inject_layers=None,
    epochs=30,
    batch_size=16,
    device=3,
    project_name='runs/lut_experiment',
    run_name='experiment',
    freeze_backbone=True,
    data_yaml=None,
):
    if data_yaml is None:
        data_yaml = prepare_dataset_config()

    if inject_layers is None:
        inject_layers = [10]

    print(f"  [train] loading model yolov8n.pt ...")
    model = YOLO('yolov8n.pt')

    if freeze_backbone:
        for p in model.model.parameters():
            p.requires_grad = False
        print(f"  [train] backbone: FROZEN")
    else:
        print(f"  [train] backbone: UNFROZEN (joint finetune)")

    layers_list = list(model.model.model.children())

    for layer_idx in inject_layers:
        c_in = 256 if layer_idx >= 6 else 128
        lut = SliceLUT_FiLM(c_in=c_in, lut_size=lut_size, addr_dim=addr_dim, addr_indices=addr_indices)
        for p in lut.parameters():
            p.requires_grad = True

        setattr(model.model, f'lut_layer{layer_idx}', lut)

        def make_hook(lut_mod):
            return lambda m, i, o: lut_mod(o)

        if layer_idx < len(layers_list):
            layers_list[layer_idx].register_forward_hook(make_hook(lut))
            print(f"  [train] LUT injected at layer {layer_idx}")
        else:
            print(f"  [train] WARN: layer {layer_idx} out of range")

    n_layers = len(inject_layers)
    total_params = n_layers * lut_size * 256 * 2
    storage_kb = total_params * 4 / 1024
    print(f"  [train] config: lut_size={lut_size}, layers={inject_layers}, storage={storage_kb:.1f} KB")

    train_kwargs = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': device,
        'project': project_name,
        'name': run_name,
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

    if hasattr(results, 'box') and results.box is not None:
        mAP = float(results.box.map)
        mAP50 = float(results.box.map50)
    else:
        mAP, mAP50 = 0.0, 0.0

    print(f"  [train] result: mAP50-95={mAP:.4f}, mAP50={mAP50:.4f}")
    return mAP, mAP50


# =====================================================================
# 5. 实验函数 (返回结果字典)
# =====================================================================

def run_channel_select(device=3):
    """EXP 1: 通道选择"""
    print("=" * 60)
    print("EXP 1/4: Channel Selection")
    print("=" * 60)

    data_yaml = prepare_dataset_config()
    model = YOLO('yolov8n.pt')
    top_indices = analyze_channel_importance(model, data_yaml, device=device)
    selected = top_indices[:8]
    default = list(range(8))

    print(f"  [EXP1] default  channels: {default}")
    print(f"  [EXP1] selected channels: {selected}")

    print(f"\n  [EXP1] --- Training with DEFAULT channels (0-7) ---")
    map_def, map50_def = train_single_config(
        addr_indices=default, epochs=30, device=device,
        project_name='runs/lut_ch_select', run_name='default_ch',
    )

    print(f"\n  [EXP1] --- Training with SELECTED channels ---")
    map_sel, map50_sel = train_single_config(
        addr_indices=selected, epochs=30, device=device,
        project_name='runs/lut_ch_select', run_name='selected_ch',
    )

    result = {
        "name": "channel_selection",
        "selected_channels": selected,
        "default_channels": default,
        "default": {"mAP50-95": map_def, "mAP50": map50_def},
        "selected": {"mAP50-95": map_sel, "mAP50": map50_sel},
        "delta_mAP": map_sel - map_def,
    }
    print(f"  [EXP1] SUMMARY: default={map_def:.4f}, selected={map_sel:.4f}, delta={map_sel-map_def:+.4f}")
    return result


def run_compress(device=3):
    """EXP 2: LUT 压缩"""
    print("=" * 60)
    print("EXP 2/4: LUT Compression")
    print("=" * 60)

    lut_sizes = [256, 128, 64, 32]
    entries = []

    for ls in lut_sizes:
        kb = ls * 512 * 4 / 1024
        print(f"\n  [EXP2] --- lut_size={ls} ({kb:.1f} KB) ---")
        mAP, mAP50 = train_single_config(
            lut_size=ls, epochs=30, device=device,
            project_name='runs/lut_compress', run_name=f'size_{ls}',
        )
        entries.append({"lut_size": ls, "storage_kb": kb, "mAP50-95": mAP, "mAP50": mAP50})

    # 找 knee point (mAP 不低于最优的 95% 的最小表)
    best_map = max(e["mAP50-95"] for e in entries)
    threshold = best_map * 0.95
    knee = None
    for e in entries:
        if e["mAP50-95"] >= threshold:
            knee = e
            break

    result = {
        "name": "compression",
        "baseline_mAP": best_map,
        "knee_point": knee,
        "grid": entries,
    }
    print(f"\n  [EXP2] SUMMARY:")
    for e in entries:
        print(f"    lut_size={e['lut_size']:>4}  storage={e['storage_kb']:>6.1f} KB  mAP={e['mAP50-95']:.4f}")
    print(f"  [EXP2] knee point: lut_size={knee['lut_size']} ({knee['storage_kb']:.1f} KB), mAP={knee['mAP50-95']:.4f}")
    return result


def run_multilayer(device=3):
    """EXP 3: 多层注入"""
    print("=" * 60)
    print("EXP 3/4: Multi-Layer LUT Injection")
    print("=" * 60)

    configs = [
        ("single_256",     [10],      256),
        ("dual_128",       [6, 10],   128),
        ("triple_64",      [4, 7, 10], 64),
    ]
    entries = []

    for name, layers, ls in configs:
        kb = len(layers) * ls * 256 * 2 * 4 / 1024
        print(f"\n  [EXP3] --- {name}: layers={layers}, lut_size={ls}, total={kb:.1f} KB ---")
        mAP, mAP50 = train_single_config(
            lut_size=ls, inject_layers=layers, epochs=30, device=device,
            project_name='runs/lut_multilayer', run_name=name,
        )
        entries.append({
            "name": name, "layers": layers, "lut_size": ls,
            "total_storage_kb": kb, "mAP50-95": mAP, "mAP50": mAP50,
        })

    result = {"name": "multilayer", "configs": entries}
    print(f"\n  [EXP3] SUMMARY:")
    for e in entries:
        print(f"    {e['name']:>15}  layers={e['layers']}  size={e['lut_size']}  storage={e['total_storage_kb']:.1f} KB  mAP={e['mAP50-95']:.4f}")
    return result


def run_joint_finetune(device=3):
    """EXP 4: 联合微调"""
    print("=" * 60)
    print("EXP 4/4: Joint Finetune")
    print("=" * 60)

    mAP, mAP50 = train_single_config(
        epochs=30, device=device, freeze_backbone=False,
        project_name='runs/lut_joint', run_name='joint',
    )

    result = {"name": "joint_finetune", "mAP50-95": mAP, "mAP50": mAP50}
    print(f"  [EXP4] SUMMARY: mAP50-95={mAP:.4f}, mAP50={mAP50:.4f}")
    return result


# =====================================================================
# 6. 入口
# =====================================================================

RESULTS_FILE = "lut_v2_results.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all',
                        choices=['channel_select', 'compress', 'multilayer', 'joint_finetune', 'all'])
    parser.add_argument('--device', type=int, default=3)
    args = parser.parse_args()

    # 强制进程只看见一张 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = 0  # 进程内编号

    LOG_FILE = "lut_v2_all.log"
    print(f"[INFO] GPU       : physical={args.device}, internal={device}")
    print(f"[INFO] Log file   : {LOG_FILE}")
    print(f"[INFO] Results file: {RESULTS_FILE}")
    print(f"[INFO] Start time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_handle = open(LOG_FILE, "a", buffering=1)
    sys.stdout = log_handle
    sys.stderr = log_handle

    mode_fn = {
        'channel_select': run_channel_select,
        'compress': run_compress,
        'multilayer': run_multilayer,
        'joint_finetune': run_joint_finetune,
    }

    all_results = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "baseline": {"model": "yolov8n", "mAP50-95": 0.373, "mAP50": 0.528},
        "experiments": {},
    }

    if args.mode == 'all':
        modes = ['channel_select', 'compress', 'multilayer', 'joint_finetune']
    else:
        modes = [args.mode]

    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"# START: {mode}")
        print(f"{'#'*60}")
        try:
            result = mode_fn[mode](device=device)
            all_results["experiments"][mode] = result
            print(f"# DONE: {mode}")
        except Exception as e:
            print(f"# FAILED: {mode} - {e}")
            import traceback
            traceback.print_exc()
            all_results["experiments"][mode] = {"name": mode, "status": "failed", "error": str(e)}

        # 每完成一个实验就写入一次，防止中间崩溃丢失结果
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    log_handle.close()