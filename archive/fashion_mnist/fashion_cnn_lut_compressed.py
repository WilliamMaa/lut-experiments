"""
Fashion-MNIST: CNN前端 + 压缩LUT动态注入
目标：保持多尺度结构 + O(1)查表，压缩query_proj参数量

压缩策略：
  1. query_proj从 16→128→1020(133K) 压缩到 16→16→340(5.7K)
     - 3尺度共享同一组坐标，减少坐标输出维度
  2. 保持3尺度[256,64,16]不变（泛化能力核心）

预期效果：
  - LUT总参数：248K → 120K（压缩2倍）
  - 保持多分辨率查表能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# CNN 前端：输出 16-dim 特征向量
# ==========================================
class CNNBackbone(nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.proj = nn.Linear(7 * 7 * 32, out_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.proj(feat)


# ==========================================
# 原始 LUT-3：用于对比
# ==========================================
class LUT3_Original(nn.Module):
    def __init__(self, target_dim=340):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.02)
        
        # 原始：每个尺度独立坐标 → 340×3=1020维输出
        self.query_proj = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, target_dim * 3)
        )

    def _interp_1d(self, pos, table, size):
        B = pos.size(0)
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c
        table_exp = table.unsqueeze(0).expand(B, -1, -1)
        vf = torch.gather(table_exp, 2, idx_f.unsqueeze(-1)).squeeze(-1)
        vc = torch.gather(table_exp, 2, idx_c.unsqueeze(-1)).squeeze(-1)
        return vf * w_f + vc * w_c

    def forward(self, feat_16):
        B = feat_16.size(0)
        # 每个尺度独立坐标
        coords = torch.sigmoid(self.query_proj(feat_16)).view(B, 3, self.target_dim)
        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords[:, i, :], self.tables[i], size)
        return out


# ==========================================
# 压缩 LUT-3：共享坐标 + query_proj压缩
# ==========================================
class LUT3_Compressed(nn.Module):
    def __init__(self, target_dim=340, table_sizes=None, hidden_dim=16):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = table_sizes if table_sizes else [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.02)
        
        # 压缩：3尺度共享坐标 → 只输出340维
        self.query_proj = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def _interp_1d(self, pos, table, size):
        B = pos.size(0)
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c
        table_exp = table.unsqueeze(0).expand(B, -1, -1)
        vf = torch.gather(table_exp, 2, idx_f.unsqueeze(-1)).squeeze(-1)
        vc = torch.gather(table_exp, 2, idx_c.unsqueeze(-1)).squeeze(-1)
        return vf * w_f + vc * w_c

    def forward(self, feat_16):
        # 3尺度共享同一组坐标
        coords = torch.sigmoid(self.query_proj(feat_16))  # (B, 340)
        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords, self.tables[i], size)
        return out


# ==========================================
# 静态基线模型
# ==========================================
class StaticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone(out_dim=16)
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat = self.cnn(x)
        out = self.backbone_main(feat)
        return self.classifier(out)


# ==========================================
# LUT动态模型（可切换原始/压缩版本）
# ==========================================
class LUTModel(nn.Module):
    def __init__(self, lut_version="compressed", **lut_kwargs):
        super().__init__()
        self.cnn = CNNBackbone(out_dim=16)
        
        if lut_version == "original":
            self.lut = LUT3_Original(target_dim=340)
        elif lut_version == "compressed":
            self.lut = LUT3_Compressed(target_dim=340, **lut_kwargs)
        else:
            raise ValueError(f"Unknown lut_version: {lut_version}")
        
        self.base_weight = nn.Parameter(torch.randn(20, 16) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(20))
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat = self.cnn(x)
        dyn = self.lut(feat)
        dyn_w = dyn[:, :320].view(-1, 20, 16)
        dyn_b = dyn[:, 320:]
        weight = self.base_weight.unsqueeze(0) + dyn_w
        bias = self.base_bias.unsqueeze(0) + dyn_b
        out = torch.bmm(weight, feat.unsqueeze(-1)).squeeze(-1) + bias
        return self.classifier(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_lut_params(lut_model):
    """只统计LUT部分的参数（query_proj + tables + base）"""
    return count_params(lut_model.lut) + lut_model.base_weight.numel() + lut_model.base_bias.numel()


def train_model(model, trainloader, testloader, device, epochs=100, desc="Train"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(trainloader, desc=f"{desc} Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = 100.0 * correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  [{desc} Epoch {epoch+1:3d}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_test_acc:.2f}%")

    return best_test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    trainloader = DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=256, shuffle=True
    )
    testloader = DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False
    )

    # ===== 实验配置 =====
    experiments = [
        {"name": "静态CNN+FC", "type": "static"},
        {"name": "原始LUT-3(对照)", "type": "lut", "version": "original"},
        {"name": "压缩LUT-3(共享坐标)", "type": "lut", "version": "compressed", "hidden_dim": 16},
        {"name": "压缩LUT-3(hidden=8)", "type": "lut", "version": "compressed", "hidden_dim": 8},
    ]

    results = []

    for exp in experiments:
        print("\n" + "=" * 60)
        print(f">>> 实验: {exp['name']}")
        print("=" * 60)

        if exp["type"] == "static":
            model = StaticModel().to(device)
            total_params = count_params(model)
            lut_params = 0
            print(f"总参数量: {total_params:,}")
        else:
            model = LUTModel(lut_version=exp["version"], **{k: v for k, v in exp.items() if k not in ["name", "type", "version"]}).to(device)
            total_params = count_params(model)
            lut_params = count_lut_params(model)
            print(f"总参数量: {total_params:,}")
            print(f"  CNN部分: {count_params(model.cnn):,}")
            print(f"  LUT部分: {lut_params:,}")

        best_acc = train_model(model, trainloader, testloader, device, epochs=100, desc=exp["name"])
        
        results.append({
            "name": exp["name"],
            "total_params": total_params,
            "lut_params": lut_params,
            "best_acc": best_acc
        })
        print(f"\n  >>> {exp['name']} 最佳测试准确率: {best_acc:.2f}%")

    # ===== 汇总结果 =====
    print("\n" + "=" * 70)
    print("【实验结果汇总】")
    print("=" * 70)
    print(f"{'实验名称':<25} {'总参数':<10} {'LUT参数':<10} {'测试集':<10}")
    print("-" * 70)
    
    static_acc = None
    for r in results:
        if r["lut_params"] == 0:
            static_acc = r["best_acc"]
        gain_str = f"{r['best_acc'] - static_acc:+.2f}%" if static_acc and r["lut_params"] > 0 else "—"
        print(f"{r['name']:<25} {r['total_params']:>8,}  {r['lut_params']:>8,}  {r['best_acc']:>6.2f}%  {gain_str:>8}")
    
    print("=" * 70)
    
    # 压缩比分析
    original_lut = next((r for r in results if "原始" in r["name"]), None)
    compressed_luts = [r for r in results if "压缩" in r["name"]]
    
    if original_lut and compressed_luts:
        print("\n【压缩效果分析】")
        print(f"原始LUT参数: {original_lut['lut_params']:,}")
        for c in compressed_luts:
            ratio = original_lut['lut_params'] / c['lut_params']
            acc_diff = c['best_acc'] - original_lut['best_acc']
            print(f"{c['name']}: {c['lut_params']:,} (压缩{ratio:.1f}x), 准确率差异: {acc_diff:+.2f}%")


if __name__ == '__main__':
    main()
