"""
Fashion-MNIST: CNN前端 + FC后端 + LUT-3 main动态注入
两步实验：
  1. 静态CNN+FC基线（目标 >= 92%）
  2. 接入LUT-3 main动态注入（目标 >= 93%，且优于同参数量静态）

CNN设计（按用户要求）：
  Conv(8,3x3) → ReLU → MaxPool(2)
  Conv(32,3x3) → ReLU → MaxPool(2)  [32通道控制总参数量在20-30K]
  Flatten → Linear → 16维特征向量(query)
  
  总参数量约28K（CNN 27.5K + FC 0.5K）
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
        # 7*7*32 = 1568
        self.proj = nn.Linear(7 * 7 * 32, out_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.proj(feat)


# ==========================================
# LUT-3 main：340-dim 查表（与之前一致）
# ==========================================
class LUT3_MultiScale_1D_main(nn.Module):
    def __init__(self, target_dim=340):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.02)
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
        coords = torch.sigmoid(self.query_proj(feat_16)).view(B, 3, self.target_dim)
        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords[:, i, :], self.tables[i], size)
        return out


# ==========================================
# 阶段 1：静态 CNN + FC
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
# 阶段 2：CNN + LUT-3 main 动态注入
# ==========================================
class LUTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone(out_dim=16)
        self.lut = LUT3_MultiScale_1D_main(target_dim=340)
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
            print(f"  [{desc} Epoch {epoch+1:3d}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

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

    # ===== 阶段 1：静态 CNN + FC =====
    print("\n" + "=" * 60)
    print(">>> 阶段 1：训练静态 CNN + FC 基线")
    print("=" * 60)
    static_model = StaticModel().to(device)
    print(f"静态模型参数量: {count_params(static_model):,}")
    static_acc = train_model(static_model, trainloader, testloader, device, epochs=100, desc="Static")

    # ===== 阶段 2：CNN + LUT 动态注入 =====
    print("\n" + "=" * 60)
    print(">>> 阶段 2：CNN + LUT-3 main 动态注入")
    print("=" * 60)
    lut_model = LUTModel().to(device)
    print(f"LUT 动态模型参数量: {count_params(lut_model):,}")
    print(f"  其中 CNN 部分: {count_params(lut_model.cnn):,}")
    lut_part_params = count_params(lut_model.lut) + lut_model.base_weight.numel() + lut_model.base_bias.numel()
    print(f"  其中 LUT 部分: {lut_part_params:,}")
    lut_acc = train_model(lut_model, trainloader, testloader, device, epochs=100, desc="LUT")

    print("\n" + "=" * 60)
    print("【CNN + LUT 动态注入 - 最终结果】")
    print("=" * 60)
    print(f"静态 CNN+FC:      {static_acc:.2f}%")
    print(f"CNN + LUT 动态:   {lut_acc:.2f}%")
    print(f"LUT 增益:         {lut_acc - static_acc:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
