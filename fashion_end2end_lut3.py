"""
Fashion-MNIST 端到端 LUT-3 查表验证
核心：用 LUT3_MultiScale_1D 替代 MLP 生成参数，保持 O(1) 查表属性
架构与 MNIST 两阶段 LUT-3 完全一致，仅改为端到端联合训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# LUT-3 多尺度 1D 查表（与 mnist_val_lut_bottleneck16.py 完全一致）
# ==========================================
class LUT3_MultiScale_1D(nn.Module):
    def __init__(self, query_dim=60, target_dim=20):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.05)
        # query_proj 直接内嵌，和两阶段版本一致
        self.query_proj = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, query_dim)
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
# 端到端模型（LUT 查表动态注入）
# ==========================================
class End2EndLUTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU()
        )
        self.backbone_main = nn.Linear(16, 20)
        self.lut = LUT3_MultiScale_1D(query_dim=60, target_dim=20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_16 = self.backbone_feat(x)
        features = self.backbone_main(feat_16)
        dyn_weights = self.lut(feat_16)  # O(1) LUT 查表
        return self.classifier(features * dyn_weights)


# ==========================================
# 基线模型（无 LUT，同容量 backbone 用于公平比较）
# ==========================================
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, trainloader, testloader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
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
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1:3d}] Train Loss: {total_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return best_test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    print("\n" + "=" * 60)
    print("【Fashion-MNIST LUT-3 端到端查表验证】")
    print("=" * 60)

    # 基线
    print("\n>>> 训练基线模型...")
    baseline = BaselineModel().to(device)
    print(f"基线参数量: {count_params(baseline):,}")
    baseline_acc = train_model(baseline, trainloader, testloader, device, epochs=100)

    # LUT-3 动态注入
    print("\n>>> 训练 LUT-3 端到端动态注入模型...")
    lut_model = End2EndLUTModel().to(device)
    print(f"LUT-3 参数量: {count_params(lut_model):,}")
    lut_acc = train_model(lut_model, trainloader, testloader, device, epochs=100)

    print("\n" + "=" * 60)
    print("【最终结果】")
    print("=" * 60)
    print(f"基线 (静态):        {baseline_acc:.2f}%")
    print(f"LUT-3 (O(1) 查表):  {lut_acc:.2f}%")
    print(f"绝对增益:           {lut_acc - baseline_acc:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
