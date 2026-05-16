"""
方向1：端到端联合训练（layer_beta 动态注入）
不再分两阶段，backbone + query_proj + LUT 联合优化
验证端到端训练是否能突破两阶段解耦的天花板
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 端到端模型：backbone + LUT 联合优化
# ==========================================
class End2EndLayerBetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        self.backbone_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU()
        )
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

        # LUT-3 组件：从 16 维特征生成 20 维动态权重
        self.query_proj = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 60)  # 3层 × 20坐标
        )
        self.sizes = [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(20, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.05)

    def _lut_query(self, feat_16):
        """从 16 维特征查表得到 20 维动态权重"""
        B = feat_16.size(0)
        coords = torch.sigmoid(self.query_proj(feat_16)).view(B, 3, 20)
        out = 0
        for i, size in enumerate(self.sizes):
            pos = coords[:, i, :]
            pos_scaled = pos * (size - 1)
            idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
            idx_c = idx_f + 1
            w_c = pos_scaled - idx_f.float()
            w_f = 1.0 - w_c

            table = self.tables[i]
            table_exp = table.unsqueeze(0).expand(B, -1, -1)
            vf = torch.gather(table_exp, 2, idx_f.unsqueeze(-1)).squeeze(-1)
            vc = torch.gather(table_exp, 2, idx_c.unsqueeze(-1)).squeeze(-1)
            out = out + vf * w_f + vc * w_c
        return out

    def forward(self, x):
        feat_16 = self.backbone_feat(x)
        dyn_weights = self._lut_query(feat_16)
        features = self.backbone_main(feat_16)
        features = features * dyn_weights
        return self.classifier(features)


# ==========================================
# 主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    model = End2EndLayerBetaModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print("\n--- 端到端联合训练 ---")
    best_test_acc = 0.0
    for epoch in range(100):
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

        # 测试集验证
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

    print("\n" + "=" * 60)
    print("【端到端联合训练 - layer_beta 动态注入 - 最终结果】")
    print("=" * 60)
    print(f"最佳测试集准确率: {best_test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
