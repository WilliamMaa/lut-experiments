"""
方向2：动态注入 backbone_main 层（16→20 的 weight + bias）
验证"细胞级 MOE"可以应用到网络的任意一层，不只是最后一层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 端到端模型：动态注入 backbone_main（16→20）
# ==========================================
class End2EndMainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取（固定/可学习）
        self.backbone_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU()
        )

        # 分类器
        self.classifier = nn.Linear(20, 10)

        # 动态生成 backbone_main 的 weight 残差（20*16=320）和 bias 残差（20），共 340
        self.query_proj = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, 340)
        )

        # 基线 weight 和 bias（可学习）
        self.base_weight = nn.Parameter(torch.randn(20, 16) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(20))

    def forward(self, x):
        feat_16 = self.backbone_feat(x)  # (B, 16)

        # 生成动态残差
        dyn_residual = self.query_proj(feat_16)  # (B, 340)
        dyn_w_residual = dyn_residual[:, :320].view(-1, 20, 16)  # (B, 20, 16)
        dyn_b_residual = dyn_residual[:, 320:]  # (B, 20)

        # 基线 + 残差注入
        weight = self.base_weight.unsqueeze(0) + dyn_w_residual  # (B, 20, 16)
        bias = self.base_bias.unsqueeze(0) + dyn_b_residual      # (B, 20)

        # 批量矩阵乘法：feat_16 @ W^T + b
        # feat_16: (B, 16), weight: (B, 20, 16)
        features = torch.bmm(weight, feat_16.unsqueeze(-1)).squeeze(-1) + bias  # (B, 20)

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

    model = End2EndMainModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print("\n--- 端到端联合训练（动态注入 backbone_main）---")
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
    print("【端到端联合训练 - backbone_main 动态注入 - 最终结果】")
    print("=" * 60)
    print(f"最佳测试集准确率: {best_test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
