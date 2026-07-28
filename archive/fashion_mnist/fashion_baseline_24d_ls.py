"""
Fashion-MNIST 24-dim 静态基线 + Label Smoothing
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineModel(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, dim),
            nn.ReLU(),
            nn.Linear(dim, 20),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))


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

    model = BaselineModel(dim=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("--- 24-dim 基线 + Label Smoothing ---")

    best_test_acc = 0.0
    for epoch in range(100):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
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
            print(f"[Epoch {epoch+1:3d}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    print("\n" + "=" * 60)
    print(f"24-dim + Label Smoothing 最佳测试集: {best_test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
