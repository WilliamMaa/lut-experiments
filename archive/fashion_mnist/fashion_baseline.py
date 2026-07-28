"""
Fashion-MNIST 基线：与端到端模型同容量，但无动态注入
backbone: 784 -> 16 -> 20 -> 10
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    model = BaselineModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Fashion-MNIST 基线（无动态注入）---")
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
    print("【Fashion-MNIST 基线 - 最终结果】")
    print("=" * 60)
    print(f"最佳测试集准确率: {best_test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
