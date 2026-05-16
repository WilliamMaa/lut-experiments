"""
Fashion-MNIST backbone 维度缩放实验
纯静态基线，不加任何动态注入
维度：16 / 24 / 32 / 48
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineModel(nn.Module):
    def __init__(self, dim):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, trainloader, testloader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    history = []

    for epoch in range(epochs):
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
        history.append((epoch + 1, train_acc, test_acc))

        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:3d}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    return best_test_acc, history


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

    dims = [16, 24, 32, 48]
    results = {}

    for dim in dims:
        print("\n" + "=" * 60)
        print(f">>> Backbone dim = {dim}")
        print("=" * 60)
        model = BaselineModel(dim=dim).to(device)
        print(f"参数量: {count_params(model):,}")
        best_acc, history = train_model(model, trainloader, testloader, device, epochs=100)
        results[dim] = {"best": best_acc, "history": history}

    print("\n" + "=" * 60)
    print("【backbone 维度缩放 - 最终结果】")
    print("=" * 60)
    print(f"{'Dim':>6} | {'Params':>10} | {'Best Test Acc':>14}")
    print("-" * 40)
    for dim in dims:
        model = BaselineModel(dim=dim)
        n_params = count_params(model)
        best_acc = results[dim]["best"]
        print(f"{dim:>6} | {n_params:>10,} | {best_acc:>13.2f}%")
    print("=" * 60)

    # 保存详细历史供后续画图分析
    torch.save(results, "fashion_baseline_scaling_history.pt")
    print("\n详细训练历史已保存到 fashion_baseline_scaling_history.pt")


if __name__ == '__main__':
    main()
