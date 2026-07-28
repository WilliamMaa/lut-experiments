"""
Fashion-MNIST VQ-LUT：离散码本查表，从 per-sample 到 per-cluster
核心：64 个码本向量，query_proj 输出查询向量，argmin 硬选择最近码本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class VQLUT(nn.Module):
    def __init__(self, num_codes=64, code_dim=40):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        # 离散码本
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.normal_(self.codebook, mean=0.0, std=0.05)

        self.query_proj = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, code_dim)
        )

    def forward(self, feat_16):
        z = self.query_proj(feat_16)  # (B, 40) 查询向量

        # 计算与所有码本的距离，硬选择最近的一个
        distances = torch.cdist(z, self.codebook)  # (B, num_codes)
        idx = distances.argmin(dim=-1)  # (B,)
        e = self.codebook[idx]  # (B, 40) 选中的码本向量

        # STE：前向用离散码本 e，反向梯度传给 z
        z_q = e + (z - e).detach()
        return z_q, z, e


class End2EndVQModel(nn.Module):
    def __init__(self, num_codes=64):
        super().__init__()
        self.backbone_feat = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.ReLU())
        self.backbone_main = nn.Linear(16, 20)
        self.vq_lut = VQLUT(num_codes=num_codes, code_dim=40)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_16 = self.backbone_feat(x)
        features = self.backbone_main(feat_16)
        dyn, z, e = self.vq_lut(feat_16)
        dyn_mul = dyn[:, :20]
        dyn_add = dyn[:, 20:]
        features = features * dyn_mul + dyn_add
        logits = self.classifier(features)
        return logits, z, e


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 16), nn.ReLU(),
            nn.Linear(16, 20), nn.ReLU(),
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x)), None, None


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, trainloader, testloader, device, epochs=100, beta=0.25):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, ce_loss_sum, vq_loss_sum = 0, 0, 0
        correct, total = 0, 0

        for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, z, e = model(imgs)
            ce_loss = ce_criterion(outputs, labels)
            # Commitment loss：约束 z 靠近选中的码本 e
            if z is not None and e is not None:
                vq_loss = F.mse_loss(z, e.detach())
                loss = ce_loss + beta * vq_loss
            else:
                vq_loss = torch.tensor(0.0, device=device)
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ce_loss_sum += ce_loss.item()
            vq_loss_sum += vq_loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs, _, _ = model(imgs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1:3d}] CE:{ce_loss_sum/len(trainloader):.4f} VQ:{vq_loss_sum/len(trainloader):.4f} "
                  f"Train:{train_acc:.2f}% Test:{test_acc:.2f}%")

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

    print("\n>>> 训练基线...")
    baseline = BaselineModel().to(device)
    print(f"基线参数量: {count_params(baseline):,}")
    base_acc = train_model(baseline, trainloader, testloader, device)

    print("\n>>> 训练 VQ-LUT (64 码本)...")
    model = End2EndVQModel(num_codes=64).to(device)
    print(f"VQ-LUT 参数量: {count_params(model):,}")
    vq_acc = train_model(model, trainloader, testloader, device)

    print("\n" + "=" * 60)
    print("【VQ-LUT (64 码本) 结果】")
    print(f"基线:    {base_acc:.2f}%")
    print(f"VQ-LUT:  {vq_acc:.2f}%")
    print(f"增益:    {vq_acc - base_acc:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
