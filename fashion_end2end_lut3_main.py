"""
Fashion-MNIST LUT-3 main：340-dim 查表，直接替换 backbone_main 的 weight+bias
LUT 输出 320-dim weight 残差 + 20-dim bias 残差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


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


class End2EndLUTModel_main(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_feat = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.ReLU())
        self.lut = LUT3_MultiScale_1D_main(target_dim=340)
        self.base_weight = nn.Parameter(torch.randn(20, 16) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(20))
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_16 = self.backbone_feat(x)
        dyn = self.lut(feat_16)
        dyn_w = dyn[:, :320].view(-1, 20, 16)
        dyn_b = dyn[:, 320:]

        weight = self.base_weight.unsqueeze(0) + dyn_w
        bias = self.base_bias.unsqueeze(0) + dyn_b

        features = torch.bmm(weight, feat_16.unsqueeze(-1)).squeeze(-1) + bias
        return self.classifier(features)


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 16), nn.ReLU(),
            nn.Linear(16, 20), nn.ReLU(),
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, trainloader, testloader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
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
            print(f"[Epoch {epoch+1:3d}] Train Loss: {total_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    return best_test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    trainloader = DataLoader(torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    testloader = DataLoader(torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform), batch_size=256, shuffle=False)

    print("\n>>> 训练基线...")
    baseline = BaselineModel().to(device)
    print(f"基线参数量: {count_params(baseline):,}")
    base_acc = train_model(baseline, trainloader, testloader, device)

    print("\n>>> 训练 LUT-3 main (340-dim)...")
    model = End2EndLUTModel_main().to(device)
    print(f"LUT-3 main 参数量: {count_params(model):,}")
    lut_acc = train_model(model, trainloader, testloader, device)

    print("\n" + "=" * 60)
    print("【LUT-3 main (340-dim) 结果】")
    print(f"基线:      {base_acc:.2f}%")
    print(f"LUT-3 main:{lut_acc:.2f}%")
    print(f"增益:      {lut_acc - base_acc:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
