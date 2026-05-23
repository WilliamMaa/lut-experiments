"""
Kuzushiji-MNIST: 零 query_proj + FiLM 动态注入
目标：验证方案在类内差异更大的数据集上的泛化能力

Kuzushiji-MNIST 特点：
- 日文书法字体，类内差异极大（不同人写法完全不同）
- 比 Fashion-MNIST 更难（基线准确率更低）
- 如果 LUT 在这个数据集上有效，说明方案普适

实验流程：
  实验0: KMNIST 静态基线（16-dim）
  实验1: KMNIST + FiLM [32,8,2]
  实验2: KMNIST 静态基线（8-dim）
  实验3: KMNIST 8-dim + FiLM [32,8,2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# CNN 双头 Backbone
# ==========================================
class CNNBackboneConfigurable(nn.Module):
    def __init__(self, class_dim=16, lut_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(7 * 7 * 32, class_dim)
        self.lut_dim = lut_dim
        self.lut_bn = nn.BatchNorm1d(lut_dim, momentum=0.1)

    def forward(self, x):
        conv_feat = self.conv(x)
        flat = conv_feat.view(conv_feat.size(0), -1)
        feat_class = self.proj(flat)
        feat_lut = flat[:, :self.lut_dim]
        feat_lut_coords = torch.sigmoid(self.lut_bn(feat_lut))
        return feat_class, feat_lut_coords


# ==========================================
# LUT-FiLM
# ==========================================
class LUT3_FiLM(nn.Module):
    def __init__(self, query_dim=16, film_dim=20, table_sizes=None):
        super().__init__()
        self.query_dim = query_dim
        self.film_dim = film_dim
        self.output_dim = film_dim * 2
        self.sizes = table_sizes if table_sizes else [32, 8, 2]

        self.tables = nn.ParameterList()
        for size in self.sizes:
            scale_tables = nn.Parameter(torch.randn(query_dim, self.output_dim, size))
            nn.init.normal_(scale_tables, mean=0.0, std=0.02)
            self.tables.append(scale_tables)

    def _interp_1d(self, pos, tables, size):
        B, D = pos.size()
        inj = self.output_dim
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c

        tables_b = tables.unsqueeze(0).expand(B, -1, -1, -1)
        idx_f_exp = idx_f.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)
        idx_c_exp = idx_c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)

        vf = torch.gather(tables_b, 3, idx_f_exp).squeeze(3)
        vc = torch.gather(tables_b, 3, idx_c_exp).squeeze(3)
        out = (vf * w_f.unsqueeze(-1) + vc * w_c.unsqueeze(-1)).sum(dim=1)
        return out

    def forward(self, feat_lut_coords):
        out = 0
        for i, tables in enumerate(self.tables):
            out = out + self._interp_1d(feat_lut_coords, tables, self.sizes[i])
        return out


def uniformity_loss(feat):
    feat_flat = feat.view(-1)
    hist = torch.histc(feat_flat, bins=16, min=0.0, max=1.0)
    hist = hist / hist.sum()
    uniform = torch.ones_like(hist) / 16.0
    return F.mse_loss(hist, uniform)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 静态基线模型
# ==========================================
class StaticModel(nn.Module):
    def __init__(self, class_dim=16):
        super().__init__()
        self.backbone = CNNBackboneConfigurable(class_dim=class_dim, lut_dim=class_dim)
        self.backbone_main = nn.Linear(class_dim, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, _ = self.backbone(x)
        out = self.backbone_main(feat_class)
        return self.classifier(out)


# ==========================================
# FiLM 动态注入模型
# ==========================================
class LUTFiLMModel(nn.Module):
    def __init__(self, class_dim=16, lut_dim=16, table_sizes=None):
        super().__init__()
        self.backbone = CNNBackboneConfigurable(class_dim=class_dim, lut_dim=lut_dim)
        self.lut = LUT3_FiLM(query_dim=lut_dim, film_dim=20, table_sizes=table_sizes)
        
        self.base_gamma = nn.Parameter(torch.zeros(20))
        self.base_beta = nn.Parameter(torch.zeros(20))
        self.backbone_main = nn.Linear(class_dim, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, feat_lut_coords = self.backbone(x)
        dyn = self.lut(feat_lut_coords)
        
        gamma = dyn[:, :20] + self.base_gamma
        beta = dyn[:, 20:] + self.base_beta
        
        base_out = self.backbone_main(feat_class)
        features = base_out * (1 + gamma) + beta
        return self.classifier(features)


def train_model(model, trainloader, testloader, device, epochs=100, 
                desc="Train", uniformity_lambda=0.0, use_uniformity=False, is_static=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        uni_loss_sum = 0.0
        
        for imgs, labels in tqdm(trainloader, desc=f"{desc} Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if is_static:
                logits = model(imgs)
                loss = criterion(logits, labels)
            else:
                feat_class, feat_lut_coords = model.backbone(imgs)
                dyn = model.lut(feat_lut_coords)
                
                gamma = dyn[:, :20] + model.base_gamma
                beta = dyn[:, 20:] + model.base_beta
                
                base_out = model.backbone_main(feat_class)
                features = base_out * (1 + gamma) + beta
                logits = model.classifier(features)
                
                loss = criterion(logits, labels)
                
                if use_uniformity and uniformity_lambda > 0:
                    loss_uni = uniformity_loss(feat_lut_coords)
                    loss = loss + uniformity_lambda * loss_uni
                    uni_loss_sum += loss_uni.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_uni_loss = uni_loss_sum / len(trainloader) if use_uniformity else 0.0

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
            uni_str = f" | UniLoss: {avg_uni_loss:.4f}" if use_uniformity else ""
            print(f"  [{desc} Epoch {epoch+1:3d}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_test_acc:.2f}%{uni_str}")

    return best_test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Kuzushiji-MNIST 均值/方差（需要自己计算，这里用 Fashion-MNIST 的近似值）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Kuzushiji-MNIST 数据集
    print("Downloading Kuzushiji-MNIST...")
    trainset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    results = []

    # ===== 实验0：KMNIST 静态基线（16-dim） =====
    print("\n" + "=" * 60)
    print(">>> 实验0：KMNIST 静态基线（16-dim）")
    print("=" * 60)
    model0 = StaticModel(class_dim=16).to(device)
    print(f"参数量: {count_params(model0):,}")
    acc0 = train_model(model0, trainloader, testloader, device, epochs=100, desc="KMNIST-Static16", is_static=True)
    results.append({"name": "KMNIST静态(16-dim)", "params": count_params(model0), "lut": 0, "acc": acc0, "type": "baseline"})
    print(f"\n  >>> 实验0 最佳测试准确率: {acc0:.2f}%")

    # ===== 实验1：KMNIST + FiLM [32,8,2] =====
    print("\n" + "=" * 60)
    print(">>> 实验1：KMNIST + FiLM [32,8,2]")
    print("=" * 60)
    model1 = LUTFiLMModel(class_dim=16, lut_dim=16, table_sizes=[32, 8, 2]).to(device)
    print(f"参数量: {count_params(model1):,}")
    lut_params = count_params(model1.lut) + model1.base_gamma.numel() + model1.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc1 = train_model(model1, trainloader, testloader, device, epochs=100, desc="KMNIST-FiLM16", is_static=False)
    results.append({"name": "KMNIST+FiLM[32,8,2]", "params": count_params(model1), "lut": lut_params, "acc": acc1, "type": "film"})
    print(f"\n  >>> 实验1 最佳测试准确率: {acc1:.2f}%")
    print(f"  >>> 16-dim LUT 增益: {acc1 - acc0:+.2f}%")

    # ===== 实验2：KMNIST 静态基线（8-dim） =====
    print("\n" + "=" * 60)
    print(">>> 实验2：KMNIST 静态基线（8-dim）")
    print("=" * 60)
    model2 = StaticModel(class_dim=8).to(device)
    print(f"参数量: {count_params(model2):,}")
    acc2 = train_model(model2, trainloader, testloader, device, epochs=100, desc="KMNIST-Static8", is_static=True)
    results.append({"name": "KMNIST静态(8-dim)", "params": count_params(model2), "lut": 0, "acc": acc2, "type": "baseline"})
    print(f"\n  >>> 实验2 最佳测试准确率: {acc2:.2f}%")

    # ===== 实验3：KMNIST 8-dim + FiLM [32,8,2] =====
    print("\n" + "=" * 60)
    print(">>> 实验3：KMNIST 8-dim + FiLM [32,8,2]")
    print("=" * 60)
    model3 = LUTFiLMModel(class_dim=8, lut_dim=8, table_sizes=[32, 8, 2]).to(device)
    print(f"参数量: {count_params(model3):,}")
    lut_params = count_params(model3.lut) + model3.base_gamma.numel() + model3.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc3 = train_model(model3, trainloader, testloader, device, epochs=100, desc="KMNIST-FiLM8", is_static=False)
    results.append({"name": "KMNIST8-dim+FiLM", "params": count_params(model3), "lut": lut_params, "acc": acc3, "type": "film"})
    print(f"\n  >>> 实验3 最佳测试准确率: {acc3:.2f}%")
    print(f"  >>> 8-dim LUT 增益: {acc3 - acc2:+.2f}%")

    # ===== 汇总结果 =====
    print("\n" + "=" * 75)
    print("【Kuzushiji-MNIST 实验结果汇总】")
    print("=" * 75)
    print(f"{'实验名称':<25} {'总参数':<10} {'LUT部分':<10} {'测试集':<10} {'LUT增益':<8}")
    print("-" * 75)
    
    for r in results:
        if r['type'] == 'baseline':
            gain_str = "—"
        else:
            gain_str = ""
        print(f"{r['name']:<25} {r['params']:>8,}  {r['lut']:>8,}  {r['acc']:>6.2f}%  {gain_str:>8}")
    
    print("=" * 75)


if __name__ == '__main__':
    main()
