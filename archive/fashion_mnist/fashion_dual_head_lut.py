"""
Fashion-MNIST: 双头解耦表示 + LUT 动态注入
目标：推理阶段零额外计算，feat_lut 直接作为 SRAM 地址线

核心设计：
  - CNN conv 输出 32×7×7 = 1568-dim
  - flat[:, :16] → feat_lut（查表地址，零计算切片）
  - Linear(1568→32) → feat_class（分类特征）
  - uniformity loss 约束 feat_lut 分布均匀

实验流程：
  实验0: 双头静态基线（无LUT，无uniformity loss）
  实验1: 双头 + LUT（无uniformity loss）
  实验2: 双头 + LUT + uniformity loss（lambda扫描）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


# ==========================================
# CNN 双头 Backbone
# ==========================================
class CNNBackboneDualHead(nn.Module):
    def __init__(self, class_dim=32, lut_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1×28×28 → 8×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8×28×28 → 8×14×14
            nn.Conv2d(8, 32, kernel_size=3, padding=1),  # 8×14×14 → 32×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32×14×14 → 32×7×7
        )
        self.proj = nn.Linear(7 * 7 * 32, class_dim)     # 1568 → 32
        self.lut_dim = lut_dim

    def forward(self, x):
        conv_feat = self.conv(x)
        flat = conv_feat.view(conv_feat.size(0), -1)     # 1568-dim

        feat_class = self.proj(flat)                      # 32-dim，分类用
        feat_lut = flat[:, :self.lut_dim]                 # 16-dim，查表用，零计算

        return feat_class, feat_lut


# ==========================================
# LUT-3 多尺度查表（适配双头方案）
# ==========================================
class LUT3_MultiScale_1D(nn.Module):
    def __init__(self, query_dim=16, inject_dim=340, table_sizes=None):
        super().__init__()
        self.query_dim = query_dim
        self.inject_dim = inject_dim
        self.sizes = table_sizes if table_sizes else [256, 64, 16]
        
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(inject_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.02)
        
        # query_proj：将 feat_lut 映射到查表坐标
        # 注意：这里仍然需要 query_proj，因为 feat_lut 是 16-dim，
        # 但查表需要为每个尺度生成 inject_dim 个坐标
        # 后续可替换为"feat_lut 直接作为坐标"的硬件方案
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, 64), nn.ReLU(),
            nn.Linear(64, inject_dim * len(self.sizes))
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

    def forward(self, feat_lut):
        B = feat_lut.size(0)
        coords = torch.sigmoid(self.query_proj(feat_lut)).view(B, len(self.sizes), self.inject_dim)
        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords[:, i, :], self.tables[i], size)
        return out


def uniformity_loss(feat):
    """
    Wang & Isola, ICML 2020
    让特征均匀分布在超球面上，最大化表空间利用率
    """
    feat = F.normalize(feat, dim=-1)
    # 采样一批计算 pairwise distance（避免 O(N^2)）
    n = min(feat.size(0), 256)
    idx = torch.randperm(feat.size(0))[:n]
    feat_sample = feat[idx]
    sq_dists = torch.pdist(feat_sample, p=2).pow(2)
    return sq_dists.mul(-2).exp().mean().log()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 模型定义
# ==========================================
class DualHeadStaticModel(nn.Module):
    """实验0：双头静态基线"""
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackboneDualHead(class_dim=32, lut_dim=16)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        feat_class, feat_lut = self.backbone(x)
        return self.classifier(feat_class)


class DualHeadLUTModel(nn.Module):
    """实验1/2：双头 + LUT 动态注入"""
    def __init__(self, table_sizes=None):
        super().__init__()
        self.backbone = CNNBackboneDualHead(class_dim=32, lut_dim=16)
        # inject_dim = 20×32(weight) + 20(bias) = 660
        self.inject_dim = 660
        self.lut = LUT3_MultiScale_1D(query_dim=16, inject_dim=self.inject_dim, table_sizes=table_sizes)
        
        # base weight: backbone_main 从 32-dim → 20-dim
        self.base_weight = nn.Parameter(torch.randn(20, 32) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(20))
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, feat_lut = self.backbone(x)
        dyn = self.lut(feat_lut)
        dyn_w = dyn[:, :640].view(-1, 20, 32)    # 20×32 = 640
        dyn_b = dyn[:, 640:]                      # 20
        
        weight = self.base_weight.unsqueeze(0) + dyn_w
        bias = self.base_bias.unsqueeze(0) + dyn_b
        
        out = torch.bmm(weight, feat_class.unsqueeze(-1)).squeeze(-1) + bias
        return self.classifier(out)


def train_model(model, trainloader, testloader, device, epochs=100, 
                desc="Train", uniformity_lambda=0.0, use_uniformity=False, is_static=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
                # 静态模型直接前向传播
                logits = model(imgs)
                loss = criterion(logits, labels)
            else:
                # LUT模型：双头 + 动态注入
                feat_class, feat_lut = model.backbone(imgs)
                dyn = model.lut(feat_lut)
                dyn_w = dyn[:, :640].view(-1, 20, 32)
                dyn_b = dyn[:, 640:660]
                
                weight = model.base_weight.unsqueeze(0) + dyn_w
                bias = model.base_bias.unsqueeze(0) + dyn_b
                
                features = torch.bmm(weight, feat_class.unsqueeze(-1)).squeeze(-1) + bias
                logits = model.classifier(features)
                
                loss = criterion(logits, labels)
                
                if use_uniformity and uniformity_lambda > 0:
                    loss_uni = uniformity_loss(feat_lut)
                    loss = loss + uniformity_lambda * loss_uni
                    uni_loss_sum += loss_uni.item()
            
            loss.backward()
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
                if is_static:
                    outputs = model(imgs)
                else:
                    feat_class, feat_lut = model.backbone(imgs)
                    dyn = model.lut(feat_lut)
                    dyn_w = dyn[:, :640].view(-1, 20, 32)
                    dyn_b = dyn[:, 640:660]
                    
                    weight = model.base_weight.unsqueeze(0) + dyn_w
                    bias = model.base_bias.unsqueeze(0) + dyn_b
                    
                    features = torch.bmm(weight, feat_class.unsqueeze(-1)).squeeze(-1) + bias
                    outputs = model.classifier(features)
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

    results = []

    # ===== 实验0：双头静态基线 =====
    print("\n" + "=" * 60)
    print(">>> 实验0：双头静态基线（无LUT，无uniformity loss）")
    print("=" * 60)
    model0 = DualHeadStaticModel().to(device)
    print(f"参数量: {count_params(model0):,}")
    acc0 = train_model(model0, trainloader, testloader, device, epochs=100, desc="Exp0-Static", is_static=True)
    results.append({"name": "双头静态基线", "params": count_params(model0), "acc": acc0, "lambda": "N/A"})
    print(f"\n  >>> 实验0 最佳测试准确率: {acc0:.2f}%")

    # ===== 实验1：双头 + LUT（无uniformity loss） =====
    print("\n" + "=" * 60)
    print(">>> 实验1：双头 + LUT（无uniformity loss）")
    print("=" * 60)
    model1 = DualHeadLUTModel().to(device)
    print(f"参数量: {count_params(model1):,}")
    acc1 = train_model(model1, trainloader, testloader, device, epochs=100, desc="Exp1-LUT", is_static=False)
    results.append({"name": "双头+LUT(无uniformity)", "params": count_params(model1), "acc": acc1, "lambda": 0.0})
    print(f"\n  >>> 实验1 最佳测试准确率: {acc1:.2f}%")

    # ===== 实验2：双头 + LUT + uniformity loss（lambda扫描） =====
    lambdas = [0.01, 0.05, 0.1]
    for lam in lambdas:
        print("\n" + "=" * 60)
        print(f">>> 实验2：双头 + LUT + uniformity loss (lambda={lam})")
        print("=" * 60)
        model2 = DualHeadLUTModel().to(device)
        print(f"参数量: {count_params(model2):,}")
        acc2 = train_model(model2, trainloader, testloader, device, epochs=100, 
                          desc=f"Exp2-LUT-lam{lam}", uniformity_lambda=lam, use_uniformity=True, is_static=False)
        results.append({"name": f"双头+LUT+uniformity(λ={lam})", "params": count_params(model2), "acc": acc2, "lambda": lam})
        print(f"\n  >>> 实验2 (λ={lam}) 最佳测试准确率: {acc2:.2f}%")

    # ===== 汇总结果 =====
    print("\n" + "=" * 70)
    print("【实验结果汇总】")
    print("=" * 70)
    print(f"{'实验名称':<35} {'总参数':<10} {'lambda':<10} {'测试集':<10} {'增益':<8}")
    print("-" * 70)
    
    baseline_acc = results[0]["acc"]
    for r in results:
        gain_str = f"{r['acc'] - baseline_acc:+.2f}%" if r != results[0] else "—"
        print(f"{r['name']:<35} {r['params']:>8,}  {str(r['lambda']):>8}  {r['acc']:>6.2f}%  {gain_str:>8}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
