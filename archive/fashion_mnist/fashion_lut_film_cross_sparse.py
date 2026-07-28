"""
Fashion-MNIST: 零 query_proj + 坐标交织 + Top-K 稀疏 + LUT-FiLM
目标：在零额外计算的基础上，通过特征交叉和动态仿射调制提升表达力

核心设计：
  P0: Top-K 稀疏查表（推理时只查 Top-K 个坐标的表，降低功耗+抑制过拟合）
  P1: 坐标交织（16 原生 + 8 交织 = 24 坐标，纯加减法零 MAC）
  P2: LUT-FiLM（输出 20γ + 20β，动态仿射调制）

实验流程：
  实验0: 基线 + FiLM（无交织，无 Top-K）
  实验1: 基线 + FiLM + 坐标交织
  实验2: 基线 + FiLM + 坐标交织 + Top-K=12
  实验3: 基线 + FiLM + 坐标交织 + Top-K=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# CNN 双头 Backbone（零 query_proj 版）
# ==========================================
class CNNBackboneNoQueryProj(nn.Module):
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
# LUT-FiLM：坐标交织 + Top-K 稀疏 + 动态仿射调制
# ==========================================
class LUT3_FiLM(nn.Module):
    def __init__(self, query_dim=16, film_dim=20, table_sizes=None,
                 use_coordinate_cross=False, top_k=None):
        super().__init__()
        self.query_dim = query_dim
        self.film_dim = film_dim  # 20
        self.output_dim = film_dim * 2  # 40 (gamma + beta)
        self.sizes = table_sizes if table_sizes else [64, 16, 4]
        self.use_coordinate_cross = use_coordinate_cross
        self.top_k = top_k

        # 坐标总数：原生 + 交织
        if use_coordinate_cross:
            self.total_coords = query_dim + query_dim // 2  # 16 + 8 = 24
        else:
            self.total_coords = query_dim

        # 表：total_coords × output_dim × table_size
        self.tables = nn.ParameterList()
        for size in self.sizes:
            scale_tables = nn.Parameter(torch.randn(self.total_coords, self.output_dim, size))
            nn.init.normal_(scale_tables, mean=0.0, std=0.02)
            self.tables.append(scale_tables)

    def _coordinate_cross(self, coords):
        """
        16 原生坐标 → 24 坐标（+8 交织坐标）
        交织方式：(c1+c2)/2, |c3-c4|, (c5+c6)/2, |c7-c8|, ...
        纯加减法，硬件上零 MAC
        """
        c = coords  # (B, 16)
        cross = torch.stack([
            (c[:, 0] + c[:, 1]) / 2,
            torch.abs(c[:, 2] - c[:, 3]),
            (c[:, 4] + c[:, 5]) / 2,
            torch.abs(c[:, 6] - c[:, 7]),
            (c[:, 8] + c[:, 9]) / 2,
            torch.abs(c[:, 10] - c[:, 11]),
            (c[:, 12] + c[:, 13]) / 2,
            torch.abs(c[:, 14] - c[:, 15]),
        ], dim=1)
        return torch.cat([c, cross], dim=1)  # (B, 24)

    def _interp_1d(self, pos, tables, size):
        """
        pos: (B, D) - 每个维度一个坐标
        tables: (D, output_dim, size)
        输出: (B, output_dim)
        """
        B, D = pos.size()
        inj = self.output_dim
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c

        # 查表
        tables_b = tables.unsqueeze(0).expand(B, -1, -1, -1)  # (B, D, inj, size)
        idx_f_exp = idx_f.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)  # (B, D, inj, 1)
        idx_c_exp = idx_c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)

        vf = torch.gather(tables_b, 3, idx_f_exp).squeeze(3)  # (B, D, inj)
        vc = torch.gather(tables_b, 3, idx_c_exp).squeeze(3)

        out = (vf * w_f.unsqueeze(-1) + vc * w_c.unsqueeze(-1)).sum(dim=1)  # (B, inj)
        return out

    def forward(self, feat_lut_coords):
        B = feat_lut_coords.size(0)

        # 1. 坐标交织（P1）
        if self.use_coordinate_cross:
            coords = self._coordinate_cross(feat_lut_coords)
        else:
            coords = feat_lut_coords

        # 2. Top-K 稀疏（P0）
        if self.top_k is not None:
            coords_abs = coords.abs()
            topk_vals, topk_idx = torch.topk(coords_abs, self.top_k, dim=1)
            mask = torch.zeros_like(coords_abs).scatter_(1, topk_idx, 1.0)
            coords = coords * mask

        # 3. 多尺度查表 + 求和
        out = 0
        for i, tables in enumerate(self.tables):
            out = out + self._interp_1d(coords, tables, self.sizes[i])
        return out  # (B, 40)


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
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackboneNoQueryProj(class_dim=16, lut_dim=16)
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, _ = self.backbone(x)
        out = self.backbone_main(feat_class)
        return self.classifier(out)


# ==========================================
# LUT-FiLM 动态注入模型
# ==========================================
class LUTFiLMModel(nn.Module):
    def __init__(self, table_sizes=None, use_coordinate_cross=False, top_k=None):
        super().__init__()
        self.backbone = CNNBackboneNoQueryProj(class_dim=16, lut_dim=16)
        self.lut = LUT3_FiLM(query_dim=16, film_dim=20, table_sizes=table_sizes,
                            use_coordinate_cross=use_coordinate_cross, top_k=top_k)
        
        # FiLM 参数初始化
        # gamma 初始为 0（即 1+gamma=1，恒等变换）
        # beta 初始为 0
        self.base_gamma = nn.Parameter(torch.zeros(20))
        self.base_beta = nn.Parameter(torch.zeros(20))
        
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, feat_lut_coords = self.backbone(x)
        dyn = self.lut(feat_lut_coords)  # (B, 40)
        
        gamma = dyn[:, :20] + self.base_gamma  # (B, 20)
        beta = dyn[:, 20:] + self.base_beta    # (B, 20)
        
        # FiLM: out = backbone_main(x) * (1 + gamma) + beta
        base_out = self.backbone_main(feat_class)
        out = base_out * (1 + gamma) + beta
        return self.classifier(out)


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
                dyn = model.lut(feat_lut_coords)  # (B, 40)
                
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

    # ===== 实验0：基线 + FiLM（无交织，无 Top-K） =====
    print("\n" + "=" * 60)
    print(">>> 实验0：基线 + FiLM（无交织，无 Top-K）")
    print("=" * 60)
    model0 = LUTFiLMModel(table_sizes=[64, 16, 4], use_coordinate_cross=False, top_k=None).to(device)
    print(f"参数量: {count_params(model0):,}")
    lut_params = count_params(model0.lut) + model0.base_gamma.numel() + model0.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc0 = train_model(model0, trainloader, testloader, device, epochs=100, desc="Exp0-FiLM", is_static=False)
    results.append({"name": "FiLM(无交织)", "params": count_params(model0), "lut": lut_params, "acc": acc0, "top_k": "None"})
    print(f"\n  >>> 实验0 最佳测试准确率: {acc0:.2f}%")

    # ===== 实验1：基线 + FiLM + 坐标交织 =====
    print("\n" + "=" * 60)
    print(">>> 实验1：基线 + FiLM + 坐标交织（16→24 坐标）")
    print("=" * 60)
    model1 = LUTFiLMModel(table_sizes=[64, 16, 4], use_coordinate_cross=True, top_k=None).to(device)
    print(f"参数量: {count_params(model1):,}")
    lut_params = count_params(model1.lut) + model1.base_gamma.numel() + model1.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc1 = train_model(model1, trainloader, testloader, device, epochs=100, desc="Exp1-FiLM+Cross", is_static=False)
    results.append({"name": "FiLM+坐标交织", "params": count_params(model1), "lut": lut_params, "acc": acc1, "top_k": "None"})
    print(f"\n  >>> 实验1 最佳测试准确率: {acc1:.2f}%")

    # ===== 实验2：基线 + FiLM + 坐标交织 + Top-K=12 =====
    print("\n" + "=" * 60)
    print(">>> 实验2：基线 + FiLM + 坐标交织 + Top-K=12")
    print("=" * 60)
    model2 = LUTFiLMModel(table_sizes=[64, 16, 4], use_coordinate_cross=True, top_k=12).to(device)
    print(f"参数量: {count_params(model2):,}")
    lut_params = count_params(model2.lut) + model2.base_gamma.numel() + model2.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc2 = train_model(model2, trainloader, testloader, device, epochs=100, desc="Exp2-FiLM+Cross+Top12", is_static=False)
    results.append({"name": "FiLM+交织+Top12", "params": count_params(model2), "lut": lut_params, "acc": acc2, "top_k": 12})
    print(f"\n  >>> 实验2 最佳测试准确率: {acc2:.2f}%")

    # ===== 实验3：基线 + FiLM + 坐标交织 + Top-K=8 =====
    print("\n" + "=" * 60)
    print(">>> 实验3：基线 + FiLM + 坐标交织 + Top-K=8")
    print("=" * 60)
    model3 = LUTFiLMModel(table_sizes=[64, 16, 4], use_coordinate_cross=True, top_k=8).to(device)
    print(f"参数量: {count_params(model3):,}")
    lut_params = count_params(model3.lut) + model3.base_gamma.numel() + model3.base_beta.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc3 = train_model(model3, trainloader, testloader, device, epochs=100, desc="Exp3-FiLM+Cross+Top8", is_static=False)
    results.append({"name": "FiLM+交织+Top8", "params": count_params(model3), "lut": lut_params, "acc": acc3, "top_k": 8})
    print(f"\n  >>> 实验3 最佳测试准确率: {acc3:.2f}%")

    # ===== 汇总结果 =====
    print("\n" + "=" * 75)
    print("【实验结果汇总】")
    print("=" * 75)
    print(f"{'实验名称':<20} {'总参数':<10} {'LUT部分':<10} {'Top-K':<8} {'测试集':<10} {'增益':<8}")
    print("-" * 75)
    
    baseline_acc = results[0]["acc"]
    for r in results:
        gain_str = f"{r['acc'] - baseline_acc:+.2f}%" if r != results[0] else "—"
        top_k_str = str(r['top_k'])
        print(f"{r['name']:<20} {r['params']:>8,}  {r['lut']:>8,}  {top_k_str:>6}  {r['acc']:>6.2f}%  {gain_str:>8}")
    
    print("=" * 75)


if __name__ == '__main__':
    main()
