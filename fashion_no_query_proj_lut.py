"""
Fashion-MNIST: 零 query_proj + LUT 动态注入
目标：feat_lut 直接作为查表坐标，彻底消除 query_proj

核心设计：
  - CNN conv 输出 1568-dim
  - flat[:16] → feat_lut，用 BatchNorm + Sigmoid 归一化到 [0,1]
  - feat_lut 直接作为 LUT 查表坐标（零额外计算）
  - LUT 表输出 20 维 bias 残差（只注入 bias，weight 用静态）
  - 压缩表 + 大 weight_decay 解决过拟合

实验流程：
  实验0: 零 query_proj 静态基线
  实验1: 零 query_proj + 单尺度 LUT（64）
  实验2: 零 query_proj + 多尺度 LUT [64,16,4]
  实验3: 零 query_proj + 多尺度 LUT + uniformity loss
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
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1×28×28 → 8×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8×28×28 → 8×14×14
            nn.Conv2d(8, 32, kernel_size=3, padding=1),  # 8×14×14 → 32×14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32×14×14 → 32×7×7
        )
        self.proj = nn.Linear(7 * 7 * 32, class_dim)     # 1568 → 16（分类用）
        self.lut_dim = lut_dim
        
        # BatchNorm 约束 feat_lut 分布，使其更容易归一化到 [0,1]
        self.lut_bn = nn.BatchNorm1d(lut_dim, momentum=0.1)

    def forward(self, x):
        conv_feat = self.conv(x)
        flat = conv_feat.view(conv_feat.size(0), -1)     # 1568-dim

        feat_class = self.proj(flat)                      # 16-dim，分类用
        feat_lut = flat[:, :self.lut_dim]                 # 16-dim，查表用
        
        # BatchNorm 使特征分布更稳定
        feat_lut = self.lut_bn(feat_lut)
        
        # Sigmoid 映射到 (0, 1)，作为查表坐标
        feat_lut_coords = torch.sigmoid(feat_lut)

        return feat_class, feat_lut_coords


# ==========================================
# 零 query_proj LUT 查表
# ==========================================
class LUT3_NoQueryProj(nn.Module):
    def __init__(self, query_dim=16, inject_dim=20, table_sizes=None):
        super().__init__()
        self.query_dim = query_dim
        self.inject_dim = inject_dim
        self.sizes = table_sizes if table_sizes else [64, 16, 4]
        
        # 表：inject_dim × table_size
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(inject_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.02)

        # 无 query_proj！feat_lut 的每个维度直接作为查表坐标
        # query_dim 和 inject_dim 不需要相等：
        # 16 个坐标 → 查表得到 20 维输出
        # 每个坐标对应表的一行，但表有 inject_dim 行？不对
        # 正确理解：16 个坐标查 16 个表条目，但每个条目是 inject_dim 维
        # 所以需要 16 张表？这不对
        
        # 重新设计：
        # feat_lut_coords: (B, 16) → 16 个坐标
        # 输出: (B, 20) → 20 维 bias 残差
        # 方法：16 个坐标分别查 16 张小表，每张表 20 维，然后求和
        # 存储: 16 × 20 × table_size = 16 × 20 × 64 = 20K
        
        # 更简单的方法：
        # feat_lut_coords 作为"全局坐标"，查一张大表
        # 表: (20, 64) → 用 16 个坐标的平均位置去查？
        
        # 最终方案：用 16 维坐标的线性组合
        # 表: (20, 16) → 每列是一个 20 维的"基向量"
        # 输出 = coords @ table.T = (B,16) @ (16,20) = (B,20)
        # 但这变成了矩阵乘法，不是查表
        
        # 正确的零 query_proj 查表：
        # feat_lut_coords (B, 16) 的每个维度对应一个"表索引"
        # 16 个维度 → 16 次查表 → 每次查 20 维 → 20×16 输出
        # 但这样表会很大：16 × 20 × table_size
        
        # 简化：只输出 20 维，用 16 个坐标的平均值去查
        # 或者：每个坐标查表后加权平均
        pass

    def forward(self, feat_lut_coords):
        # feat_lut_coords: (B, query_dim)
        # 简单方案：加权平均
        # 输出 = Σ coords_i * table_i
        B = feat_lut_coords.size(0)
        
        # 表: (inject_dim, query_dim) → 每个 query_dim 对应一个 inject_dim 维的列
        # 输出 = feat_lut_coords @ tables[0].T = (B, query_dim) @ (query_dim, inject_dim) = (B, inject_dim)
        # 但这变成了线性层！
        
        # 真正查表：每个坐标离散化后查表
        out = 0
        for i, size in enumerate(self.sizes):
            table = self.tables[i]  # (inject_dim, query_dim)
            out = out + feat_lut_coords @ table.T  # (B, inject_dim)
        return out


class LUT3_ProperNoQueryProj(nn.Module):
    """
    正确的零 query_proj 查表实现：
    feat_lut_coords (B, 16) 的每个维度作为独立的查表坐标
    每个坐标查一张表（inject_dim 维），16 次查表后求和
    """
    def __init__(self, query_dim=16, inject_dim=20, table_sizes=None):
        super().__init__()
        self.query_dim = query_dim
        self.inject_dim = inject_dim
        self.sizes = table_sizes if table_sizes else [64]
        
        # 表：每个尺度有 query_dim 张表，每张表 inject_dim 维
        # 存储: query_dim × inject_dim × table_size
        self.tables = nn.ParameterList()
        for size in self.sizes:
            scale_tables = nn.Parameter(torch.randn(query_dim, inject_dim, size))
            nn.init.normal_(scale_tables, mean=0.0, std=0.02)
            self.tables.append(scale_tables)

    def _interp_1d(self, pos, tables, size):
        """
        pos: (B, D) - 每个维度一个坐标
        tables: (D, inj, size) - 每个维度一张表
        输出: (B, inj)
        """
        B, D = pos.size()
        inj = self.inject_dim
        pos_scaled = pos * (size - 1)  # (B, D)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)  # (B, D)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()  # (B, D)
        w_f = 1.0 - w_c
        
        # tables: (D, inj, size) → (B, D, inj, size)
        tables_b = tables.unsqueeze(0).expand(B, -1, -1, -1)
        
        # idx: (B, D) → (B, D, inj, 1) 用于在 dim=3 上 gather
        idx_f_exp = idx_f.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)  # (B, D, inj, 1)
        idx_c_exp = idx_c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inj, 1)
        
        vf = torch.gather(tables_b, 3, idx_f_exp).squeeze(3)  # (B, D, inj)
        vc = torch.gather(tables_b, 3, idx_c_exp).squeeze(3)  # (B, D, inj)
        
        # 插值 + 求和: w (B, D) → (B, D, 1)
        out = (vf * w_f.unsqueeze(-1) + vc * w_c.unsqueeze(-1)).sum(dim=1)  # (B, inj)
        return out

    def forward(self, feat_lut_coords):
        out = 0
        for i, tables in enumerate(self.tables):
            out = out + self._interp_1d(feat_lut_coords, tables, self.sizes[i])
        return out


def uniformity_loss(feat):
    """
    约束 feat_lut 的分布接近 Uniform[0,1]
    使用直方图匹配
    """
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
class StaticModelNoQueryProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackboneNoQueryProj(class_dim=16, lut_dim=16)
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, feat_lut_coords = self.backbone(x)
        out = self.backbone_main(feat_class)
        return self.classifier(out)


# ==========================================
# 零 query_proj + LUT 动态注入模型
# ==========================================
class LUTModelNoQueryProj(nn.Module):
    def __init__(self, table_sizes=None):
        super().__init__()
        self.backbone = CNNBackboneNoQueryProj(class_dim=16, lut_dim=16)
        # LUT 输出 20 维 bias 残差
        self.lut = LUT3_ProperNoQueryProj(query_dim=16, inject_dim=20, table_sizes=table_sizes)
        
        self.base_bias = nn.Parameter(torch.zeros(20))
        self.backbone_main = nn.Linear(16, 20)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        feat_class, feat_lut_coords = self.backbone(x)
        dyn_b = self.lut(feat_lut_coords)  # (B, 20)
        bias = self.base_bias.unsqueeze(0) + dyn_b
        
        out = self.backbone_main(feat_class) + bias
        return self.classifier(out)


def train_model(model, trainloader, testloader, device, epochs=100, 
                desc="Train", uniformity_lambda=0.0, use_uniformity=False, is_static=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # 大 weight_decay
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
                dyn_b = model.lut(feat_lut_coords)
                bias = model.base_bias.unsqueeze(0) + dyn_b
                
                out = model.backbone_main(feat_class) + bias
                logits = model.classifier(out)
                
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
                if is_static:
                    outputs = model(imgs)
                else:
                    feat_class, feat_lut_coords = model.backbone(imgs)
                    dyn_b = model.lut(feat_lut_coords)
                    bias = model.base_bias.unsqueeze(0) + dyn_b
                    
                    out = model.backbone_main(feat_class) + bias
                    outputs = model.classifier(out)
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

    # ===== 实验0：零 query_proj 静态基线 =====
    print("\n" + "=" * 60)
    print(">>> 实验0：零 query_proj 静态基线")
    print("=" * 60)
    model0 = StaticModelNoQueryProj().to(device)
    print(f"参数量: {count_params(model0):,}")
    acc0 = train_model(model0, trainloader, testloader, device, epochs=100, desc="Exp0-Static", is_static=True)
    results.append({"name": "零query_proj静态", "params": count_params(model0), "acc": acc0, "lambda": "N/A"})
    print(f"\n  >>> 实验0 最佳测试准确率: {acc0:.2f}%")

    # ===== 实验1：零 query_proj + 单尺度 LUT（64） =====
    print("\n" + "=" * 60)
    print(">>> 实验1：零 query_proj + 单尺度 LUT（64，只注入bias）")
    print("=" * 60)
    model1 = LUTModelNoQueryProj(table_sizes=[64]).to(device)
    print(f"参数量: {count_params(model1):,}")
    lut_params = count_params(model1.lut) + model1.base_bias.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc1 = train_model(model1, trainloader, testloader, device, epochs=100, desc="Exp1-LUT-single", is_static=False)
    results.append({"name": "零query_proj+LUT(单64)", "params": count_params(model1), "acc": acc1, "lambda": 0.0})
    print(f"\n  >>> 实验1 最佳测试准确率: {acc1:.2f}%")

    # ===== 实验2：零 query_proj + 多尺度 LUT [64,16,4] =====
    print("\n" + "=" * 60)
    print(">>> 实验2：零 query_proj + 多尺度 LUT [64,16,4]")
    print("=" * 60)
    model2 = LUTModelNoQueryProj(table_sizes=[64, 16, 4]).to(device)
    print(f"参数量: {count_params(model2):,}")
    lut_params = count_params(model2.lut) + model2.base_bias.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc2 = train_model(model2, trainloader, testloader, device, epochs=100, desc="Exp2-LUT-multi", is_static=False)
    results.append({"name": "零query_proj+LUT(64,16,4)", "params": count_params(model2), "acc": acc2, "lambda": 0.0})
    print(f"\n  >>> 实验2 最佳测试准确率: {acc2:.2f}%")

    # ===== 实验3：零 query_proj + LUT + uniformity loss =====
    print("\n" + "=" * 60)
    print(">>> 实验3：零 query_proj + LUT + uniformity loss (λ=0.01)")
    print("=" * 60)
    model3 = LUTModelNoQueryProj(table_sizes=[64, 16, 4]).to(device)
    print(f"参数量: {count_params(model3):,}")
    lut_params = count_params(model3.lut) + model3.base_bias.numel()
    print(f"  LUT部分: {lut_params:,}")
    acc3 = train_model(model3, trainloader, testloader, device, epochs=100, 
                      desc="Exp3-LUT-uni", uniformity_lambda=0.01, use_uniformity=True, is_static=False)
    results.append({"name": "零query_proj+LUT+uniformity(0.01)", "params": count_params(model3), "acc": acc3, "lambda": 0.01})
    print(f"\n  >>> 实验3 最佳测试准确率: {acc3:.2f}%")

    # ===== 汇总结果 =====
    print("\n" + "=" * 70)
    print("【实验结果汇总】")
    print("=" * 70)
    print(f"{'实验名称':<35} {'总参数':<10} {'LUT部分':<10} {'测试集':<10} {'增益':<8}")
    print("-" * 70)
    
    baseline_acc = results[0]["acc"]
    for r in results:
        gain_str = f"{r['acc'] - baseline_acc:+.2f}%" if r != results[0] else "—"
        lut_part = ""
        if r['name'] == "零query_proj+LUT(单64)":
            lut_part = f"{16 * 20 * 64 + 20:,}"
        elif '64,16,4' in r['name']:
            lut_part = f"{16 * 20 * (64+16+4) + 20:,}"
        else:
            lut_part = "—"
        print(f"{r['name']:<35} {r['params']:>8,}  {lut_part:>10}  {r['acc']:>6.2f}%  {gain_str:>8}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
