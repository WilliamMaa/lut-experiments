"""
流程二（LUT 升级版）：四种可学习 O1 查表方案
核心改进：
  1. 用可学习的 QueryNet + LUT 查表替代 sklearn 的 k-NN/RBF/VQ
  2. 查表是常数时间 O1，不依赖训练集大小，硬件友好
  3. 端到端训练，query_net 和查表参数联合优化
"""

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np


# ==========================================
# 0. 基座模型（与流程一严格一致，加法残差）
# ==========================================
class LayerBeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(20))

    def forward(self, x):
        return x * self.weights


class SmallBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 3),
            nn.ReLU(),
            nn.Linear(3, 20)
        )
        self.layer_beta = LayerBeta()
        self.classifier = nn.Linear(20, 10)

    def forward(self, x, dynamic_beta_weights=None):
        features = self.backbone(x)
        if dynamic_beta_weights is not None:
            features = features * dynamic_beta_weights
        else:
            features = self.layer_beta(features)
        return self.classifier(features)


# ==========================================
# 1. 查询网络（轻量 CNN，从图像提取查询坐标）
# ==========================================
class QueryNet(nn.Module):
    """
    极简 CNN：1x28x28 -> 128 维查询向量
    参数量约 420K，计算量极小
    """
    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 14x14
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. 四种 LUT 查表方案
# ==========================================

# --------------------------------------------------
# 方案 1：20头独立 2D LUT
# query_net 输出 40 个数 = 20 对 (x,y) 坐标
# 每张 32x32 的表负责查一个参数的值
# --------------------------------------------------
class LUT1_20Head_2D(nn.Module):
    def __init__(self, target_dim=20, grid_size=32):
        super().__init__()
        self.G = grid_size
        self.target_dim = target_dim
        # 20 张独立的二维表，每张 32x32 = 1024 个格子
        self.tables = nn.Parameter(torch.randn(target_dim, self.G * self.G))
        nn.init.normal_(self.tables, mean=0.0, std=0.1)
        self.query_net = QueryNet(target_dim * 2)

    def forward(self, x):
        B = x.size(0)
        coords = torch.sigmoid(self.query_net(x)).view(B, self.target_dim, 2) * (self.G - 1)
        cx, cy = coords[:, :, 0], coords[:, :, 1]

        fx, fy = torch.floor(cx).long(), torch.floor(cy).long()
        cx_w, cy_w = cx - fx.float(), cy - fy.float()
        fx = torch.clamp(fx, 0, self.G - 2)
        fy = torch.clamp(fy, 0, self.G - 2)

        idx_00 = fx * self.G + fy
        idx_10 = (fx + 1) * self.G + fy
        idx_01 = fx * self.G + (fy + 1)
        idx_11 = (fx + 1) * self.G + (fy + 1)

        tables_exp = self.tables.unsqueeze(0).expand(B, -1, -1)  # (B, 20, G*G)

        def _gather(idx):
            return torch.gather(tables_exp, 2, idx.unsqueeze(-1)).squeeze(-1)

        v00, v10 = _gather(idx_00), _gather(idx_10)
        v01, v11 = _gather(idx_01), _gather(idx_11)

        w00 = (1 - cx_w) * (1 - cy_w)
        w10 = cx_w * (1 - cy_w)
        w01 = (1 - cx_w) * cy_w
        w11 = cx_w * cy_w

        return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11


# --------------------------------------------------
# 方案 2：多分辨率 2D LUT（NGP 风格）
# 1 个全局坐标，查 3 张不同分辨率表（8x8, 16x16, 32x32）后相加
# --------------------------------------------------
class LUT2_MultiRes_2D(nn.Module):
    def __init__(self, target_dim=20):
        super().__init__()
        self.sizes = [8, 16, 32]
        self.target_dim = target_dim
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(s * s, target_dim)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.1)
        self.query_net = QueryNet(2)

    def forward(self, x):
        B = x.size(0)
        pos = torch.sigmoid(self.query_net(x))  # (B, 2)
        out = 0
        for i, G in enumerate(self.sizes):
            cx, cy = pos[:, 0] * (G - 1), pos[:, 1] * (G - 1)
            fx, fy = torch.floor(cx).long(), torch.floor(cy).long()
            cx_w, cy_w = cx - fx.float(), cy - fy.float()
            fx = torch.clamp(fx, 0, G - 2)
            fy = torch.clamp(fy, 0, G - 2)

            idx_00 = fx * G + fy
            idx_10 = (fx + 1) * G + fy
            idx_01 = fx * G + (fy + 1)
            idx_11 = (fx + 1) * G + (fy + 1)

            t = self.tables[i]  # (G*G, target_dim)
            t_exp = t.unsqueeze(0).expand(B, -1, -1)  # (B, G*G, target_dim)

            def _gather(idx):
                idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.target_dim)
                return torch.gather(t_exp, 1, idx_exp).squeeze(1)

            v00, v10 = _gather(idx_00), _gather(idx_10)
            v01, v11 = _gather(idx_01), _gather(idx_11)

            w00 = ((1 - cx_w) * (1 - cy_w)).unsqueeze(-1)
            w10 = (cx_w * (1 - cy_w)).unsqueeze(-1)
            w01 = ((1 - cx_w) * cy_w).unsqueeze(-1)
            w11 = (cx_w * cy_w).unsqueeze(-1)

            out = out + (v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11)
        return out


# --------------------------------------------------
# 方案 3：连续多尺度 1D 级联
# 3 层 1D 表（256, 64, 16），query_net 输出 60 个坐标
# --------------------------------------------------
class LUT3_MultiScale_1D(nn.Module):
    def __init__(self, target_dim=20):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = [256, 64, 16]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=0.0, std=0.05)
        self.query_net = QueryNet(target_dim * 3)

    def _interp_1d(self, pos, table, size):
        """pos: (B, target_dim), table: (target_dim, size)"""
        B = pos.size(0)
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c

        table_exp = table.unsqueeze(0).expand(B, -1, -1)  # (B, target_dim, size)
        vf = torch.gather(table_exp, 2, idx_f.unsqueeze(-1)).squeeze(-1)
        vc = torch.gather(table_exp, 2, idx_c.unsqueeze(-1)).squeeze(-1)
        return vf * w_f + vc * w_c

    def forward(self, x):
        B = x.size(0)
        coords = torch.sigmoid(self.query_net(x)).view(B, 3, self.target_dim)
        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords[:, i, :], self.tables[i], size)
        return out


# --------------------------------------------------
# 方案 4：MoE 连续 1D
# 4 个专家各 256 长度，query_net 输出 20 个坐标 + 4 个专家权重
# --------------------------------------------------
class LUT4_MoE_1D(nn.Module):
    def __init__(self, target_dim=20, table_size=256, num_experts=4):
        super().__init__()
        self.target_dim = target_dim
        self.size = table_size
        self.num_experts = num_experts
        # (4个专家, 20维, 256长)
        self.tables = nn.Parameter(torch.randn(num_experts, target_dim, table_size))
        nn.init.normal_(self.tables, mean=0.0, std=0.1)
        self.query_net = QueryNet(target_dim + num_experts)

    def forward(self, x):
        B = x.size(0)
        logits = self.query_net(x)
        pos = torch.sigmoid(logits[:, :self.target_dim])               # (B, 20)
        expert_weights = F.softmax(logits[:, self.target_dim:], dim=-1)  # (B, 4)

        pos_scaled = pos * (self.size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, self.size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.float()
        w_f = 1.0 - w_c

        out = 0
        for i in range(self.num_experts):
            table = self.tables[i].unsqueeze(0).expand(B, -1, -1)  # (B, 20, 256)
            vf = torch.gather(table, 2, idx_f.unsqueeze(-1)).squeeze(-1)
            vc = torch.gather(table, 2, idx_c.unsqueeze(-1)).squeeze(-1)
            interp = vf * w_f + vc * w_c
            out = out + interp * expert_weights[:, i].unsqueeze(-1)
        return out


# ==========================================
# 3. 数据集定义
# ==========================================
class DynamicFloatDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = torch.tensor(item['image'], dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        label = torch.tensor(item['label'], dtype=torch.long)
        dyn_weights = torch.tensor(item['target_20_floats'], dtype=torch.float32)
        return img, label, dyn_weights


# ==========================================
# 4. 训练与验证函数
# ==========================================
def train_lut(lut_model, train_loader, val_loader, device, epochs=100):
    optimizer = torch.optim.AdamW(lut_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        lut_model.train()
        total_loss = 0
        for imgs, _, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = lut_model(imgs)
            loss = mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # 验证 MSE
        lut_model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, _, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = lut_model(imgs)
                val_loss += mse_loss(preds, targets).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch + 1:3d}/{epochs}] Train MSE: {total_loss / len(train_loader):.4f}, Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = lut_model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  >> Early stopping at epoch {epoch + 1}, best val MSE: {best_val_loss:.4f}")
            break

    if best_state is not None:
        lut_model.load_state_dict(best_state)
    return lut_model


def evaluate_on_testset(lut_model, base_model, test_loader, device):
    correct = 0
    total = 0
    lut_model.eval()
    base_model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            dyn_residual = lut_model(imgs)
            logits = base_model(imgs, dynamic_beta_weights=dyn_residual)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ==========================================
# 5. 主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载阶段一数据集
    try:
        data_list = torch.load('phase1_mult_dataset.pt', map_location='cpu', weights_only=False)
        full_dataset = DynamicFloatDataset(data_list)
        print(f"成功加载数据集，共 {len(full_dataset)} 个样本")
    except FileNotFoundError:
        print("错误: 找不到 'phase1_mult_dataset.pt'，请先运行 mnist_traindata_mult.py")
        return

    # 加载基座模型
    try:
        base_model = SmallBaseModel().to(device)
        base_model.load_state_dict(torch.load('model_B_mult_base.pth', map_location=device, weights_only=False))
        base_model.eval()
        print("成功加载基座模型 'model_B_mult_base.pth'")
    except FileNotFoundError:
        print("错误: 找不到 'model_B_mult_base.pth'")
        return

    # 80/20 划分：80% 训练 LUT，20% 验证 LUT（早停用）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 官方测试集（最终泛化评估）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    # Baseline（静态基座）
    print("\n" + "=" * 60)
    print("计算 Baseline 静态基座准确率...")
    base_correct = 0
    base_total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = base_model(imgs)
            base_correct += (logits.argmax(dim=1) == labels).sum().item()
            base_total += labels.size(0)
    base_acc = 100.0 * base_correct / base_total
    print(f"Baseline 静态基座: {base_acc:.2f}%")

    # 四种 LUT 方案
    schemes = {
        "LUT-1: 20头独立 2D 贴图": LUT1_20Head_2D,
        "LUT-2: 多分辨率 2D (NGP风格)": LUT2_MultiRes_2D,
        "LUT-3: 连续多尺度 1D 级联": LUT3_MultiScale_1D,
        "LUT-4: MoE 连续 1D 专家混合": LUT4_MoE_1D,
    }

    results = {}
    for name, BrainClass in schemes.items():
        print("\n" + "-" * 60)
        print(f"🚀 训练: {name}")
        brain = BrainClass().to(device)
        brain = train_lut(brain, train_loader, val_loader, device, epochs=100)

        # 在阶段一验证集上算分类准确率（公平对比同事版本）
        correct_val = 0
        total_val = 0
        brain.eval()
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                dyn_residual = brain(imgs)
                logits = base_model(imgs, dynamic_beta_weights=dyn_residual)
                correct_val += (logits.argmax(dim=1) == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100.0 * correct_val / total_val
        
        print(f"  >> 在官方测试集上验证...")
        test_acc = evaluate_on_testset(brain, base_model, testloader, device)
        results[name] = test_acc
        print(f"✅ {name} 阶段一验证集: {val_acc:.2f}% | 官方测试集: {test_acc:.2f}%")

    # 最终汇总
    print("\n" + "=" * 60)
    print("【LUT 查表方案终极战报】")
    print("=" * 60)
    print(f"0. Baseline 静态基座     : {base_acc:.2f}%")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🥇" if acc == max(results.values()) else "   "
        print(f"{marker} {name.ljust(35)}: {acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
