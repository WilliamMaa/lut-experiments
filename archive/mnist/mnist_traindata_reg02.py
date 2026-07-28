"""
流程一：基座训练 + 单样本残差提取
设计理念：细胞级 MOE —— 仅动态改变 20 个参数，冻结其余所有层
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np


# ==========================================
# 1. 模型定义：极度受限的基座 + 可加性动态残差
# ==========================================
class LayerBeta(nn.Module):
    """
    细胞级动态参数层：仅 20 个浮点数。
    加法残差注入：Feature = X * W_base + alpha * ΔW
    相比乘法残差更稳定，预测误差不会指数级放大。
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(20))
        self.alpha = alpha

    def forward(self, x, dynamic_residual=None):
        if dynamic_residual is not None:
            return x * self.weights + self.alpha * dynamic_residual
        return x * self.weights


class SmallBaseModel(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        # 信息瓶颈：784 → 3 → 20
        # 3 维空间强制大量信息丢失，基座准确率存在物理上限
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 3),
            nn.ReLU(),
            nn.Linear(3, 20)
        )
        self.layer_beta = LayerBeta(alpha=alpha)
        self.classifier = nn.Linear(20, 10)

    def forward(self, x, dynamic_beta_weights=None):
        features = self.backbone(x)
        if dynamic_beta_weights is not None:
            features = self.layer_beta(features, dynamic_residual=dynamic_beta_weights)
        else:
            features = self.layer_beta(features)
        return self.classifier(features)


# ==========================================
# 2. 训练基座模型 B
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

SUBSET_SIZE = 60000
trainset = Subset(full_trainset, range(SUBSET_SIZE))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

model = SmallBaseModel(alpha=1.0).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("=" * 60)
print("步骤 1: 全量训练基座模型 B（解锁所有参数）")
print("=" * 60)

for epoch in range(30):
    model.train()
    total_loss = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1:2d}, Loss: {total_loss / len(trainloader):.4f}")

# 保存基座
torch.save(model.state_dict(), 'model_B_reg02_base.pth')
global_beta_weights = model.layer_beta.weights.detach().clone()
print("\n基座模型已保存至 model_B_reg01_base.pth")

# ==========================================
# 3. 单样本残差提取（阶段一核心）
# ==========================================
print("\n" + "=" * 60)
print("步骤 2: 细胞级过拟合 —— 逐样本提取最优残差 ΔW")
print("=" * 60)

# 冻结 backbone、classifier 和基座 layer_beta 权重
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = False
model.layer_beta.weights.requires_grad = False  # 基座权重不动

custom_dataset = []
single_loader = DataLoader(trainset, batch_size=1, shuffle=False)

# 正则化系数：强烈约束残差大小，迫使参数空间平滑
LAMBDA_REG = 0.2

for idx, (img, target) in enumerate(tqdm(single_loader, desc="单样本过拟合")):
    img, target = img.to(device), target.to(device)

    # 恢复基座权重
    model.layer_beta.weights.data = global_beta_weights.clone()

    # 每个样本从零初始化独立残差（从基座出发）
    residual = torch.zeros(20, requires_grad=True, device=device)
    optimizer_single = optim.SGD([residual], lr=0.1)

    # 过拟合直到收敛
    prev_loss = float('inf')
    for step in range(50):
        optimizer_single.zero_grad()
        output = model(img, dynamic_beta_weights=residual)
        ce_loss = criterion(output, target)

        # L2 正则化：限制残差偏离 0 的程度
        reg_loss = LAMBDA_REG * torch.sum(residual ** 2)
        total_loss = ce_loss + reg_loss
        total_loss.backward()
        optimizer_single.step()

        if abs(prev_loss - ce_loss.item()) < 1e-4 or ce_loss.item() < 0.01:
            break
        prev_loss = ce_loss.item()

    # 提取 backbone 特征（供阶段二验证注入使用）
    with torch.no_grad():
        backbone_feat = model.backbone(img).cpu().numpy().flatten()

    custom_dataset.append({
        'image': img.squeeze().cpu().numpy(),
        'label': target.item(),
        'target_20_floats': residual.detach().cpu().numpy(),
        'backbone_feature': backbone_feat
    })

# 保存
torch.save(custom_dataset, 'phase1_reg02_dataset.pt')
print("\n阶段一完成：数据集已保存至 phase1_reg01_dataset.pt")
