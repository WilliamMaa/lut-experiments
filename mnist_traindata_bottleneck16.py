"""
阶段一（Bottleneck 16 维版）：基座训练 + 单样本绝对权重提取
核心改进：backbone 改为 784→16→20，保留更丰富的中间层信息
为"复用上一层输出"提供足够的信息量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# ==========================================
# 模型定义（bottleneck 16 维，支持层级复用）
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
        # 拆分 backbone：前半段输出 16 维（可复用），后半段 16→20
        self.backbone_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU()
        )
        self.backbone_main = nn.Linear(16, 20)
        self.layer_beta = LayerBeta()
        self.classifier = nn.Linear(20, 10)

    def forward(self, x, dynamic_beta_weights=None):
        feat_16 = self.backbone_feat(x)      # (B, 16)  ← 可复用的中间层
        features = self.backbone_main(feat_16)  # (B, 20)
        if dynamic_beta_weights is not None:
            features = features * dynamic_beta_weights
        else:
            features = self.layer_beta(features)
        return self.classifier(features)


# ==========================================
# 训练基座
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

SUBSET_SIZE = 60000
trainset = Subset(full_trainset, range(SUBSET_SIZE))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

model = SmallBaseModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("--- 步骤 1: 全量训练基座模型 B（bottleneck 16 维）---")
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
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trainloader):.4f}")

torch.save(model.state_dict(), 'model_B_b16_base.pth')
global_beta_weights = model.layer_beta.weights.detach().clone()
print("\n基座模型已保存至 model_B_b16_base.pth")

# ==========================================
# 单样本过拟合（乘法残差，无正则化）
# ==========================================
print("\n--- 步骤 2: 单样本过拟合 layer_beta 权重 ---")
for param in model.backbone_feat.parameters(): param.requires_grad = False
for param in model.backbone_main.parameters(): param.requires_grad = False
for param in model.classifier.parameters(): param.requires_grad = False

custom_dataset = []
single_loader = DataLoader(trainset, batch_size=1, shuffle=False)

for idx, (img, target) in enumerate(tqdm(single_loader, desc="单样本过拟合")):
    img, target = img.to(device), target.to(device)
    model.layer_beta.weights.data = global_beta_weights.clone()
    optimizer_single = optim.SGD([model.layer_beta.weights], lr=0.1)

    prev_loss = float('inf')
    for step in range(50):
        optimizer_single.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        optimizer_single.step()
        if abs(prev_loss - loss.item()) < 1e-4 or loss.item() < 0.01:
            break
        prev_loss = loss.item()

    optimized_weights = model.layer_beta.weights.detach().cpu().numpy()
    custom_dataset.append({
        'image': img.squeeze().cpu().numpy(),
        'label': target.item(),
        'target_20_floats': optimized_weights
    })

torch.save(custom_dataset, 'phase1_b16_dataset.pt')
print("\n阶段一完成：数据集已保存至 phase1_b16_dataset.pt")
