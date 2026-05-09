import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


# ==========================================
# 1. 保持与训练时完全一致的模型结构
# ==========================================
class LayerBeta(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(20))
        self.alpha = alpha  # 残差缩放系数

    def forward(self, x, dynamic_residual=None):
        # 残差动态注入：基座参数 + alpha * 动态残差
        if dynamic_residual is not None:
            return x * self.weights + self.alpha * dynamic_residual
        return x * self.weights


class SmallBaseModel(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        # 你的 70% 准确率瓶颈结构
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

        # 如果传入了动态参数，则以残差方式注入
        if dynamic_beta_weights is not None:
            features = self.layer_beta(features, dynamic_residual=dynamic_beta_weights)
        else:
            features = self.layer_beta(features)

        return self.classifier(features)


# ==========================================
# 2. 定义 Dataset 读取你保存的自定义数据
# ==========================================
class DynamicFloatDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 恢复图像的通道维度: (28, 28) -> (1, 28, 28)
        img = torch.tensor(item['image'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
        dyn_weights = torch.tensor(item['target_20_floats'], dtype=torch.float32)
        return img, label, dyn_weights


# ==========================================
# 3. 评估主程序
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = SmallBaseModel().to(device)
    try:
        model.load_state_dict(torch.load('model_B_base.pth', map_location=device))
        print("成功加载基座模型 'model_B_base.pth'")
    except FileNotFoundError:
        print("错误: 找不到 'model_B_base.pth'，请确保路径正确。")
        return
    model.eval()  # 切换到评估模式

    # 加载数据
    try:
        data_list = torch.load('phase1_20_floats_dataset.pt', map_location='cpu', weights_only=False)
        print(f"成功加载动态参数数据集 'phase1_20_floats_dataset.pt'，共 {len(data_list)} 个样本")
    except FileNotFoundError:
        print("错误: 找不到 'phase1_20_floats_dataset.pt'。")
        return

    dataset = DynamicFloatDataset(data_list)
    # 使用较大的 batch_size 加快评估速度
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # 记录变量
    base_total_loss = 0.0
    base_correct = 0

    dyn_total_loss = 0.0
    dyn_correct = 0

    total_samples = 0

    print("\n开始对比评估...")
    with torch.no_grad():
        for imgs, labels, dyn_weights in tqdm(dataloader, desc="Evaluating"):
            imgs, labels, dyn_weights = imgs.to(device), labels.to(device), dyn_weights.to(device)
            batch_size = imgs.size(0)
            total_samples += batch_size

            # --- 测试 1：纯基座模型 (使用全局 Layer β) ---
            base_outputs = model(imgs, dynamic_beta_weights=None)
            base_loss = criterion(base_outputs, labels)
            base_total_loss += base_loss.item() * batch_size
            _, base_preds = base_outputs.max(1)
            base_correct += base_preds.eq(labels).sum().item()

            # --- 测试 2：动态注入 (使用过拟合的 20个浮点数残差) ---
            dyn_outputs = model(imgs, dynamic_beta_weights=dyn_weights)
            dyn_loss = criterion(dyn_outputs, labels)
            dyn_total_loss += dyn_loss.item() * batch_size
            _, dyn_preds = dyn_outputs.max(1)
            dyn_correct += dyn_preds.eq(labels).sum().item()

    # 计算平均值
    base_avg_loss = base_total_loss / total_samples
    base_accuracy = 100.0 * base_correct / total_samples

    dyn_avg_loss = dyn_total_loss / total_samples
    dyn_accuracy = 100.0 * dyn_correct / total_samples

    # 打印最终对比结果
    print("\n" + "=" * 50)
    print("【评估结果对比】")
    print(f"测试样本总数: {total_samples}")
    print("-" * 50)
    print("1. 基座模型 (使用固定全局参数):")
    print(f"   - 平均 Loss : {base_avg_loss:.4f}")
    print(f"   - 准 确 率  : {base_accuracy:.2f}%")
    print("-" * 50)
    print("2. 动态模型 (注入单样本过拟合残差):")
    print(f"   - 平均 Loss : {dyn_avg_loss:.4f}")
    print(f"   - 准 确 率  : {dyn_accuracy:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()
