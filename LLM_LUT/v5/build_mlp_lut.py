import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. 定义网络结构
# ==========================================

# 目标 FFN (2048 -> 4096 -> 2048)
class TargetFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048)
        )

    def forward(self, x):
        return self.net(x)


# 降维映射 MLP (2048 -> 1)
class MappingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 可以是一个简单的线性映射，也可以带隐藏层
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)


# 局部插值拟合模型
# 输入特征维度: 左右坐标(2) + 中心坐标(1) + 左右GT2048(2048*2) = 4099
# 输出维度: 2048 (拟合中心的GT)
class InterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 2048 + 2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048)
        )

    def forward(self, c_L, c_i, c_R, Y_L, Y_R):
        # 将所有输入拼接在一起
        x = torch.cat([c_L, c_i, c_R, Y_L, Y_R], dim=-1)
        return self.net(x)


# ==========================================
# 2. 实验初始化与数据准备
# ==========================================
print("初始化网络与数据...")
torch.manual_seed(42)  # 固定随机种子以保证结果可复现

target_ffn = TargetFFN()
mapping_mlp = MappingMLP()
target_ffn.eval()  # 目标网络不需要训练
mapping_mlp.eval()  # 映射网络在这里只用来提取相对位置，也不需要训练

# 生成1000条随机数据
num_samples = 1000
X = torch.randn(num_samples, 2048)

# 获取 GT 2048 (目标输出)
with torch.no_grad():
    Y_gt = target_ffn(X)

# ==========================================
# 3. 特殊手段：映射到1D并强制均匀分布
# ==========================================
print("进行1D坐标映射与均匀分布处理...")
with torch.no_grad():
    # 1. 用映射MLP打分
    scores = mapping_mlp(X).squeeze()  # shape: (1000,)

    # 2. 获取排序索引
    sort_indices = torch.argsort(scores)

    # 3. 对 X 和 Y_gt 按照1D空间的顺序进行排序
    X_sorted = X[sort_indices]
    Y_gt_sorted = Y_gt[sort_indices]

    # 4. 强制赋予均匀分布的坐标 (0.0 到 1.0 的等差数列)
    # 这一步完美保证了这1000条数据在1D坐标上是完全均匀的
    C_sorted = torch.linspace(0.0, 1.0, num_samples).unsqueeze(1)  # shape: (1000, 1)

# ==========================================
# 4. 构建插值模型的训练数据集
# ==========================================
# 对于排好序的数据，除头尾（0和999）外，中间的998条数据都可以作为训练样本
# i 的左邻居是 i-1，右邻居是 i+1
print("构建插值模型的训练集...")

c_L_list, c_i_list, c_R_list = [], [], []
Y_L_list, Y_R_list = [], []
Y_target_list = []

for i in range(1, num_samples - 1):
    c_L_list.append(C_sorted[i - 1])
    c_i_list.append(C_sorted[i])
    c_R_list.append(C_sorted[i + 1])

    Y_L_list.append(Y_gt_sorted[i - 1])
    Y_R_list.append(Y_gt_sorted[i + 1])

    Y_target_list.append(Y_gt_sorted[i])

# 转为 Tensor
c_L_tensor = torch.stack(c_L_list)
c_i_tensor = torch.stack(c_i_list)
c_R_tensor = torch.stack(c_R_list)
Y_L_tensor = torch.stack(Y_L_list)
Y_R_tensor = torch.stack(Y_R_list)
Y_target_tensor = torch.stack(Y_target_list)

print(f"训练集大小: {Y_target_tensor.shape[0]} 条数据")

# ==========================================
# 5. 训练局部插值模型
# ==========================================
print("开始训练插值模型...")
interp_model = InterpolationModel()
optimizer = optim.Adam(interp_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 500
# 这里数据量小（998条），直接使用 Full-Batch 训练
for epoch in range(epochs):
    interp_model.train()
    optimizer.zero_grad()

    # 前向传播
    Y_pred = interp_model(c_L_tensor, c_i_tensor, c_R_tensor, Y_L_tensor, Y_R_tensor)

    # 计算损失 (拟合输入的真实 GT 2048)
    loss = criterion(Y_pred, Y_target_tensor)

    # 反向传播
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], MSE Loss: {loss.item():.6f}")

# ==========================================
# 6. 对比 Baseline：简单的线性插值
# ==========================================
# 为了证明模型确实学到了东西，我们可以计算一下如果仅仅使用数学上的线性插值的误差
# 因为坐标是严格均匀分布的，c_i 刚好在 c_L 和 c_R 的正中间
# 所以线性插值直接就是 (Y_L + Y_R) / 2
baseline_Y_pred = (Y_L_tensor + Y_R_tensor) / 2.0
baseline_loss = criterion(baseline_Y_pred, Y_target_tensor)

print("-" * 30)
print(f"训练结束最终模型 MSE Loss: {loss.item():.6f}")
print(f"纯数学线性插值 (Baseline) MSE Loss: {baseline_loss.item():.6f}")