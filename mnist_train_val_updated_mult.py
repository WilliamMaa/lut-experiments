import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


# ==========================================
# 0. 基座模型 (严格保持原样)
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
# 1. 强化版视觉脑 (保持极简 ResNet 结构)
# ==========================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


def build_resnet_query_net(out_features):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        BasicBlock(32, 64, stride=2),
        BasicBlock(64, 128, stride=2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, out_features)
    )


# ==========================================
# 2. 四大极限 O(1) 连续查表方案 (必破 95%)
# ==========================================

# 【巅峰方案 1】20头独立 2D 贴图矩阵 (Ultimate 2D Decoupling)
# 暴力且优雅！为 20 个参数各自配备一张 32x32 的独立 2D 地图！
# 相当于生成 20 组不同的 (x,y) 坐标，分别在 20 张图上进行 O(1) 双线性查表。
class Scheme1_20Head_2D_LUT(nn.Module):
    def __init__(self, target_dim=20, grid_size=32):
        super().__init__()
        self.G = grid_size
        self.target_dim = target_dim
        # 20张独立的二维表 [20, 32*32]
        self.tables = nn.Parameter(torch.randn(target_dim, self.G * self.G))
        nn.init.normal_(self.tables, mean=1.0, std=0.1)
        self.query_net = build_resnet_query_net(target_dim * 2)  # 20对(x,y) = 40个输出

    def forward(self, x):
        # 极速计算坐标
        coords = torch.sigmoid(self.query_net(x)).view(-1, self.target_dim, 2) * (self.G - 1)
        cx, cy = coords[:, :, 0], coords[:, :, 1]

        fx, fy = torch.floor(cx).long(), torch.floor(cy).long()
        cx_w, cy_w = cx - fx.detach().float(), cy - fy.detach().float()
        fx, fy = torch.clamp(fx, 0, self.G - 2), torch.clamp(fy, 0, self.G - 2)

        # O(1) 绝对寻址
        idx_00 = fx * self.G + fy
        idx_10 = (fx + 1) * self.G + fy
        idx_01 = fx * self.G + (fy + 1)
        idx_11 = (fx + 1) * self.G + (fy + 1)

        dim_idx = torch.arange(self.target_dim, device=x.device).unsqueeze(0)

        # 瞬间抽取出 20 个参数的 4 个角点数值
        v00 = self.tables[dim_idx, idx_00]
        v10 = self.tables[dim_idx, idx_10]
        v01 = self.tables[dim_idx, idx_01]
        v11 = self.tables[dim_idx, idx_11]

        # 双线性插值叠加
        w00 = ((1 - cx_w) * (1 - cy_w))
        w10 = (cx_w * (1 - cy_w))
        w01 = ((1 - cx_w) * cy_w)
        w11 = (cx_w * cy_w)

        return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11


# 【巅峰方案 2】全局单点 NGP 多分辨率 2D 表 (Global Multi-Res 2D)
# 参考 NVIDIA Instant-NGP：提取器只输出 1 个全局的 (X,Y) 坐标，
# 但用这一个坐标，去查 3 张不同精度的二维表 (8x8, 16x16, 32x32)，然后全部加起来。
class Scheme2_NGP_Style_2D_LUT(nn.Module):
    def __init__(self, target_dim=20):
        super().__init__()
        self.sizes = [8, 16, 32]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(s * s, target_dim)) for s in self.sizes
        ])
        for t in self.tables:
            nn.init.normal_(t, mean=1.0 / len(self.sizes), std=0.1)

        self.query_net = build_resnet_query_net(2)  # 仅输出 1对 全局(x,y) 坐标

    def forward(self, x):
        pos = torch.sigmoid(self.query_net(x))
        out = 0
        for i, G in enumerate(self.sizes):
            cx, cy = pos[:, 0] * (G - 1), pos[:, 1] * (G - 1)
            fx, fy = torch.floor(cx).long(), torch.floor(cy).long()
            cx_w, cy_w = cx - fx.detach().float(), cy - fy.detach().float()
            fx, fy = torch.clamp(fx, 0, G - 2), torch.clamp(fy, 0, G - 2)

            idx_00 = fx * G + fy
            idx_10 = (fx + 1) * G + fy
            idx_01 = fx * G + (fy + 1)
            idx_11 = (fx + 1) * G + (fy + 1)

            t = self.tables[i]
            v00, v10 = t[idx_00], t[idx_10]
            v01, v11 = t[idx_01], t[idx_11]

            w00 = ((1 - cx_w) * (1 - cy_w)).unsqueeze(-1)
            w10 = (cx_w * (1 - cy_w)).unsqueeze(-1)
            w01 = ((1 - cx_w) * cy_w).unsqueeze(-1)
            w11 = (cx_w * cy_w).unsqueeze(-1)

            out = out + (v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11)
        return out


# 【巅峰方案 3】连续多尺度 1D 级联瀑布 (Continuous RVQ-1D)
# 把上一代拉垮的“离散 3 层相加”变成“连续 3 层插值”。
# 彻底解决梯度断层，保留庞大的组合爆炸表达力。
class Scheme3_MultiScale_Continuous_1D(nn.Module):
    def __init__(self, target_dim=20):
        super().__init__()
        self.target_dim = target_dim
        self.sizes = [256, 64, 16]

        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(target_dim, s)) for s in self.sizes
        ])
        self.tables[0].data.normal_(1.0, 0.1)  # 粗表定基调
        self.tables[1].data.normal_(0.0, 0.05)  # 中表微调
        self.tables[2].data.normal_(0.0, 0.01)  # 细表精修

        self.query_net = build_resnet_query_net(target_dim * 3)  # 为每层分别输出 20 个坐标

    def _interp_1d(self, pos, table, size):
        pos_scaled = pos * (size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.detach().float()
        w_f = 1.0 - w_c

        dim_idx = torch.arange(self.target_dim, device=pos.device).unsqueeze(0)
        return table[dim_idx, idx_f] * w_f + table[dim_idx, idx_c] * w_c

    def forward(self, x):
        B = x.size(0)
        coords = torch.sigmoid(self.query_net(x)).view(B, 3, self.target_dim)

        out = 0
        for i, size in enumerate(self.sizes):
            out = out + self._interp_1d(coords[:, i, :], self.tables[i], size)
        return out


# 【巅峰方案 4】专家混合态连续 1D 脑 (MoE Continuous 1D)
# 准备 4 套完全不同的 256 长 1D 表。
# 网络不仅输出坐标，还输出对这 4 套表的“信任权重”，进行动态融合。
class Scheme4_MoE_Continuous_1D(nn.Module):
    def __init__(self, target_dim=20, table_size=256, num_experts=4):
        super().__init__()
        self.target_dim = target_dim
        self.size = table_size
        self.num_experts = num_experts

        # [4个专家, 20维, 256长]
        self.tables = nn.Parameter(torch.randn(num_experts, target_dim, table_size))
        nn.init.normal_(self.tables, mean=1.0, std=0.1)

        # 输出: 20个坐标 + 4个全局专家权重 = 24
        self.query_net = build_resnet_query_net(target_dim + num_experts)

    def forward(self, x):
        logits = self.query_net(x)

        pos = torch.sigmoid(logits[:, :self.target_dim])  # [B, 20]
        expert_weights = F.softmax(logits[:, self.target_dim:], dim=-1)  # [B, 4]

        pos_scaled = pos * (self.size - 1)
        idx_f = torch.clamp(torch.floor(pos_scaled).long(), 0, self.size - 2)
        idx_c = idx_f + 1
        w_c = pos_scaled - idx_f.detach().float()
        w_f = 1.0 - w_c

        dim_idx = torch.arange(self.target_dim, device=x.device).unsqueeze(0)

        out = 0
        for i in range(self.num_experts):
            vec_f = self.tables[i, dim_idx, idx_f]
            vec_c = self.tables[i, dim_idx, idx_c]
            interp_val = vec_f * w_f + vec_c * w_c
            # 乘以对应专家的权重
            out = out + interp_val * expert_weights[:, i].unsqueeze(-1)

        return out


# ==========================================
# 3. 严格解耦的数据与训练评估流程
# ==========================================
class DynamicFloatDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return (torch.tensor(item['image'], dtype=torch.float32).unsqueeze(0),
                torch.tensor(item['label'], dtype=torch.long),
                torch.tensor(item['target_20_floats'], dtype=torch.float32))


def train_and_evaluate(scheme_name, brain_class, full_dataset, base_path, device):
    print(f"\n{'-' * 65}")
    print(f"🚀 正在训练极限外挂脑: {scheme_name}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    brain = brain_class().to(device)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    mse_loss_fn = nn.MSELoss()

    epochs = 100
    for epoch in range(epochs):
        brain.train()
        total_loss = 0
        for imgs, _, target_weights in train_loader:
            imgs, target_weights = imgs.to(device), target_weights.to(device)
            optimizer.zero_grad()
            loss = mse_loss_fn(brain(imgs), target_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch + 1:3d}/{epochs}] MSE 拟合 Loss: {total_loss / len(train_loader):.4f}")

    print("  >> 连接基座模型，准备见证奇迹...")
    base_model = SmallBaseModel().to(device)
    base_model.load_state_dict(torch.load(base_path, map_location=device, weights_only=False))
    base_model.eval()
    brain.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            dyn_weights = brain(imgs)
            logits = base_model(imgs, dynamic_beta_weights=dyn_weights)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total
    print(f"✅ {scheme_name} 训练集子集准确率: {val_acc:.2f}%")
    
    # 官方测试集验证（公平对比）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            dyn_weights = brain(imgs)
            logits = base_model(imgs, dynamic_beta_weights=dyn_weights)
            correct_test += (logits.argmax(dim=1) == labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100.0 * correct_test / total_test
    print(f"✅ {scheme_name} 官方测试集准确率: {test_acc:.2f}%")
    return test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data_list = torch.load('phase1_mult_dataset.pt', map_location='cpu', weights_only=False)
        full_dataset = DynamicFloatDataset(data_list)
        base_path = 'model_B_mult_base.pth'
    except:
        print("数据加载失败，请检查文件。")
        return

    schemes = {
        "巅峰 1: 20头独立 2D 贴图矩阵 (Ultimate 2D Decoupling)": Scheme1_20Head_2D_LUT,
        "巅峰 2: 全局单点 NGP 多分辨率 2D 表 (Global Multi-Res)": Scheme2_NGP_Style_2D_LUT,
        "巅峰 3: 连续多尺度 1D 级联瀑布 (Continuous RVQ-1D)": Scheme3_MultiScale_Continuous_1D,
        "巅峰 4: 专家混合态连续 1D 脑 (MoE Continuous 1D)": Scheme4_MoE_Continuous_1D
    }

    results = {}
    for name, brain_class in schemes.items():
        results[name] = train_and_evaluate(name, brain_class, full_dataset, base_path, device)

    print("\n" + "💎" * 15)
    print("【破壁之战：冲击 95% 的极致 O(1) 终极战报】")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name.ljust(50)}: {acc:.2f}%")
    print("💎" * 15)


if __name__ == '__main__':
    main()