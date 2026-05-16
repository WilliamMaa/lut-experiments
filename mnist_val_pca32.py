"""
流程二：四种 O1 动态查询方案 + 验证集泛化对比
设计理念：
  - O1 查询：PCA 降维 + 查表/回归均为常数/线性时间，无大网络前传
  - 细胞级 MOE：仅改变 20 个参数，主网络其余部分完全冻结
  - 加法残差：即使查询有误，影响有界，基座保底
"""

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==========================================
# 模型定义（必须与流程一严格一致）
# ==========================================
class LayerBeta(nn.Module):
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
# 1. 加载基座与阶段一数据集
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_B = SmallBaseModel(alpha=1.0).to(device)
model_B.load_state_dict(torch.load('model_B_base.pth', map_location=device))
model_B.eval()

dataset_phase1 = torch.load('phase1_20_floats_dataset.pt', map_location='cpu', weights_only=False)

# ==========================================
# 2. 构建 O1 查询空间：PCA 降维 + 标准化
# ==========================================
print("\n--- 构建 O1 查询空间（PCA 降维 + StandardScaler） ---")

# 展平图像用于 PCA
X_images = np.array([data['image'].flatten() for data in dataset_phase1])  # (N, 784)
Y_train_20floats = np.array([data['target_20_floats'] for data in dataset_phase1])

pca = PCA(n_components=32)
X_pca = pca.fit_transform(X_images)

scaler = StandardScaler()
X_query = scaler.fit_transform(X_pca)

print(f"查询空间构建完成：784 → PCA({pca.n_components_}) → StandardScaler")
print(f"保留方差比例: {sum(pca.explained_variance_ratio_):.4f}")

# ==========================================
# 方案 1：Deep k-NN（距离加权最近邻回归）
# ==========================================
print("\n训练方案 1: Deep k-NN ...", end="", flush=True)
t0 = time.time()

knn_regressor = KNeighborsRegressor(n_neighbors=7, weights='distance', algorithm='auto')
knn_regressor.fit(X_query, Y_train_20floats)
print(f" 完成 ({time.time()-t0:.2f}s)")

# ==========================================
# 方案 2：RBF Kernel Ridge（带自动 gamma）
# ==========================================
print("训练方案 2: RBF Kernel Ridge ...", end="", flush=True)
t0 = time.time()

MAX_SAMPLES_RBF = 5000
n_samples = len(X_query)
if n_samples > MAX_SAMPLES_RBF:
    indices = np.random.choice(n_samples, MAX_SAMPLES_RBF, replace=False)
    X_rbf = X_query[indices]
    Y_rbf = Y_train_20floats[indices]
else:
    X_rbf = X_query
    Y_rbf = Y_train_20floats

# 手动计算 gamma='scale'：1 / (n_features * X.var())
gamma_scale = 1.0 / (X_rbf.shape[1] * X_rbf.var()) if X_rbf.var() > 0 else 0.1
rbf_regressor = KernelRidge(kernel='rbf', gamma=gamma_scale, alpha=0.1)
rbf_regressor.fit(X_rbf, Y_rbf)
print(f" 完成 ({time.time()-t0:.2f}s)")

# ==========================================
# 方案 3：VQ Codebook（聚类 + 软查询）
# ==========================================
print("训练方案 3: VQ Codebook ...", end="", flush=True)
t0 = time.time()

NUM_CODES = 128
kmeans = KMeans(n_clusters=NUM_CODES, random_state=42, n_init='auto')
train_code_ids = kmeans.fit_predict(Y_train_20floats)
codebook = kmeans.cluster_centers_  # (128, 20)

# 用 k=3 的加权最近邻从查询空间预测码本
knn_code = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_code.fit(X_query, train_code_ids)
print(f" 完成 ({time.time()-t0:.2f}s)")

# ==========================================
# 方案 4：KV Memory Network（MLP 注意力）
# ==========================================
print("\n训练方案 4: KV Memory Network (PyTorch)")


class KVMemory(nn.Module):
    """
    改进版 KV Memory：用小型 MLP 做查询投影，参数量极小但表达能力更强。
    查询过程仍是 O1：一次前向传播即得 20 维残差。
    """
    def __init__(self, feat_dim=32, mem_size=256, out_dim=20):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mem_size)
        )
        self.values = nn.Parameter(torch.randn(mem_size, out_dim) * 0.1)

    def forward(self, query):
        attn = self.query_proj(query)          # (batch, mem_size)
        attn_weights = torch.softmax(attn, dim=-1)
        return torch.matmul(attn_weights, self.values)  # (batch, out_dim)


kv_memory = KVMemory().to(device)
kv_opt = optim.Adam(kv_memory.parameters(), lr=0.001)
kv_criterion = nn.MSELoss()

X_tensor = torch.tensor(X_query, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_train_20floats, dtype=torch.float32).to(device)

epochs = 300
pbar = tqdm(range(epochs), desc="KV Memory 训练")
for epoch in pbar:
    kv_opt.zero_grad()
    preds = kv_memory(X_tensor)
    loss = kv_criterion(preds, Y_tensor)
    loss.backward()
    kv_opt.step()
    if epoch % 20 == 0:
        pbar.set_postfix({"MSE": f"{loss.item():.4f}"})

# ==========================================
# 验证集统一评测
# ==========================================
print("\n--- 验证集泛化对比（前 1000 个样本） ---")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

correct_base = 0
correct_knn = 0
correct_rbf = 0
correct_vq = 0
correct_kv = 0
total = 1000

pbar_eval = tqdm(enumerate(testloader), total=total, desc="验证中")
for i, (img, label) in pbar_eval:
    if i >= total:
        break
    img, label = img.to(device), label.item()

    with torch.no_grad():
        # O1 查询键生成：图像 → PCA → 标准化
        img_np = img.squeeze().cpu().numpy()
        img_flat = img_np.flatten().reshape(1, -1)
        pca_feat = pca.transform(img_flat)           # (1, 16)
        query_feat = scaler.transform(pca_feat)        # (1, 16)
        query_tensor = torch.tensor(query_feat, dtype=torch.float32).to(device)

        # --- Baseline ---
        out_base = model_B(img)
        if out_base.argmax().item() == label:
            correct_base += 1

        # --- 方案 1: Deep k-NN ---
        w_knn = torch.tensor(knn_regressor.predict(query_feat), dtype=torch.float32).to(device)
        out_knn = model_B(img, dynamic_beta_weights=w_knn)
        if out_knn.argmax().item() == label:
            correct_knn += 1

        # --- 方案 2: RBF ---
        w_rbf = torch.tensor(rbf_regressor.predict(query_feat), dtype=torch.float32).to(device)
        out_rbf = model_B(img, dynamic_beta_weights=w_rbf)
        if out_rbf.argmax().item() == label:
            correct_rbf += 1

        # --- 方案 3: VQ ---
        code_id = knn_code.predict(query_feat)[0]
        w_vq = torch.tensor(codebook[code_id], dtype=torch.float32).to(device)
        out_vq = model_B(img, dynamic_beta_weights=w_vq)
        if out_vq.argmax().item() == label:
            correct_vq += 1

        # --- 方案 4: KV Memory ---
        w_kv = kv_memory(query_tensor)
        out_kv = model_B(img, dynamic_beta_weights=w_kv)
        if out_kv.argmax().item() == label:
            correct_kv += 1

# 打印结果
print("\n" + "=" * 60)
print("【验证集泛化对比结果】")
print("=" * 60)
print(f"0. Baseline 静态基座     : {correct_base / total * 100:.2f}%")
print(f"1. Deep k-NN  动态查表   : {correct_knn / total * 100:.2f}%")
print(f"2. RBF Kernel 动态生成   : {correct_rbf / total * 100:.2f}%")
print(f"3. VQ Codebook 离散查表  : {correct_vq / total * 100:.2f}%")
print(f"4. KV Memory  注意力查表 : {correct_kv / total * 100:.2f}%")
print("=" * 60)
