import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ==========================================
# 0. 超参数 (终极修Bug版)
# ==========================================
DIM = 128
NUM_CLASSES = 1024

TOTAL_TRAIN_SIZE = 1000000
TOTAL_TEST_SIZE = 50000
BATCH_SIZE = 4096

# 🚀 20 人超强智囊团集结！
NUM_COMMITTEES = 10
TABLES_PER_COM = 512
BITS = 10
ROWS = 2 ** BITS


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 🧠 物理连线器 3.0：高阶稀疏组合 (独立 Generator 保护)
# ==========================================
class HighOrderRouter:
    def __init__(self, in_features, tables, bits, device='cpu', seed=0):
        # 🚨 [神级细节] 使用独立 Generator，绝对不污染外层的 Bagging 随机性！
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        total_bits = tables * bits
        self.idx = torch.randint(0, in_features, (total_bits, 4), generator=g, device=device)

    def generate_bits(self, x):
        A = x[:, self.idx[:, 0]]
        B = x[:, self.idx[:, 1]]
        C = x[:, self.idx[:, 2]]
        D = x[:, self.idx[:, 3]]
        return (A + B > C + D).to(torch.int32)

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # ---------------------------------------------------------
    # 1. 统一宇宙法则！所有卡必须面对同一个 Teacher Model
    # ---------------------------------------------------------
    seed_everything(42)  # 🚨 修复Bug核心：固定初始全局种子

    ground_truth_model = nn.Sequential(
        nn.Linear(DIM, 1024), nn.GELU(),
        nn.Linear(1024, 1024), nn.GELU(),
        nn.Linear(1024, NUM_CLASSES)
    ).to(local_rank)
    ground_truth_model.eval()

    # ---------------------------------------------------------
    # 2. 撕裂数据集！让每张卡生成不同的数据，吃满 8xA100 的吞吐
    # ---------------------------------------------------------
    seed_everything(42 + local_rank)  # 🚨 Teacher造完后，再让各卡随机性解绑

    train_size_per_gpu = TOTAL_TRAIN_SIZE // world_size
    test_size_per_gpu = TOTAL_TEST_SIZE // world_size

    if local_rank == 0: print("[Rank 0] 正在分布式生成宇宙常量数据集...")

    with torch.no_grad():
        X_train = torch.randn(train_size_per_gpu, DIM, device=local_rank)
        X_train = F.normalize(X_train, p=2, dim=1)
        y_train = torch.zeros(train_size_per_gpu, dtype=torch.int64, device=local_rank)

        chunk_sz = 10000
        for i in range(0, train_size_per_gpu, chunk_sz):
            y_train[i:i + chunk_sz] = torch.argmax(ground_truth_model(X_train[i:i + chunk_sz]), dim=1)

        X_test = torch.randn(test_size_per_gpu, DIM, device=local_rank)
        X_test = F.normalize(X_test, p=2, dim=1)
        y_test = torch.zeros(test_size_per_gpu, dtype=torch.int64, device=local_rank)
        for i in range(0, test_size_per_gpu, chunk_sz):
            y_test[i:i + chunk_sz] = torch.argmax(ground_truth_model(X_test[i:i + chunk_sz]), dim=1)

    del ground_truth_model
    torch.cuda.empty_cache()
    dist.barrier()

    # ---------------------------------------------------------
    # 3. 串行训练：20专家接力
    # ---------------------------------------------------------
    POWERS = (2 ** torch.arange(BITS, device=local_rank)).to(torch.int32)
    OFFSETS = (torch.arange(TABLES_PER_COM, device=local_rank) * ROWS).to(torch.int32)

    ensemble_logits = torch.zeros((X_test.shape[0], NUM_CLASSES), device=local_rank)
    committee_accs = []

    for c in range(NUM_COMMITTEES):
        if local_rank == 0:
            print("\n" + "=" * 60)
            print(f"🚀 训练 [第 {c + 1}/{NUM_COMMITTEES} 号高阶专家] (Bagging保护中)")
            print("=" * 60)

        # 这里不需要再 seed_everything，因为 Router 内部自带独立 Generator
        router = HighOrderRouter(DIM, TABLES_PER_COM, BITS, local_rank, seed=999 + c)

        idx_train = torch.zeros(X_train.shape[0], TABLES_PER_COM, device=local_rank, dtype=torch.int32)
        idx_test = torch.zeros(X_test.shape[0], TABLES_PER_COM, device=local_rank, dtype=torch.int32)

        with torch.no_grad():
            for i in range(0, X_train.shape[0], chunk_sz):
                chunk = X_train[i:i + chunk_sz]
                bits_val = router.generate_bits(chunk).view(-1, TABLES_PER_COM, BITS)
                idx_train[i:i + chunk_sz] = (bits_val * POWERS).sum(dim=-1) + OFFSETS

            for i in range(0, X_test.shape[0], chunk_sz):
                chunk = X_test[i:i + chunk_sz]
                bits_val = router.generate_bits(chunk).view(-1, TABLES_PER_COM, BITS)
                idx_test[i:i + chunk_sz] = (bits_val * POWERS).sum(dim=-1) + OFFSETS

        # 🚨 Bagging：打乱，抽取 85% 数据 (各卡互不干扰)
        bag_size = int(X_train.shape[0] * 0.85)
        shuffle_idx = torch.randperm(X_train.shape[0], device=local_rank)[:bag_size]

        idx_train_bag = idx_train[shuffle_idx]
        y_train_bag = y_train[shuffle_idx]

        lut_model = nn.EmbeddingBag(TABLES_PER_COM * ROWS, NUM_CLASSES, mode='sum').to(local_rank)
        nn.init.constant_(lut_model.weight, 0.0)
        lut_model = DDP(lut_model, device_ids=[local_rank])

        # 设定健康的训练参数
        optimizer = torch.optim.AdamW(lut_model.parameters(), lr=0.05, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 训练 5 轮刚刚好，保持微量欠拟合（泛化最强）
        epochs = 5
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        dataset = TensorDataset(idx_train_bag, y_train_bag)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        disable_tqdm = (local_rank != 0)

        for epoch in range(epochs):
            lut_model.train()
            pbar = tqdm(loader, desc=f"Comm {c + 1} Epoch [{epoch + 1}/{epochs}]", disable=disable_tqdm)
            for idx_b, y_b in pbar:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(lut_model(idx_b), y_b)
                loss.backward()
                optimizer.step()
                if not disable_tqdm:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()

        lut_model.eval()
        with torch.no_grad():
            test_logits_c = torch.zeros((X_test.shape[0], NUM_CLASSES), device=local_rank)
            for i in range(0, X_test.shape[0], BATCH_SIZE):
                chunk = idx_test[i:i + BATCH_SIZE]
                test_logits_c[i:i + BATCH_SIZE] = lut_model.module(chunk)

            local_correct = (test_logits_c.argmax(dim=1) == y_test).sum().float()
            dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
            acc = (local_correct.item() / TOTAL_TEST_SIZE) * 100

            committee_accs.append(acc)
            if local_rank == 0:
                print(f"🎯 第 {c + 1} 号专家完赛！单体准确率: {acc:.2f}%")

            ensemble_logits += test_logits_c

        del lut_model, optimizer, loader, scheduler, idx_train, idx_test, test_logits_c, idx_train_bag, y_train_bag
        torch.cuda.empty_cache()
        gc.collect()


    # 🚨 把分布式同步操作放在 if 外面，确保 8 张卡一起执行！
    final_correct = (ensemble_logits.argmax(dim=1) == y_test).sum().float()
    dist.all_reduce(final_correct, op=dist.ReduceOp.SUM)
    final_acc = (final_correct.item() / TOTAL_TEST_SIZE) * 100

    if local_rank == 0:
        print("\n" + "=" * 80)
        print("👑 8x A100 集群战报：[修Bug后·高阶集成突围] 👑")
        print("=" * 80)
        for c, acc in enumerate(committee_accs):
            print(f"🔹 第 {c + 1} 号专家 (高阶特征 + Bagging) : {acc:>6.2f} %")
        print("-" * 80)
        print(f"🚀🚀🚀 [终极联合投票] Logits 纯物理求和准确率 : {final_acc:>6.2f} % !!! 🚀🚀🚀")
        print("=" * 80)

    # 兄弟们集合完毕，一起安全下线
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()