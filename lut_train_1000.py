import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

# =====================================================================
# 1. 最终量产版配置 (Ultimate Production Config)
# =====================================================================
EPOCHS = 20  # 既然架构已定，可以拉长 Epochs 进行深度炼丹 (建议最终跑 500-1000)
BATCH_SIZE = 8  # 单卡 Batch Size
ACCUMULATE_STEPS = 4  # 梯度累加，等效大 Batch
MAX_SAMPLES = 10000  # None 表示使用全部数据！火力全开！

DIR_L11 = './offline_data/layer11_out'
DIR_L12 = './offline_data/layer12_base'
DIR_IDX = './offline_data/indices'
DIR_GT = './ideal_features_fast'
OUTPUT_DIR = './offline_checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2. 数据集加载器 (稳如老狗)
# =====================================================================
class OfflineDistillationDataset(Dataset):
    def __init__(self, max_samples=None):
        self.gt_paths = sorted(glob.glob(os.path.join(DIR_GT, '*_layer12_feature.pt')))
        self.valid_samples = []
        for gt_p in self.gt_paths:
            stem = os.path.basename(gt_p).replace('_layer12_feature.pt', '')
            l11_p = os.path.join(DIR_L11, f"{stem}_layer11.pt")
            l12_p = os.path.join(DIR_L12, f"{stem}_layer12_base.pt")
            idx_p = os.path.join(DIR_IDX, f"{stem}_indices.pt")

            if os.path.exists(l11_p) and os.path.exists(l12_p) and os.path.exists(idx_p):
                self.valid_samples.append((l11_p, l12_p, idx_p, gt_p))
                if max_samples and len(self.valid_samples) >= max_samples:
                    break

        print(f"📦 成功加载 {len(self.valid_samples)} 个完美匹配的离线样本！")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        l11_p, l12_p, idx_p, gt_p = self.valid_samples[idx]
        return torch.load(l11_p, weights_only=True).float(), \
            torch.load(l12_p, weights_only=True).float(), \
            torch.load(idx_p, weights_only=True).long(), \
            torch.load(gt_p, weights_only=True).squeeze(0).float()


# =====================================================================
# 3. 终极真神架构：Ultimate_YOLO_LoRA
# =====================================================================
class Ultimate_YOLO_LoRA(nn.Module):
    """
    千锤百炼的最终版本：
    1. Rank = 32 (容量甜点区，4GB 参数量，包罗万象)
    2. std = 0.1 (激进初始化，打破早期僵局)
    3. Alpha 缩放 (通道级自适应放大器)
    4. 纯线性映射 + 完美内存对齐 (极致提取残差特征)
    """

    def __init__(self, c_in=384, c_out=128, r=32):
        super().__init__()
        # 10 亿参数的知识宝库，全在这个两张表里
        self.table_A = nn.Embedding(65536, c_in * r)
        self.table_B = nn.Embedding(65536, r * c_out)

        # 激进初始化：给予极大的初始拉扯力度
        nn.init.normal_(self.table_A.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.table_B.weight)

        self.c_in, self.c_out, self.r = c_in, c_out, r

        # Alpha 剑鞘：赋予通道瞬间改变输出幅度的能力
        self.alpha = nn.Parameter(torch.ones(1, c_out, 1, 1))

    def forward(self, x, base, idx):
        B, C, H, W = x.shape

        # O(1) 极速查表，抽出对应 Token 的专属权重
        wA = self.table_A(idx).view(B, H, W, self.c_in, self.r)
        wB = self.table_B(idx).view(B, H, W, self.r, self.c_out)

        # 严格执行空间通道对齐，防止特征坍塌，纯线性矩阵乘法重构完美特征
        x_spatial = x.permute(0, 2, 3, 1).contiguous()
        hidden = torch.einsum('bhwc,bhwcr->bhwr', x_spatial, wA)
        lora_out = torch.einsum('bhwr,bhwro->bhwo', hidden, wB).permute(0, 3, 1, 2).contiguous()

        return base + (self.alpha * lora_out)


# =====================================================================
# 4. 主控训练引擎 (专为量产设计)
# =====================================================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    if local_rank == 0:
        print("🔥 [生产环境启动] Ultimate YOLO LoRA 长程炼丹开始...")

    dataset = OfflineDistillationDataset(max_samples=MAX_SAMPLES)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    # 实例化终极模型
    model = Ultimate_YOLO_LoRA().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    # 因为 Epoch 变长，使用平滑的余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = 999.0

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False) if local_rank == 0 else dataloader

        for i, (l11, l12_base, idx, gt) in enumerate(pbar):
            l11, l12_base, idx, gt = l11.to(device), l12_base.to(device), idx.to(device), gt.to(device)

            with torch.cuda.amp.autocast():
                pred = model(l11, l12_base, idx)
                # 复合损失函数：方向与幅度的双重逼近
                loss_dir = (1.0 - F.cosine_similarity(pred, gt, dim=1)).mean()
                loss_mag = F.smooth_l1_loss(pred, gt)
                loss = (3.0 * loss_dir + loss_mag) / ACCUMULATE_STEPS

            scaler.scale(loss).backward()

            # 梯度累加逻辑
            if (i + 1) % ACCUMULATE_STEPS == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if local_rank == 0:
                real_loss = loss.item() * ACCUMULATE_STEPS
                epoch_loss += real_loss
                steps += 1
                pbar.set_postfix({'Avg Loss': f"{epoch_loss / steps:.4f}"})

        scheduler.step()

        # 每轮总结与保存模型
        if local_rank == 0:
            final_loss = epoch_loss / steps
            print(f"📊 [Epoch {epoch + 1}/{EPOCHS}] 平均 Loss: {final_loss:.5f}")

            if final_loss < best_loss:
                best_loss = final_loss
                save_path = os.path.join(OUTPUT_DIR, "Ultimate_YOLO_LoRA_Best.pt")
                # 只保存模型参数，剥离 DDP 的 module 壳子
                torch.save(model.module.state_dict(), save_path)
                print(f"   🌟 新纪录！模型已保存至: {save_path}")

    # =================================================================
    # 优雅退出
    # =================================================================
    dist.barrier()
    if local_rank == 0:
        print("\n🎉 训练圆满结束！")
        print(f"🏆 历史最低 Loss: {best_loss:.5f}")
        print(f"💾 最终神级权重获取地址: {os.path.join(OUTPUT_DIR, 'Ultimate_YOLO_LoRA_Best.pt')}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()