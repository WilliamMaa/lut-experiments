import os
import glob
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import jvp
import torch.multiprocessing as mp


# ==============================================================================
# 1. 基础 Qwen MoE 专家 (Teacher 模型)
# ==============================================================================
class QwenMoEExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def load_real_teacher(pt_path, device):
    print(f"\n🧠 正在解析 Teacher 权重: {pt_path}")
    state_dict = torch.load(pt_path, map_location='cpu')
    gate_key = next(k for k in state_dict.keys() if 'gate_proj' in k and 'weight' in k)
    intermediate_size, hidden_size = state_dict[gate_key].shape
    expert = QwenMoEExpert(hidden_size, intermediate_size)
    clean_state_dict = {k.split('expert.')[-1] if 'expert.' in k else k: v for k, v in state_dict.items()}
    expert.load_state_dict(clean_state_dict, strict=False)
    expert.to(device).eval()
    return expert, hidden_size


# ==============================================================================
# 2. 真实流式数据引擎 (修复了内存 OOM Bug)
# ==============================================================================
def real_file_stream_generator(file_paths, batch_size, device):
    buffer = []
    buffer_len = 0
    for fpath in file_paths:
        try:
            tensor = torch.load(fpath, map_location='cpu')
        except Exception:
            continue

        buffer.append(tensor)
        buffer_len += tensor.shape[0]

        # 🚀 修复核心：从 if 改为 while，大文件会被正确循环切分，内存绝不会堆积爆炸！
        while buffer_len >= batch_size:
            cat_tensor = torch.cat(buffer, dim=0).to(torch.float32)
            yield cat_tensor[:batch_size].to(device, non_blocking=True)
            leftover = cat_tensor[batch_size:]
            buffer = [leftover]
            buffer_len = leftover.shape[0]

    if buffer_len > 0:
        yield torch.cat(buffer, dim=0).to(torch.float32).to(device, non_blocking=True)


# ==============================================================================
# 3. 核心算法：单GPU工作进程 (Worker)
# ==============================================================================
def gpu_search_worker(gpu_id, file_subset, test_batch_cpu, stream_batch_size=100000):
    """
    运行在单个 A100 上的搜索进程
    """
    device = torch.device(f'cuda:{gpu_id}')

    # 🔥 A100 黑魔法：开启 TF32 加速，计算距离速度直接翻倍！
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    test_batch = test_batch_cpu.to(device)
    B_test = test_batch.shape[0]

    best_distances = torch.full((B_test,), float('inf'), device=device)
    best_anchors = torch.zeros_like(test_batch)

    train_gen = real_file_stream_generator(file_subset, stream_batch_size, device)

    # 🚀 修复核心：增加工作进度打印，消除卡死错觉
    for i, train_chunk in enumerate(train_gen):
        if i % 10 == 0:  # 每处理 10 个数据块打印一次，不刷屏
            print(f"  [GPU {gpu_id}] 正在高速计算第 {i} 个数据块 (Chunk)...")

        # 使用 cdist，底层在 A100 会自动调用 Tensor Core (TF32) 矩阵乘法
        dists = torch.cdist(test_batch, train_chunk, p=2.0)
        min_dists_in_chunk, min_indices = torch.min(dists, dim=1)

        update_mask = min_dists_in_chunk < best_distances
        if update_mask.any():
            best_distances[update_mask] = min_dists_in_chunk[update_mask]
            best_anchors[update_mask] = train_chunk[min_indices[update_mask]]

    # 返回回 CPU 内存，避免多进程间的显存冲突
    return best_anchors.cpu(), best_distances.cpu()


# ==============================================================================
# 4. 多卡分布式调度器
# ==============================================================================
@torch.no_grad()
def multi_gpu_find_exact_nearest_anchors(test_batch_cpu, train_files, num_gpus=8):
    print(f"\n🚀 启动 {num_gpus} 卡并行搜索引擎...")

    chunk_size = math.ceil(len(train_files) / num_gpus)
    file_chunks = [train_files[i:i + chunk_size] for i in range(0, len(train_files), chunk_size)]
    actual_gpus = min(num_gpus, torch.cuda.device_count())
    file_chunks = file_chunks[:actual_gpus]

    results_anchors = []
    results_dists = []

    ctx = mp.get_context('spawn')
    start_time = time.time()

    with ctx.Pool(processes=actual_gpus) as pool:
        jobs = []
        for i in range(actual_gpus):
            jobs.append(pool.apply_async(gpu_search_worker, (i, file_chunks[i], test_batch_cpu)))

        # 等待所有卡完成并汇总结果
        for job in tqdm(jobs, desc="🔥 并行地毯式搜索总进度", total=actual_gpus):
            anc, dist = job.get()
            results_anchors.append(anc)
            results_dists.append(dist)

    print(f"✅ 底层搜索完毕，耗时: {time.time() - start_time:.2f}s，正在合并全局最优解...")

    all_anchors = torch.stack(results_anchors)
    all_dists = torch.stack(results_dists)

    global_min_indices = torch.argmin(all_dists, dim=0)
    B = test_batch_cpu.shape[0]

    final_anchors = all_anchors[global_min_indices, torch.arange(B)]
    final_dists = all_dists[global_min_indices, torch.arange(B)]

    return final_anchors, final_dists


# ==============================================================================
# 5. 主函数 (修复了死锁问题)
# ==============================================================================
def main():
    torch.manual_seed(42)
    # 🚨 警告：在这个阶段绝对不要初始化任何 CUDA 上下文，也不要把数据 .to('cuda:0')
    num_gpus = torch.cuda.device_count()
    print(f"🌟 侦测到硬件环境: {num_gpus} 张 GPU 准备就绪！")

    # ==============================================
    TEACHER_WEIGHT_PATH = "/root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt"
    DATASET_DIR = "/data/ai2/datasets/lut_distill_dataset/input_qwen3_layer1_ffn_3y_0711"
    TEST_BATCH_SIZE = 256
    # ==============================================

    print("\n[1/4] 获取数据集文件列表...")
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.pt")))
    if not all_files: raise FileNotFoundError(f"没有找到 .pt 文件: {DATASET_DIR}")
    train_files = all_files[:-100]
    test_files = all_files[-100:]

    print("\n[2/4] 抽取测试向量 (纯 CPU 执行，防止多卡冲突)...")
    test_gen = real_file_stream_generator(test_files, TEST_BATCH_SIZE, device='cpu')
    test_input_cpu = next(test_gen).clone().detach()  # 加 clone 彻底脱离生成器

    print("\n[3/4] 开始 8 卡地毯式搜索...")
    # 🚀 启动 8 卡并行搜索！此时主进程完全没有显存占用，子进程可以放心使用全部显卡。
    anchor_input_cpu, anchor_distances_cpu = multi_gpu_find_exact_nearest_anchors(
        test_input_cpu, train_files, num_gpus=num_gpus
    )

    # ============================== 分水岭 ==============================
    # 搜索任务彻底结束，子进程已销毁。此时可以安全地占用主卡（cuda:0）加载大模型了！
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n🌟 多卡释放完毕，主控节点 {device} 接管。加载 Qwen 专家 Teacher...")

    teacher, DIM = load_real_teacher(TEACHER_WEIGHT_PATH, device)
    teacher_single = teacher.module if hasattr(teacher, 'module') else teacher

    # 把搜到的答案搬回主卡计算
    anchor_input = anchor_input_cpu.to(device)
    anchor_distances = anchor_distances_cpu.to(device)
    test_input = test_input_cpu.to(device)

    print(f"📊 锚点与测试向量的平均 L2 距离: {anchor_distances.mean().item():.4f}")

    print("\n[4/4] 🚀 核心实验：雅可比矩阵独立叠加推理 (JVP)")

    def model_fn(x):
        return teacher_single(x)

    delta_x = test_input - anchor_input

    # 🔥 顶级黑魔法：torch.func.jvp
    anchor_out, delta_out = jvp(model_fn, (anchor_input,), (delta_x,))

    student_out = anchor_out + delta_out
    teacher_out = model_fn(test_input)

    print("\n" + "=" * 70)
    print("📐 实验结果：雅可比叠加法 vs 真实 FFN")
    total_mse = F.mse_loss(student_out, teacher_out).item()
    total_rel_error = torch.norm(student_out - teacher_out).item() / (torch.norm(teacher_out).item() + 1e-8)
    total_cos_sim = F.cosine_similarity(student_out, teacher_out, dim=-1).mean().item()

    print(f"✨ 雅可比叠加绝对均方误差 (MSE)         : {total_mse:.6f}")
    print(f"📉 雅可比叠加相对误差 (Relative Error)  : {total_rel_error:.2%}")
    print(f"🎯 雅可比叠加余弦相似度 (Cosine Sim)    : {total_cos_sim:.4f}")

    naive_mse = F.mse_loss(anchor_out, teacher_out).item()
    naive_rel_error = torch.norm(anchor_out - teacher_out).item() / (torch.norm(teacher_out).item() + 1e-8)
    naive_cos_sim = F.cosine_similarity(anchor_out, teacher_out, dim=-1).mean().item()

    print("\n[对比] 如果只用锚点直接替代(不加雅可比微调)：")
    print(f"✨ 裸锚点绝对均方误差 (MSE)         : {naive_mse:.6f}")
    print(f"📉 裸锚点相对误差 (Relative Error)  : {naive_rel_error:.2%}")
    print(f"🎯 裸锚点余弦相似度 (Cosine Sim)    : {naive_cos_sim:.4f}")


if __name__ == "__main__":
    # 必须强制使用 spawn 以保障 CUDA 正常初始化
    mp.set_start_method('spawn', force=True)
    main()