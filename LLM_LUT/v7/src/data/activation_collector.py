"""
activation_collector.py

从目标 FFN 层收集激活值和输出。
Phase 0: 数据固化与基线复现
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import json

import torch
import torch.nn as nn
from tqdm import tqdm


class FFNLayerExtractor(nn.Module):
    """
    包装器：提取 FFN 层的输入和输出。
    用于 Qwen2.5 MoE 模型的 expert FFN。
    """

    def __init__(self, expert_module: nn.Module):
        super().__init__()
        self.expert = expert_module
        self.inputs = []
        self.outputs = []
        self.hooks = []

    def register_hooks(self):
        """注册 forward hook 来捕获输入输出。"""
        def hook_fn(module, input, output):
            # input 是 tuple，取第一个元素
            x = input[0].detach().cpu()
            y = output.detach().cpu()
            self.inputs.append(x)
            self.outputs.append(y)

        handle = self.expert.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return self

    def remove_hooks(self):
        """移除所有 hook。"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def clear_cache(self):
        """清空缓存的输入输出。"""
        self.inputs.clear()
        self.outputs.clear()

    def get_collected_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回收集到的所有输入输出。
        Returns:
            inputs: [N, hidden_size]
            outputs: [N, hidden_size]
        """
        if not self.inputs:
            return torch.empty(0), torch.empty(0)
        x_all = torch.cat(self.inputs, dim=0)
        y_all = torch.cat(self.outputs, dim=0)
        return x_all, y_all

    def forward(self, *args, **kwargs):
        return self.expert(*args, **kwargs)


class QwenMoEExpert(nn.Module):
    """
    Qwen2.5 MoE Expert FFN。
    与 v6 保持一致。
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size]
        Returns:
            output: [..., hidden_size]
        """
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def load_expert_from_state_dict(
    pt_path: str,
    device: torch.device
) -> Tuple[QwenMoEExpert, int, int]:
    """
    从 .pt 文件加载 expert 权重。

    Args:
        pt_path: 权重文件路径
        device: 目标设备（明确指定单卡，如 cuda:0 或 cpu）

    Returns:
        expert: 加载好的模型
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度
    """
    print(f"Loading teacher weights: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu")

    # 推断维度
    gate_key = next(k for k in state_dict.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = state_dict[gate_key].shape

    # 创建模型
    expert = QwenMoEExpert(hidden_size, intermediate_size)

    # 清理 state_dict 键名
    clean_state_dict = {}
    for k, v in state_dict.items():
        if "expert." in k:
            new_k = k.split("expert.")[-1]
        else:
            new_k = k
        clean_state_dict[new_k] = v

    expert.load_state_dict(clean_state_dict, strict=False)
    expert.to(device).eval()

    print(f"  hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    return expert, hidden_size, intermediate_size


def collect_from_pt_files(
    input_paths: List[str],
    output_paths: Optional[List[str]],
    needed: int,
    teacher: nn.Module,
    batch_size: int,
    device: torch.device,
    desc: str = "collecting"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 .pt 文件收集输入输出数据。

    Args:
        input_paths: 输入文件路径列表
        output_paths: 预计算输出文件路径列表（可选）
        needed: 需要收集的样本数
        teacher: 教师模型
        batch_size: 批大小
        device: 设备
        desc: 进度条描述

    Returns:
        inputs: [needed, hidden_size]
        outputs: [needed, hidden_size]
    """
    use_precomputed = output_paths is not None
    weight_dtype = next(teacher.parameters()).dtype

    inputs = []
    outputs = []
    collected = 0

    pbar = tqdm(total=needed, desc=desc, unit="sample")

    for idx, in_path in enumerate(sorted(input_paths)):
        if collected >= needed:
            break

        try:
            x_tensor = torch.load(in_path, map_location="cpu")
            # 处理维度
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            elif x_tensor.dim() != 2:
                print(f"  skip {in_path}: unexpected shape {x_tensor.shape}")
                continue
        except Exception as e:
            print(f"  skip {in_path}: {e}")
            continue

        # 获取输出
        if use_precomputed:
            out_path = output_paths[idx]
            try:
                y_tensor = torch.load(out_path, map_location="cpu")
                if y_tensor.dim() == 1:
                    y_tensor = y_tensor.unsqueeze(0)
                elif y_tensor.dim() != 2:
                    print(f"  skip {out_path}: unexpected shape {y_tensor.shape}")
                    continue
                if y_tensor.shape != x_tensor.shape:
                    print(f"  skip {in_path}: shape mismatch")
                    continue
            except Exception as e:
                print(f"  skip {out_path}: {e}")
                continue
        else:
            y_tensor = None

        # 分批处理
        n_samples = x_tensor.shape[0]
        for start in range(0, n_samples, batch_size):
            if collected >= needed:
                break

            end = min(start + batch_size, n_samples)
            x_batch = x_tensor[start:end].to(device, dtype=weight_dtype)

            if y_tensor is not None:
                y_batch = y_tensor[start:end].float().cpu()
            else:
                with torch.no_grad():
                    y_batch = teacher(x_batch).float().cpu()

            x_batch = x_batch.float().cpu()

            inputs.append(x_batch)
            outputs.append(y_batch)
            collected += x_batch.shape[0]
            pbar.update(x_batch.shape[0])

    pbar.close()

    if collected == 0:
        raise RuntimeError(f"No valid samples found")

    x_all = torch.cat(inputs, dim=0)[:needed]
    y_all = torch.cat(outputs, dim=0)[:needed]

    return x_all, y_all


def estimate_eval_files_needed(input_files: List[str], eval_size: int) -> int:
    """
    估算需要预留多少个文件给 eval 集。
    """
    n = len(input_files)
    if n <= 1:
        return n

    probe = min(100, n)
    counts = []

    for path in input_files[-probe:]:
        try:
            t = torch.load(path, map_location="cpu")
            if t.dim() == 1:
                counts.append(1)
            elif t.dim() == 2:
                counts.append(t.shape[0])
            else:
                counts.append(1)
        except Exception:
            counts.append(1)

    avg = sum(counts) / len(counts) if counts else 1.0
    n_eval = max(100, int((eval_size / avg) + 0.5))

    return min(n_eval, n - 1)


def prepare_data_paths(
    dataset_dir: str,
    output_dataset_dir: Optional[str],
    eval_size: int
) -> Tuple[List[str], List[str], Optional[List[str]], Optional[List[str]]]:
    """
    准备数据文件路径。

    Returns:
        train_input_files, test_input_files, train_output_files, test_output_files
    """
    input_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {dataset_dir}")

    if output_dataset_dir:
        output_files_map = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(output_dataset_dir, "*.pt"))
        }
        paired = [(inp, output_files_map.get(os.path.basename(inp))) for inp in input_files]
        paired = [pair for pair in paired if pair[1] is not None]

        if not paired:
            raise FileNotFoundError(f"No matching .pt files between {dataset_dir} and {output_dataset_dir}")

        input_files = [p[0] for p in paired]
        output_files = [p[1] for p in paired]

        n_eval = estimate_eval_files_needed(input_files, eval_size)
        train_input = input_files[:-n_eval]
        test_input = input_files[-n_eval:]
        train_output = output_files[:-n_eval]
        test_output = output_files[-n_eval:]

        print(f"Found {len(paired)} paired input/output .pt files")
        print(f"  using {len(train_input)} files for calibration, {len(test_input)} files for eval")

        return train_input, test_input, train_output, test_output
    else:
        n_eval = estimate_eval_files_needed(input_files, eval_size)
        print(f"Found {len(input_files)} .pt input files")
        print(f"  using {len(input_files) - n_eval} files for calibration, {n_eval} files for eval")

        train_input = input_files[:-n_eval]
        test_input = input_files[-n_eval:]

        return train_input, test_input, None, None
