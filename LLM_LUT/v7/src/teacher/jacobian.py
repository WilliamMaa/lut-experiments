"""
Jacobian 计算与存储

提供多种计算 Jacobian 的方法：
1. 自动微分（autograd）
2. 有限差分
3. JVP（Jacobian-vector product）采样
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
import time


def compute_jacobian_autograd(
    model: Callable,
    x: torch.Tensor,
    create_graph: bool = False
) -> torch.Tensor:
    """
    使用自动微分计算完整 Jacobian。
    
    Args:
        model: 可微函数，输入 [d_in]，输出 [d_out]
        x: [d_in] 输入点
        create_graph: 是否创建计算图（用于高阶导数）
    Returns:
        J: [d_out, d_in] Jacobian 矩阵
    """
    x = x.detach().requires_grad_(True)
    
    # 前向
    y = model(x)
    d_out = y.shape[0]
    
    # 逐行计算梯度
    jacobian_rows = []
    for i in range(d_out):
        if x.grad is not None:
            x.grad.zero_()
        y[i].backward(retain_graph=(i < d_out - 1), create_graph=create_graph)
        jacobian_rows.append(x.grad.clone())
    
    J = torch.stack(jacobian_rows, dim=0)
    return J


def compute_jacobian_finite_diff(
    model: Callable,
    x: torch.Tensor,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    使用有限差分计算 Jacobian。
    不需要可微性，但精度较低。
    
    Args:
        model: 函数，输入 [d_in]，输出 [d_out]
        x: [d_in] 输入点
        epsilon: 差分步长
    Returns:
        J: [d_out, d_in] Jacobian 矩阵
    """
    x = x.detach()
    d_in = x.shape[0]
    
    # 基准输出
    y0 = model(x)
    d_out = y0.shape[0]
    
    # 逐维扰动
    J = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)
    for i in range(d_in):
        x_plus = x.clone()
        x_plus[i] += epsilon
        y_plus = model(x_plus)
        J[:, i] = (y_plus - y0) / epsilon
    
    return J


def estimate_jacobian_low_rank(
    model: Callable,
    x: torch.Tensor,
    n_directions: int = 64,
    method: str = "random"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 JVP 估计 Jacobian 的低秩结构。
    
    J ≈ U @ Σ @ V^T
    其中 V 是随机/精心选择的探测方向，U 是输出方向。
    
    Args:
        model: 函数，输入 [d_in]，输出 [d_out]
        x: [d_in] 输入点
        n_directions: JVP 探测方向数
        method: "random" 或 "orthogonal"
    Returns:
        U: [d_out, n_directions] 输出方向
        S: [n_directions] 奇异值
        V: [d_in, n_directions] 输入方向
    """
    x = x.detach().requires_grad_(True)
    d_in = x.shape[0]
    
    # 生成探测方向
    if method == "random":
        V = torch.randn(d_in, n_directions, device=x.device, dtype=x.dtype)
        V = torch.linalg.qr(V)[0]  # 正交化
    elif method == "orthogonal":
        # 使用标准基的子集
        V = torch.eye(d_in, device=x.device, dtype=x.dtype)[:n_directions].T
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 计算 JVP: J @ v for each v in V
    jvps = []
    for i in range(n_directions):
        v = V[:, i]
        if x.grad is not None:
            x.grad.zero_()
        
        # 前向模式微分（如果可用）
        # 这里使用双反向技巧模拟
        y = model(x)
        grad_y = torch.autograd.grad(
            y, x, v, create_graph=False, retain_graph=True
        )[0]
        jvps.append(grad_y)
    
    J_V = torch.stack(jvps, dim=1)  # [d_out, n_directions]
    
    # SVD 得到低秩近似
    U, S, _ = torch.linalg.svd(J_V, full_matrices=False)
    
    return U, S, V


class JacobianComputer:
    """
    Jacobian 计算管理器。
    """
    
    def __init__(
        self,
        method: str = "autograd",
        device: str = "cpu",
        cache_results: bool = True
    ):
        self.method = method
        self.device = device
        self.cache = {} if cache_results else None
    
    def compute(
        self,
        model: Callable,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算单个点的 Jacobian。
        """
        x_key = None
        if self.cache is not None:
            x_key = hash(x.cpu().numpy().tobytes())
            if x_key in self.cache:
                return self.cache[x_key]
        
        if self.method == "autograd":
            J = compute_jacobian_autograd(model, x, **kwargs)
        elif self.method == "finite_diff":
            J = compute_jacobian_finite_diff(model, x, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.cache is not None and x_key is not None:
            self.cache[x_key] = J
        
        return J
    
    def compute_batch(
        self,
        model: Callable,
        xs: torch.Tensor,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        批量计算 Jacobian。
        
        Args:
            model: 函数
            xs: [N, d_in] 多个输入点
        Returns:
            Js: [N, d_out, d_in] Jacobian 矩阵
        """
        N = xs.shape[0]
        
        # 先计算一个点确定维度
        x0 = xs[0]
        y0 = model(x0)
        d_out = y0.shape[0]
        d_in = x0.shape[0]
        
        Js = torch.zeros(N, d_out, d_in, device=self.device)
        
        start_time = time.time()
        for i in range(N):
            Js[i] = self.compute(model, xs[i])
            
            if show_progress and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (N - i - 1)
                print(f"[Jacobian] Computed {i+1}/{N}, "
                      f"ETA: {eta:.1f}s")
        
        return Js
    
    def compute_jvp(
        self,
        model: Callable,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Jacobian-vector product: J(x) @ v。
        比完整 Jacobian 更高效。
        """
        x = x.detach().requires_grad_(True)
        y = model(x)
        
        # 使用 torch.autograd.grad 计算 JVP
        jvp = torch.autograd.grad(
            y, x, v, create_graph=False, retain_graph=False
        )[0]
        
        return jvp


def precompute_anchors_with_jacobians(
    ffn_layer: Callable,
    anchors: torch.Tensor,
    method: str = "autograd",
    device: str = "cpu",
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    预计算所有 anchor 的输出和 Jacobian。
    
    Args:
        ffn_layer: FFN 层函数
        anchors: [N, d_in] anchor 输入
        method: Jacobian 计算方法
        device: 计算设备
        save_path: 可选的保存路径
    Returns:
        Dict with 'outputs' and 'jacobians'
    """
    print(f"[Precompute] Computing outputs and Jacobians for {anchors.shape[0]} anchors")
    
    anchors = anchors.to(device)
    N, d_in = anchors.shape
    
    # 先计算一个点确定输出维度
    y0 = ffn_layer(anchors[0])
    d_out = y0.shape[0]
    
    outputs = torch.zeros(N, d_out, device=device)
    jacobians = torch.zeros(N, d_out, d_in, device=device)
    
    computer = JacobianComputer(method=method, device=device)
    
    start_time = time.time()
    for i in range(N):
        x = anchors[i]
        
        # 计算输出
        outputs[i] = ffn_layer(x)
        
        # 计算 Jacobian
        jacobians[i] = computer.compute(ffn_layer, x)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / N
            eta = elapsed / progress * (1 - progress)
            print(f"[Precompute] {i+1}/{N} ({progress*100:.1f}%), ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"[Precompute] Done in {total_time:.1f}s")
    print(f"[Precompute] Storage: outputs {outputs.element_size() * outputs.nelement() / 1e6:.2f} MB, "
          f"Jacobians {jacobians.element_size() * jacobians.nelement() / 1e6:.2f} MB")
    
    result = {
        'inputs': anchors.cpu(),
        'outputs': outputs.cpu(),
        'jacobians': jacobians.cpu()
    }
    
    if save_path:
        torch.save(result, save_path)
        print(f"[Precompute] Saved to {save_path}")
    
    return result
