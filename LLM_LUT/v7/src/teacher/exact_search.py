"""
exact_search.py

Teacher Path: 暴力最近邻搜索 + 完整 Jacobian 修正。
仅用于离线评估基线，不用于在线部署。
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def brute_force_nearest_neighbor(
    query: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    暴力最近邻搜索。

    Args:
        query: [N, d] 查询向量
        anchors: [M, d] anchor 向量

    Returns:
        indices: [N] 最近邻的 anchor 索引
    """
    # 计算所有 pair 距离 [N, M]
    # ||q - a||^2 = ||q||^2 + ||a||^2 - 2*q@a^T
    q_norm = (query ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    a_norm = (anchors ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
    distances = q_norm + a_norm.T - 2 * query @ anchors.T
    indices = distances.argmin(dim=-1)
    return indices


def compute_jacobian_finite_diff(
    model: torch.nn.Module,
    x: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    使用有限差分计算 Jacobian 矩阵。

    Args:
        model: 模型
        x: 输入 [d_in]
        epsilon: 扰动大小

    Returns:
        J: [d_out, d_in] Jacobian 矩阵
    """
    model.eval()
    x = x.unsqueeze(0)  # [1, d_in]

    with torch.no_grad():
        y0 = model(x)  # [1, d_out]
        d_out = y0.shape[-1]
        d_in = x.shape[-1]

        J = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)

        for i in range(d_in):
            x_plus = x.clone()
            x_plus[0, i] += epsilon
            y_plus = model(x_plus)
            J[:, i] = (y_plus - y0).squeeze(0) / epsilon

    return J


def compute_jacobian_autograd(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    使用 autograd 计算 Jacobian 矩阵（更精确但更慢）。

    Args:
        model: 模型
        x: 输入 [d_in]

    Returns:
        J: [d_out, d_in] Jacobian 矩阵
    """
    model.eval()
    x = x.requires_grad_(True)

    y = model(x.unsqueeze(0)).squeeze(0)  # [d_out]
    d_out = y.shape[0]
    d_in = x.shape[0]

    J = torch.zeros(d_out, d_in, device=x.device, dtype=x.dtype)

    for i in range(d_out):
        if x.grad is not None:
            x.grad.zero_()
        y[i].backward(retain_graph=True)
        J[i] = x.grad.clone()

    x.requires_grad_(False)
    return J


class TeacherOracle:
    """
    Teacher Oracle: 提供暴力 NN + Jacobian 修正的上界。
    """

    def __init__(
        self,
        anchors: torch.Tensor,
        anchor_outputs: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        jacobian_method: str = "finite_diff",
        epsilon: float = 1e-4,
    ):
        """
        Args:
            anchors: [N, d_in] anchor 输入
            anchor_outputs: [N, d_out] anchor 输出
            model: 用于计算 Jacobian 的模型（如果为 None，则不计算 Jacobian）
            jacobian_method: "finite_diff" 或 "autograd"
            epsilon: 有限差分步长
        """
        self.anchors = anchors
        self.anchor_outputs = anchor_outputs
        self.model = model
        self.jacobian_method = jacobian_method
        self.epsilon = epsilon

        # 预计算所有 anchor 的 Jacobian（如果提供了模型）
        self.jacobians = None
        if model is not None:
            self.jacobians = self._precompute_jacobians()

    def _precompute_jacobians(self) -> torch.Tensor:
        """预计算所有 anchor 的 Jacobian。"""
        N = self.anchors.shape[0]
        d_in = self.anchors.shape[1]
        d_out = self.anchor_outputs.shape[1]

        jacobians = torch.zeros(N, d_out, d_in, device=self.anchors.device)

        print(f"Precomputing Jacobians for {N} anchors...")
        for i in tqdm(range(N)):
            x = self.anchors[i]
            if self.jacobian_method == "finite_diff":
                J = compute_jacobian_finite_diff(self.model, x, self.epsilon)
            else:
                J = compute_jacobian_autograd(self.model, x)
            jacobians[i] = J

        return jacobians

    def query(
        self,
        x: torch.Tensor,
        use_jacobian: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询 Teacher Oracle。

        Args:
            x: [..., d_in] 查询输入
            use_jacobian: 是否使用 Jacobian 修正

        Returns:
            y_pred: [..., d_out] 预测输出
            indices: [...] 选中的 anchor 索引
        """
        original_shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        # 暴力最近邻搜索
        indices = brute_force_nearest_neighbor(x_flat, self.anchors)

        # 获取 anchor 输出
        y_pred = self.anchor_outputs[indices]

        # Jacobian 修正
        if use_jacobian and self.jacobians is not None:
            for i in range(x_flat.shape[0]):
                idx = indices[i]
                delta = x_flat[i] - self.anchors[idx]
                J = self.jacobians[idx]  # [d_out, d_in]
                correction = J @ delta  # [d_out]
                y_pred[i] += correction

        y_pred = y_pred.view(*original_shape, -1)
        indices = indices.view(*original_shape)

        return y_pred, indices

    def evaluate(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        use_jacobian: bool = True,
    ) -> dict:
        """
        评估 Teacher Oracle。

        Args:
            x: [N, d_in] 输入
            y_true: [N, d_out] 真实输出
            use_jacobian: 是否使用 Jacobian 修正

        Returns:
            metrics: 评估指标字典
        """
        y_pred, indices = self.query(x, use_jacobian)

        mse = F.mse_loss(y_pred, y_true).item()
        rmse = math.sqrt(mse)
        var = y_true.var().item()
        rel_mse = mse / (var + 1e-8)
        rel_l2 = torch.norm(y_pred - y_true).item() / (torch.norm(y_true).item() + 1e-8)
        cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1).mean().item()

        # 统计最近邻距离
        x_flat = x.view(-1, x.shape[-1])
        selected_anchors = self.anchors[indices.view(-1)]
        distances = torch.norm(x_flat - selected_anchors, dim=-1)
        mean_dist = distances.mean().item()
        median_dist = distances.median().item()

        return {
            "mse": mse,
            "rmse": rmse,
            "relative_mse": rel_mse,
            "relative_l2": rel_l2,
            "cosine_similarity": cos_sim,
            "mean_nn_distance": mean_dist,
            "median_nn_distance": median_dist,
        }


def evaluate_bare_anchor(
    x: torch.Tensor,
    y_true: torch.Tensor,
    anchors: torch.Tensor,
    anchor_outputs: torch.Tensor,
) -> dict:
    """
    评估裸 anchor 基线（无 Jacobian 修正）。

    Args:
        x: [N, d_in] 输入
        y_true: [N, d_out] 真实输出
        anchors: [M, d_in] anchor 输入
        anchor_outputs: [M, d_out] anchor 输出

    Returns:
        metrics: 评估指标字典
    """
    oracle = TeacherOracle(anchors, anchor_outputs, model=None)
    return oracle.evaluate(x, y_true, use_jacobian=False)


def evaluate_with_jacobian(
    x: torch.Tensor,
    y_true: torch.Tensor,
    anchors: torch.Tensor,
    anchor_outputs: torch.Tensor,
    model: torch.nn.Module,
    jacobian_method: str = "finite_diff",
) -> dict:
    """
    评估 anchor + Jacobian 修正。

    Args:
        x: [N, d_in] 输入
        y_true: [N, d_out] 真实输出
        anchors: [M, d_in] anchor 输入
        anchor_outputs: [M, d_out] anchor 输出
        model: 用于计算 Jacobian 的模型
        jacobian_method: "finite_diff" 或 "autograd"

    Returns:
        metrics: 评估指标字典
    """
    oracle = TeacherOracle(
        anchors, anchor_outputs, model,
        jacobian_method=jacobian_method
    )
    return oracle.evaluate(x, y_true, use_jacobian=True)
