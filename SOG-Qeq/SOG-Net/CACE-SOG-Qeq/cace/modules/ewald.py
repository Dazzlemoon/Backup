from typing import Dict

import torch
import torch.nn as nn


class EwaldPotential(nn.Module):
    """
    精简版 Ewald 势，用于给 Qeq/ChargeEq 提供核矩阵 A。

    与原始 CACE 实现不同，这里只实现：
    - compute_potential_triclinic: 返回
        * pot: 对给定电荷 q 的库伦能（简单 1/r 形式）
        * A_mat: A_ij = 1 / |r_i - r_j| 的核矩阵
    目前不做真正的 Ewald 周期展开，只是直接 1/r 核，主要用于演示和开发。
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        feature_key: str = "q",
        output_key: str = "ewald_potential",
        aggregation_mode: str = "sum",
        compute_field: bool = False,
    ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.compute_field = compute_field
        self.model_outputs = [output_key]

    def compute_potential_triclinic(
        self,
        r: torch.Tensor,
        q: torch.Tensor,
        cell: torch.Tensor,
        compute_field: bool = False,
    ):
        """
        简单 1/r 库伦核：
        - r: (N, 3)
        - q: (N,) 或 (N, n_q)
        - cell: (3, 3)，此处不显式使用，仅保留接口

        返回：
        - pot: 长度为 n_q 的能量向量（每一列 q 的能量）
        - A_mat: (N, N) 的核矩阵，用于 Qeq 中的 A
        """
        device = r.device

        if q.dim() == 1:
            q = q.unsqueeze(1)  # (N, 1)

        # 计算 pairwise 距离
        diff = r.unsqueeze(0) - r.unsqueeze(1)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)

        eps = 1e-6
        A_mat = 1.0 / (dist + eps)

        if self.remove_self_interaction:
            diag_idx = torch.arange(A_mat.size(0), device=device)
            A_mat[diag_idx, diag_idx] = 0.0

        # 对每一列 q 计算能量 E = 1/2 q^T A q
        Aq = A_mat @ q  # (N, n_q)
        pot = 0.5 * torch.sum(q * Aq, dim=0)  # (n_q,)

        return pot, A_mat

