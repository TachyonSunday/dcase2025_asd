"""
损失函数与异常分数计算模块 —— MSE 重构损失 + 基于统计的异常判定。
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis


class ReconstructionLoss(nn.Module):
    """
    重构损失 —— 均方误差 (MSE), 衡量输入与重建之间的差异。

    异常声音片段因偏离正常训练分布, 在重构时产生更大的误差。

    参数
    ----
    reduction : str
        归约方式, ``"mean"`` 返回标量, ``"none"`` 返回逐元素误差。
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        计算重构误差。

        参数
        ----
        x : torch.Tensor
            原始输入, shape=(B, C, H, W)。
        x_recon : torch.Tensor
            重建输出, shape=(B, C, H, W)。

        返回
        ----
        torch.Tensor
            MSE 损失值。
        """
        return F.mse_loss(x_recon, x, reduction=self.reduction)


def compute_anomaly_score(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    基于重构误差计算逐帧异常分数。

    参数
    ----
    x : torch.Tensor
        原始输入, shape=(N, C, H, W) 或 (C, H, W)。
    x_recon : torch.Tensor
        重建输出, shape 同输入。
    reduction : str
        ``"mean"`` 返回每帧的平均 MSE,
        ``"none"`` 返回逐元素平方误差。

    返回
    ----
    torch.Tensor
        异常分数 — 值越大越可能异常。
    """
    sq_error = (x_recon - x) ** 2
    if reduction == "mean":
        # 在通道与空间维度上取均值, 得到每帧的异常分数
        return sq_error.mean(dim=tuple(range(1, sq_error.ndim)))
    elif reduction == "none":
        return sq_error
    else:
        raise ValueError(f"不支持的 reduction 方式: {reduction}")


def compute_file_score(
    frame_scores: torch.Tensor,
    aggregation: str = "mean",
    top_k: int = 10,
) -> float:
    """
    将逐帧异常分数聚合为单个文件的异常分数。

    参数
    ----
    frame_scores : torch.Tensor
        逐帧异常分数, shape=(N,), 其中 N 为帧数。
    aggregation : str
        聚合方式: ``"mean"``, ``"max"``, ``"topk_mean"``。
    top_k : int
        当 ``aggregation="topk_mean"`` 时, 取前 top_k 个最高分的均值。

    返回
    ----
    float
        文件的异常分数。
    """
    scores = frame_scores.detach().cpu()
    if aggregation == "mean":
        return float(scores.mean().item())
    elif aggregation == "max":
        return float(scores.max().item())
    elif aggregation == "topk_mean":
        k = min(top_k, len(scores))
        return float(scores.topk(k).values.mean().item())
    else:
        raise ValueError(f"不支持的聚合方式: {aggregation}")


def compute_mahalanobis_score(
    z: torch.Tensor,
    mean: torch.Tensor,
    cov_inv: torch.Tensor,
) -> torch.Tensor:
    """
    基于马氏距离 (Mahalanobis Distance) 计算异常分数。

    使用正常样本隐向量的均值与协方差矩阵的逆,
    计算测试样本隐向量的马氏距离作为异常分数。

    参数
    ----
    z : torch.Tensor
        测试样本隐向量, shape=(N, latent_dim)。
    mean : torch.Tensor
        训练集隐向量均值, shape=(latent_dim,)。
    cov_inv : torch.Tensor
        训练集隐向量协方差矩阵的逆, shape=(latent_dim, latent_dim)。

    返回
    ----
    torch.Tensor
        马氏距离, shape=(N,), 值越大越异常。
    """
    z_centered = z - mean.unsqueeze(0)  # (N, D)
    # 马氏距离平方: (z-μ)^T Σ^{-1} (z-μ)
    dist_sq = torch.sum((z_centered @ cov_inv) * z_centered, dim=1)
    return torch.sqrt(torch.clamp(dist_sq, min=0.0))


def compute_normal_statistics(
    latent_vectors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从正常样本的隐向量集合中估计均值与协方差逆矩阵。

    参数
    ----
    latent_vectors : torch.Tensor
        正常样本隐向量, shape=(N, latent_dim)。

    返回
    ----
    tuple[torch.Tensor, torch.Tensor]
        (均值, 协方差逆矩阵)。
    """
    mean = latent_vectors.mean(dim=0)
    # 计算协方差矩阵, 并添加小的正则化项防止奇异
    cov = torch.cov(latent_vectors.T)  # (D, D)
    reg = 1e-4 * torch.eye(cov.size(0), device=cov.device)
    cov_inv = torch.linalg.inv(cov + reg)
    return mean, cov_inv
