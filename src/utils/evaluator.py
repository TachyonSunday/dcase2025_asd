"""
评估器模块 —— AUC 计算、阈值选取、推理接口。
"""

import os
from typing import Optional, Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm

from src.models.conv_ae import ConvAE
from src.models.losses import (
    compute_anomaly_score,
    compute_file_score,
    compute_mahalanobis_score,
    compute_normal_statistics,
)


class Evaluator:
    """
    ConvAE 评估器 — 计算异常分数、选取决策阈值、评估 AUC。

    使用方式::

        evaluator = Evaluator(model, config_path="config.yaml")
        # 加载检查点
        evaluator.load_checkpoint("best_model.pt")
        # 在测试集上推理
        scores, labels = evaluator.predict(test_loader)
        auc = evaluator.compute_auc(scores, labels)
        print(f"AUC: {auc:.4f}")

    参数
    ----
    model : ConvAE
        已训练的卷积自编码器。
    config_path : str
        YAML 配置文件的路径。
    device : str
        推理设备。
    """

    def __init__(
        self,
        model: ConvAE,
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device or self.config.get("device", "cuda"))
        self.model = model.to(self.device)
        self.model.eval()

        # 用于 Mahalanobis 模式的统计量
        self._normal_mean: Optional[torch.Tensor] = None
        self._normal_cov_inv: Optional[torch.Tensor] = None

        # 决策阈值
        self.threshold: Optional[float] = None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载模型权重。

        参数
        ----
        checkpoint_path : str
            检查点文件路径。
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[Evaluator] 已加载检查点: {checkpoint_path}")

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        score_aggregation: str = "mean",
        top_k: int = 10,
    ) -> Tuple[List[float], List[int]]:
        """
        在数据集上运行推理, 返回异常分数与真实标签。

        参数
        ----
        dataloader : DataLoader
            数据加载器。
        score_aggregation : str
            帧分数聚合方式 (传给 compute_file_score)。
        top_k : int
            topk_mean 聚合时的 k 值。

        返回
        ----
        tuple[list[float], list[int]]
            (异常分数列表, 标签列表)。
        """
        all_scores: List[float] = []
        all_labels: List[int] = []

        for batch_x, batch_y in tqdm(dataloader, desc="推理中"):
            batch_x = batch_x.to(self.device)
            x_recon, _ = self.model(batch_x)

            # 逐帧 MSE
            frame_scores = compute_anomaly_score(batch_x, x_recon, reduction="mean")

            # 聚合为文件分数 (此处将每帧视为独立样本)
            for i in range(len(frame_scores)):
                all_scores.append(float(frame_scores[i].item()))

            all_labels.extend(batch_y.tolist())

        return all_scores, all_labels

    @torch.no_grad()
    def predict_mahalanobis(
        self,
        dataloader: DataLoader,
    ) -> Tuple[List[float], List[int]]:
        """
        使用马氏距离模式进行推理 (需要先调用 fit_normal_statistics)。

        参数
        ----
        dataloader : DataLoader
            数据加载器。

        返回
        ----
        tuple[list[float], list[int]]
            (异常分数列表, 标签列表)。
        """
        if self._normal_mean is None or self._normal_cov_inv is None:
            raise RuntimeError("请先调用 fit_normal_statistics() 计算正常样本统计量")

        all_scores: List[float] = []
        all_labels: List[int] = []

        for batch_x, batch_y in tqdm(dataloader, desc="推理 (Mahalanobis)"):
            batch_x = batch_x.to(self.device)
            z = self.model.encode(batch_x)
            scores = compute_mahalanobis_score(
                z, self._normal_mean.to(self.device), self._normal_cov_inv.to(self.device)
            )
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(batch_y.tolist())

        return all_scores, all_labels

    @torch.no_grad()
    def fit_normal_statistics(self, train_loader: DataLoader) -> None:
        """
        从训练集计算正常样本的隐向量均值与协方差逆矩阵。

        参数
        ----
        train_loader : DataLoader
            正常样本数据加载器。
        """
        all_z: List[torch.Tensor] = []
        for batch_x, _ in tqdm(train_loader, desc="计算隐向量统计量"):
            batch_x = batch_x.to(self.device)
            z = self.model.encode(batch_x)
            all_z.append(z.cpu())

        latent_all = torch.cat(all_z, dim=0)
        self._normal_mean, self._normal_cov_inv = compute_normal_statistics(latent_all)
        print(f"[Evaluator] 已计算隐向量统计量 (样本数: {len(latent_all)})")

    def compute_auc(self, scores: List[float], labels: List[int]) -> float:
        """
        计算 ROC-AUC 分数。

        参数
        ----
        scores : list[float]
            异常分数列表。
        labels : list[int]
            真实标签 (0=正常, 1=异常)。

        返回
        ----
        float
            AUC 值。
        """
        return float(roc_auc_score(labels, scores))

    def find_threshold(
        self,
        scores: List[float],
        labels: List[int],
        method: str = "best_f1",
    ) -> float:
        """
        根据验证集寻找最优决策阈值。

        参数
        ----
        scores : list[float]
            验证集异常分数。
        labels : list[int]
            验证集真实标签。
        method : str
            阈值选取方法:
            - ``"best_f1"``: 最大化 F1 分数的阈值
            - ``"percentile"``: 使用分位数阈值 (需 config 配置)

        返回
        ----
        float
            最优决策阈值。
        """
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        if method == "best_f1":
            precision, recall, thresholds = precision_recall_curve(labels_arr, scores_arr)
            # precision, recall 比 thresholds 多一个元素
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = int(np.argmax(f1_scores))
            # thresholds 比 precision/recall 少一个元素
            if best_idx >= len(thresholds):
                best_idx = len(thresholds) - 1
            self.threshold = float(thresholds[best_idx])
        elif method == "percentile":
            percentile = self.config["anomaly"].get("threshold_percentile", 90)
            self.threshold = float(np.percentile(scores_arr, percentile))
        else:
            raise ValueError(f"不支持的阈值选取方法: {method}")

        print(f"[Evaluator] 决策阈值: {self.threshold:.6f} (method={method})")
        return self.threshold

    def classify(self, score: float) -> Tuple[int, float]:
        """
        对单个异常分数进行分类判定。

        参数
        ----
        score : float
            异常分数值。

        返回
        ----
        tuple[int, float]
            (判定结果: 0=正常/1=异常, 归一化置信度)
        """
        if self.threshold is None:
            raise RuntimeError("请先调用 find_threshold() 计算决策阈值")
        prediction = int(score > self.threshold)
        # 简单置信度: 分数与阈值的相对距离
        confidence = min(abs(score - self.threshold) / (abs(self.threshold) + 1e-8), 1.0)
        return prediction, confidence
