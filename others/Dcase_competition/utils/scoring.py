"""
utils/scoring.py
============================================================
DCASE 2025 Task 2 - 异常打分 & 评估指标
============================================================

功能说明:
    1. DomainWiseDensityScorer: 基于 GenRep 的域感知异常打分器
       - 内存库拆分: source_memory_bank 和 target_memory_bank
       - Target 域 Mixup 增强: 动态扩充 target 域样本
       - L2 归一化: 对高维 Embedding 应用 L2 归一化，配合余弦距离
       - 全局 Z-Score 标准化: 对齐 source 和 target 域的分数分布
       - 推理: 测试样本到各域的 K 近邻平均距离，取 min
       - [修复] 单样本推理时使用训练集统计量，避免 Z-Score 坍塌

    2. KNNScorer: 传统 KNN 异常打分器 (保留用于对比实验)

    3. compute_auc / compute_pauc / compute_all_metrics: 评估指标

    DCASE 官方评估使用 AUC 和 pAUC (max_fpr=0.1) 作为核心指标。
    分数越高表示越异常，与 sklearn 的约定一致。

参考文献:
    - GenRep: Generative Representation Learning for Domain-Generalized
      Anomalous Sound Detection (DCASE 2025)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from typing import Optional, Tuple


# ============================================================
# Domain-wise Scorer with Mixup and Z-Score (GenRep)
# ============================================================
class DomainWiseDensityScorer:
    """
    基于 GenRep 的域感知异常打分器（Mixup + Z-Score 版本）

    核心原理:
        - 将训练集按域拆分为 source_memory_bank 和 target_memory_bank
        - 对 target 域执行 Mixup 增强，动态扩充样本
        - 对特征进行 L2 归一化，使高维 Embedding 更适合余弦距离度量
        - 推理时，计算测试样本到各域的 K 近邻平均距离
        - 对 source 和 target 域的分数执行全局 Z-Score 标准化
        - 最终异常得分取两个域的极小值

    公式:
        score_s(y) = mean_{k=1..K} [ d(y, x_k^s) ]
        score_t(y) = mean_{k=1..K} [ d(y, x_k^t) ]

        standardized_s = (score_s - train_mean_s) / (train_std_s + 1e-8)
        standardized_t = (score_t - train_mean_t) / (train_std_t + 1e-8)

        score(y) = min(standardized_s, standardized_t)

    其中:
        - d(y, x) 是测试样本 y 到参考样本 x 的距离（原始欧氏距离）
        - x_k^s 是 source 域中离 y 第 k 近的样本
        - x_k^t 是 target 域中离 y 第 k 近的样本
        - train_mean 和 train_std 是基于训练集计算的基准统计量
          [重要] 用于支持单样本在线推理，避免 batch_size=1 时 Z-Score 坍塌

    Mixup 增强原理:
        - 对每个 target 样本，找到最近的 K 个 source 样本
        - 线性插值: mixed = alpha * target + (1 - alpha) * nearest_source
        - 将生成的混合样本拼接到 target_memory_bank
        - 目的: 增强 target 域的表征能力，减少域偏移

    Z-Score 标准化原理:
        - 对 source 和 target 域的原始分数分别标准化
        - 消除域间的尺度差异，使得 min() 操作公平
        - [修复] 使用训练集基准统计量，支持单样本在线推理

    参数:
        k_source (int): 源域近邻数，默认 16
        k_target (int): 目标域近邻数，默认 9
        k_score (int): 推理时取 K 近邻平均，默认 5
        metric (str): 距离度量方式，默认 "euclidean"
        normalize_features (bool): 是否对特征进行 L2 归一化，默认 False
        n_mix_support (int): Mixup 时每个 target 样本使用的 source 近邻数，默认 None (不开启)
        alpha (float): Mixup 插值系数，默认 0.90
        algorithm (str): KNN 算法，默认 "auto"
        n_jobs (int): 并行计算线程数，默认 -1
    """

    def __init__(
        self,
        k_source: int = 16,
        k_target: int = 9,
        k_score: int = 5,
        metric: str = "euclidean",
        normalize_features: bool = False,
        n_mix_support: int = None,
        alpha: float = 0.90,
        algorithm: str = "auto",
        n_jobs: int = -1,
        score_normalization: str = "local_density",
    ):
        self.k_source = k_source
        self.k_target = k_target
        self.k_score = k_score
        self.metric = metric
        self.normalize_features = normalize_features
        self.n_mix_support = n_mix_support
        self.alpha = alpha
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.score_normalization = score_normalization  # "z_score" or "local_density"

        # 内存库 (L2 归一化后)
        self._source_memory_bank: Optional[np.ndarray] = None
        self._target_memory_bank: Optional[np.ndarray] = None

        # KNN 索引 (用于 z_score 模式的 K 近邻查询)
        self._source_knn: Optional[NearestNeighbors] = None
        self._target_knn: Optional[NearestNeighbors] = None

        # [local_density] 局部密度数组 (每个参考样本的平均 KNN 距离)
        self._source_density: Optional[np.ndarray] = None
        self._target_density: Optional[np.ndarray] = None

        # [z_score] 训练集基准统计量（用于单样本在线推理）
        self._train_score_mean_source: Optional[float] = None
        self._train_score_std_source: Optional[float] = None
        self._train_score_mean_target: Optional[float] = None
        self._train_score_std_target: Optional[float] = None

        self._is_fitted = False

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        对特征进行 L2 归一化

        Args:
            features: 特征矩阵, shape = (N, D)

        Returns:
            L2 归一化后的特征矩阵, shape = (N, D)
        """
        if self.normalize_features:
            # 使用 sklearn 的 normalize 函数，按行 (样本) 进行 L2 归一化
            return normalize(features, norm='l2', axis=1)
        return features

    def fit(
        self,
        train_features: np.ndarray,
        domain_labels: np.ndarray,
    ) -> "DomainWiseDensityScorer":
        """
        构建域感知内存库，执行 Mixup 增强 (恢复为得分最高的高斯噪声抗自匹配版本)
        """
        assert train_features.ndim == 2, (
            f"[Error] 训练特征应为 2D 数组, 当前 shape: {train_features.shape}"
        )
        assert len(train_features) == len(domain_labels), (
            f"[Error] 特征数 ({len(train_features)}) 与标签数 ({len(domain_labels)}) 不匹配"
        )

        domain_labels = np.asarray(domain_labels)
        train_features_normalized = self._normalize(train_features)

        if self.normalize_features:
            print(
                f"[DomainWise] 特征已进行 L2 归一化 | "
                f"原始维度: {train_features.shape[1]} | "
                f"度量方式: {self.metric} (L2 归一化后等效于余弦距离)"
            )

        # ---- 内存库拆分 ----
        source_indices = np.where(domain_labels == 0)[0]
        target_indices = np.where(domain_labels == 1)[0]

        self._source_memory_bank = train_features_normalized[source_indices].copy()
        self._target_memory_bank = train_features_normalized[target_indices].copy()

        # ---- Target 域 Mixup 增强 (原始顺序：先增强，再建库，再算基准) ----
        if (
            self.n_mix_support is not None
            and len(self._source_memory_bank) > 0
            and len(self._target_memory_bank) > 0
        ):
            print(
                f"[DomainWise] 执行 Target 域 Mixup 增强 | "
                f"n_mix_support={self.n_mix_support} | alpha={self.alpha:.2f}"
            )

            from scipy.spatial.distance import cdist
            dist_matrix = cdist(
                self._target_memory_bank,
                self._source_memory_bank,
                metric='euclidean'
            )

            k_nearest = min(self.n_mix_support, len(self._source_memory_bank))
            nearest_indices = np.argsort(dist_matrix, axis=1)[:, :k_nearest]

            mixed_samples = []
            for i in range(len(self._target_memory_bank)):
                target_sample = self._target_memory_bank[i]
                for j in range(k_nearest):
                    source_idx = nearest_indices[i, j]
                    source_sample = self._source_memory_bank[source_idx]
                    # 线性插值
                    mixed_sample = (
                        self.alpha * target_sample
                        + (1 - self.alpha) * source_sample
                    )
                    mixed_samples.append(mixed_sample)

            mixed_samples = np.array(mixed_samples)
            self._target_memory_bank = np.concatenate(
                [self._target_memory_bank, mixed_samples], axis=0
            )

            print(
                f"[DomainWise] Mixup 完成 | "
                f"生成 {len(mixed_samples)} 个混合样本 | "
                f"Target 域扩充至 {len(self._target_memory_bank)} 样本"
            )

        # ---- 构建最终的 KNN 索引 ----
        if len(self._source_memory_bank) > 0:
            self._source_knn = NearestNeighbors(
                n_neighbors=min(self.k_score, len(self._source_memory_bank)),
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=self.n_jobs,
            )
            self._source_knn.fit(self._source_memory_bank)

        if len(self._target_memory_bank) > 0:
            self._target_knn = NearestNeighbors(
                n_neighbors=min(self.k_score, len(self._target_memory_bank)),
                metric=self.metric,
                algorithm=self.algorithm,
                n_jobs=self.n_jobs,
            )
            self._target_knn.fit(self._target_memory_bank)

        # ---- [local_density] 计算局部密度 ----
        if self.score_normalization == "local_density":
            print("[DomainWise] 计算局部密度 (Local Density)...")
            self._source_density = self._compute_density(
                self._source_memory_bank, self.k_source
            )
            self._target_density = self._compute_density(
                self._target_memory_bank, self.k_target
            )
            if self._source_density is not None:
                print(
                    f"  Source 密度: shape={self._source_density.shape}, "
                    f"mean={self._source_density.mean():.6f}, "
                    f"min={self._source_density.min():.6f}"
                )
            if self._target_density is not None:
                print(
                    f"  Target 密度: shape={self._target_density.shape}, "
                    f"mean={self._target_density.mean():.6f}, "
                    f"min={self._target_density.min():.6f}"
                )

        # ---- 计算训练集基准统计量 (恢复微小高斯噪声抗自匹配逻辑) ----
        # 直接用训练特征查询 KNN 会导致自匹配距离为 0，std 坍塌为极小值。
        # 这里恢复原始的最佳实践：加微小噪声进行查询。
        print("[DomainWise] 计算训练集基准统计量 (含高斯噪声抗自匹配)...")
        noise = np.random.normal(0, 0.05, train_features_normalized.shape).astype(np.float32)
        noisy_train = train_features_normalized + noise

        train_scores_source = self._calculate_domain_scores(noisy_train, domain='source')
        train_scores_target = self._calculate_domain_scores(noisy_train, domain='target')

        # 撤销了所有人为硬截断，还原最纯粹的数学统计
        if len(train_scores_source) > 0:
            self._train_score_mean_source = float(np.mean(train_scores_source))
            self._train_score_std_source = float(np.std(train_scores_source))
            print(f"[DomainWise] Source 域基准统计 | mean={self._train_score_mean_source:.6f}, std={self._train_score_std_source:.6f}")
        else:
            self._train_score_mean_source = 0.0
            self._train_score_std_source = 1.0

        if len(train_scores_target) > 0:
            self._train_score_mean_target = float(np.mean(train_scores_target))
            self._train_score_std_target = float(np.std(train_scores_target))
            print(f"[DomainWise] Target 域基准统计 | mean={self._train_score_mean_target:.6f}, std={self._train_score_std_target:.6f}")
        else:
            self._train_score_mean_target = 0.0
            self._train_score_std_target = 1.0

        self._is_fitted = True
        print("[DomainWise] 拟合完成")
        return self

    def score(self, test_features: np.ndarray) -> np.ndarray:
        """
        计算测试样本的域感知异常分数
        """
        anomaly_scores, _, _ = self.score_with_details(test_features)
        return anomaly_scores

    def _calculate_domain_scores(
        self, features: np.ndarray, domain: str
    ) -> np.ndarray:
        """
        计算样本到指定域内存库的 K 近邻平均距离 (z_score 模式)
        """
        if domain == 'source':
            memory_bank = self._source_memory_bank
            knn = self._source_knn
        else:
            memory_bank = self._target_memory_bank
            knn = self._target_knn

        if memory_bank is None or len(memory_bank) == 0:
            return np.array([])

        distances, _ = knn.kneighbors(features)
        return distances.mean(axis=1)

    def _compute_density(
        self, memory_bank: Optional[np.ndarray], k: int
    ) -> Optional[np.ndarray]:
        """
        计算内存库中每个参考样本的局部密度 (GenRep Eq.4/5)

        密度 = SUM_{k=1..K} d(f, f_k) + 1e-8

        GenRep 论文使用 SUM 而非 MEAN:
        - K_s=16 (source) 和 K_t=9 (target) 的 SUM 值自然不同
        - 这个差异作为隐式域权重，平衡 source 和 target 的密度量级
        - 如果用 MEAN，会消除这个权重机制

        Args:
            memory_bank: 参考样本矩阵, shape = (N, D)
            k: 近邻数 (k_source 或 k_target)

        Returns:
            density: shape = (N,), 每个样本的局部密度值 (SUM of KNN distances)
                     如果 memory_bank 为空或样本不足, 返回 None
        """
        if memory_bank is None or len(memory_bank) == 0:
            return None

        # k+1 因为 kneighbors 包含样本自身 (距离=0)
        n_neighbors = min(k + 1, len(memory_bank))

        density_knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        density_knn.fit(memory_bank)

        # 查询内存库自身 → distances[:, 0] = 0 (自身), distances[:, 1:k+1] = k 近邻
        distances, _ = density_knn.kneighbors(memory_bank)

        # 取第 1 到第 k 个近邻 (排除自身), 计算 SQUARED SUM 距离 (GenRep Eq.1/4/5)
        # GenRep Eq.1: d(f_i, f_j) = ||f_i - f_j||^2 (squared Euclidean)
        # sklearn euclidean 返回非平方距离，需要手动平方
        sq_distances = distances[:, 1:] ** 2

        # 使用 SUM 而非 MEAN: K_s=16 vs K_t=9 的差异作为隐式域权重
        if n_neighbors > 1:
            density = sq_distances.sum(axis=1) + 1e-8
        else:
            # 极端情况: 只有 1 个样本, 密度设为 1e-8
            density = np.full(len(memory_bank), 1e-8)

        return density.astype(np.float32)

    def _calculate_domain_scores_density(
        self, features: np.ndarray, domain: str
    ) -> np.ndarray:
        """
        局部密度归一化打分 (local_density 模式)

        对每个测试样本 y, 计算到域内所有参考样本 f 的距离,
        除以 f 的局部密度, 取最小值:
            score_d(y) = min_{f ∈ bank_d} [ d(y, f) / density(f) ]

        Args:
            features: 测试样本特征, shape = (M, D)
            domain: "source" 或 "target"

        Returns:
            scores: shape = (M,), 每个测试样本的密度归一化异常分数
        """
        if domain == 'source':
            memory_bank = self._source_memory_bank
            density = self._source_density
        else:
            memory_bank = self._target_memory_bank
            density = self._target_density

        if memory_bank is None or len(memory_bank) == 0:
            return np.full(len(features), np.inf)

        # 计算所有测试样本到所有参考样本的 SQUARED Euclidean 距离
        # GenRep Eq.1: d(f_i, f_j) = ||f_i - f_j||^2
        # cdist: (M, D) × (N, D) → (M, N), metric='sqeuclidean'
        from scipy.spatial.distance import cdist
        all_sq_distances = cdist(features, memory_bank, metric='sqeuclidean')
        # all_sq_distances shape: (M, N)

        # 密度归一化: 每列除以对应参考样本的密度 (density 也是 squared SUM)
        # normalized_distances[i, j] = sq_d(y, f_j) / density(f_j)
        normalized_distances = all_sq_distances / density[np.newaxis, :]

        # 取每个测试样本到所有参考样本的最小归一化距离
        scores = normalized_distances.min(axis=1)

        return scores

    def score_with_details(
        self, test_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算测试样本的异常分数，并返回各域详细分数。

        支持两种打分模式:
            - "local_density": 局部密度归一化 (GenRep 推荐)
              score_d(y) = min_{f} [ d(y,f) / density(f) ]
              score(y) = min(score_s, score_t)

            - "z_score": KNN + Z-Score 标准化 (原有逻辑)
              score_d(y) = mean KNN distance
              Z-score 标准化后取 min
        """
        assert self._is_fitted, "[Error] 请先调用 fit() 构建内存库"
        assert test_features.ndim == 2, (
            f"[Error] 测试特征应为 2D 数组, 当前 shape: {test_features.shape}"
        )

        test_features_normalized = self._normalize(test_features)

        # ================================================================
        # 模式分支: local_density vs z_score
        # ================================================================
        if self.score_normalization == "local_density":
            return self._score_local_density(test_features_normalized)
        else:
            return self._score_z_score(test_features_normalized)

    def _score_local_density(
        self, test_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        局部密度归一化打分

        score_d(y) = min_{f ∈ bank_d} [ d(y, f) / density(f) ]
        score(y) = min(score_source, score_target)
        """
        scores_source = self._calculate_domain_scores_density(
            test_features, domain='source'
        )
        scores_target = self._calculate_domain_scores_density(
            test_features, domain='target'
        )

        # 最终异常分数: 取两个域的极小值
        anomaly_scores = np.minimum(scores_source, scores_target)

        return anomaly_scores, scores_source, scores_target

    def _score_z_score(
        self, test_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        KNN + 智能双模 Z-Score 标准化打分 (原有逻辑)
        智能兼容 DCASE 官方评测 (Batch) 与 Web 在线推理 (Single-sample)

        注意: test_features 已经经过 _normalize(), 不再重复归一化
        """

        # ---- 1. 距离计算 (特征已在 score_with_details 中归一化) ----
        scores_source = self._calculate_domain_scores(
            test_features, domain='source'
        )
        scores_target = self._calculate_domain_scores(
            test_features, domain='target'
        )

        epsilon = 1e-8
        n_samples = len(test_features)

        # ================================================================
        # 核心：智能双模 Z-Score 标准化
        # ================================================================
        if n_samples > 1:
            # 模式 A: 官方评估 / 盲测模式 (Transductive Batch Normalization)
            # 复刻 GenRep 原版逻辑，利用当前 Test Batch 全局分布计算均值和方差
            mean_s = np.mean(scores_source)
            std_s = np.std(scores_source)
            mean_t = np.mean(scores_target)
            std_t = np.std(scores_target)
        else:
            # 模式 B: Web API 在线单样本推理模式 (Single-Sample Fallback)
            # n=1 时 std 为 0，回退使用 fit() 时保存的训练集基准统计量
            mean_s = self._train_score_mean_source
            std_s = self._train_score_std_source
            mean_t = self._train_score_mean_target
            std_t = self._train_score_std_target

        # ---- 2. 执行纯净的无损缩放 ----
        if len(scores_source) > 0 and mean_s is not None:
            standardized_source = (scores_source - mean_s) / (std_s + epsilon)
        else:
            standardized_source = np.full(n_samples, np.inf)

        if len(scores_target) > 0 and mean_t is not None:
            standardized_target = (scores_target - mean_t) / (std_t + epsilon)
        else:
            standardized_target = np.full(n_samples, np.inf)

        # ---- 3. 极小值融合 ----
        anomaly_scores = np.minimum(standardized_source, standardized_target)

        return anomaly_scores, standardized_source, standardized_target

    def fit_score(
        self,
        train_features: np.ndarray,
        domain_labels: np.ndarray,
        test_features: np.ndarray,
    ) -> np.ndarray:
        """
        便捷方法：一步完成 fit + score

        Args:
            train_features: 训练集特征
            domain_labels: 域标签
            test_features: 测试集特征

        Returns:
            anomaly_scores: 异常分数数组
        """
        self.fit(train_features, domain_labels)
        return self.score(test_features)

    def get_memory_bank_stats(self) -> dict:
        """
        获取内存库统计信息

        Returns:
            dict: 包含 source/target 域的样本数、Mixup 参数等
        """
        stats = {
            "source_samples": len(self._source_memory_bank)
            if self._source_memory_bank is not None
            else 0,
            "target_samples": len(self._target_memory_bank)
            if self._target_memory_bank is not None
            else 0,
            "k_source": self.k_source,
            "k_target": self.k_target,
            "k_score": self.k_score,
            "n_mix_support": self.n_mix_support,
            "alpha": self.alpha,
            "train_score_mean_source": self._train_score_mean_source,
            "train_score_std_source": self._train_score_std_source,
            "train_score_mean_target": self._train_score_mean_target,
            "train_score_std_target": self._train_score_std_target,
        }
        return stats


# ============================================================
# 传统 KNN Scorer (保留用于对比实验)
# ============================================================
class KNNScorer:
    """
    基于 KNN 的异常打分器 (传统方法)

    核心原理:
        - 无监督异常检测: 训练阶段只有正常样本
        - KNN 学习正常样本的特征分布
        - 测试时，如果一个样本远离正常样本的特征空间，
          则被判定为异常（异常分数高）

    参数:
        k (int):           KNN 的 K 值（使用第 K 个最近邻的距离）
        metric (str):      距离度量方式，可选 "euclidean", "cosine", "minkowski" 等
        algorithm (str):   KNN 算法，可选 "auto", "ball_tree", "kd_tree", "brute"
        n_jobs (int):      并行计算线程数
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        n_jobs: int = -1,
    ):
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs

        self.knn = NearestNeighbors(
            n_neighbors=k,
            metric=metric,
            algorithm=algorithm,
            n_jobs=n_jobs,
        )

        self._is_fitted = False
        self._train_features = None

    def fit(self, train_features: np.ndarray) -> "KNNScorer":
        """使用训练集特征构建 KNN 索引"""
        assert train_features.ndim == 2, (
            f"[Error] 训练特征应为 2D 数组, 当前 shape: {train_features.shape}"
        )

        self._train_features = train_features.copy()
        self.knn.fit(train_features)
        self._is_fitted = True

        print(
            f"[KNN] 索引构建完成 | "
            f"样本数: {len(train_features)} | K={self.k} | 度量: {self.metric}"
        )
        return self

    def score(self, test_features: np.ndarray) -> np.ndarray:
        """计算测试样本的异常分数 (第 K 个近邻距离)"""
        assert self._is_fitted, "[Error] 请先调用 fit()"

        distances, _ = self.knn.kneighbors(test_features)
        anomaly_scores = distances[:, -1]
        return anomaly_scores

    def score_with_mean(self, test_features: np.ndarray) -> np.ndarray:
        """计算测试样本的异常分数 (K 近邻平均距离)"""
        assert self._is_fitted, "[Error] 请先调用 fit()"

        distances, _ = self.knn.kneighbors(test_features)
        anomaly_scores = distances.mean(axis=1)
        return anomaly_scores

    def fit_score(
        self, train_features: np.ndarray, test_features: np.ndarray
    ) -> np.ndarray:
        """便捷方法：fit + score"""
        self.fit(train_features)
        return self.score(test_features)


# ============================================================
# 评估指标
# ============================================================
def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 AUC (Area Under the ROC Curve)

    Args:
        y_true: 真实标签, 0=正常, 1=异常
        y_score: 异常分数（越高越异常）

    Returns:
        AUC 值, 范围 [0, 1]
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        print("[Warning] 标签中只有一个类别，AUC 无意义")
        return 0.0

    return float(roc_auc_score(y_true, y_score))


def compute_pauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.1,
) -> float:
    """
    计算 pAUC (partial Area Under the ROC Curve)

    DCASE 使用 max_fpr=0.1，只关注低误报率下的检测性能。

    Args:
        y_true: 真实标签
        y_score: 异常分数
        max_fpr: 最大假阳性率

    Returns:
        pAUC 值
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        print("[Warning] 标签中只有一个类别，pAUC 无意义")
        return 0.0

    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.1,
) -> dict:
    """
    一次性计算所有评估指标

    Args:
        y_true: 真实标签
        y_score: 异常分数
        max_fpr: pAUC 的最大假阳性率

    Returns:
        dict: {"AUC": float, "pAUC": float}
    """
    return {
        "AUC": compute_auc(y_true, y_score),
        "pAUC": compute_pauc(y_true, y_score, max_fpr=max_fpr),
    }


# ============================================================
# 调试入口
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Domain-wise Scorer with Mixup and Z-Score 打分模块测试")
    print("=" * 70)

    np.random.seed(42)
    embedding_dim = 384

    # 模拟训练集: source 域 300 样本 + target 域 200 样本
    source_train = np.random.randn(300, embedding_dim).astype(np.float32)
    target_train = np.random.randn(200, embedding_dim).astype(np.float32) + 0.5
    train_features = np.concatenate([source_train, target_train], axis=0)
    domain_labels = np.concatenate([
        np.zeros(300),  # source
        np.ones(200),   # target
    ])

    # 模拟测试集: 50 正常 + 50 异常
    normal_test = np.random.randn(50, embedding_dim).astype(np.float32) + 0.25
    anomaly_test = np.random.randn(50, embedding_dim).astype(np.float32) * 2.0 + 3.0
    test_features = np.concatenate([normal_test, anomaly_test], axis=0)
    y_true = np.concatenate([np.zeros(50), np.ones(50)])

    # ---- Domain-wise 打分 (with Mixup) ----
    print("\n--- Domain-wise Scorer with Mixup ---")
    scorer = DomainWiseDensityScorer(
        k_source=16,
        k_target=9,
        k_score=5,
        metric="euclidean",
        n_mix_support=3,  # 每个 target 样本使用 3 个最近的 source 样本进行 Mixup
        alpha=0.90,
    )
    scores = scorer.fit_score(train_features, domain_labels, test_features)

    # 详细分数
    final_scores, source_scores, target_scores = scorer.score_with_details(test_features)

    metrics = compute_all_metrics(y_true, final_scores, max_fpr=0.1)

    print(f"\n异常分数统计:")
    print(f"  正常样本 (final): mean={final_scores[:50].mean():.4f}, std={final_scores[:50].std():.4f}")
    print(f"  异常样本 (final): mean={final_scores[50:].mean():.4f}, std={final_scores[50:].std():.4f}")
    print(f"  正常样本 (source): mean={source_scores[:50].mean():.4f}")
    print(f"  异常样本 (source): mean={source_scores[50:].mean():.4f}")
    print(f"  正常样本 (target): mean={target_scores[:50].mean():.4f}")
    print(f"  异常样本 (target): mean={target_scores[50:].mean():.4f}")

    print(f"\n评估指标:")
    print(f"  AUC:  {metrics['AUC']:.4f}")
    print(f"  pAUC: {metrics['pAUC']:.4f}")

    # 内存库统计
    stats = scorer.get_memory_bank_stats()
    print(f"\n内存库统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ---- [新增] 单样本在线推理测试 ----
    print("\n--- 单样本在线推理测试 (验证 Z-Score 坍塌修复) ---")
    single_sample = test_features[0:1]  # batch_size=1
    single_score = scorer.score(single_sample)
    print(f"  单样本原始分数: {single_score[0]:.6f}")
    print(f"  单样本分数不为 0: {abs(single_score[0]) > 1e-6}")

    # 与批量推理对比
    batch_scores = scorer.score(test_features[:10])
    print(f"  批量推理前 10 个样本分数: {batch_scores[:5]}")
    print(f"  单样本与批量推理一致: {np.allclose(single_score[0], batch_scores[0])}")

    # ---- 传统 KNN 对比 ----
    print("\n--- 传统 KNN Scorer (对比) ---")
    knn_scorer = KNNScorer(k=5, metric="euclidean")
    knn_scores = knn_scorer.fit_score(train_features, test_features)
    knn_metrics = compute_all_metrics(y_true, knn_scores, max_fpr=0.1)

    print(f"\n评估指标:")
    print(f"  AUC:  {knn_metrics['AUC']:.4f}")
    print(f"  pAUC: {knn_metrics['pAUC']:.4f}")
