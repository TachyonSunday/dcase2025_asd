#!/usr/bin/env python3
"""
generate_submission.py
============================================================
DCASE 2025 Task 2 - 官方盲测提交文件生成脚本 (BEATs GenRep 版本)
============================================================

功能说明:
    1. 加载预提取的 add_data 训练特征和 test 评估特征
       - 支持 GenRep 3D 多层特征 [num_layers, N, 6144]
       - 通过 --best_layer 参数选择最优层
    2. 针对 8 种新机器分别训练 DomainWiseDensityScorer
       - 使用 local_density 归一化 (GenRep)
    3. 对 test 数据进行异常分数预测
    4. 生成 DCASE 官方提交格式的 CSV 文件

使用方法:
    # 使用 evaluate.py 输出的最优层 (默认 layer=4)
    python generate_submission.py --best_layer 4

    # 指定特征目录
    python generate_submission.py --best_layer 5 --features_dir ./features_eval

输出:
    ./submission/anomaly_score_<machine_type>_section_00.csv
    - 无表头
    - 每行格式: [filename],[score]
    - 例如: section_00_0000.wav,0.123456
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.scoring import DomainWiseDensityScorer


# DCASE 2025 评估集的 8 种新机器类型
MACHINE_TYPES = [
    "AutoTrash",
    "BandSealer",
    "CoffeeGrinder",
    "HomeCamera",
    "Polisher",
    "ScrewFeeder",
    "ToyPet",
    "ToyRCCar"
]


def load_features(features_dir, mode):
    """
    加载预提取的特征和元数据 (支持 GenRep 3D 多层特征)

    Args:
        features_dir: 特征目录路径
        mode: "train" 或 "test"

    Returns:
        embeddings: numpy array
            - 3D: shape = (num_layers, N, per_layer_dim)  GenRep 多层格式
            - 2D: shape = (N, D)                         兼容旧格式
        metadata: dict, 包含 file_paths, domain_labels 等
    """
    embeddings_path = os.path.join(features_dir, f"{mode}_embeddings.npy")
    metadata_path = os.path.join(features_dir, f"{mode}_metadata.pkl")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"特征文件不存在: {embeddings_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

    print(f"[Load] 正在加载 {mode} 特征...")
    embeddings = np.load(embeddings_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # 兼容 2D 和 3D 格式
    if embeddings.ndim == 3:
        num_layers, n_samples, per_layer_dim = embeddings.shape
        print(f"[Load] {mode} 特征已加载 (GenRep 3D):")
        print(f"  层数: {num_layers}")
        print(f"  样本数: {n_samples}")
        print(f"  每层维度: {per_layer_dim}")
    else:
        n_samples, embedding_dim = embeddings.shape
        print(f"[Load] {mode} 特征已加载 (2D):")
        print(f"  样本数: {n_samples}")
        print(f"  特征维度: {embedding_dim}")

    print(f"  元数据键: {list(metadata.keys())}")

    return embeddings, metadata


def filter_by_machine(embeddings, metadata, machine_type):
    """
    根据机器类型过滤样本

    Args:
        embeddings: 全部特征, shape = (N, D)
        metadata: 全部元数据, 包含 file_paths 等列表
        machine_type: 机器类型名称 (如 "AutoTrash")

    Returns:
        filtered_embeddings: 过滤后的特征
        filtered_metadata: 过滤后的元数据
    """
    file_paths = metadata['file_paths']

    # 找出包含该机器名称的索引
    indices = [i for i, path in enumerate(file_paths) if machine_type in path]

    if len(indices) == 0:
        return None, None

    # 过滤特征
    filtered_embeddings = embeddings[indices]

    # 过滤元数据（处理列表和标量）
    filtered_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            filtered_metadata[key] = [value[i] for i in indices]
        else:
            filtered_metadata[key] = value

    return filtered_embeddings, filtered_metadata


def generate_submission(args):
    """生成 DCASE 2025 官方提交文件"""

    print("="*70)
    print("DCASE 2025 Task 2 - 官方盲测提交文件生成 (BEATs GenRep)")
    print("="*70)

    # 1. 加载配置
    config_path = args.config
    print(f"\n[Config] 加载配置: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取特征目录
    features_dir = args.features_dir or \
                   config.get('dataset', {}).get('features_dir', './features_eval')

    output_dir = args.output_dir

    print(f"[Config] 特征目录: {features_dir}")
    print(f"[Config] 输出目录: {output_dir}")
    print(f"[Config] 最优层: Layer {args.best_layer}")

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 加载全部特征
    print("\n" + "="*70)
    print("步骤 1: 加载特征数据")
    print("="*70)

    try:
        train_embeddings_raw, train_metadata = load_features(features_dir, 'train')
        test_embeddings_raw, test_metadata = load_features(features_dir, 'test')
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("[Hint] 请先运行特征提取脚本:")
        print(f"  python extract_features.py --config {config_path} --mode both")
        return

    # 4. 从 3D 张量中切出最优层
    print("\n" + "="*70)
    print("步骤 2: 选择最优层")
    print("="*70)

    if train_embeddings_raw.ndim == 3:
        num_layers = train_embeddings_raw.shape[0]
        best_layer = args.best_layer

        if best_layer < 0 or best_layer >= num_layers:
            print(
                f"[Error] best_layer={best_layer} 超出范围 [0, {num_layers - 1}]"
            )
            return

        if test_embeddings_raw.ndim != 3 or test_embeddings_raw.shape[0] != num_layers:
            print(
                f"[Error] 训练集 ({train_embeddings_raw.shape}) 和测试集 "
                f"({test_embeddings_raw.shape}) 层数不匹配"
            )
            return

        # 切出最优层 → 2D: (N, per_layer_dim)
        train_embeddings = train_embeddings_raw[best_layer]
        test_embeddings = test_embeddings_raw[best_layer]

        print(f"  GenRep 3D 特征: {train_embeddings_raw.shape}")
        print(f"  选择 Layer {best_layer} / {num_layers - 1}")
        print(f"  训练集切片: {train_embeddings.shape}")
        print(f"  测试集切片: {test_embeddings.shape}")

    else:
        # 2D 兼容模式 (旧版特征)
        print(f"  2D 兼容模式, 跳过层选择")
        train_embeddings = train_embeddings_raw
        test_embeddings = test_embeddings_raw

    # 5. 遍历每种机器类型
    print("\n" + "="*70)
    print("步骤 3: 生成提交文件")
    print("="*70)

    success_count = 0
    skip_count = 0

    for machine_type in MACHINE_TYPES:
        print(f"\n{'─'*70}")
        print(f"[Machine] {machine_type}")
        print(f"{'─'*70}")

        # 5.1 过滤该机器的样本
        train_emb_machine, train_meta_machine = filter_by_machine(
            train_embeddings, train_metadata, machine_type
        )
        test_emb_machine, test_meta_machine = filter_by_machine(
            test_embeddings, test_metadata, machine_type
        )

        # 检查是否找到样本
        if train_emb_machine is None or train_emb_machine.shape[0] == 0:
            print(f"  [Warning] 未找到 {machine_type} 的训练样本，跳过")
            skip_count += 1
            continue

        if test_emb_machine is None or test_emb_machine.shape[0] == 0:
            print(f"  [Warning] 未找到 {machine_type} 的测试样本，跳过")
            skip_count += 1
            continue

        print(f"  训练样本数: {train_emb_machine.shape[0]}")
        print(f"  测试样本数: {test_emb_machine.shape[0]}")

        # 5.2 获取该机器的 domain_labels
        if 'domain_labels' in train_meta_machine:
            train_domain_labels = np.array(train_meta_machine['domain_labels'])
            print(f"  训练域分布: source={np.sum(train_domain_labels == 0)}, "
                  f"target={np.sum(train_domain_labels == 1)}")
        else:
            train_domain_labels = np.zeros(train_emb_machine.shape[0], dtype=int)
            print(f"  训练域: 全部默认为 source (无 domain_labels 字段)")

        # 5.3 实例化打分器 (读取 score_normalization 配置)
        knn_config = config.get('knn', {})
        score_norm = knn_config.get('score_normalization', 'local_density')

        print(f"  打分器配置:")
        print(f"    k_source: {knn_config.get('k_source', 16)}")
        print(f"    k_target: {knn_config.get('k_target', 9)}")
        print(f"    k_score: {knn_config.get('k_score', 5)}")
        print(f"    metric: {knn_config.get('metric', 'cosine')}")
        print(f"    n_mix_support: {knn_config.get('n_mix_support', 50)}")
        print(f"    alpha: {knn_config.get('alpha', 0.90)}")
        print(f"    score_normalization: {score_norm}")

        scorer = DomainWiseDensityScorer(
            k_source=knn_config.get('k_source', 16),
            k_target=knn_config.get('k_target', 9),
            k_score=knn_config.get('k_score', 5),
            metric=knn_config.get('metric', 'cosine'),
            n_mix_support=knn_config.get('n_mix_support', 50),
            alpha=knn_config.get('alpha', 0.90),
            n_jobs=knn_config.get('n_jobs', -1),
            score_normalization=score_norm,
        )

        # 5.4 训练（建库）
        print(f"  训练打分器...")
        scorer.fit(train_emb_machine, train_domain_labels)

        # 5.5 预测异常分数
        print(f"  预测异常分数...")
        anomaly_scores = scorer.score(test_emb_machine)

        print(f"  分数统计:")
        print(f"    最小值: {anomaly_scores.min():.6f}")
        print(f"    最大值: {anomaly_scores.max():.6f}")
        print(f"    平均值: {anomaly_scores.mean():.6f}")
        print(f"    标准差: {anomaly_scores.std():.6f}")

        # 5.6 生成 CSV 文件
        csv_filename = f"anomaly_score_{machine_type}_section_00.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        test_file_paths = test_meta_machine['file_paths']

        print(f"  生成 CSV: {csv_filename}")

        with open(csv_path, 'w', encoding='utf-8') as f:
            for file_path, score in zip(test_file_paths, anomaly_scores):
                filename = os.path.basename(file_path)
                f.write(f"{filename},{score:.6f}\n")

        print(f"  ✓ 成功: {len(anomaly_scores)} 条记录")
        success_count += 1

    # 6. 完成提示
    print("\n" + "="*70)
    print("完成")
    print("="*70)

    print(f"\n所有提交文件已生成至 {output_dir} 目录")
    print(f"使用层: Layer {args.best_layer}")
    print(f"\n统计:")
    print(f"  成功: {success_count} / {len(MACHINE_TYPES)}")
    print(f"  跳过: {skip_count} / {len(MACHINE_TYPES)}")

    print(f"\n生成的文件:")
    for machine_type in MACHINE_TYPES:
        csv_filename = f"anomaly_score_{machine_type}_section_00.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        if os.path.exists(csv_path):
            size = os.path.getsize(csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"  ✓ {csv_filename:40s} {size:8d} bytes  {line_count:4d} 条")
        else:
            print(f"  ✗ {csv_filename:40s} (未生成)")

    print("\n" + "="*70)
    print("下一步: 打包提交")
    print("="*70)
    print(f"\n将 {output_dir} 目录下的所有 CSV 文件打包为 ZIP:")
    print(f"  cd {output_dir}")
    print(f"  zip ../dcase2025_submission.zip *.csv")
    print("\n然后上传至 DCASE 2025 官方提交系统:")
    print("  https://dcase.community/challenge2025/task-unsupervised-anomalous-sound-detection")
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCASE 2025 Task 2 - 盲测提交文件生成 (BEATs GenRep)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)",
    )
    parser.add_argument(
        "--best_layer",
        type=int,
        default=4,
        help="GenRep 最优层索引 (0-indexed, 默认: 4 = 第5层, BEATs 典型最优层)",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="特征目录 (默认: 从 config.yaml 读取 dataset.features_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./submission",
        help="提交文件输出目录 (默认: ./submission)",
    )
    args = parser.parse_args()

    generate_submission(args)
