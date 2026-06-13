"""
extract_features.py
============================================================
DCASE 2025 Task 2 - 特征提取脚本 (BEATs GenRep 版本)
============================================================

功能说明:
    1. 加载配置文件
    2. 扫描训练集/测试集目录
    3. 通过 BEATs 特征提取器提取多层 Embedding
       - 输出 shape: [num_layers, N, 6144]
       - 11 层 encoder 特征, 每层 6144 维 (GenRep 时序池化)
    4. 将特征、标签 (anomaly + domain)、文件路径保存为 .npy / .pkl 文件
    5. 这些特征将用于后续 Layer Search + Domain-wise Local Density Scorer

使用方法:
    python extract_features.py --config configs/config.yaml --mode train
    python extract_features.py --config configs/config.yaml --mode test
    python extract_features.py --config configs/config.yaml --mode both
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle
import torch

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import AudioAnomalyDataset, create_dataloader
from models.feature_extractor import get_feature_extractor


def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[Config] 已加载配置文件: {config_path}")
    return config


def extract_features_for_dataset(
    config: dict,
    mode: str,
    device: str,
) -> dict:
    """
    对指定模式的数据集提取 BEATs 多层特征

    Args:
        config: 配置字典
        mode: "train" 或 "test"
        device: 计算设备

    Returns:
        dict: {
            "embeddings": np.ndarray, shape = (num_layers, N, 6144),
            "anomaly_labels": list[int],
            "domain_labels": list[int],
            "file_paths": list[str],
            "filenames": list[str],
        }
    """
    dataset_cfg = config["dataset"]

    # ---- 根据模式选择目录 ----
    if mode == "train":
        audio_dir = dataset_cfg["train_dir"]
    else:
        audio_dir = dataset_cfg["test_dir"]

    if not os.path.exists(audio_dir):
        print(f"[Error] 目录不存在: {audio_dir}")
        print(f"        请将音频文件放入对应目录后再运行")
        return None

    # ---- 创建 DataLoader ----
    print(f"\n{'='*60}")
    print(f"正在加载 {mode} 数据集: {audio_dir}")
    print(f"{'='*60}")

    dataloader = create_dataloader(
        audio_dir=audio_dir,
        config=config,
        mode=mode,
        shuffle=False,  # 特征提取时不打乱顺序
    )

    if len(dataloader.dataset) == 0:
        print(f"[Warning] {mode} 数据集为空，跳过")
        return None

    # ---- 创建特征提取器 ----
    model = get_feature_extractor(config)
    model = model.to(device)
    model.eval()

    # ---- 使用 get_embeddings 批量提取 ----
    # 返回: (embeddings, anomaly_labels, domain_labels, file_paths)
    # embeddings shape: (num_layers, N, per_layer_dim) = (11, N, 6144)
    print(f"\n[Extract] 开始提取 BEATs 多层特征 (共 {len(dataloader.dataset)} 个样本)...")

    embeddings, anomaly_labels, domain_labels, file_paths = model.get_embeddings(
        dataloader, device=device
    )

    # ---- 从 dataset 获取文件名列表 (保持顺序一致) ----
    filenames = [os.path.basename(fp) for fp in file_paths]

    print(f"\n[Extract] 特征提取完成!")
    print(f"  Embeddings shape: {embeddings.shape}  (num_layers, N, per_layer_dim)")
    print(f"  正常样本: {anomaly_labels.count(0)}")
    print(f"  异常样本: {anomaly_labels.count(1)}")
    print(f"  Source 域: {domain_labels.count(0)}")
    print(f"  Target 域: {domain_labels.count(1)}")

    return {
        "embeddings": embeddings,
        "anomaly_labels": anomaly_labels,
        "domain_labels": domain_labels,
        "file_paths": file_paths,
        "filenames": filenames,
    }


def save_features(
    features_dict: dict,
    mode: str,
    features_dir: str,
) -> None:
    """
    保存提取的特征到本地文件

    保存格式:
        - {mode}_embeddings.npy:  3D 特征张量 [num_layers, N, per_layer_dim]
        - {mode}_metadata.pkl:    元数据（标签、路径等）

    Args:
        features_dict: extract_features_for_dataset 返回的字典
        mode: "train" 或 "test"
        features_dir: 特征保存目录
    """
    os.makedirs(features_dir, exist_ok=True)

    embeddings = features_dict["embeddings"]

    # 保存 3D Embedding 张量 (.npy)
    # shape: (num_layers, N, per_layer_dim) e.g. (11, N, 6144)
    embeddings_path = os.path.join(features_dir, f"{mode}_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"[Save] Embeddings 已保存: {embeddings_path}")
    print(f"       shape: {embeddings.shape} (num_layers, N, per_layer_dim)")

    # 保存元数据 (.pkl)
    metadata = {
        "anomaly_labels": features_dict["anomaly_labels"],
        "domain_labels": features_dict["domain_labels"],
        "file_paths": features_dict["file_paths"],
        "filenames": features_dict["filenames"],
        "num_samples": len(features_dict["anomaly_labels"]),
        "num_layers": embeddings.shape[0],
        "per_layer_dim": embeddings.shape[2],
        "feature_format": "beats_genrep_multilayer",
    }
    metadata_path = os.path.join(features_dir, f"{mode}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[Save] 元数据已保存: {metadata_path}")


def main():
    # ---- 命令行参数解析 ----
    parser = argparse.ArgumentParser(
        description="DCASE 2025 Task 2 - 特征提取脚本"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="提取模式: train/test/both (默认: both)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备: cuda/cpu/mps (默认: 自动检测)",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)

    # ---- 确定设备 ----
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[Device] 使用设备: {device}")

    features_dir = config["dataset"]["features_dir"]

    # ---- 提取训练集特征 ----
    if args.mode in ("train", "both"):
        print("\n" + "=" * 60)
        print(" 阶段 1: 提取训练集特征 (仅正常样本)")
        print("=" * 60)
        train_features = extract_features_for_dataset(config, "train", device)
        if train_features is not None:
            save_features(train_features, "train", features_dir)

    # ---- 提取测试集特征 ----
    if args.mode in ("test", "both"):
        print("\n" + "=" * 60)
        print(" 阶段 2: 提取测试集特征 (正常 + 异常样本)")
        print("=" * 60)
        test_features = extract_features_for_dataset(config, "test", device)
        if test_features is not None:
            save_features(test_features, "test", features_dir)

    print("\n" + "=" * 60)
    print(" 特征提取流程全部完成!")
    print("=" * 60)
    print(f"\n后续步骤: 运行 evaluate.py 进行 KNN 异常打分与评估")
    print(f"  python evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
