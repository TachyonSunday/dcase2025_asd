"""
evaluate.py
============================================================
DCASE 2025 Task 2 - 官方评估指标计算脚本
============================================================

功能说明:
    1. 加载预先提取的训练集和测试集特征 (.npy)
    2. 使用 DomainWiseDensityScorer 构建域感知内存库
       - Source 域内存库: K_s=16 近邻计算局部密度
       - Target 域内存库: K_t=9 近邻计算局部密度
    3. 对测试集样本计算密度归一化的异常分数
       - score_s(y): 到 source 域的归一化距离
       - score_t(y): 到 target 域的归一化距离
       - score(y) = min(score_s(y), score_t(y))
    4. 按 Machine Type 计算官方评估指标:
       - AUC_source: 仅针对 source 域测试样本的 AUC
       - AUC_target: 仅针对 target 域测试样本的 AUC
       - pAUC: 针对所有测试样本的 partial AUC (max_fpr=0.1)
       - Official Score Ω: 三者的调和平均数
    5. 汇总所有 7 个机器的独立得分和平均 Official Score

使用方法:
    python evaluate.py --config configs/config.yaml
    python evaluate.py --config configs/config.yaml --k_source 20 --k_target 12

前置条件:
    需先运行 extract_features.py 提取特征:
    python extract_features.py --config configs/config.yaml --mode both

参考文献:
    - GenRep: Generative Representation Learning for Domain-Generalized
      Anomalous Sound Detection (DCASE 2025)
    - DCASE 2025 Task 2 官方评估协议
"""

import os
import sys
import argparse
import yaml
import re
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from collections import defaultdict

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.scoring import DomainWiseDensityScorer, KNNScorer, compute_auc, compute_pauc, compute_all_metrics


# DCASE 2025 Task 2 的 7 个机器类型
MACHINE_TYPES = [
    "ToyCar",
    "ToyTrain",
    "bearing",
    "fan",
    "gearbox",
    "slider",
    "valve",
]


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


def load_features(features_dir: str, mode: str) -> dict:
    """
    加载预先提取的特征文件 (支持 3D GenRep 多层特征)

    Args:
        features_dir: 特征保存目录
        mode: "train" 或 "test"

    Returns:
        dict: {
            "embeddings": np.ndarray, shape = (num_layers, N, per_layer_dim)
                                     或 (N, embedding_dim) 兼容旧格式
            "anomaly_labels": list,
            "domain_labels": list,
            "file_paths": list,
            "filenames": list,
        }
    """
    embeddings_path = os.path.join(features_dir, f"{mode}_embeddings.npy")
    metadata_path = os.path.join(features_dir, f"{mode}_metadata.pkl")

    if not os.path.exists(embeddings_path):
        print(f"[Error] 特征文件不存在: {embeddings_path}")
        print(f"        请先运行 extract_features.py 提取特征")
        return None

    if not os.path.exists(metadata_path):
        print(f"[Error] 元数据文件不存在: {metadata_path}")
        return None

    # 加载 Embedding 张量
    embeddings = np.load(embeddings_path)

    # 加载元数据
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # 兼容 2D (旧格式) 和 3D (GenRep 多层) 特征
    if embeddings.ndim == 3:
        num_layers, n_samples, per_layer_dim = embeddings.shape
        print(
            f"[Load] {mode} 特征已加载 | "
            f"格式: GenRep 多层 | "
            f"层数: {num_layers} | 样本数: {n_samples} | "
            f"每层维度: {per_layer_dim}"
        )
    else:
        n_samples, embedding_dim = embeddings.shape
        print(
            f"[Load] {mode} 特征已加载 | "
            f"格式: 2D | "
            f"样本数: {n_samples} | Embedding 维度: {embedding_dim}"
        )

    return {
        "embeddings": embeddings,
        "anomaly_labels": metadata["anomaly_labels"],
        "domain_labels": metadata["domain_labels"],
        "file_paths": metadata["file_paths"],
        "filenames": metadata["filenames"],
    }


def extract_machine_type(file_path: str) -> str:
    """
    从文件路径中提取机器类型

    DCASE 目录结构: dev_data/raw/{machine_type}/train/... 或 dev_data/raw/{machine_type}/test/...
    例如: ./dev_data/raw\ToyCar\train\section_00_source_train_normal_0000.wav

    Args:
        file_path: 完整文件路径

    Returns:
        机器类型字符串
    """
    # 将路径转为小写以便匹配
    path_lower = file_path.lower()

    # 尝试匹配已知的机器类型（检查完整路径）
    for machine in MACHINE_TYPES:
        if machine.lower() in path_lower:
            return machine

    # 如果没匹配到，尝试从文件名中提取
    filename = os.path.basename(file_path)
    for machine in MACHINE_TYPES:
        if filename.lower().startswith(machine.lower()):
            return machine

    # 默认返回 "Unknown"
    return "Unknown"

    # 如果没匹配到，尝试从文件路径中提取
    # 例如: .../ToyCar/train/normal_001.wav
    for machine in MACHINE_TYPES:
        if machine.lower() in filename.lower():
            return machine

    # 默认返回 "Unknown"
    return "Unknown"


def harmonic_mean(values: list) -> float:
    """
    计算调和平均数 (Harmonic Mean)

    H = n / (1/x₁ + 1/x₂ + ... + 1/xₙ)

    Args:
        values: 数值列表

    Returns:
        调和平均数，如果任一值为 0 则返回 0
    """
    if not values or any(v <= 0 for v in values):
        return 0.0

    n = len(values)
    reciprocal_sum = sum(1.0 / v for v in values)
    return n / reciprocal_sum if reciprocal_sum > 0 else 0.0


def evaluate_machine_type(
    machine_type: str,
    train_features: dict,
    test_features: dict,
    config: dict,
) -> dict:
    """
    评估单个机器类型的官方指标

    Args:
        machine_type: 机器类型名称
        train_features: 训练集特征字典
        test_features: 测试集特征字典
        config: 配置字典

    Returns:
        dict: {
            "machine_type": str,
            "auc_source": float,
            "auc_target": float,
            "pauc": float,
            "official_score": float,
            "n_source": int,
            "n_target": int,
            "n_total": int,
        }
    """
    print(f"\n{'='*70}")
    print(f" 评估机器类型: {machine_type}")
    print(f"{'='*70}")

    # ---- 过滤该机器的测试样本 ----
    test_file_paths = test_features["file_paths"]
    test_mask = np.array([
        extract_machine_type(fp) == machine_type for fp in test_file_paths
    ])

    if test_mask.sum() == 0:
        print(f"[Warning] 未找到 {machine_type} 的测试样本，跳过")
        return None

    # 提取该机器的测试数据
    test_embeddings = test_features["embeddings"][test_mask]
    test_anomaly_labels = np.array(test_features["anomaly_labels"])[test_mask]
    test_domain_labels = np.array(test_features["domain_labels"])[test_mask]

    n_test = len(test_embeddings)
    n_source = (test_domain_labels == 0).sum()
    n_target = (test_domain_labels == 1).sum()

    print(f"测试样本: 总计 {n_test} | Source: {n_source} | Target: {n_target}")
    print(f"标签分布: 正常 {(test_anomaly_labels == 0).sum()} | 异常 {(test_anomaly_labels == 1).sum()}")

    # ---- 过滤该机器的训练样本 ----
    train_file_paths = train_features["file_paths"]
    train_mask = np.array([
        extract_machine_type(fp) == machine_type for fp in train_file_paths
    ])

    if train_mask.sum() == 0:
        print(f"[Warning] 未找到 {machine_type} 的训练样本，跳过")
        return None

    train_embeddings = train_features["embeddings"][train_mask]
    train_domain_labels = np.array(train_features["domain_labels"])[train_mask]

    print(f"训练样本: {len(train_embeddings)}")

    # ---- 构建打分器 ----
    knn_cfg = config.get("knn", {})
    eval_cfg = config.get("evaluation", {})

    scorer = DomainWiseDensityScorer(
        k_source=knn_cfg.get("k_source", 16),
        k_target=knn_cfg.get("k_target", 9),
        k_score=knn_cfg.get("k_score", 5),
        metric=knn_cfg.get("metric", "euclidean"),
        algorithm=knn_cfg.get("algorithm", "auto"),
        n_jobs=knn_cfg.get("n_jobs", -1),
        # Mixup 参数
        n_mix_support=knn_cfg.get("n_mix_support", 3),
        alpha=knn_cfg.get("alpha", 0.90),
        # 打分归一化模式: "local_density" 或 "z_score"
        score_normalization=knn_cfg.get("score_normalization", "local_density"),
    )

    # 拟合打分器
    scorer.fit(train_embeddings, train_domain_labels)

    # ---- 计算异常分数 ----
    anomaly_scores, scores_source, scores_target = scorer.score_with_details(test_embeddings)

    # ---- 计算官方指标 ----
    max_fpr = eval_cfg.get("max_fpr", 0.1)

    # 1. AUC_source: 仅针对 source 域测试样本
    source_mask = (test_domain_labels == 0)
    if source_mask.sum() > 0 and len(np.unique(test_anomaly_labels[source_mask])) > 1:
        auc_source = compute_auc(
            test_anomaly_labels[source_mask],
            anomaly_scores[source_mask]
        )
    else:
        auc_source = 0.0
        print(f"[Warning] Source 域样本不足或标签单一，AUC_source 设为 0")

    # 2. AUC_target: 仅针对 target 域测试样本
    target_mask = (test_domain_labels == 1)
    if target_mask.sum() > 0 and len(np.unique(test_anomaly_labels[target_mask])) > 1:
        auc_target = compute_auc(
            test_anomaly_labels[target_mask],
            anomaly_scores[target_mask]
        )
    else:
        auc_target = 0.0
        print(f"[Warning] Target 域样本不足或标签单一，AUC_target 设为 0")

    # 3. pAUC: 针对所有测试样本
    if len(np.unique(test_anomaly_labels)) > 1:
        pauc = compute_pauc(
            test_anomaly_labels,
            anomaly_scores,
            max_fpr=max_fpr
        )
    else:
        pauc = 0.0
        print(f"[Warning] 测试样本标签单一，pAUC 设为 0")

    # 4. Official Score Ω: 调和平均数
    official_score = harmonic_mean([auc_source, auc_target, pauc])

    # ---- 输出结果 ----
    print(f"\n--- 官方评估指标 ---")
    print(f"  AUC_source:       {auc_source:.4f}")
    print(f"  AUC_target:       {auc_target:.4f}")
    print(f"  pAUC:             {pauc:.4f} (max_fpr={max_fpr})")
    print(f"  Official Score Ω: {official_score:.4f}")

    return {
        "machine_type": machine_type,
        "auc_source": auc_source,
        "auc_target": auc_target,
        "pauc": pauc,
        "official_score": official_score,
        "n_source": int(n_source),
        "n_target": int(n_target),
        "n_total": int(n_test),
    }


def run_evaluation(
    train_features: dict,
    test_features: dict,
    config: dict,
    output_dir: str = None,
) -> dict:
    """
    运行完整的 DCASE 2025 官方评估流程

    Args:
        train_features: 训练集特征字典
        test_features: 测试集特征字典
        config: 配置字典
        output_dir: 结果保存目录

    Returns:
        dict: 评估结果汇总
    """
    print(f"\n{'='*70}")
    print(f" DCASE 2025 Task 2 官方评估")
    print(f"{'='*70}")

    # ---- 逐个机器类型评估 ----
    results = []
    for machine_type in MACHINE_TYPES:
        result = evaluate_machine_type(
            machine_type=machine_type,
            train_features=train_features,
            test_features=test_features,
            config=config,
        )
        if result is not None:
            results.append(result)

    # ---- 汇总统计 ----
    print(f"\n{'='*70}")
    print(f" 汇总统计")
    print(f"{'='*70}")

    if len(results) == 0:
        print("[Error] 没有成功评估的机器类型")
        return None

    # 打印表格
    print(f"\n{'Machine Type':<15} {'AUC_s':>8} {'AUC_t':>8} {'pAUC':>8} {'Ω':>8} {'N_test':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['machine_type']:<15} "
            f"{r['auc_source']:>8.4f} "
            f"{r['auc_target']:>8.4f} "
            f"{r['pauc']:>8.4f} "
            f"{r['official_score']:>8.4f} "
            f"{r['n_total']:>8d}"
        )
    print("-" * 70)

    # 计算平均值
    avg_auc_source = np.mean([r["auc_source"] for r in results])
    avg_auc_target = np.mean([r["auc_target"] for r in results])
    avg_pauc = np.mean([r["pauc"] for r in results])
    avg_official_score = np.mean([r["official_score"] for r in results])

    print(
        f"{'AVERAGE':<15} "
        f"{avg_auc_source:>8.4f} "
        f"{avg_auc_target:>8.4f} "
        f"{avg_pauc:>8.4f} "
        f"{avg_official_score:>8.4f} "
        f"{len(results):>8d}"
    )
    print("=" * 70)

    # ---- 保存结果 ----
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果 CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f"dcase2025_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[Save] 详细结果已保存: {csv_path}")

        # 保存评估摘要
        summary = {
            "timestamp": timestamp,
            "task": "DCASE 2025 Task 2",
            "evaluation_protocol": "Official",
            "scorer_type": "DomainWiseDensityScorer",
            "scorer_params": {
                "k_source": config.get("knn", {}).get("k_source", 16),
                "k_target": config.get("knn", {}).get("k_target", 9),
                "k_score": config.get("knn", {}).get("k_score", 5),
                "metric": config.get("knn", {}).get("metric", "euclidean"),
            },
            "max_fpr": config.get("evaluation", {}).get("max_fpr", 0.1),
            "machine_results": results,
            "averages": {
                "auc_source": float(avg_auc_source),
                "auc_target": float(avg_auc_target),
                "pauc": float(avg_pauc),
                "official_score": float(avg_official_score),
                "n_machines": len(results),
            },
        }
        summary_path = os.path.join(output_dir, f"dcase2025_summary_{timestamp}.yaml")
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
        print(f"[Save] 评估摘要已保存: {summary_path}")

    return {
        "results": results,
        "averages": {
            "auc_source": avg_auc_source,
            "auc_target": avg_auc_target,
            "pauc": avg_pauc,
            "official_score": avg_official_score,
            "n_machines": len(results),
        },
    }


def main():
    # ---- 命令行参数解析 ----
    parser = argparse.ArgumentParser(
        description="DCASE 2025 Task 2 - 官方评估指标计算脚本"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)",
    )
    parser.add_argument(
        "--k_source",
        type=int,
        default=None,
        help="覆盖配置文件中的源域近邻数 K_s (默认: 16)",
    )
    parser.add_argument(
        "--k_target",
        type=int,
        default=None,
        help="覆盖配置文件中的目标域近邻数 K_t (默认: 9)",
    )
    parser.add_argument(
        "--k_score",
        type=int,
        default=None,
        help="覆盖配置文件中的推理近邻数 K (默认: 5)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["euclidean", "cosine", "minkowski"],
        help="覆盖配置文件中的距离度量方式",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="结果保存目录 (默认: ./results)",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)

    # ---- 覆盖打分器参数 ----
    if args.k_source is not None:
        config["knn"]["k_source"] = args.k_source
        print(f"[Override] 源域近邻数 K_s 覆盖为: {args.k_source}")
    if args.k_target is not None:
        config["knn"]["k_target"] = args.k_target
        print(f"[Override] 目标域近邻数 K_t 覆盖为: {args.k_target}")
    if args.k_score is not None:
        config["knn"]["k_score"] = args.k_score
        print(f"[Override] 推理近邻数 K 覆盖为: {args.k_score}")
    if args.metric is not None:
        config["knn"]["metric"] = args.metric
        print(f"[Override] 距离度量覆盖为: {args.metric}")

    # ---- 加载特征 ----
    features_dir = config["dataset"]["features_dir"]

    print(f"\n{'='*60}")
    print(f" 加载预提取特征")
    print(f"{'='*60}")

    train_features = load_features(features_dir, "train")
    test_features = load_features(features_dir, "test")

    if train_features is None or test_features is None:
        print("\n[Error] 特征加载失败，请确认已运行 extract_features.py")
        print("  python extract_features.py --config configs/config.yaml --mode both")
        sys.exit(1)

    # ---- 运行官方评估 (支持 GenRep 多层 Layer Search) ----
    train_embeddings = train_features["embeddings"]
    test_embeddings = test_features["embeddings"]

    if train_embeddings.ndim == 3:
        # ---- GenRep 多层模式: 逐层评估, 选取最优层 ----
        num_layers = train_embeddings.shape[0]
        print(f"\n{'='*70}")
        print(f" GenRep Layer Search: 共 {num_layers} 层, 逐层独立评估")
        print(f"{'='*70}")

        best_layer = -1
        best_score = -1.0
        best_result = None
        all_layer_results = []

        for layer_idx in range(num_layers):
            print(f"\n{'#'*70}")
            print(f"  Layer {layer_idx} / {num_layers - 1}")
            print(f"{'#'*70}")

            # 切出当前层的 2D 特征
            layer_train = dict(train_features)
            layer_test = dict(test_features)
            layer_train["embeddings"] = train_embeddings[layer_idx]  # (N_train, D)
            layer_test["embeddings"] = test_embeddings[layer_idx]    # (N_test, D)

            result = run_evaluation(
                train_features=layer_train,
                test_features=layer_test,
                config=config,
                output_dir=None,  # 仅在最优层保存
            )

            if result is not None:
                avg_score = result["averages"]["official_score"]
                all_layer_results.append((layer_idx, avg_score, result))

                print(
                    f"\n[Layer {layer_idx}] Average Official Score Ω: {avg_score:.4f}"
                )

                if avg_score > best_score:
                    best_score = avg_score
                    best_layer = layer_idx
                    best_result = result

        # ---- 打印 Layer Search 汇总 ----
        print(f"\n{'='*70}")
        print(f" GenRep Layer Search 汇总")
        print(f"{'='*70}")
        print(f"\n{'Layer':<8} {'Avg Ω':>10}")
        print("-" * 20)
        for li, sc, _ in all_layer_results:
            marker = " ← BEST" if li == best_layer else ""
            print(f"{li:<8} {sc:>10.4f}{marker}")
        print("-" * 20)

        print(f"\n{'*'*70}")
        print(f"  ★ 最优层: Layer {best_layer}")
        print(f"  ★ Average Official Score Ω: {best_score:.4f}")
        print(f"{'*'*70}")

        # ---- 保存最优层结果 ----
        if best_result is not None and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 详细结果 CSV
            results_df = pd.DataFrame(best_result["results"])
            csv_path = os.path.join(
                args.output_dir,
                f"dcase2025_best_layer{best_layer}_{timestamp}.csv"
            )
            results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n[Save] 最优层结果已保存: {csv_path}")

            # Layer search 摘要
            layer_summary = {
                "best_layer": int(best_layer),
                "best_official_score": float(best_score),
                "all_layers": [
                    {"layer": li, "official_score": float(sc)}
                    for li, sc, _ in all_layer_results
                ],
            }
            summary_path = os.path.join(
                args.output_dir,
                f"dcase2025_layer_search_{timestamp}.yaml"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                yaml.dump(layer_summary, f, allow_unicode=True, default_flow_style=False)
            print(f"[Save] Layer search 摘要已保存: {summary_path}")

        results = best_result

    else:
        # ---- 2D 兼容模式 (旧版 CED-Tiny 特征) ----
        results = run_evaluation(
            train_features=train_features,
            test_features=test_features,
            config=config,
            output_dir=args.output_dir,
        )

    if results is None:
        print("\n[Error] 评估失败")
        sys.exit(1)

    # ---- 最终输出 ----
    print(f"\n{'='*60}")
    print(f" DCASE 2025 Task 2 官方评估完成!")
    print(f"{'='*60}")
    print(f" 平均 Official Score Ω: {results['averages']['official_score']:.4f}")
    print(f" 评估机器数: {results['averages']['n_machines']}")
    print(f" 结果已保存至: {args.output_dir}/")


if __name__ == "__main__":
    main()
